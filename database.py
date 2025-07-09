import psycopg
from config import DB_PARAMS
import logging
from datetime import datetime
import random
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_db():
    try:
        conn = psycopg.connect(**DB_PARAMS)
        logger.info("Database connection established")
        return conn
    except psycopg.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def init_database(conn):
    try:
        with conn.cursor() as cur:
            # Check if each table exists
            tables = [
                'employees', 'sessions', 'emotion_details', 
                'consolidated_emotion_data', 'feedback_users', 'feedback_responses'
            ]
            existing_tables = []
            for table in tables:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table,))
                if cur.fetchone()[0]:
                    existing_tables.append(table)
            
            # Create or validate tables
            if 'employees' not in existing_tables:
                logger.info("Creating employees table")
                cur.execute("""
                    CREATE TABLE employees (
                        employee_id VARCHAR(50) PRIMARY KEY,
                        department VARCHAR(50) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
            if 'sessions' not in existing_tables:
                logger.info("Creating sessions table")
                cur.execute("""
                    CREATE TABLE sessions (
                        id SERIAL PRIMARY KEY,
                        employee_id VARCHAR(50) REFERENCES employees(employee_id),
                        duration_seconds FLOAT NOT NULL,
                        dominant_emotion VARCHAR(20),
                        session_date DATE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
            if 'emotion_details' not in existing_tables:
                logger.info("Creating emotion_details table")
                cur.execute("""
                    CREATE TABLE emotion_details (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES sessions(id),
                        emotion VARCHAR(20) NOT NULL,
                        count INTEGER NOT NULL,
                        percentage FLOAT NOT NULL
                    )
                """)
            
            if 'consolidated_emotion_data' not in existing_tables:
                logger.info("Creating consolidated_emotion_data table")
                cur.execute("""
                    CREATE TABLE consolidated_emotion_data (
                        id SERIAL PRIMARY KEY,
                        employee_id VARCHAR(50) REFERENCES employees(employee_id),
                        department VARCHAR(50) NOT NULL,
                        session_id INTEGER REFERENCES sessions(id),
                        emotion VARCHAR(20) NOT NULL,
                        count INTEGER NOT NULL,
                        percentage FLOAT NOT NULL,
                        session_date DATE NOT NULL,
                        time_stamp TIME NOT NULL
                    )
                """)
            else:
                # Validate and alter consolidated_emotion_data schema
                logger.info("Validating consolidated_emotion_data schema")
                required_columns = {
                    'employee_id': ['VARCHAR(50)', 'CHARACTER VARYING'],
                    'department': ['VARCHAR(50)', 'CHARACTER VARYING'],
                    'session_id': ['INTEGER'],
                    'emotion': ['VARCHAR(20)', 'CHARACTER VARYING'],
                    'count': ['INTEGER'],
                    'percentage': ['FLOAT', 'DOUBLE PRECISION'],
                    'session_date': ['DATE'],
                    'time_stamp': ['TIME', 'TIME WITHOUT TIME ZONE']
                }
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'consolidated_emotion_data'
                """)
                existing_columns = {row[0]: row[1].upper() for row in cur.fetchall()}
                
                for col, expected_types in required_columns.items():
                    if col not in existing_columns:
                        logger.info(f"Adding column {col} to consolidated_emotion_data")
                        col_type = expected_types[0]  # Use first type as default
                        cur.execute(f"""
                            ALTER TABLE consolidated_emotion_data
                            ADD COLUMN {col} {col_type}
                            {' NOT NULL' if col not in ['employee_id', 'session_id'] else ''}
                        """)
                    elif existing_columns[col] not in [t.upper() for t in expected_types]:
                        logger.warning(f"Column {col} has type {existing_columns[col]}, expected one of {expected_types}")
                
                # Add foreign key constraints if missing
                if 'employee_id' in existing_columns:
                    cur.execute("""
                        SELECT constraint_name 
                        FROM information_schema.table_constraints 
                        WHERE table_name = 'consolidated_emotion_data' 
                        AND constraint_type = 'FOREIGN KEY'
                        AND constraint_name LIKE '%employee_id%'
                    """)
                    if not cur.fetchone():
                        logger.info("Adding foreign key constraint for employee_id")
                        cur.execute("""
                            ALTER TABLE consolidated_emotion_data
                            ADD CONSTRAINT fk_employee_id
                            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
                        """)
                if 'session_id' in existing_columns:
                    cur.execute("""
                        SELECT constraint_name 
                        FROM information_schema.table_constraints 
                        WHERE table_name = 'consolidated_emotion_data' 
                        AND constraint_type = 'FOREIGN KEY'
                        AND constraint_name LIKE '%session_id%'
                    """)
                    if not cur.fetchone():
                        logger.info("Adding foreign key constraint for session_id")
                        cur.execute("""
                            ALTER TABLE consolidated_emotion_data
                            ADD CONSTRAINT fk_session_id
                            FOREIGN KEY (session_id) REFERENCES sessions(id)
                        """)
            
            if 'feedback_users' not in existing_tables:
                logger.info("Creating feedback_users table")
                cur.execute("""
                    CREATE TABLE feedback_users (
                        emp_id VARCHAR(50) PRIMARY KEY,
                        department VARCHAR(50) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
            if 'feedback_responses' not in existing_tables:
                logger.info("Creating feedback_responses table")
                cur.execute("""
                    CREATE TABLE feedback_responses (
                        id SERIAL PRIMARY KEY,
                        emp_id VARCHAR(50) REFERENCES feedback_users(emp_id),
                        q1 VARCHAR(50) NOT NULL,
                        q2 VARCHAR(50) NOT NULL,
                        q3 VARCHAR(50) NOT NULL,
                        comments TEXT,
                        feedback_date DATE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
            conn.commit()
            logger.info("Database tables verified/created successfully")
            
            # Add sample data only if no employee data exists
            cur.execute("SELECT COUNT(*) FROM employees")
            if cur.fetchone()[0] == 0:
                add_sample_data(conn)
            else:
                logger.info("Employee data already exists, skipping sample data insertion")
                
    except psycopg.Error as e:
        logger.error(f"Database initialization error: {e}")
        conn.rollback()
        raise

def add_sample_data(conn):
    try:
        with conn.cursor() as cur:
            logger.info("Adding sample data...")
            employees = [
                ("EMP001", "IT"),
                ("EMP002", "IT"),
                ("EMP003", "Accounting"),
                ("EMP004", "Accounting"),
                ("EMP005", "Marketing"),
                ("EMP006", "Marketing")
            ]
            for emp_id, dept in employees:
                cur.execute(
                    "INSERT INTO employees (employee_id, department) VALUES (%s, %s)",
                    (emp_id, dept)
                )
                cur.execute(
                    "INSERT INTO feedback_users (emp_id, department) VALUES (%s, %s)",
                    (emp_id, dept)
                )
            
            today = datetime.now().date()
            emotions = ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"]
            for emp_id, _ in employees:
                num_sessions = random.randint(3, 5)
                for _ in range(num_sessions):
                    days_ago = random.randint(0, 29)
                    session_date = today - timedelta(days=days_ago)
                    duration = random.randint(300, 1800)
                    dominant_emotion = random.choice(emotions)
                    cur.execute(
                        """
                        INSERT INTO sessions 
                        (employee_id, duration_seconds, dominant_emotion, session_date) 
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (emp_id, duration, dominant_emotion, session_date)
                    )
                    session_id = cur.fetchone()[0]
                    dominant_percentage = random.uniform(40, 70)
                    remaining_percentage = 100 - dominant_percentage
                    other_emotions = [e for e in emotions if e != dominant_emotion]
                    selected_emotions = random.sample(other_emotions, random.randint(2, 4))
                    other_percentages = []
                    remaining = remaining_percentage
                    for _ in range(len(selected_emotions) - 1):
                        p = random.uniform(5, remaining - 5 * (len(selected_emotions) - len(other_percentages) - 1))
                        other_percentages.append(p)
                        remaining -= p
                    other_percentages.append(remaining)
                    cur.execute(
                        """
                        INSERT INTO emotion_details 
                        (session_id, emotion, count, percentage) 
                        VALUES (%s, %s, %s, %s)
                        """,
                        (session_id, dominant_emotion, int(dominant_percentage), dominant_percentage)
                    )
                    for emotion, percentage in zip(selected_emotions, other_percentages):
                        cur.execute(
                            """
                            INSERT INTO emotion_details 
                            (session_id, emotion, count, percentage) 
                            VALUES (%s, %s, %s, %s)
                            """,
                            (session_id, emotion, int(percentage), percentage)
                        )
            conn.commit()
            logger.info("Sample data added successfully")
    except psycopg.Error as e:
        logger.error(f"Error adding sample data: {e}")
        conn.rollback()

def consolidate_data(conn):
    try:
        with conn.cursor() as cur:
            # Check if consolidated_emotion_data exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'consolidated_emotion_data'
                )
            """)
            if not cur.fetchone()[0]:
                logger.error("consolidated_emotion_data table does not exist, cannot consolidate data")
                return
            
            # Verify required columns
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'consolidated_emotion_data'
            """)
            columns = [row[0] for row in cur.fetchall()]
            required_columns = [
                'id', 'employee_id', 'department', 'session_id', 
                'emotion', 'count', 'percentage', 'session_date', 'time_stamp'
            ]
            if not all(col in columns for col in required_columns):
                logger.error(f"Missing required columns in consolidated_emotion_data: {set(required_columns) - set(columns)}")
                return
            
            cur.execute("TRUNCATE TABLE consolidated_emotion_data")
            cur.execute("""
                INSERT INTO consolidated_emotion_data (
                    employee_id, department, session_id, 
                    emotion, count, percentage, session_date, time_stamp
                )
                SELECT 
                    e.employee_id,
                    e.department,
                    s.id AS session_id,
                    ed.emotion,
                    ed.count,
                    ed.percentage,
                    s.session_date,
                    s.created_at::TIME AS time_stamp
                FROM 
                    employees e
                JOIN 
                    sessions s ON e.employee_id = s.employee_id
                JOIN 
                    emotion_details ed ON s.id = ed.session_id
            """)
            conn.commit()
            logger.info("Data successfully consolidated into consolidated_emotion_data table")
    except psycopg.Error as e:
        logger.error(f"Error consolidating data: {e}")
        conn.rollback()