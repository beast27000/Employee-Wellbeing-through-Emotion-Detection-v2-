const socket = io('http://localhost:8000', { transports: ['websocket'], reconnectionAttempts: 5 });
let statsChart, complianceChart, acceptanceChart;
let isCameraActive = false;
let employeeId = null; // Store employee_id after login

socket.on('connect', () => {
    console.log('Socket.IO connected');
    if (employeeId) {
        socket.emit('set_employee_id', { employee_id: employeeId });
    }
});

socket.on('connection_success', (data) => {
    console.log('Server confirmed connection:', data.message);
});

socket.on('connect_error', (error) => {
    console.error('Socket.IO connection failed:', error);
    alert('Failed to connect to real-time server. Please ensure the server is running and try again.');
});

socket.on('frame', (data) => {
    document.getElementById('video-feed').src = `data:image/jpeg;base64,${data.frame}`;
    document.getElementById('video-feed').classList.remove('hidden');
    document.getElementById('camera-off').classList.add('hidden');
    updateStatsChart(data.stats);
    document.getElementById('status').textContent = `Status: ${data.status}`;
    document.getElementById('session-time').textContent = `Session Time: ${data.sessionTime}`;
});

socket.on('error', (data) => {
    alert(data.message);
    if (isCameraActive) {
        isCameraActive = false;
        document.getElementById('video-feed').classList.add('hidden');
        document.getElementById('camera-off').classList.remove('hidden');
    }
});

socket.on('camera_stopped', (data) => {
    isCameraActive = false;
    document.getElementById('video-feed').classList.add('hidden');
    document.getElementById('camera-off').classList.remove('hidden');
    document.getElementById('status').textContent = 'Status: Camera stopped';
    document.getElementById('session-time').textContent = 'Session Time: 00:00:00';
    alert(data.message);
});

function showScreen(screen) {
    document.querySelectorAll('#login-screen, #selection-screen, #main-screen, #admin-screen, #about-screen, #description-screen, #feedback-login-screen, #feedback-screen, #admin-feedback-screen').forEach(el => el.classList.add('hidden'));
    document.getElementById(`${screen}-screen`).classList.remove('hidden');
}

function promptAdminLogin() {
    const password = prompt('Enter admin password:');
    if (password === 'admin123') {
        showScreen('admin');
    } else {
        alert('Incorrect admin password');
    }
}

async function handleLogin() {
    const employeeIdInput = document.getElementById('employee-id').value.trim();
    const department = document.getElementById('department').value;
    if (!employeeIdInput || !employeeIdInput.match(/^\d+$/)) {
        alert('Please enter a valid numeric Employee ID');
        return;
    }
    if (!['IT', 'Accounting', 'Marketing', 'All'].includes(department)) {
        alert('Please select a valid department');
        return;
    }
    try {
        const response = await axios.post('http://localhost:8000/login', { employee_id: employeeIdInput, department });
        employeeId = response.data.employee_id; // Store employee_id
        socket.emit('set_employee_id', { employee_id: employeeId }); // Send to server
        document.getElementById('user-info').textContent = `Employee ID: ${employeeIdInput} | Department: ${department} | Model: ${document.getElementById('model')?.value || 'Not selected'}`;
        showScreen('selection');
    } catch (error) {
        alert(`Login failed: ${error.response?.data?.detail || error.message}`);
    }
}

async function handleProceed() {
    const model = document.getElementById('model').value;
    try {
        await axios.post('http://localhost:8000/select', { model });
        document.getElementById('user-info').textContent = `Employee ID: ${document.getElementById('employee-id').value} | Department: ${document.getElementById('department').value} | Model: ${model}`;
        showScreen('main');
    } catch (error) {
        alert(`Failed to proceed: ${error.response?.data?.detail || error.message}`);
    }
}

async function handleFeedbackLogin() {
    const employeeId = document.getElementById('feedback-employee-id').value.trim();
    const department = document.getElementById('feedback-department').value;
    if (!employeeId || !employeeId.match(/^[A-Za-z]{3}$/)) {
        alert('Please enter a valid 3-letter Employee ID');
        return;
    }
    if (!['IT', 'Accounting', 'Marketing', 'All'].includes(department)) {
        alert('Please select a valid department');
        return;
    }
    try {
        await axios.post('http://localhost:8000/feedback_login', { employee_id: employeeId, department });
        showScreen('feedback');
    } catch (error) {
        alert(`Feedback login failed: ${error.response?.data?.detail || error.message}`);
    }
}

async function handleFeedbackAdminLogin() {
    const password = document.getElementById('feedback-admin-password').value;
    if (!password) {
        alert('Please enter the admin password');
        return;
    }
    try {
        await axios.post('http://localhost:8000/feedback_login', { password });
        showScreen('admin-feedback');
    } catch (error) {
        alert(`Admin login failed: ${error.response?.data?.detail || error.message}`);
    }
}

async function submitFeedback() {
    const employeeId = document.getElementById('feedback-employee-id').value.trim();
    const department = document.getElementById('feedback-department').value;
    const feedback = {
        employee_id: employeeId,
        department: department,
        question_1: parseInt(document.getElementById('question-1').value),
        question_2: parseInt(document.getElementById('question-2').value),
        question_3: parseInt(document.getElementById('question-3').value),
        question_4: parseInt(document.getElementById('question-4').value),
        question_5: parseInt(document.getElementById('question-5').value),
        text_feedback: document.getElementById('text-feedback').value.trim()
    };
    try {
        await axios.post('http://localhost:8000/submit_feedback', feedback);
        alert('Feedback submitted successfully');
        showScreen('login');
    } catch (error) {
        alert(`Feedback submission failed: ${error.response?.data?.detail || error.message}`);
    }
}

async function fetchAdminFeedback() {
    const employeeId = document.getElementById('admin-feedback-employee-id').value.trim();
    if (!employeeId || !employeeId.match(/^[A-Za-z]{3}$/)) {
        alert('Please enter a valid 3-letter Employee ID');
        return;
    }
    try {
        const response = await axios.get(`http://localhost:8000/admin_feedback?employee_id=${employeeId}`);
        const feedback = response.data;
        const resultsDiv = document.getElementById('feedback-results');
        resultsDiv.innerHTML = feedback.map(f => `
            <div class="bg-[#A8DADC] p-4 rounded-lg mb-4">
                <p><strong>Department:</strong> ${f.department}</p>
                <p><strong>Comfort with Monitoring:</strong> ${f.question_1}</p>
                <p><strong>Feeling Supported:</strong> ${f.question_2}</p>
                <p><strong>Accuracy of Detection:</strong> ${f.question_3}</p>
                <p><strong>Comfort Sharing Feedback:</strong> ${f.question_4}</p>
                <p><strong>Likelihood to Recommend:</strong> ${f.question_5}</p>
                <p><strong>Additional Feedback:</strong> ${f.text_feedback || 'None'}</p>
                <p><strong>Submitted At:</strong> ${new Date(f.submitted_at).toLocaleString()}</p>
            </div>
        `).join('');
    } catch (error) {
        alert(`Failed to fetch feedback: ${error.response?.data?.detail || error.message}`);
    }
}

function updateStatsChart(stats) {
    if (!statsChart) {
        statsChart = new ApexCharts(document.getElementById('stats-chart'), {
            chart: { type: 'pie', height: 320 },
            series: Object.values(stats),
            labels: Object.keys(stats),
            colors: ['#FF0000', '#FFA500', '#800080', '#008000', '#FFFF00', '#0000FF', '#FFC0CB'],
            responsive: [{ breakpoint: 480, options: { chart: { height: 280 }, legend: { position: 'bottom' } } }]
        });
        statsChart.render();
    } else {
        statsChart.updateSeries(Object.values(stats));
    }
}

async function startCamera() {
    if (!socket.connected) {
        alert('Cannot start camera: Not connected to server');
        return;
    }
    if (isCameraActive) {
        alert('Camera is already active');
        return;
    }
    isCameraActive = true;
    socket.emit('start_camera');
}

async function stopCamera() {
    if (!socket.connected) {
        alert('Cannot stop camera: Not connected to server');
        return;
    }
    if (!isCameraActive) {
        alert('Camera is not active');
        return;
    }
    socket.emit('stop_camera');
}

async function uploadVideo(event) {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await axios.post('http://localhost:8000/upload_video', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        alert('Video analysis complete');
        updateStatsChart(response.data.stats);
    } catch (error) {
        alert(`Video upload failed: ${error.response?.data?.detail || error.message}`);
    }
}

async function testSamples() {
    try {
        const response = await axios.get('http://localhost:8000/test_samples');
        updateStatsChart(response.data.stats);
        const sampleImages = document.getElementById('sample-images');
        sampleImages.innerHTML = response.data.sample_images.map(img => `
            <div class="sample-card">
                <div class="label">${img.label}</div>
                <img src="data:image/jpeg;base64,${img.data}" alt="Sample Image">
            </div>
        `).join('');
        document.getElementById('sample-strip').classList.remove('hidden');
    } catch (error) {
        alert(`Test samples failed: ${error.response?.data?.detail || error.message}`);
    }
}

async function fetchAdminStats() {
    const dept = document.getElementById('admin-dept').value;
    const dateRange = document.getElementById('admin-date').value;
    try {
        document.getElementById('admin-loading').classList.remove('hidden');
        const response = await axios.get(`http://localhost:8000/admin_stats?department=${dept}&date_range=${dateRange}`);
        const { compliance, employees } = response.data;

        if (!complianceChart) {
            complianceChart = new ApexCharts(document.getElementById('compliance-chart'), {
                chart: { type: 'bar', height: 320 },
                series: [{ name: 'Count', data: compliance.map(c => c.total) }],
                xaxis: { categories: compliance.map(c => c.status) },
                colors: ['#457B9D'],
                responsive: [{ breakpoint: 480, options: { chart: { height: 280 } } }]
            });
            complianceChart.render();
        } else {
            complianceChart.updateOptions({
                xaxis: { categories: compliance.map(c => c.status) },
                series: [{ data: compliance.map(c => c.total) }]
            });
        }

        if (!acceptanceChart) {
            acceptanceChart = new ApexCharts(document.getElementById('acceptance-chart'), {
                chart: { type: 'bar', height: 320 },
                series: [{ name: 'Avg Duration (min)', data: employees.map(e => Math.min(e.avg_duration, 5)) }],
                xaxis: { categories: employees.map(e => e.employee_id) },
                yaxis: { max: 5 },
                colors: ['#52B788'],
                responsive: [{ breakpoint: 480, options: { chart: { height: 280 } } }]
            });
            acceptanceChart.render();
        } else {
            acceptanceChart.updateOptions({
                xaxis: { categories: employees.map(e => e.employee_id) },
                series: [{ data: employees.map(e => Math.min(e.avg_duration, 5)) }]
            });
        }

        const tableBody = document.getElementById('employee-table');
        tableBody.innerHTML = employees.map(e => `
            <tr class="hover:bg-[#B7E4C7]">
                <td class="border p-3">${e.employee_id}</td>
                <td class="border p-3">${e.department}</td>
                <td class="border p-3">${e.session_count}</td>
                <td class="border p-3">${e.avg_duration.toFixed(2)}</td>
                <td class="border p-3">${e.dominant_emotion}</td>
            </tr>
        `).join('');
    } catch (error) {
        alert(`Failed to fetch stats: ${error.response?.data?.detail || error.message}`);
    } finally {
        document.getElementById('admin-loading').classList.add('hidden');
    }
}