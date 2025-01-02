function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show the selected section
    document.getElementById(sectionId).style.display = 'block';
}

function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('imageResult');
        resultDiv.innerHTML = `<img src="${data.processed_image}" alt="Processed Image">`;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error processing image');
    });
}

function uploadVideo() {
    const fileInput = document.getElementById('videoInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a video file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('videoResult');
        resultDiv.innerHTML = `
            <video controls>
                <source src="${data.processed_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>`;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error processing video');
    });
}

function startLiveDetection() {
    fetch('/start_camera')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('startLiveBtn').style.display = 'none';
                document.getElementById('stopLiveBtn').style.display = 'block';
                document.getElementById('liveVideo').style.display = 'block';
                document.getElementById('liveVideo').src = '/video_feed';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error starting camera');
        });
}

function stopLiveDetection() {
    fetch('/stop_camera')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('startLiveBtn').style.display = 'block';
                document.getElementById('stopLiveBtn').style.display = 'none';
                document.getElementById('liveVideo').style.display = 'none';
                document.getElementById('liveVideo').src = '';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error stopping camera');
        });
} 