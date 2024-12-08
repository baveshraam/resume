const dragDropZone = document.getElementById('drag-drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const fileList = document.getElementById('uploaded-files-list');
const confirmBtn = document.getElementById('confirm-btn');
const dataInput = document.getElementById('data-input');  // New field for additional data

let uploadedFiles = [];

// Handle Drag and Drop
dragDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dragDropZone.style.background = '#dcefff';
});

dragDropZone.addEventListener('dragleave', () => {
    dragDropZone.style.background = '#eef4ff';
});

dragDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDropZone.style.background = '#eef4ff';
    handleFiles(e.dataTransfer.files);
});

// Open File Input
browseBtn.addEventListener('click', () => fileInput.click());

// Handle File Input
fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Handle Uploaded Files
function handleFiles(files) {
    Array.from(files).forEach((file) => {
        const fileId = generateUniqueId();
        uploadedFiles.push({ file, id: fileId });

        const listItem = document.createElement('li');
        listItem.className = 'file-item';
        listItem.innerHTML = `
            <span>${file.name}</span>
            <div class="progress-container">
                <div class="progress-bar" id="progress-${fileId}" style="width: 0%;"></div>
            </div>
            <button class="delete-btn">&times;</button>
        `;

        listItem.querySelector('.delete-btn').addEventListener('click', () => {
            uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
            listItem.remove();
            toggleConfirmButton();
        });

        fileList.appendChild(listItem);
    });

    toggleConfirmButton();
}

// Generate unique ID for files
function generateUniqueId() {
    return '_' + Math.random().toString(36).substr(2, 9);
}

// Enable/Disable Confirm Button
function toggleConfirmButton() {
    confirmBtn.disabled = uploadedFiles.length === 0;
}

// Confirm and Upload
confirmBtn.addEventListener('click', () => {
    const additionalData = dataInput.value; // Collect additional data

    uploadedFiles.forEach((fileData) => {
        const formData = new FormData();
        formData.append('file', fileData.file);
        formData.append('data', additionalData);  // Add additional data to form

        const progressBar = document.getElementById(`progress-${fileData.id}`);

        // Upload the file and update progress bar
        fetchWithProgress('http://127.0.0.1:5000/upload', formData, (progress) => {
            progressBar.style.width = `${progress}%`;
        })
        .then((response) => {
            progressBar.style.background = '#4caf50'; // Set progress bar to green upon success
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });
});

// Fetch with Progress Tracking
function fetchWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url, true);

        xhr.upload.onprogress = (event) => {
            if (event.lengthComputable) {
                const percentComplete = Math.round((event.loaded / event.total) * 100);
                onProgress(percentComplete);
            }
        };

        xhr.onload = () => {
            if (xhr.status === 200) {
                resolve(xhr.responseText);
            } else {
                reject(new Error('Upload failed.'));
            }
        };

        xhr.onerror = () => reject(new Error('Network Error'));
        xhr.send(formData);
    });
}
