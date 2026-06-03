document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');
    const resultCard = document.getElementById('resultCard');
    const resultTitle = document.getElementById('resultTitle');
    const probBar = document.getElementById('probBar');
    const probValue = document.getElementById('probValue');
    const thresholdValue = document.getElementById('thresholdValue');

    let currentFile = null;

    // Drag and drop events
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Lütfen sadece resim dosyası (JPEG, PNG, WEBP) yükleyin.');
            return;
        }
        
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadZone.classList.add('hidden');
            previewSection.classList.remove('hidden');
            resultSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        previewSection.classList.add('hidden');
        resultSection.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        probBar.style.width = '0%';
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI state update
        analyzeBtn.disabled = true;
        loader.classList.remove('hidden');
        resultSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Bir hata oluştu.');
            }

            const data = await response.json();
            displayResult(data);
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            analyzeBtn.disabled = false;
            loader.classList.add('hidden');
        }
    });

    function displayResult(data) {
        resultSection.classList.remove('hidden');
        
        const probPercent = (data.prob * 100).toFixed(1);
        probValue.textContent = probPercent;
        thresholdValue.textContent = data.threshold;
        
        setTimeout(() => {
            probBar.style.width = `${probPercent}%`;
        }, 100);

        resultCard.classList.remove('safe', 'refer');
        
        if (data.label === 0) {
            resultTitle.textContent = 'Healthy (No DR Detected)';
            resultCard.classList.add('safe');
        } else {
            resultTitle.textContent = 'Refer (DR Detected)';
            resultCard.classList.add('refer');
        }
    }
});
