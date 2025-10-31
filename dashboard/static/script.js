// Dashboard JavaScript - Handle Image Upload and Predictions

let startTime;

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    // File input handling
    const fileInput = document.getElementById('imageInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // Drag and drop
    const uploadZone = document.getElementById('upload-zone');
    if (uploadZone) {
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
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
    }
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size should be less than 10MB');
        return;
    }
    
    // Upload and predict
    uploadAndPredict(file);
}

async function uploadAndPredict(file) {
    // Show loading spinner
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    
    if (loading) loading.classList.add('active');
    if (resultsSection) resultsSection.classList.remove('active');
    
    startTime = performance.now();
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Send to API
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Calculate processing time
            const endTime = performance.now();
            const processingTime = ((endTime - startTime) / 1000).toFixed(2);
            
            // Display results
            displayResults(result, processingTime);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to process image. Please try again.');
    } finally {
        if (loading) loading.classList.remove('active');
    }
}

function displayResults(result, processingTime) {
    // Show results section
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.classList.add('active');
    }
    
    // Set images
    const originalImg = document.getElementById('original-img');
    const probabilityImg = document.getElementById('probability-img');
    const binaryImg = document.getElementById('binary-img');
    const overlayImg = document.getElementById('overlay-img');
    
    if (originalImg) originalImg.src = 'data:image/png;base64,' + result.original;
    if (probabilityImg) probabilityImg.src = 'data:image/png;base64,' + result.probability;
    if (binaryImg) binaryImg.src = 'data:image/png;base64,' + result.binary;
    if (overlayImg) overlayImg.src = 'data:image/png;base64,' + result.overlay;
    
    // Set statistics
    const coverageValue = document.getElementById('coverage-value');
    const confidenceValue = document.getElementById('confidence-value');
    const sizeValue = document.getElementById('size-value');
    
    if (coverageValue) coverageValue.textContent = result.stats.vessel_coverage + '%';
    if (confidenceValue) confidenceValue.textContent = result.stats.mean_confidence;
    if (sizeValue) sizeValue.textContent = result.stats.image_size;
    
    // Scroll to results
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

function resetUpload() {
    // Hide results
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.classList.remove('active');
    }
    
    // Reset file input
    const fileInput = document.getElementById('imageInput');
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Load stats on page load
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        console.log('Model Stats:', data);
        
        // You can update any dynamic stats here
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load stats when page loads
window.addEventListener('load', loadStats);

// Add smooth scrolling for all internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});
