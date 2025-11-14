// RetinaAI Platform JavaScript

// Navigation System
class NavigationController {
    constructor() {
        this.currentPage = 'dashboard'; // Start with dashboard page
        this.init();
    }

    init() {
        // Add click listeners to nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const targetPage = e.currentTarget.dataset.page;
                this.navigateTo(targetPage);
            });
        });

        // Set initial page
        this.showPage(this.currentPage);
    }

    navigateTo(pageName) {
        if (this.currentPage === pageName) return;
        
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.page === pageName) {
                item.classList.add('active');
            }
        });

        // Show page
        this.showPage(pageName);
        this.currentPage = pageName;
    }

    showPage(pageName) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });

        // Show target page
        const targetPage = document.getElementById(`page-${pageName}`);
        if (targetPage) {
            targetPage.classList.add('active');
        }
    }
}

// Sparkline Chart Generator
class SparklineChart {
    constructor(canvasId, data, color = '#C2185B') {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        
        this.ctx = this.canvas.getContext('2d');
        this.data = data;
        this.color = color;
        this.draw();
    }

    draw() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const points = this.data.length;
        
        const max = Math.max(...this.data);
        const min = Math.min(...this.data);
        const range = max - min || 1;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // Draw line
        this.ctx.beginPath();
        this.ctx.strokeStyle = this.color;
        this.ctx.lineWidth = 2;
        
        for (let i = 0; i < points; i++) {
            const x = (i / (points - 1)) * width;
            const y = height - ((this.data[i] - min) / range) * height;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
        
        // Draw fill
        this.ctx.lineTo(width, height);
        this.ctx.lineTo(0, height);
        this.ctx.closePath();
        
        const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, `${this.color}40`);
        gradient.addColorStop(1, `${this.color}00`);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
    }
}

// Image Upload Handler
class ImageUploader {
    constructor() {
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('imageInput');
        this.uploadCard = document.querySelector('.upload-card');
        this.resultsPreview = document.getElementById('resultsPreview');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.recentUploads = [];
        this.init();
    }

    init() {
        if (!this.uploadZone || !this.fileInput) {
            console.error('Upload elements not found');
            return;
        }

        // Click to upload
        this.uploadZone.addEventListener('click', () => {
            this.fileInput.click();
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.uploadImage(file);
            }
        });

        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('dragover');
        });

        this.uploadZone.addEventListener('dragleave', () => {
            this.uploadZone.classList.remove('dragover');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.uploadImage(file);
            }
        });
    }

    async uploadImage(file) {
        // Show loading
        this.uploadCard.style.display = 'none';
        this.loadingIndicator.style.display = 'block';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/segment', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const result = await response.json();
            
            // Display results
            this.displayResults(result, file);
            
            // Add to recent uploads
            this.addToRecentUploads(result, file);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to process image. Please try again.');
            
            // Reset UI
            this.uploadCard.style.display = 'block';
            this.loadingIndicator.style.display = 'none';
        }
    }

    displayResults(result, file) {
        // Hide loading, show results
        this.loadingIndicator.style.display = 'none';
        this.resultsPreview.style.display = 'block';

        // Set images (only 2 in preview)
        document.getElementById('previewOriginal').src = result.original_image;
        document.getElementById('previewSegmentation').src = result.mask;

        // Store current result for modal
        window.currentResult = result;
    }

    addToRecentUploads(result, file) {
        const upload = {
            id: Date.now(),
            name: file.name,
            date: new Date().toISOString(),
            originalImage: result.original_image,
            mask: result.mask,
            overlay: result.overlay,
            heatmap: result.heatmap,
            metrics: {
                dice: (result.dice * 100).toFixed(2),
                iou: (result.iou * 100).toFixed(2),
                pixelAccuracy: (result.pixel_accuracy * 100).toFixed(2)
            }
        };

        this.recentUploads.unshift(upload);
        if (this.recentUploads.length > 5) {
            this.recentUploads.pop();
        }

        this.renderRecentUploads();
        this.updateHistoryTable();
    }

    renderRecentUploads() {
        const recentList = document.getElementById('recentList');
        if (!recentList) {
            console.error('Recent list element not found');
            return;
        }

        if (this.recentUploads.length === 0) {
            recentList.innerHTML = '<p class="empty-state">No recent analyses</p>';
            return;
        }

        recentList.innerHTML = this.recentUploads.map(upload => `
            <div class="recent-item" onclick="imageUploader.loadUpload(${upload.id})">
                <img src="${upload.originalImage}" alt="${upload.name}" class="recent-thumbnail">
                <div class="recent-info">
                    <div class="recent-name">${this.truncateName(upload.name, 20)}</div>
                    <div class="recent-date">${this.formatDate(upload.date)}</div>
                </div>
            </div>
        `).join('');
    }

    loadUpload(id) {
        const upload = this.recentUploads.find(u => u.id === id);
        if (!upload) return;

        // Display in preview
        document.getElementById('preview-original').src = upload.originalImage;
        document.getElementById('preview-mask').src = upload.mask;
        document.getElementById('preview-overlay').src = upload.overlay;
        document.getElementById('preview-heatmap').src = upload.heatmap;

        this.uploadCard.style.display = 'none';
        this.loadingIndicator.style.display = 'none';
        this.resultsPreview.style.display = 'block';

        window.currentResult = upload;
    }

    updateHistoryTable() {
        const tbody = document.querySelector('#historyTableBody');
        if (!tbody) return;

        if (this.recentUploads.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No analysis history available</td></tr>';
            return;
        }

        tbody.innerHTML = this.recentUploads.map((upload, index) => `
            <tr>
                <td><img src="${upload.originalImage}" class="history-thumbnail" alt="${upload.name}"></td>
                <td>${this.formatDateTime(upload.date)}</td>
                <td>${this.truncateName(upload.name, 30)}</td>
                <td>${upload.metrics.dice}%</td>
                <td>${(upload.metrics.iou || 0).toFixed(2)}%</td>
                <td><span class="status-badge success">Complete</span></td>
                <td>
                    <button class="action-button" onclick="imageUploader.viewDetails(${upload.id})">View</button>
                </td>
            </tr>
        `).join('');
    }

    viewDetails(id) {
        const upload = this.recentUploads.find(u => u.id === id);
        if (!upload) return;

        modalController.showResults(upload);
    }

    resetUpload() {
        this.uploadCard.style.display = 'block';
        this.loadingIndicator.style.display = 'none';
        this.resultsPreview.style.display = 'none';
        this.fileInput.value = '';
    }

    truncateName(name, maxLength) {
        if (name.length <= maxLength) return name;
        const ext = name.split('.').pop();
        const nameWithoutExt = name.substring(0, name.lastIndexOf('.'));
        const truncated = nameWithoutExt.substring(0, maxLength - ext.length - 4) + '...';
        return truncated + '.' + ext;
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
        return date.toLocaleDateString();
    }

    formatDateTime(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

// Modal Controller
class ModalController {
    constructor() {
        this.modal = document.getElementById('resultsModal');
        this.init();
    }

    init() {
        if (!this.modal) {
            console.error('Modal element not found');
            return;
        }

        // Close button
        document.getElementById('modal-close')?.addEventListener('click', () => {
            this.close();
        });

        // Close on backdrop click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.close();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modal.classList.contains('active')) {
                this.close();
            }
        });

        // View full results button
        document.getElementById('view-full-results')?.addEventListener('click', () => {
            if (window.currentResult) {
                this.showResults(window.currentResult);
            }
        });
    }

    showResults(result) {
        // Set images - match HTML IDs
        const resultOriginal = document.getElementById('resultOriginal');
        const resultProbability = document.getElementById('resultProbability');
        const resultBinary = document.getElementById('resultBinary');
        const resultOverlay = document.getElementById('resultOverlay');
        
        if (resultOriginal) resultOriginal.src = result.original_image;
        if (resultProbability) resultProbability.src = result.heatmap;
        if (resultBinary) resultBinary.src = result.mask;
        if (resultOverlay) resultOverlay.src = result.overlay;

        // Set metrics
        const statCoverage = document.getElementById('statCoverage');
        const statConfidence = document.getElementById('statConfidence');
        const statSize = document.getElementById('statSize');
        
        if (result.vessel_coverage && statCoverage) {
            statCoverage.textContent = `${result.vessel_coverage.toFixed(2)}%`;
        }
        if (result.mean_confidence && statConfidence) {
            statConfidence.textContent = `${(result.mean_confidence * 100).toFixed(2)}%`;
        }
        if (result.image_size && statSize) {
            statSize.textContent = result.image_size;
        }

        // Show modal
        this.modal.classList.add('active');
    }

    close() {
        this.modal.classList.remove('active');
    }
}

// Search and Filter Controller
class SearchController {
    constructor() {
        this.searchInput = document.getElementById('historySearch');
        this.filterSelect = document.getElementById('historyFilter');
        this.init();
    }

    init() {
        if (!this.searchInput || !this.filterSelect) return;

        this.searchInput.addEventListener('input', () => {
            this.filterTable();
        });

        this.filterSelect.addEventListener('change', () => {
            this.filterTable();
        });
    }

    filterTable() {
        const searchTerm = this.searchInput.value.toLowerCase();
        const filterValue = this.filterSelect.value;
        
        const rows = document.querySelectorAll('#history-table tbody tr');
        
        rows.forEach(row => {
            if (row.cells.length === 1) return; // Skip empty state row
            
            const name = row.cells[3].textContent.toLowerCase();
            const status = row.cells[5].textContent.toLowerCase();
            
            const matchesSearch = name.includes(searchTerm);
            const matchesFilter = filterValue === 'all' || status === filterValue;
            
            row.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
        });
    }
}

// Initialize Sparklines
function initializeSparklines() {
    // Generate sample data for sparklines
    const generateData = (base, variance) => {
        return Array.from({ length: 20 }, () => 
            base + (Math.random() - 0.5) * variance
        );
    };

    // Create sparklines with correct IDs
    new SparklineChart('sparkline-dice', generateData(83.82, 2), '#C2185B');
    new SparklineChart('sparkline-accuracy', generateData(96.08, 1.5), '#4CAF50');
    new SparklineChart('sparkline-sensitivity', generateData(82.91, 3), '#FF9800');
    new SparklineChart('sparkline-specificity', generateData(97.97, 1), '#2196F3');
}

// Initialize Progress Bars with Animation
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill');
    
    progressBars.forEach(bar => {
        const targetWidth = bar.style.width;
        bar.style.width = '0%';
        
        setTimeout(() => {
            bar.style.width = targetWidth;
        }, 100);
    });
}

// Download Functions
function downloadImage(imageId, filename) {
    const img = document.getElementById(imageId);
    if (!img || !img.src) {
        console.error(`Image not found: ${imageId}`);
        return;
    }
    
    fetch(img.src)
        .then(res => res.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        })
        .catch(err => console.error('Download failed:', err));
}

function downloadResults() {
    if (!window.currentResult) {
        alert('No results available to download');
        return;
    }
    
    // Use correct modal image IDs
    downloadImage('resultOriginal', 'retina_original.png');
    setTimeout(() => downloadImage('resultBinary', 'retina_mask.png'), 200);
    setTimeout(() => downloadImage('resultOverlay', 'retina_overlay.png'), 400);
    setTimeout(() => downloadImage('resultProbability', 'retina_heatmap.png'), 600);
}

// Global instances
let navigationController;
let imageUploader;
let modalController;
let searchController;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize controllers
    navigationController = new NavigationController();
    imageUploader = new ImageUploader();
    modalController = new ModalController();
    searchController = new SearchController();
    
    // Initialize UI elements
    initializeSparklines();
    initializeProgressBars();
    
    console.log('RetinaAI Platform initialized');
});

// Expose functions globally
window.imageUploader = imageUploader;
window.modalController = modalController;
window.downloadResults = downloadResults;
window.downloadImage = downloadImage;

// Global functions for HTML onclick handlers
function showFullResults() {
    if (window.currentResult && modalController) {
        modalController.showResults(window.currentResult);
    }
}

function closeResults() {
    if (modalController) {
        modalController.close();
    }
}

function navigateTo(page) {
    if (navigationController) {
        navigationController.navigateTo(page);
    }
}

// Expose these functions globally
window.showFullResults = showFullResults;
window.closeResults = closeResults;
window.navigateTo = navigateTo;
