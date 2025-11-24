// RetinaAI Platform JavaScript

// Navigation System
class NavigationController {
    constructor() {
        this.currentPage = 'dashboard'; // Start with dashboard page
        this.init();
    }

    init() {
        // Add click listeners to nav items (both sidebar and mobile bottom nav)
        document.querySelectorAll('.nav-item, .mobile-nav-item').forEach(item => {
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

        // Update active nav item in sidebar
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.page === pageName) {
                item.classList.add('active');
            }
        });

        // Update active nav item in mobile bottom nav
        document.querySelectorAll('.mobile-nav-item').forEach(item => {
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
        // Show loading immediately
        console.log('Starting upload process...');
        this.uploadCard.style.display = 'none';
        this.resultsPreview.style.display = 'none';
        this.loadingIndicator.style.display = 'block';
        
        // Reset progress
        this.resetLoadingSteps();
        this.updateProgress(0);
        
        // Add loading class to body for global loading state
        document.body.classList.add('loading-active');

        try {
            // Step 1: Validating image
            this.updateLoadingStep(1, 'active');
            document.getElementById('loadingText').textContent = 'Validating image...';
            console.log('Step 1: Validating image');
            await this.simulateProgress(25, 1000);

            // Step 2: Preprocessing
            this.updateLoadingStep(1, 'completed');
            this.updateLoadingStep(2, 'active');
            document.getElementById('loadingText').textContent = 'Preprocessing image...';
            console.log('Step 2: Preprocessing');
            await this.simulateProgress(50, 1500);

            // Step 3: AI Analysis
            this.updateLoadingStep(2, 'completed');
            this.updateLoadingStep(3, 'active');
            document.getElementById('loadingText').textContent = 'AI Analysis in progress...';
            console.log('Step 3: AI Analysis');
            await this.simulateProgress(75, 2000);

            // Step 4: Generating results
            this.updateLoadingStep(3, 'completed');
            this.updateLoadingStep(4, 'active');
            document.getElementById('loadingText').textContent = 'Generating results...';
            console.log('Step 4: Generating results');

            const formData = new FormData();
            formData.append('file', file);

            console.log('Sending request to /api/segment...');
            const response = await fetch('/api/segment', {
                method: 'POST',
                body: formData
            });

            console.log('Response received:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', response.status, errorText);
                throw new Error(`Server error: ${response.status} - ${errorText}`);
            }

            const result = await response.json();
            console.log('Result received:', result);
            
            // Complete progress
            this.updateProgress(100);
            this.updateLoadingStep(4, 'completed');
            document.getElementById('loadingText').textContent = 'Complete!';
            
            // Small delay to show completion
            setTimeout(() => {
                console.log('Displaying results...');
                // Display results
                this.displayResults(result, file);
                
                // Add to recent uploads
                this.addToRecentUploads(result, file);
                
                // Remove loading class
                document.body.classList.remove('loading-active');
            }, 500);
            
        } catch (error) {
            console.error('Upload error:', error);
            document.getElementById('loadingText').textContent = `Error: ${error.message}. Please try again.`;
            
            // Reset UI after error
            setTimeout(() => {
                this.uploadCard.style.display = 'block';
                this.loadingIndicator.style.display = 'none';
                document.body.classList.remove('loading-active');
            }, 3000);
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

    resetLoadingSteps() {
        for (let i = 1; i <= 4; i++) {
            const step = document.getElementById(`step${i}`);
            if (step) step.className = 'loading-step';
        }
    }

    updateLoadingStep(stepNumber, status) {
        const step = document.getElementById(`step${stepNumber}`);
        if (step) step.className = `loading-step ${status}`;
    }

    updateProgress(percentage) {
        const progressBar = document.getElementById('progressBar');
        if (progressBar) progressBar.style.width = `${percentage}%`;
    }

    simulateProgress(targetPercentage, duration) {
        return new Promise(resolve => {
            const startPercentage = parseFloat(document.getElementById('progressBar').style.width) || 0;
            const increment = (targetPercentage - startPercentage) / (duration / 50);
            let currentPercentage = startPercentage;

            const interval = setInterval(() => {
                currentPercentage += increment;
                if (currentPercentage >= targetPercentage) {
                    currentPercentage = targetPercentage;
                    clearInterval(interval);
                    resolve();
                }
                this.updateProgress(currentPercentage);
            }, 50);
        });
    }

    resetUpload() {
        this.uploadCard.style.display = 'block';
        this.loadingIndicator.style.display = 'none';
        this.resultsPreview.style.display = 'none';
        this.fileInput.value = '';
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
        const statTime = document.getElementById('statTime');
        
        if (result.vessel_coverage && statCoverage) {
            statCoverage.textContent = `${result.vessel_coverage.toFixed(2)}%`;
        }
        if (result.mean_confidence && statConfidence) {
            statConfidence.textContent = `${(result.mean_confidence * 100).toFixed(2)}%`;
        }
        if (result.processing_time && statTime) {
            statTime.textContent = `${result.processing_time}s`;
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
let onboardingController;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize controllers
    navigationController = new NavigationController();
    imageUploader = new ImageUploader();
    modalController = new ModalController();
    searchController = new SearchController();
    onboardingController = new OnboardingController();
    
    // Initialize UI elements
    initializeSparklines();
    initializeProgressBars();
    
    console.log('🎉 RetinaAI Platform v1.1.0 initialized successfully');
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

// Premium Onboarding System
class OnboardingController {
    constructor() {
        this.currentStep = 0;
        this.isActive = false;
        this.touchStartX = 0;
        this.touchStartY = 0;
        this.touchEndX = 0;
        this.touchEndY = 0;
        this.steps = [
            {
                target: '.upload-zone',
                title: 'Upload Your Retina Images',
                content: 'Drag and drop or click to upload retina images for AI-powered segmentation analysis.',
                position: 'bottom',
                icon: '📤'
            },
            {
                target: '.metric-cards',
                title: 'Real-time Performance Metrics',
                content: 'Monitor your AI model\'s performance with live Dice coefficient, accuracy, and other key metrics.',
                position: 'top',
                icon: '📊'
            },
            {
                target: '.recent-uploads',
                title: 'Analysis History',
                content: 'Quickly access your recent analyses and compare results across different images.',
                position: 'left',
                icon: '📚'
            },
            {
                target: '.mobile-bottom-nav .mobile-nav-item[data-page="history"]',
                title: 'Detailed History & Reports',
                content: 'View comprehensive analysis history, download reports, and track your progress over time.',
                position: 'top',
                icon: '📋'
            },
            {
                target: '.mobile-bottom-nav .mobile-nav-item[data-page="settings"]',
                title: 'Customize Your Experience',
                content: 'Adjust AI parameters, configure notifications, and personalize your dashboard settings.',
                position: 'top',
                icon: '⚙️'
            }
        ];
        this.overlay = null;
        this.tooltip = null;
        this.init();
    }

    init() {
        // Check if user has seen onboarding
        const hasSeenOnboarding = localStorage.getItem('retinaAI_onboarding_completed');
        
        if (!hasSeenOnboarding) {
            // Show onboarding after a short delay
            setTimeout(() => {
                this.start();
            }, 1500);
        }

        // Add restart onboarding option (could be in settings)
        this.addRestartOption();

        // Add touch gesture support
        this.addTouchGestures();
    }

    addTouchGestures() {
        // Add touch event listeners for swipe gestures
        document.addEventListener('touchstart', (e) => {
            this.touchStartX = e.changedTouches[0].screenX;
            this.touchStartY = e.changedTouches[0].screenY;
        }, { passive: true });

        document.addEventListener('touchend', (e) => {
            this.touchEndX = e.changedTouches[0].screenX;
            this.touchEndY = e.changedTouches[0].screenY;
            this.handleSwipeGesture();
        }, { passive: true });
    }

    handleSwipeGesture() {
        if (!this.isActive) return;

        const deltaX = this.touchEndX - this.touchStartX;
        const deltaY = this.touchEndY - this.touchStartY;
        const minSwipeDistance = 50;

        // Check if it's a horizontal swipe (more horizontal than vertical)
        if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > minSwipeDistance) {
            if (deltaX > 0) {
                // Swipe right - previous step
                this.previous();
            } else {
                // Swipe left - next step
                this.next();
            }
        }
    }

    start() {
        this.isActive = true;
        this.currentStep = 0;
        this.createOverlay();
        this.showStep(0);
    }

    createOverlay() {
        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'onboarding-overlay';
        this.overlay.innerHTML = `
            <div class="onboarding-tooltip" id="onboardingTooltip">
                <div class="tooltip-arrow" id="tooltipArrow"></div>
                <div class="tooltip-content">
                    <div class="tooltip-header">
                        <span class="tooltip-icon" id="tooltipIcon"></span>
                        <h4 id="tooltipTitle"></h4>
                    </div>
                    <div class="tooltip-body">
                        <p id="tooltipContent"></p>
                        <div class="tooltip-progress">
                            <div class="progress-dots" id="progressDots"></div>
                            <span class="progress-text" id="progressText"></span>
                        </div>
                    </div>
                    <div class="tooltip-actions">
                        <button class="btn-skip" id="skipOnboarding">Skip Tour</button>
                        <div class="primary-actions">
                            <button class="btn-prev" id="prevStep" style="display: none;">Previous</button>
                            <button class="btn-next" id="nextStep">Next</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(this.overlay);
        this.tooltip = document.getElementById('onboardingTooltip');
        
        // Add event listeners
        document.getElementById('skipOnboarding').addEventListener('click', () => this.end());
        document.getElementById('nextStep').addEventListener('click', () => this.next());
        document.getElementById('prevStep').addEventListener('click', () => this.previous());
        
        // Prevent interaction with background
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                e.stopPropagation();
            }
        });
    }

    showStep(stepIndex) {
        const step = this.steps[stepIndex];
        if (!step) return;

        // Update tooltip content
        document.getElementById('tooltipIcon').textContent = step.icon;
        document.getElementById('tooltipTitle').textContent = step.title;
        document.getElementById('tooltipContent').textContent = step.content;
        document.getElementById('progressText').textContent = `${stepIndex + 1} of ${this.steps.length}`;

        // Update progress dots
        this.updateProgressDots(stepIndex);

        // Position tooltip
        this.positionTooltip(step);

        // Update button visibility
        const prevBtn = document.getElementById('prevStep');
        const nextBtn = document.getElementById('nextStep');
        
        prevBtn.style.display = stepIndex > 0 ? 'block' : 'none';
        nextBtn.textContent = stepIndex === this.steps.length - 1 ? 'Get Started' : 'Next';

        // Highlight target element
        this.highlightTarget(step.target);
    }

    positionTooltip(step) {
        const targetElement = document.querySelector(step.target);
        if (!targetElement) return;

        const rect = targetElement.getBoundingClientRect();
        const tooltip = this.tooltip;
        const arrow = document.getElementById('tooltipArrow');

        // Reset tooltip position
        tooltip.style.top = '';
        tooltip.style.left = '';
        tooltip.style.transform = '';

        // Calculate position based on step.position
        let top, left, arrowPosition;

        switch (step.position) {
            case 'top':
                top = rect.top - 10;
                left = rect.left + (rect.width / 2);
                arrowPosition = 'bottom';
                break;
            case 'bottom':
                top = rect.bottom + 10;
                left = rect.left + (rect.width / 2);
                arrowPosition = 'top';
                break;
            case 'left':
                top = rect.top + (rect.height / 2);
                left = rect.left - 10;
                arrowPosition = 'right';
                break;
            case 'right':
                top = rect.top + (rect.height / 2);
                left = rect.right + 10;
                arrowPosition = 'left';
                break;
            default:
                top = rect.bottom + 10;
                left = rect.left + (rect.width / 2);
                arrowPosition = 'top';
        }

        // Position tooltip
        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
        tooltip.style.transform = 'translate(-50%, -100%)';

        // Position arrow
        arrow.className = `tooltip-arrow arrow-${arrowPosition}`;
    }

    updateProgressDots(activeIndex) {
        const dotsContainer = document.getElementById('progressDots');
        dotsContainer.innerHTML = '';

        this.steps.forEach((_, index) => {
            const dot = document.createElement('div');
            dot.className = `dot ${index === activeIndex ? 'active' : ''}`;
            dotsContainer.appendChild(dot);
        });
    }

    highlightTarget(selector) {
        // Remove previous highlights
        document.querySelectorAll('.onboarding-highlight').forEach(el => {
            el.classList.remove('onboarding-highlight');
        });

        // Add highlight to current target
        const target = document.querySelector(selector);
        if (target) {
            target.classList.add('onboarding-highlight');
        }
    }

    next() {
        if (this.currentStep < this.steps.length - 1) {
            this.currentStep++;
            this.showStep(this.currentStep);
        } else {
            this.end();
        }
    }

    previous() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.showStep(this.currentStep);
        }
    }

    end() {
        this.isActive = false;
        
        // Remove overlay
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
        }

        // Remove highlights
        document.querySelectorAll('.onboarding-highlight').forEach(el => {
            el.classList.remove('onboarding-highlight');
        });

        // Mark as completed
        localStorage.setItem('retinaAI_onboarding_completed', 'true');
    }

    restart() {
        this.end();
        setTimeout(() => {
            this.start();
        }, 300);
    }

    addRestartOption() {
        // Could add a restart button in settings or footer
        // For now, add a global function for testing
        window.restartOnboarding = () => this.restart();
    }
}

// Expose these functions globally
window.showFullResults = showFullResults;
window.closeResults = closeResults;
window.navigateTo = navigateTo;
window.restartOnboarding = window.restartOnboarding || (() => {});
