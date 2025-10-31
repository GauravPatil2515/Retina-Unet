# Run Dashboard Script
# Quick start script for the Retina U-Net Dashboard

Write-Host ""
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "  🩺 RETINA U-NET++ DASHBOARD" -ForegroundColor Green
Write-Host "  Medical AI Blood Vessel Segmentation" -ForegroundColor Yellow
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

Write-Host "📦 Checking requirements..." -ForegroundColor Yellow
Write-Host ""

# Check if model exists
if (Test-Path "results/checkpoints_unetpp/best.pth") {
    Write-Host "   ✓ Model checkpoint found" -ForegroundColor Green
} else {
    Write-Host "   ⚠ Model checkpoint not found!" -ForegroundColor Red
    Write-Host "   Run: python scripts/train_unetpp.py" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "🚀 Starting dashboard server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "   Dashboard will be available at:" -ForegroundColor Cyan
Write-Host "   → http://localhost:8000" -ForegroundColor Green
Write-Host ""
Write-Host "   Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# Start the server
cd dashboard
python app.py
