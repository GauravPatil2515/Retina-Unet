# 🎨 Dashboard Guide - Retina U-Net++

## 🚀 Quick Start

### 1. Install Dashboard Dependencies

```bash
pip install -r dashboard/requirements.txt
```

### 2. Run the Dashboard

```bash
cd dashboard
python app.py
```

or

```bash
uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open Browser

Navigate to: **http://localhost:8000**

---

## 🎨 Color Scheme

The dashboard uses a **dark medical theme** inspired by modern healthcare applications:

### Primary Colors
- **Primary Purple:** `#8B5CF6` - Main brand color
- **Primary Pink:** `#EC4899` - Accent for highlights
- **Primary Blue:** `#3B82F6` - Information elements

### Background Colors
- **BG Primary:** `#0F0F1E` - Main background (very dark blue-black)
- **BG Secondary:** `#1A1A2E` - Sidebar background
- **BG Card:** `#1E1E3F` - Card backgrounds (dark purple-blue)
- **BG Tertiary:** `#252541` - Hover states

### Status Colors
- **Success Green:** `#10B981` - Positive metrics
- **Warning Yellow:** `#F59E0B` - Alerts
- **Error Red:** `#EF4444` - Errors
- **Info Blue:** `#3B82F6` - Information

### Text Colors
- **Text Primary:** `#F9FAFB` - Main text (almost white)
- **Text Secondary:** `#D1D5DB` - Secondary text (light gray)
- **Text Muted:** `#9CA3AF` - Muted text (gray)

---

## 📁 Dashboard Structure

```
dashboard/
├── app.py                  # FastAPI backend
├── requirements.txt        # Python dependencies
├── static/
│   ├── style.css          # Dark theme styles
│   └── script.js          # Frontend logic
└── templates/
    └── index.html         # Main dashboard page
```

---

## 🎯 Features

### ✅ Implemented
- [x] Dark medical theme UI
- [x] Real-time image upload (drag & drop)
- [x] Model inference with progress indicator
- [x] 4-panel result visualization
- [x] Performance metrics dashboard
- [x] Model status indicator
- [x] Responsive design

### 🔜 Coming Soon
- [ ] Batch processing
- [ ] Download results
- [ ] History/Analytics
- [ ] User authentication
- [ ] API key management
- [ ] Export reports (PDF)

---

## 🛠️ Customization

### Change Colors

Edit `dashboard/static/style.css`:

```css
:root {
    --primary: #8B5CF6;        /* Change primary color */
    --accent-pink: #EC4899;    /* Change accent */
    --bg-primary: #0F0F1E;     /* Change background */
}
```

### Add New Sections

1. Edit `dashboard/templates/index.html`
2. Add new section HTML
3. Update `dashboard/static/script.js` for interactivity

### Modify Metrics

Edit `dashboard/app.py` → `load_metrics()` function

---

## 🚀 Deployment Options

### Option 1: Local Development
```bash
python dashboard/app.py
```

### Option 2: Production (Gunicorn)
```bash
pip install gunicorn
gunicorn dashboard.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Option 3: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r dashboard/requirements.txt
RUN pip install -r requirements_unetpp.txt
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 4: Cloud Platforms
- **Render:** Connect GitHub repo, deploy as Web Service
- **Railway:** One-click deploy
- **Heroku:** `Procfile`: `web: uvicorn dashboard.app:app --host 0.0.0.0 --port $PORT`
- **AWS/Azure:** Deploy with container service

---

## 📊 API Endpoints

### GET `/`
- **Description:** Main dashboard page
- **Returns:** HTML page

### POST `/predict`
- **Description:** Upload image for segmentation
- **Body:** FormData with `file` field
- **Returns:** JSON with results and base64 images

### GET `/api/stats`
- **Description:** Get model statistics
- **Returns:** JSON with performance metrics

### GET `/api/model-info`
- **Description:** Get model details
- **Returns:** JSON with architecture info

---

## 🎨 Design Inspiration

The dashboard is inspired by:
- **nixhd** - Modern dark theme statistics dashboard
- **Medical UI** - Healthcare application standards
- **Glassmorphism** - Subtle glass effects
- **Purple/Pink gradient** - Medical/healthcare branding

---

## 💡 Tips

1. **Performance:** Images are processed in ~1 second on GPU
2. **File Size:** Max 10MB per image
3. **Format:** PNG, JPG, JPEG supported
4. **Best Results:** Use 512x512 retinal fundus images
5. **Mobile:** Responsive design works on tablets/phones

---

## 🔧 Troubleshooting

### Dashboard won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Use different port
uvicorn dashboard.app:app --port 8080
```

### Model not loading
```bash
# Check if model checkpoint exists
ls results/checkpoints_unetpp/best.pth

# Train model first if not found
python scripts/train_unetpp.py
```

### Images not displaying
- Check browser console for errors
- Ensure base64 encoding is working
- Verify file permissions

---

## 📞 Support

- **GitHub Issues:** Report bugs
- **Email:** Your email
- **Docs:** See `README.md` for full documentation

---

## 🎉 Ready to Use!

Your professional medical AI dashboard is ready. Simply run:

```bash
cd dashboard
python app.py
```

Then open: **http://localhost:8000**

Upload a retinal image and see the magic happen! ✨
