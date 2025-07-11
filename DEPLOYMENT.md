# Deployment Guide - NFL Schedule App (Flask)

This guide provides step-by-step instructions for deploying the Flask version of the NFL Schedule App to various cloud platforms.

## 🚀 Quick Start

The app is currently running locally on `http://localhost:5001`. To deploy to production, choose one of the platforms below.

## 📋 Prerequisites

- Python 3.8+ installed
- Git repository with the Flask app
- Cloud platform account (Heroku, Railway, Render, etc.)

## 🎯 Deployment Options

### 1. Heroku Deployment

**Step 1: Install Heroku CLI**
```bash
# macOS
brew install heroku/brew/heroku

# Windows
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

**Step 2: Create Heroku App**
```bash
heroku login
heroku create your-nfl-schedule-app
```

**Step 3: Add Procfile**
Create a file named `Procfile` (no extension) in the root directory:
```
web: gunicorn app:app
```

**Step 4: Update requirements.txt**
Add gunicorn to your requirements:
```
flask>=2.3.0
gunicorn>=20.1.0
pandas
numpy
joblib>=1.2.0
matplotlib
scikit-learn
```

**Step 5: Deploy**
```bash
git add .
git commit -m "Add Heroku deployment files"
git push heroku main
```

**Step 6: Open App**
```bash
heroku open
```

### 2. Railway Deployment

**Step 1: Connect Repository**
1. Go to [Railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your repository

**Step 2: Configure**
Railway will automatically detect Flask and deploy. No additional configuration needed.

**Step 3: Access**
Your app will be available at the provided Railway URL.

### 3. Render Deployment

**Step 1: Create Web Service**
1. Go to [Render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository

**Step 2: Configure Settings**
- **Name**: `nfl-schedule-app`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

**Step 3: Deploy**
Click "Create Web Service" and wait for deployment.

### 4. Vercel Deployment

**Step 1: Install Vercel CLI**
```bash
npm i -g vercel
```

**Step 2: Create vercel.json**
Create `vercel.json` in the root directory:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

**Step 3: Deploy**
```bash
vercel
```

### 5. DigitalOcean App Platform

**Step 1: Create App**
1. Go to DigitalOcean App Platform
2. Click "Create App"
3. Connect your GitHub repository

**Step 2: Configure**
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Run Command**: `gunicorn app:app`

**Step 3: Deploy**
Click "Create Resources" to deploy.

## 🔧 Environment Variables

For production deployment, consider setting these environment variables:

```bash
FLASK_ENV=production
FLASK_DEBUG=False
```

## 📊 Performance Optimization

### 1. Enable Caching
Add Redis caching for better performance:

```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@app.route('/schedule')
@cache.cached(timeout=3600)  # Cache for 1 hour
def schedule():
    # Your existing code
```

### 2. Static File Optimization
- Compress CSS and JavaScript
- Use CDN for Tailwind CSS
- Enable gzip compression

### 3. Database Optimization
- Use connection pooling
- Implement query caching
- Optimize data loading

## 🔒 Security Considerations

### 1. Environment Variables
Never commit sensitive data to version control:
```bash
# .env file (not committed)
SECRET_KEY=your-secret-key
DATABASE_URL=your-database-url
```

### 2. HTTPS
Enable HTTPS on your domain:
- Heroku: Automatic
- Railway: Automatic
- Render: Automatic
- Vercel: Automatic

### 3. Rate Limiting
Add rate limiting to prevent abuse:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

## 📈 Monitoring

### 1. Logging
Add structured logging:
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/nfl_app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('NFL Schedule App startup')
```

### 2. Health Checks
Add a health check endpoint:
```python
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in app.py
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Loading Errors**
   - Ensure all model files are in the `results/` directory
   - Check file permissions

4. **Memory Issues**
   - Optimize data loading
   - Use lazy loading for large datasets
   - Consider using a database instead of CSV files

### Debug Mode
For local development:
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

## 📞 Support

If you encounter issues:
1. Check the logs: `heroku logs --tail` (Heroku)
2. Review the deployment platform's documentation
3. Ensure all dependencies are properly installed
4. Verify environment variables are set correctly

---

**Note**: This is a prototype application. For production use, consider implementing additional security measures, monitoring, and backup strategies. 