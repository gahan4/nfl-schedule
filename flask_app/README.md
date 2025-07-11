# NFL Schedule App - Flask Version

A modern, responsive web application for viewing and analyzing NFL schedules, built with Flask and Tailwind CSS.

## 🚀 Features

- **Modern UI**: Beautiful, responsive design using Tailwind CSS
- **Interactive Schedule**: Color-coded schedule grid showing all teams and weeks
- **Team Analysis**: Detailed individual team pages with statistics and schedules
- **Viewership Projections**: Machine learning-based viewership predictions
- **Mobile Responsive**: Works perfectly on desktop, tablet, and mobile devices

## 🏗️ Architecture

- **Backend**: Flask web framework
- **Frontend**: Tailwind CSS for styling, vanilla JavaScript for interactions
- **Data**: Pandas for data processing, scikit-learn for ML models
- **Optimization**: Google OR-Tools for mathematical optimization

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gahan4/nfl-schedule.git
   cd nfl-schedule
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 🎯 Usage

### Home Page
- Overview of the project and methodology
- Explanation of the intrigue score system
- Links to all major sections

### League Schedule
- Complete 2025 NFL schedule for all 32 teams
- Color-coded by game slot (MNF, SNF, TNF, Sunday)
- Home/away status indicated by background/border colors
- Interactive hover effects for game details

### Team Analysis
- Individual team statistics and rankings
- Complete team schedule with viewership projections
- Game intrigue percentile rankings
- Detailed explanations of metrics

### Analysis Page
- Deep dive into the mathematical optimization process
- Explanation of the two-step modeling approach
- Technical implementation details
- Limitations and future work

## 🎨 Design Features

- **Modern Gradient Navigation**: Beautiful blue gradient header
- **Card-based Layout**: Clean, organized information presentation
- **Hover Effects**: Interactive elements with smooth transitions
- **Color-coded Schedule**: Intuitive visual representation of game slots
- **Responsive Grid**: Adapts to any screen size
- **Professional Typography**: Clear, readable text hierarchy

## 🚀 Deployment Options

### Heroku
1. Create a `Procfile`:
   ```
   web: gunicorn app:app
   ```
2. Add `gunicorn` to requirements.txt
3. Deploy to Heroku

### Railway
1. Connect your GitHub repository
2. Railway will automatically detect Flask and deploy

### Render
1. Create a new Web Service
2. Connect your repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`

### Vercel
1. Install Vercel CLI
2. Create `vercel.json`:
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

## 🔧 Customization

### Styling
- Modify `templates/base.html` for global styling changes
- Update Tailwind classes in individual templates
- Add custom CSS in the `<style>` blocks

### Data
- Update CSV files in the `results/` directory
- Modify the data loading functions in `app.py`
- Add new routes for additional data views

### Features
- Add new pages by creating templates and routes
- Implement additional interactive features with JavaScript
- Extend the optimization model with new constraints

## 📊 Data Sources

- **Teams**: Team statistics, market data, social media metrics
- **Schedule**: Optimized 2025 NFL schedule
- **Models**: Pre-trained scikit-learn models for viewership prediction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- NFL for the inspiration
- Google OR-Tools for optimization capabilities
- Tailwind CSS for the beautiful design system
- Flask community for the excellent web framework

---

**Note**: This is a prototype schedule and should not be used for actual NFL scheduling decisions. The methodology demonstrates the potential for data-driven schedule optimization. 