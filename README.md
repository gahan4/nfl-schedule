# NFL Scheduling Project

## 📖 Overview
This repository contains code, data, and results related to creating an optimal schedule for the 2025 NFL season. The goal of the problem was to maximize viewership of primetime games during the 2025 NFL season, subject to logistical and competitive balance constraints that the league has publicized.

1. **Viewership Model:** For each of the 272 games on the schedule, an estimate is made for how many viewers will watch the game if shown in a primetime slot (TNF, SNF, or MNF). 
2. **Schedule Creation:** Each game is assigned to a week/slot pairing. 

The approach leverages mathematical optimization techniques to balance expected viewership while adhering to league-mandated constraints.

A (still-in-testing) app that displays the optimal schedule and provides insight into the expected number of viewers for any individual game is available at https://schedule-app.streamlit.app/.

---

## 🎯 Project Goals
- **Maximize Viewership:** Assign time slots to maximize expected viewership in primetime slots.
- **Respect Constraints:**
  - Max 2 Thursday games per team, with no more than 1 at home.
  - No back-to-back road games after Monday night or before Thursday night games.
  - Ensure fairness in prime-time exposure across all teams.

---

## How It Works
### **Data Collection**
- Data retrieved to help for modeling viewership data and .
- Historical NFL viewership data.
- Team popularity metrics.
- Game-specific intrigue scores (min and max intrigue).

### **Modeling Approach**
- Integer programming is used to optimize time slot assignments while respecting constraints.
- The objective function maximizes expected viewership, weighted by intrigue and historical trends.

### **Key Outputs**
- Optimal schedule with time slot assignments.
- Projected viewership for each game.
- Intrigue scores to highlight the most exciting matchups.
- App deployed in experimental stages at https://schedule-app.streamlit.app/.

## 🚀 How to Reproduce
1. **Clone the Repository:**
```bash
git clone https://github.com/yourusername/NFL-Scheduling-Project.git
cd NFL-Scheduling-Project
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Optimization:**
```bash
python src/schedule_optimizer.py
```

---

## 📝 Repository Structure
```
nfl-schedule/
│
├─ 📄 README.md        # Project overview
├─ 📄 data/            # Code to scrape data, csv's with manually scraped data
├─ 🟢 src/             # Python scripts for data processing and modeling
├─ 📊 results/         # A directory containing stored results information
└─ 📈 app/             # Python scripts to run the Streamlit app
```

---

## Acknowledgments
- NFL viewership data sourced from publicly available datasets.
- NFL and Gurobi Videos
- https://www.youtube.com/watch?v=bS1xGetyrh0&ab_channel=NFL, 


---

💬 *For questions, feedback, or collaboration opportunities, feel free to reach out via GitHub issues or pull requests!*



Note also that this project does not utilize Gurobi, Cplex, or any
other commercial solver.
