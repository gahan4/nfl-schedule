# NFL Scheduling Project

## 📖 Overview
This repository contains code, data, and results related to the creation of an optimal schedule for the 2025 NFL season. The goal of the problem is to maximize viewership of primetime games during the 2025 NFL season, subject to logistical and competitive balance constraints that the league has publicized.

1. **Viewership Model:** For each of the 272 games on the schedule, an estimate is made for how many viewers will watch the game if shown in a primetime slot (TNF, SNF, or MNF), based on simple factors like the popularity and strength of each team.
2. **Schedule Creation:** Each game is assigned to a week/slot pairing in a manner that satisfies scheduling constraints while maximizing primetime viewership.

Technically, the viewership model chosen is a linear regression, while the schedule is created using integer programming.

A (still-in-testing) app that displays the optimal schedule and provides insight into the expected number of viewers for any individual game is available at https://schedule-app.streamlit.app/.

---

## How It Works
### **Data Collection**
- Data retrieved to help understand viewership trends and find league-mandated matchups for 2025.
- Historical NFL viewership data for 2021-22 primetime games was manually added. 
- As a proxy for popularity, team Twitter follower counts were retrieved from Sports M

### **Modeling Approach**
- Two-step process for determining number of likely viewers for primetime game. Firstly, "intrigue score" for each team calculated based on their win percentage in previous season and  
- Integer programming is used to optimize time slot assignments while respecting constraints.
- 

---


## 🚀 How to Reproduce
1. **Clone the Repository:**
```bash
git clone https://github.com/gahan4/nfl-schedule.git
cd nfl-schedule
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Retrieve Data, Create Viewership Models, Run Optimization:**
```bash
python src/main.py
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

### **Packages Used**


---

💬 *For questions, feedback, or collaboration opportunities, feel free to reach out via GitHub issues or pull requests!*



Note also that this project does not utilize Gurobi, Cplex, or any
other commercial solver.
