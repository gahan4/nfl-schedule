# NFL Scheduling Project

## 📖 Overview
This repository contains code, data, and results related to the creation of an optimal schedule for the 2025 NFL season. The goal of the problem is to maximize viewership of primetime games during the 2025 NFL season, subject to logistical and competitive balance constraints that the league has publicized.

1. **Viewership Model:** For each of the 272 games on the schedule, an estimate is made for how many viewers will watch the game if shown in a primetime slot (TNF, SNF, or MNF), based on simple factors like the popularity and strength of each team.
2. **Schedule Creation:** Each game is assigned to a week/slot pairing in a manner that satisfies scheduling constraints while maximizing primetime viewership.

A (still-in-testing) app that displays the optimal schedule and provides insight into the expected number of viewers for any individual game is available at https://schedule-app.streamlit.app/.

---

## How It Works
### **Data Collection**
** Using a variety of public sources, historical data was collected to aid in projecting viewership numbers, such as:
** - Record of each team in each season
- Number of twitter followers of each team
- Size of each team's home market
Additionally, the list of 272 matchups scheduled for the 2025 NFL Regular Season was pulled from the league website.

### **Modeling Approach**
Two-step process for determining number of likely viewers for primetime game. 
- Firstly, an "intrigue score" for each team was derived. This was a single number that represented how intriguing an individual team was to viewers, based on actual viewership trends. The intrigue model credited teams with higher scores when they had more twitter followers and a better record. For interpretability, a score of 100 was considered average and larger numbers being better.
- Secondly, a model was to predict the number of viewers in any particular primetime game was created. This model considered the intrigue scores of the teams in question and the game's slot (TNF, SNF, or MNF). 
- Using the intrigue and viewership models, a projected viewership number for all 2025 games was created.
- Using integer optimization techniques, an optimal schedule that satisfied a host of the league's scheduling constraints was created.

### Where It Falls Short
This schedule probably isn't ready for the prime time. Here are some areas where it falls short, relative to what would be required for a real NFL schedule:
- Viewership data was collected from public sources from just 2 seasons of games (2022-23), and only for games in the traditional primetime windows. Real practitioners would hopefully have a much more robust viewership dataset. 
- Only a small number of variables were tested to create the viewership model, and just 2 were included in the final model. Real practitioners would probably spend more time collecting possible factors for their viewership model and testing different model architectures with their more robust dataset.
- To solve for the optimal schedule, a free solver (called CBC) was run on a personal laptop. Real practitioners would have access to better solvers and bigger machines.
- As a result of the limited computational power available, not every constraint that the league might consider was included. For example, this schedule does not account for international games or dates when a team's stadium might be used by other uses (e.g. concerts). Additionally, certain competition constraints, like restrictions on instances of playing a team coming off its bye, were not used in this process.

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
