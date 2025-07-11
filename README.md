# NFL Scheduling Project
[![Flask](https://img.shields.io/badge/Flask-App-blue?logo=flask&logoColor=white)](https://nfl-schedule.onrender.com/)
[![Last Updated](https://img.shields.io/github/last-commit/gahan4/nfl-schedule)](https://github.com/gahan4/nfl-schedule/commits/main)

This project builds a prototype schedule for the 2025 NFL season, optimizing game placements to maximize viewership while satisfying NFL scheduling constraints.

Please explore the app to view the results and learn more about the process used to create the schedule.
[NFL Schedule App on Render](https://nfl-schedule.onrender.com/)

---

## Project Overview

- **Viewership Estimation**  
  Models expected viewership for each of the 272 NFL games based on features like team popularity, historical success, and time slot.

- **Schedule Optimization**  
  Uses integer programming to assign each game to a week and slot in a way that maximizes primetime viewership while satisfying NFL logistics rules.

---

## Repository Structure

- `data/` – Raw and processed datasets
- `src/` – Code for data processing, modeling, and optimization  
- `app/` – Code to create Streamlit app that presents the final results  
- `results/` – Output files, trained models, and plots
  
---

## App Features

The interactive app includes:
- The full optimized 2025 schedule  
- Schedule and intrigue breakdowns for every team  
- Detailed analysis of the methods, techniques, and data sources used to create the schedule

[Open the app](https://schedule-app.streamlit.app/)

---

## License

This project is released for **educational and demonstration purposes** only.  
It is not affiliated with the NFL or its broadcast partners.
