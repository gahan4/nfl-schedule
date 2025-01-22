#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:46:08 2025

@author: neil
"""

import streamlit as st
import pandas as pd
import numpy as np

# Inject custom CSS to adjust the layout
st.markdown(
    """
    <style>
    .block-container {
        max-width: 90%; /* Increase the maximum width of the content area */
        padding-left: 5%; /* Adjust left margin */
        padding-right: 5%; /* Adjust right margin */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


teams = pd.read_csv("../results/teams.csv", index_col=False)
scheduled_games = pd.read_csv("../results/scheduled_games.csv", index_col=False)

st.session_state['teams'] = teams
st.session_state['scheduled_games'] = scheduled_games

st.title("Welcome to the NFL Schedule App")

# Create the schedule matrix
schedule_matrix = np.full((18, 32), "", dtype="U4")
for r in range(scheduled_games.shape[0]):
    stadium_num, visiting_team_num, week_num =scheduled_games.loc[r, ['Stadium', 'Team', 'Week']].values
    
    if stadium_num == visiting_team_num:
        continue
    
    home_team = teams.loc[teams['team_id'] == stadium_num, 'team_abbr'].iloc[0]
    visiting_team = teams.loc[teams['team_id'] == visiting_team_num, 'team_abbr'].iloc[0]
    
    schedule_matrix[week_num, visiting_team_num] = f'@{home_team}'
    schedule_matrix[week_num, stadium_num] = visiting_team
    
schedule_matrix = pd.DataFrame(schedule_matrix,
                               index = [i for i in range(1,19)],
                               columns = teams['team_abbr'].values)
schedule_matrix.index.name = 'Wk'

st.write(schedule_matrix)

st.session_state['schedule_matrix'] = schedule_matrix

# Provide a link to go to the team-specific schedule page
#st.write("Want to see a team's schedule? Click below:")
#st.write("[Go to Team Schedule](single_team.py)")

