#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:46:08 2025

@author: neil
"""

import streamlit as st
import pandas as pd

# Comment - yellow background for MNF, green for SNF, purple for TNF
#     appears that color highlight with white text means home, opposite for road

st.set_page_config(page_title="NFL Schedule App", layout="wide", initial_sidebar_state="collapsed")


teams = pd.read_csv("results/teams.csv", index_col=False)
scheduled_games = pd.read_csv("results/matchups_with_schedule.csv", index_col=False)
week_names = sorted(scheduled_games['Week'].unique())
team_names = sorted(teams['team_abbr'].unique())

st.session_state['teams'] = teams
st.session_state['scheduled_games'] = scheduled_games

# Create an empty DataFrame for the schedule table
schedule_df = pd.DataFrame('', index=week_names, columns=team_names)

# Populate the schedule table
for _, row in scheduled_games.iterrows():
    week, home_team, away_team, slot = row['Week'], row['home_team_abbr'], row['away_team_abbr'], row['Slot']
    schedule_df.loc[week, home_team] = f'{away_team}'
    schedule_df.loc[week, away_team] = f'{home_team}'

# Function to get colored text based on slot
def format_opponent_text(opponent, slot, home):
    color_map = {
        'MNF': '#B59410', # this is a dark gold
        'SNF': 'green',
        'TNF': 'purple',
        'Sun': 'gray'
    }
    text_color = 'white' if home else color_map[slot]
    background_color = color_map[slot] if home else 'white'

    cell_style = (
        f"background-color:{background_color}; color:{text_color};"
        f"padding:6px; border-radius:4px; border: 1px solid black;" 
        f"text-align: center; vertical-align: middle;"
        )
    return f"<td style='{cell_style}'>{opponent}</td>"

# Write info to app
st.write("### NFL Schedule Table")

col1, col2 = st.columns([1,2])
# Display legend
with col1:
    st.write("### Legend")
    st.markdown(
        """
        - 🟨 **Gold:** MNF
        - 🟩 **Green:** SNF
        - 🟪 **Purple:** TNF
        - ⚪ **White:** Sun Afternoon
        - **Colored Background:** Home Game
        - **White Background:** Away Game
        """
    )

# Write schedule df
with col2:
    html_table = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    html_table += "<tr><th>Week</th>" + "".join(f"<th>{team}</th>" for team in schedule_df.columns) + "</tr>"
    
    for week in week_names:
        html_table += f"<tr><td>{week}</td>"
        for team in team_names:
            opponent = schedule_df.loc[week, team]
            relevant_game_id = scheduled_games.loc[(scheduled_games['Week'] == week) & 
                                       ((scheduled_games['home_team_abbr'] == team) | 
                                        (scheduled_games['away_team_abbr'] == team)), 'game_id']
            if relevant_game_id.empty: # if team has bye this week
                html_table += "<td style=border: 1px solid black;> </td>"
                continue
            else:
                relevant_game_id = relevant_game_id.iloc[0]
            slot = scheduled_games.loc[scheduled_games['game_id'] == relevant_game_id, 'Slot']
            slot = slot.iloc[0] if not slot.empty else 'N/A'
            home_team_abbr = scheduled_games.loc[scheduled_games['game_id'] == relevant_game_id, 'home_team_abbr'].iloc[0]
            home = 1 if home_team_abbr == team else 0
            formatted_opponent = format_opponent_text(opponent, slot, home) if opponent else ''
            html_table += formatted_opponent
        html_table += "</tr>"
    
    html_table += "</table>"
    st.markdown(html_table, unsafe_allow_html=True)


# Provide a link to go to the team-specific schedule page
#st.write("Want to see a team's schedule? Click below:")
#st.write("[Go to Team Schedule](single_team.py)")

