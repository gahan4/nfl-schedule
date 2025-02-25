#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:46:08 2025

@author: neil
"""

import streamlit as st
import pandas as pd
import os

# Comment - yellow background for MNF, green for SNF, purple for TNF
#     appears that color highlight with white text means home, opposite for road

st.set_page_config(page_title="NFL Schedule App", layout="wide")

st.markdown(
    """
    <style>
    body {
        zoom: 0.75;
        -moz-transform: scale(0.75);
        -moz-transform-origin: 0 0;
        height: 100vh;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }
    html, body, [data-testid="stApp"] {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
    }
    div.stTabs {
        display: flex;
        padding-top: 0px; /* Move tabs higher */
        margin-bottom: 0px;
        margin-top: -50px; /* Move tabs even higher */
    }
    /* New styles for wider text display */
    h1, h2, h3, h4, h5, h6, p {
        width: 100%;
        text-align: left;
        white-space: nowrap;
    }
    .game-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 5px 10px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 12px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
    }

   </style>
    """,
    unsafe_allow_html=True
)


teams = pd.read_csv("results/teams.csv", index_col=False)
scheduled_games = pd.read_csv("results/scheduled_games.csv", index_col=False)
# intrigue percentile to be displayed later
scheduled_games['Intrigue_Percentile'] = scheduled_games['SNF_Viewers'].rank(pct=True) * 100


week_names = sorted(scheduled_games['Week'].unique())
team_names = sorted(teams['team_abbr'].unique())

st.session_state['teams'] = teams
st.session_state['scheduled_games'] = scheduled_games

# Create an empty DataFrame for the schedule table
schedule_df = pd.DataFrame('', index=week_names, columns=team_names)

# Populate the schedule table
for _, row in scheduled_games.iterrows():
    week, home_team, away_team, slot = row['Week'], row['team_abbr_home'], row['team_abbr_away'], row['Slot']
    schedule_df.loc[week, home_team] = f'{away_team}'
    schedule_df.loc[week, away_team] = f'{home_team}'

# Function to get colored text based on slot
def format_opponent_text(opponent, slot, home):
    color_map = {
        'MNF': '#B59410', # this is a dark gold
        'SNF': '#32CD32',
        'TNF': '#800080',
        'Sun': 'gray'
    }
    text_color = 'white' if home else color_map[slot]
    background_color = color_map[slot] if home else 'white'

    cell_style = (
        f"background-color:{background_color}; color:{text_color};"
        f"padding:6px; border-radius:4px; border: 1px solid black;" 
        f"text-align: center; vertical-align: middle;"
        )
    return f"<td style='{cell_style}'><b>{opponent}</b></td>"



selected_page = st.tabs(["🏈 League Schedule", "📅 Team Schedule", "📊 Analysis"])

# League Schedule page
with selected_page[0]:
    # Custom CSS to scale the entire app content to 75% and style tabs
    
    # Write info to app
    st.write("### NFL Schedule Table")
    
    # Write schedule df
    html_table = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    html_table += "<tr><th>Week</th>" + "".join(f"<th>{team}</th>" for team in schedule_df.columns) + "</tr>"
    
    for week in week_names:
        html_table += f"<tr><td style='text-align: center; vertical-align: middle;'><b>{week}</b></td>"
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
    
    st.write("### Legend")
    st.markdown("🟨 **Gold:** MNF &nbsp;&nbsp;&nbsp;&nbsp;🟩 **Green:** SNF&nbsp;&nbsp;&nbsp;&nbsp;🟪 **Purple:** TNF&nbsp;&nbsp;&nbsp;&nbsp;⬜ **Gray:** Sun Afternoon")
    st.markdown("**Colored Background:** Home Game, **White Background:** Away Game")
        
# Individual team page
with selected_page[1]:
    st.markdown(
    """
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid black;
        padding: 8px;
        text-align: center;
    }
    th {
        background-color: #f2f2f2;
    }
    .bye-week {
        text-align: center;
        font-weight: bold;
        background-color: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    team_choice = st.selectbox("Select a team:", team_names)
    team_schedule = scheduled_games[(scheduled_games['home_team_abbr'] == team_choice) |
                                    (scheduled_games['away_team_abbr'] == team_choice)]
    team_schedule = team_schedule.sort_values(by='Week')
    team_schedule['Opponent'] = team_schedule.apply(lambda row: f"@ {row['home_team_abbr']}" if team_choice == row['away_team_abbr'] else f"vs {row['away_team_abbr']}", axis=1)
    team_schedule['Date'] = 'Date'
    team_schedule['Opponent_Intrigue'] = team_schedule.apply(lambda row: row['intrigue_home'] if team_choice == row['away_team_abbr'] else row['intrigue_away'], axis=1)

    if not team_schedule.empty:
        st.write(f"### Schedule for {team_choice}")
        
        html_schedule = "<table>"
        html_schedule += "<tr><th>Week</th><th>Date</th><th>Slot</th><th>Opponent</th><th>Opponent Intrigue</th><th>Projected Viewers</th><th>Game Intrigue Percentile</th></tr>"
        
        for wk in week_names:
            if wk in team_schedule['Week'].values:
                row = team_schedule[team_schedule['Week'] == wk].iloc[0]
                html_schedule += f"<tr><td>{row['Week']}</td><td>{row['Date']}</td><td>{row['Slot']}</td><td>{row['Opponent']}</td><td>{row['Opponent_Intrigue']:.0f}</td><td>{row['SNF_Viewers']:.1f}</td><td>{row['Intrigue_Percentile']:.0f}%</td></tr>"
            else: # bye week
                html_schedule += f"<tr><td>{wk}</td><td colspan='6' class='bye-week'>BYE WEEK</td></tr>"

        html_schedule += "</table>"
        st.markdown(html_schedule, unsafe_allow_html=True)
    else:
        st.write("No schedule available for this team.")



    