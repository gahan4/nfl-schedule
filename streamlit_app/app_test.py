#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:59:40 2025

@author: neil
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="NFL Schedule App", layout="wide")
st.markdown(
    """
    <style>
    body {
        zoom: 0.7;
    }
    div.stTabs {
        display: flex;
        padding-top: 0px; /* Move tabs higher */
        margin-bottom: 0px;
        margin-top: -50px; /* Move tabs even higher */
    }
    /* New styles for wider text display */
    h1, h2, h3, h4, h5, h6, p {
        white-space: nowrap;
    }
    th {
        font-size: 18px;
    }
    .week-highlight {
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    .bye-week {
        text-align: center;
        font-weight: bold;
        background-color: #71797E;
    }
    .stApp {
        background-color: #0e1117;
        color: white
    }
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        font-weight: bold;
    }
   </style>
    """,
    unsafe_allow_html=True
)

# Function to format the color for the intrigue perentile
def get_intrigue_color(percentile):
    '''
    Function to create the HTML style for the intrigue percentile column. Green
    background is very high ranked game, red background is very low ranked game.
    
    Parameters
    ----------
    percentile : Int
        The percentile (from 0 to 100) that the game's intrigue falls in

    Returns
    -------
    str
        HTML style string.

    '''
    rgb_high = (49, 222, 40) # the color for 100th percentile
    rgb_low = (212, 36, 73) # color for 0th percentils
    # Convert percentile to a color scale from red (low) to white (mid) to green (high)
    color_int = [int((255 - rgb_low[x])/50.0 * percentile + rgb_low[x]) if percentile <= 50 else int((rgb_high[x] - 255) / 50 * percentile + 2 * 255 - rgb_high[x]) for x in range(3)]
    color_hex = [hex(color_int[x])[2:] for x in range(3)]
    color_hex = [f"0{color_hex[x]}" if len(color_hex[x]) == 1 else color_hex[x] for x in range(3)]
    color_string = "".join(color_hex).upper()
    
    text_color = 'black' if percentile >= 25 and percentile <= 75 else 'white'
    
    cell_style = (
        f"background-color: #{color_string}; color:{text_color};"
        "width: 150px;"
        f"padding:6px; border-radius:4px; border: 1px solid black;" 
        "font-weight: bold;"
        "text-align: center;"
    )
    
    return f"<td style='{cell_style}'><b>{percentile:.0f}</b></td>"



teams = pd.read_csv("results/teams.csv", index_col=False)
scheduled_games = pd.read_csv("results/scheduled_games.csv", index_col=False)
# intrigue percentile to be displayed later
scheduled_games['Intrigue_Percentile'] = scheduled_games['SNF_Viewers'].rank(pct=True) * 100

week_names = sorted(scheduled_games['Week'].unique())
team_names = sorted(teams['team_abbr'].unique())


selected_page = st.tabs(["🏈 League Schedule", "📅 Team Schedule", "📊 Analysis"])

# League Schedule page
    

team_choice = st.selectbox("Select a team:", team_names)
team_schedule = scheduled_games[(scheduled_games['home_team_abbr'] == team_choice) |
                            (scheduled_games['away_team_abbr'] == team_choice)]
team_schedule = team_schedule.sort_values(by='Week')
team_schedule['Opponent'] = team_schedule.apply(lambda row: f"@ {row['home_team_abbr']}" if team_choice == row['away_team_abbr'] else f"vs {row['away_team_abbr']}", axis=1)
team_schedule['Date'] = 'Date'
team_schedule['Opponent_Intrigue'] = team_schedule.apply(lambda row: row['intrigue_home'] if team_choice == row['away_team_abbr'] else row['intrigue_away'], axis=1)


col1, col2 = st.columns([2, 1], gap = "large") 
# In the left column, display 
with col1:  
    
    st.write(f"### Schedule for {team_choice}")
    
    html_schedule = "<table>"
    html_schedule += "<tr><th>Week</th><th>Date</th><th>Slot</th><th>Opponent</th><th>Opponent Intrigue</th><th>Projected Viewers</th><th style='min-width: 150px;'>Game Intrigue Percentile</th></tr>"
    
    for wk in week_names:
        if wk in team_schedule['Week'].values:
            row = team_schedule[team_schedule['Week'] == wk].iloc[0]
            html_schedule += f"<tr><td>{row['Week']}</td><td>{row['Date']}</td><td>{row['Slot']}</td><td>{row['Opponent']}</td><td>{row['Opponent_Intrigue']:.0f}</td><td>{row['SNF_Viewers']:.1f}</td>"
            html_schedule += get_intrigue_color(row['Intrigue_Percentile'])
            html_schedule += "</tr>"
        else: # bye week
            html_schedule += f"<tr><td>{wk}</td><td colspan='6' class='bye-week'>BYE WEEK</td></tr>"

    html_schedule += "</table>"
    st.markdown(html_schedule, unsafe_allow_html=True)


with col2:
    # Display additional team information in an HTML table        
    team_info = teams[teams['team_abbr'] == team_choice].iloc[0]
    
    record = f"{team_info['W']}-{team_info['L']}"
    record_rank = teams['WinPct'].rank(ascending=False, method='min')[teams['team_abbr'] == team_choice].values[0]
    twitter_followers = team_info['twitter_followers'] / 1000000
    if twitter_followers <= 1:
        twitter_followers = int(twitter_followers * 1000)
        twitter_followers_string = f"{twitter_followers} K"
    else:
        twitter_followers_string = f"{twitter_followers:.1f} M"
    twitter_rank = teams['twitter_followers'].rank(ascending=False, method='min')[teams['team_abbr'] == team_choice].values[0]
    intrigue = team_info['intrigue']
    intrigue_rank = teams['intrigue'].rank(ascending=False, method='min')[teams['team_abbr'] == team_choice].values[0]
    
    team_info_df = pd.DataFrame({
        "Metric": ["Record", "Record Rank", "Twitter Followers", "Twitter Rank", "Intrigue Score", "Intrigue Rank"],
        "Value": [record, record_rank, twitter_followers_string, twitter_rank, intrigue_rank, intrigue_rank]
    })        
    
    html_team_info = f"""
    <table>
        <tr>
            <th>Category</th>
            <th>Value</th>
            <th>Rank</th>
        </tr>
        <tr>
            <td>Record</td>
            <td>{record}</td>
            <td>{int(record_rank)}</td>
        </tr>
        <tr>
            <td>Twitter Followers</td>
            <td>{twitter_followers_string}</td>
            <td>{int(twitter_rank)}</td>
        </tr>
        <tr>
            <td>Intrigue Score</td>
            <td>{int(intrigue)}</td>
            <td>{int(intrigue_rank)}</td>
        </tr>
    </table>
    """
    

    st.markdown(html_team_info, unsafe_allow_html=True)

