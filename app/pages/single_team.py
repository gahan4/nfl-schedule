#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:49:44 2025

@author: neil
"""

# team_schedule.py
import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta
#from schedule_data import schedule_matrix  # Import the shared schedule data


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


opening_day = '2025-09-04' # Note that this is the Thursday of week 0
opening_day = datetime.strptime(opening_day, '%Y-%m-%d').date()

# Read in session state information
schedule_matrix = st.session_state.schedule_matrix
scheduled_games = st.session_state.scheduled_games
teams = st.session_state.teams

def get_game_date(week, slot):
    # Recall that slot 0 is regular Sunday afternoon game,
    # slot l 1 is TNF, slot is SNF, slot 3 is MNF
    week_days_to_add = 0
    if slot in [0,2]:
        week_days_to_add = 3
    elif slot == 3:
        week_days_to_add = 4
        
    total_days_to_add = int((week-1) * 7 + week_days_to_add)
    return (opening_day + timedelta(total_days_to_add))

# Add in the game date and game time for these scheduled games
scheduled_games['game_date'] = scheduled_games.apply(lambda row: get_game_date(row['Week'], row['Slot']), axis=1)



# Sidebar for selecting a team
selected_team = st.sidebar.selectbox("Select a Team", teams['team_abbr'].values)
selected_team_id = teams.loc[teams['team_abbr'] == selected_team, 'team_id'].iloc[0]

# Filter the schedule for the selected team
team_schedule = scheduled_games.loc[(scheduled_games['Stadium'] == selected_team_id) |
                                        (scheduled_games['Team'] == selected_team_id), :]
team_schedule = team_schedule.loc[team_schedule['Stadium'] != team_schedule['Team'], :]
team_schedule = team_schedule.merge(teams.loc[:, ['team_abbr', 'team_id']],
                    how='left',
                    left_on='Team', 
                    right_on='team_id')
team_schedule = team_schedule.rename(columns={'team_abbr': 'Away'}).drop('team_id', axis=1)
team_schedule = team_schedule.merge(teams.loc[:, ['team_abbr', 'team_id']],
                        how='left',
                        left_on='Stadium', 
                        right_on='team_id')
team_schedule = team_schedule.rename(columns={'team_abbr': 'Home'}).drop('team_id', axis=1)
team_schedule['str_desc'] = np.where(team_schedule['Home'] == selected_team,
                                     team_schedule['Away'],
                                     '@' + team_schedule['Home'])


# Add a column indicating the date of each game

    # CSS for the box styling
box_style = """
<style>
        .calendar-box {
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px;
        height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        text-align: center;
        background-color: #f9f9f9;
        border-radius: 5px;
    }
</style>
"""

month_box_style = """
<style>
    .month-box {
        border: 2px solid #ddd;
        padding: 10px;
        margin: 10px;
        border-radius: 10px;
    }
</style>
"""


def create_month_calendar(year, month, schedule_df):
    cal = calendar.monthcalendar(year, month)
    # note that cal returns a calendar with Monday as the first day
    # of the week
    
    # If considering Dec 2025, also need to add dates up to 1/4/26
    if year == 2025 and month == 12:
        cal[4] = [29,30,31,1,2,3,4]
    
    # Create schedule to dict for easier lookup
    schedule_dict = dict(zip(schedule_df['game_date'], schedule_df['str_desc']))

    # Inject CSS styles into the app
    st.markdown(box_style, unsafe_allow_html=True)


    # Create columns for the calendar header - note that calendar.weekheader(3)
    # returns Mon, Tue, Wed, etc (other integer values would return other ways
    # of formattning the names of days of the week)
    cols = st.columns(7)
    for i, day in enumerate(calendar.weekheader(2).split()):
        cols[i].write(day)
        
    for week in cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if day != 0:
                yr = year
                mn = month
                if (31 in week and 1 in week): # Handle Dec/Jan transition
                    yr = 2026
                    mn = 1
                date = datetime(yr, mn, day).date()
                # Add games or just the date inside the box
                if date in schedule_dict:
                    content = f"""
                    <div class="calendar-box">
                        <div><strong>{day}</strong></div>
                        <div>{schedule_dict[date]}</div>
                    </div>
                    """
                else:
                    content = f"""
                    <div class="calendar-box">
                        <div><strong>{day}</strong></div>
                    </div>
                    """
                cols[i].markdown(content, unsafe_allow_html=True)
            else:
                cols[i].write("")
        

st.title(f"{selected_team} 2025 Schedule")

# Top row contains September through November

# CSS to style the month container
# Inject CSS styles
st.markdown(month_box_style, unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    #st.markdown('<div class="month-box">', unsafe_allow_html=True)
    st.subheader("September 2025")
    create_month_calendar(2025, 9, team_schedule)
    #st.markdown('</div>', unsafe_allow_html=True)
with col2:
    #st.markdown('<div class="month-box">', unsafe_allow_html=True)
    st.subheader("October 2025")
    create_month_calendar(2025, 10, team_schedule)
    #st.markdown('</div>', unsafe_allow_html=True)
    
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="month-box">', unsafe_allow_html=True)
    st.subheader("November 2025")
    create_month_calendar(2025, 11, team_schedule)
    st.markdown('</div>', unsafe_allow_html=True)
    
with col2:
    st.markdown('<div class="month-box">', unsafe_allow_html=True)
    st.subheader("December 2025")
    create_month_calendar(2025, 12, team_schedule)
    st.markdown('</div>', unsafe_allow_html=True)


