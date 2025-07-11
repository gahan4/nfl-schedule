#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 00:13:29 2025

@author: neil
"""

import streamlit as st
import pandas as pd

teams = pd.read_csv("results/teams.csv", index_col=False)
scheduled_games = pd.read_csv("results/scheduled_games.csv", index_col=False)
# intrigue percentile to be displayed later
scheduled_games['Intrigue_Percentile'] = scheduled_games['SNF_Viewers'].rank(pct=True) * 100

week_names = sorted(scheduled_games['Week'].unique())
team_names = sorted(teams['team_abbr'].unique())

selected_page = st.tabs(["🏈 League Schedule", "📅 Team Schedule", "📊 Analysis"])


with selected_page[0]:
    st.dataframe(scheduled_games[['Week', 'Slot', 'Home', 'Away']])
    
    
with selected_page[1]:
    team_choice = st.selectbox("Select a team:", team_names)
    team_schedule = scheduled_games[(scheduled_games['home_team_abbr'] == team_choice) |
                                (scheduled_games['away_team_abbr'] == team_choice)]

    col1, col2 = st.columns(2)
    with col1:
        
    with col2:
        st.selectbox("District", ["AnotherDistrict1", "AnotherDistrict2"])