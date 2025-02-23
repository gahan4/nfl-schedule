#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:48:23 2025

@author: neil
"""

import numpy as np
import pandas as pd
import os as os

os.chdir('/Users/neil/Documents/Projects/NFL Schedule')

from data.load_data import get_teams_and_standings, add_popularity_metrics, get_matchups
from data.config import *
from src.model_viewership import model_viewership
from src.create_objective_function import create_objective_function
from src.solve_problem import get_optimal_solution
from src.define_problem import define_problem, get_index

retrain_model = True

if __name__ == "__main__":
    
    if retrain_model == True:
        intrigue_model, game_viewers_model, mean_intrigue_unscaled, std_intrigue_unscaled = model_viewership()

    teams = get_teams_and_standings(2024)
    teams = add_popularity_metrics(teams)
    matchups = get_matchups()
    
    matchups = matchups.merge(teams[['team_id', 'team_name', 'team_abbr', 'team_division']],
                              how='inner',
                       left_on='Home', right_on='team_name'
        ).rename(columns={'team_id': 'home_team_id', 'team_abbr': 'home_team_abbr',
                          'team_division': 'home_division'}
                 ).drop(columns=['team_name']).merge(
           teams[['team_id', 'team_name', 'team_abbr', 'team_division']],
                          how='inner',
                          left_on='Away', right_on='team_name'
           ).rename(columns={'team_id': 'away_team_id', 'team_abbr': 'away_team_abbr',
                             'team_division': 'away_division'}
                    ).drop(columns=['team_name'])
    matchups = matchups.sort_values(by='game_id')

    #A_eq, A_in, b_eq, b_in = create_constraints(teams)
    f, matchups = create_objective_function(teams, matchups, intrigue_model, game_viewers_model,
                                   mean_intrigue_unscaled, std_intrigue_unscaled)
    print("Completed setting up objective function")
    A_eq, A_in, b_eq, b_in = define_problem(teams, matchups)
    print("Completed setting up constraient matrices")
    opt_sol, opt_objective = get_optimal_solution(A_eq, A_in, b_eq, b_in, f)
            
    schedule_matrix = np.full((18, 32), "", dtype="U4")
    for i in range(matchups.shape[0]):
        for j in range(NUM_WEEKS):
            for k in range(NUM_SLOTS):
                this_index = get_index(i,j,k)
                if opt_sol[this_index] > .5:
                    home_team_id = matchups.loc[matchups['game_id'] == i, 'home_team_id'].iloc[0]
                    away_team_id = matchups.loc[matchups['game_id'] == i, 'away_team_id'].iloc[0]
                    if len(schedule_matrix[j, home_team_id]) > 0:
                        print(i,j,k)
                    schedule_matrix[j, home_team_id] = teams.loc[teams['team_id'] == away_team_id, 'team_abbr'].iloc[0]
                    schedule_matrix[j, away_team_id] = "@" + teams.loc[teams['team_id'] == home_team_id, 'team_abbr'].iloc[0]

    schedule_matrix = pd.DataFrame(schedule_matrix)
    schedule_matrix.columns = teams['team_abbr']
    
    matchups_with_schedule = matchups.copy()
    matchups_with_schedule['Week'] = -1
    matchups_with_schedule['Slot'] = ""
    for i in range(NUM_MATCHUPS):
        for j in range(NUM_WEEKS):
            for k in range(NUM_SLOTS):
                if opt_sol[get_index(i,j,k)] > .5:
                    matchups_with_schedule.loc[matchups_with_schedule['game_id'] == i, 'Week'] = j + 1
                    matchups_with_schedule.loc[matchups_with_schedule['game_id'] == i, 'Slot'] = slots.loc[slots['slot_id'] == k, 'slot_desc'].iloc[0]
    
