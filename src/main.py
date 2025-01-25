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
from src.model_viewership import model_viewership
#from create_constraints import get_index, create_constraints
from src.create_objective_function import create_objective_function
from src.solve_problem import get_optimal_solution
#from save_schedule import save_schedule
from src.define_problem import define_problem, get_index

retrain_model = False
num_teams = 32
num_stadiums = num_teams
num_weeks = 18
num_slots = 1

if __name__ == "__main__":
    
    if retrain_model == True:
        intrigue_model, game_viewers_model = model_viewership()

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
    A_eq, A_in, b_eq, b_in = define_problem(teams=teams, matchups=matchups,
                                            num_slots=1)
    print("Completed setting up constraint matrices")
    f = create_objective_function(teams, A_in, A_eq)
    print("Completed setting up objective function")
    opt_sol, opt_objective = get_optimal_solution(A_eq, A_in, b_eq, b_in, f, teams,
                                   num_slots=1)
            
    schedule_matrix = np.full((18, 32), "", dtype="U4")
    for i in range(matchups.shape[0]):
        for j in range(num_weeks):
            for k in range(num_slots):
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
    
    