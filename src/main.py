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

from load_data import get_teams_and_standings, add_popularity_metrics
from model_viewership import model_viewership
from create_constraints import get_index, create_constraints
from create_objective_function import create_objective_function
from solve_problem import get_optimal_solution
from save_schedule import save_schedule

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
    
    A_eq, A_in, b_eq, b_in = create_constraints(teams)
    print("Completed setting up constraint matrices")
    f = create_objective_function(teams, A_in, A_eq)
    print("Completed setting up objective function")
    opt_sol = get_optimal_solution(A_eq, A_in, b_eq, b_in, f, teams,
                                   num_slots=1)
    print("Found optimal solution")
    
    #save_schedule(teams, opt_sol.x)
    
    #returned_as_one = [flat_to_indices(x) for x in np.where(opt_sol.x)[0]]
    
    schedule_matrix = np.full((18, 32), "", dtype="U4")
    inc = 0
    for i in range(num_teams):
        for j in range(num_stadiums):
            for k in range(num_weeks):
                for l in range(num_slots):
                    if opt_sol.x[get_index(i,j,k,l)] > .5:
                        if i != j:
                            schedule_matrix[k, i] = "@" + teams.loc[teams['team_id'] == j, 'team_abbr'].iloc[0]
                            schedule_matrix[k, j] = teams.loc[teams['team_id'] == i, 'team_abbr'].iloc[0]
                        
    schedule_matrix = pd.DataFrame(schedule_matrix)
    schedule_matrix.columns = teams['team_abbr']

    
    