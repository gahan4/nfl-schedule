#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:40:13 2025

@author: neil
"""

from ortools.linear_solver import pywraplp
import pandas as pd
from data.config import *
from src.define_problem import get_index


def assign_slots(teams, matchups, opt_sol):
    """
    Second level process of problem is to assign slots to games. We will
    define x_ij = 1 if and only if game i is assigned to slot j. The week
    that each matchup will occur in is passed in via the 
    
    5. Teams that play a road game on Monday night cannot play on the road
           the following week
        6. Max 2 Thursday night games per team, max 1 of those games at home
        7. Stadium conflicts - LAR/LAC and NYG/NYJ cannot be home same weeks
        8. Each week contains 1 SNF game, 1 MNF game, and 1 TNF game, with following
            exceptions:
                - Thanksgiving, which contains 3 TNF games, and DAL/DET much each be home
                - Wk 18, which is scheduled after Wk 17, so for sake of this process
                  will have 0 MNF, TNF, or SNF games
        9. Last game of season (Wk 18) must be against divisional opponent
        10. Thursday game constraints:
            - If play road Thursday game, then need to be home previous week
            - All teams playing home Thursday games must play within division
              or same division other conference (i.e. AFC East vs NFC East)
              during previous week
        12. Cannot play same team within 3 week span (i.e. not back-to-back
                                                      or semi-repeater)"""
        
    matchups = matchups.sort_values('game_id')
    matchups['Week'] = -1
    matchups['Slot'] = ""
    num_matchups = matchups.shape[0]
    for i in range(num_matchups):
        for j in range(num_weeks):
            if opt_sol[get_index(i, j, 0)] > .5:
                matchups.at[matchups['game_id' == i], 'Week'] = j
                    
    
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
    # Define decision variables
    x = {}
    for game in matchups['game_id'].values:
        for slot in slots['slot_id'].values:
            x[game, slot] = solver.BoolVar(f'x_{game}_{slot}')
    
    # Objective: Maximize total viewership
    solver.Maximize(solver.Sum(viewership_estimate(game, slot) * x[game, slot] for game in matchups.index for slot in slots['slot_id']))
    
    # Constraint 1 - Must assign some slot to each matchup
    for game in matchups['game_id']:
        solver.Add(solver.Sum(x[game, slot] for slot in slots['slot_id']) == 1)
    
    # Constraint 2 - Max 2 Thursday games per team, max 1 of those at home
    for team in teams['team_id']:
        tnf_slot = slots.loc[slots['slot_desc'] == "TNF", 'slot_id'].iloc[0]
        solver.Add(solver.Sum(x[game, tnf_slot] for game_id in matchups['game_id'] if matchups.loc[game_id, 'home_team_id'] == team) <= 1)
        solver.Add(solver.Sum(x[game, tnf_slot] for game_id in matchups['game_id'] if matchups.loc[game_id, 'home_team_id'] == team or matchups.loc[game_id, 'away_team_id'] == team) <= 2)
    
    # Constraint 3 - Teams that play a road game on Monday cannot play on the road the following week
    mnf_slot = slots.loc[slots['slot_desc'] == "MNF", 'slot_id'].iloc[0]
    for game_id in matchups['game_id']:
        next_week_games = matchups[(matchups['week'] == matchups.loc[game_id, 'week'] + 1)]
        for next_game_id in next_week_games['game_id']:
            if matchups.loc[game_id, 'away_team_id'] == matchups.loc[next_game_id, 'away_team_id']:
                solver.Add(x[game_id, mnf_slot] + [x[next_game_id, i] for i in slots['slot_id']] <= 1)
    
    # If playing a road Thursday game, cannot have been on road the previous week
    for game in matchups['game_id']:
        prev_week_games = matchups[(matchups['Week'] == matchups.loc[game_id, 'Week'] - 1)]
        for prev_game in prev_week_games['game_id']:
            if matchups.loc[game, 'away_team_id'] == matchups.loc[prev_game, 'away_team_id']:
                solver.Add(x[game, 1] + x[prev_game, 1] <= 1)
    
    # Solve the model
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found!")
        for game in matchups.index:
            for slot in slots['slot_id']:
                if x[game, slot].solution_value() > .5:
                    print(f'Game {game} assigned to slot {slots.loc[slot, "slot_desc"]}')
    else:
        print("No optimal solution found.")
