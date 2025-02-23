#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:49:59 2025

@author: neil
"""
import numpy as np
from src.define_problem import get_index
from data.config import *

#def create_objective_function():
#    return np.zeros(A_in.shape[1])

def create_objective_function(teams, matchups, intrigue_model, game_viewers_model,
                               mean_unscaled_intrigue, std_unscaled_intrigue):
    teams['intrigue_unscaled'] = intrigue_model.params['WinPct'] * teams['WinPct'] + \
        intrigue_model.params['twitter_followers'] * teams['twitter_followers']
    teams['intrigue'] = (teams['intrigue_unscaled'] - mean_unscaled_intrigue) / \
        std_unscaled_intrigue * 20 + 100
    
    # Join in info for each teeam
    matchups = matchups.merge(teams.loc[:,['team_abbr','intrigue', 'team_division']], 
                          how='inner',
                          left_on=['away_team_abbr'], 
                          right_on=['team_abbr']).merge(
                        teams.loc[:,['team_abbr','intrigue', 'team_division']], 
                        how='inner',
                        left_on=['home_team_abbr'], 
                        right_on=['team_abbr'],
                        suffixes=("_away", "_home"))
                              
    matchups['max_intrigue'] = np.maximum(matchups['intrigue_away'],
                                                              matchups['intrigue_home'])
    matchups['min_intrigue'] = np.minimum(matchups['intrigue_away'],
                                                              matchups['intrigue_home'])
    matchups['max_above_average'] = matchups['max_intrigue'].apply(lambda x: 0 if x < 100 else x - 100)
    matchups['min_above_average'] = matchups['min_intrigue'].apply(lambda x: 0 if x < 100 else x - 100)
    matchups['two_elite_teams'] = np.where(matchups['min_intrigue'] >= 120, 1, 0)
    matchups['two_aavg_teams'] = np.where((matchups['min_intrigue'] >= 110) &
                                        (matchups['min_intrigue'] < 120), 1, 0)
    
    
    # Add projected viewers in the case of TNF, SNF, and MNF
    matchups['Window'] = 'TNF'
    matchups['TNF_Viewers'] = game_viewers_model.predict(matchups)
        
    matchups['Window'] = 'SNF'
    matchups['SNF_Viewers'] = game_viewers_model.predict(matchups)

    matchups['Window'] = 'MNF'
    matchups['MNF_Viewers'] = game_viewers_model.predict(matchups)
    
    matchups = matchups.drop(columns=['Window'])
    
    # In data set being used, there were no games in MNF or SNF with average
    # intrigue score below 85
    matchups['arithmetic_mean_intrigue'] = (matchups['intrigue_home'] +
                                            matchups['intrigue_away']) / 2.0
    
    # Create the objective function by leveraging the get_index function
    objective_function = np.zeros(int(NUM_MATCHUPS * NUM_WEEKS *  NUM_SLOTS))
    for i in range(NUM_MATCHUPS):
        for j in range(NUM_WEEKS):
            for k in range(NUM_SLOTS):
                slot_desc = slots.loc[slots['slot_id'] == k, 'slot_desc'].iloc[0]
                if slot_desc == 'MNF':
                    objective_function[get_index(i,j,k)] = matchups.loc[matchups['game_id'] == i, 'MNF_Viewers'].iloc[0]
                elif slot_desc == 'TNF':
                    objective_function[get_index(i,j,k)] = matchups.loc[matchups['game_id'] == i, 'TNF_Viewers'].iloc[0]
                elif slot_desc == "SNF":
                    objective_function[get_index(i,j,k)] = matchups.loc[matchups['game_id'] == i, 'SNF_Viewers'].iloc[0]
                else:
                    objective_function[get_index(i,j,k)] = 0
                    
    return objective_function, matchups



    

    
    
