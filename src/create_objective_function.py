#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:49:59 2025

@author: neil
"""
import numpy as np
from src.define_problem import get_index
from data.load_data import get_jersey_sales_rankings, get_draft_history, add_jersey_sales_metrics, add_draft_intrigue_metrics, \
    add_high_value_qb_metrics
from data.config import *

#def create_objective_function():
#    return np.zeros(A_in.shape[1])

def create_objective_function(teams, matchups, intrigue_model_pipeline, game_viewers_model_pipeline,
                               mean_unscaled_intrigue, std_unscaled_intrigue):
    
    # Add a couple of columns
    teams['year'] = 2024
    teams['join_year'] = 2025
    teams = add_high_value_qb_metrics(teams)
    teams = add_jersey_sales_metrics(teams)
    teams = add_draft_intrigue_metrics(teams)

    
    # First, use intrigue pipeline to find intrigue score for each team
    # Add a dummy 'Window' column with a fixed value
    teams['Window'] = 'SNF'
    teams['SharedMNFWindow'] = 0

    # Run through pipeline
    teams['intrigue_raw'] = intrigue_model_pipeline.predict(teams)
    
    teams = teams.drop(columns=['Window', 'SharedMNFWindow'])
    
    # Apply standardization
    teams["intrigue"] = 100 + 20 * (teams["intrigue_raw"] - mean_unscaled_intrigue) / std_unscaled_intrigue
    
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
                              
    # Add in a couple of additional columns required
    matchups['SharedMNFWindow'] = 0
    matchups['same_division'] = (matchups["team_division_home"] == matchups["team_division_away"]).astype(int)
    
    # Add projected viewers in the case of TNF, SNF, and MNF
    matchups['Window'] = 'TNF'
    matchups['TNF_Viewers'] = game_viewers_model_pipeline.predict(matchups)
        
    matchups['Window'] = 'SNF'
    matchups['SNF_Viewers'] = game_viewers_model_pipeline.predict(matchups)

    matchups['Window'] = 'MNF'
    matchups['MNF_Viewers'] = game_viewers_model_pipeline.predict(matchups)
    
    matchups = matchups.drop(columns=['Window'])
    
    # In data set being used, there were no games in MNF or SNF with average
    # intrigue score below 85. Include in df so that optimization can remove these
    # games from contention.
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



    

    
    
