#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:32:33 2025

@author: neil
"""

import pandas as pd
import numpy as np

def save_schedule(teams, matchups_with_schedule):
    teams.to_csv("results/teams.csv",
                 index=False)
    
    games_to_be_played = flat_to_indices(np.where(solution_vec > .5)[0])     
    
    games = list(zip(*games_to_be_played))
    games_df = pd.DataFrame(games,
                            columns=["Team", "Stadium", "Week", "Slot"])
    games_df.to_csv("results/scheduled_games.csv",
                    index=False)
