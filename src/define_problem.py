#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:22:13 2025

@author: neil
"""

import pandas as pd
import numpy as np
from data.config import *
from scipy.sparse import lil_matrix

def get_index(i, j, k,):
    return i*NUM_WEEKS*NUM_SLOTS + j*NUM_SLOTS + k

def get_inverse_index(index):
    i = index // (NUM_WEEKS * NUM_SLOTS)
    remaining_index = index % (NUM_WEEKS * NUM_SLOTS)
    j = remaining_index // NUM_SLOTS
    k = remaining_index % NUM_SLOTS
    return i, j, k

def define_problem(teams, matchups):
    '''
        1. Each matchup must occur once in season
        2. Each team must play 17 games, at most once per week, bye
           between weeks 5-14
        3. Avoid having 3 consecutive road games
        4. Stadium conflicts - LAR/LAC and NYG/NYJ cannot be home same weeks
        5. Last game of season (Wk 18) must be against divisional opponent
        6. Can't play back-to-back or semi-repeater
        7. Games where team must be home/road
        8. Each week contains 1 SNF game, 1 MNF game, and 1 TNF game, with following
            exceptions:
                - Thanksgiving, which contains 3 TNF games, and DAL/DET much each be home
                - Wk 18, which is scheduled after Wk 17, so for sake of this process
                  will have 0 MNF, TNF, or SNF games
        9.  Max 2 Thursday night games per team, max 1 of those games at home
        10. Teams that play a road game on Monday cannot play on the road the following week
        11. Thursday game constraints:
            - If play road Thursday game, then need to be home previous week
            - All teams playing home Thursday games must play within division
              or same division other conference (i.e. AFC East vs NFC East)
              during previous week
            - Teams that play Thursday after Thanksgiving must have played
                on Thanksgiving
            - Teams that play on Thursday can't have played previous SNF or MNF
            - Cannot travel more than two time zones for Thursday game
        12. Minimum threshold of game quality for primetime game
        13. Max 5 total primetime games per team, max 2 in any 5 week stretch
        14. Cannot be away for first 2 games of seaon or final two games of season
        15. In any 5 week stretch, cannot be home more than 3 games

    Parameters
    ----------
    teams : TYPE
        DESCRIPTION.
    matchups : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

        
    
    # One variable per matchup/week/slot pairing
    num_vars = NUM_MATCHUPS * NUM_WEEKS * NUM_SLOTS
    A_eq = lil_matrix((10000, num_vars))
    A_in = lil_matrix((20000, num_vars))
    b_eq = np.zeros(10000)
    b_in = np.zeros(20000)
    r_eq = -1
    r_in = -1
    
    
    # 1. Each matchup must occur at some point in season
    for i in range(NUM_MATCHUPS):
        r_eq = r_eq + 1
        for j in range(NUM_WEEKS):
            for k in range(NUM_SLOTS):
                A_eq[r_eq, get_index(i,j,k)] = 1
        b_eq[r_eq] = 1
        
                
    # 2. Each team must play at most 1 game per week, exactly one game per week
    #    in the period before week 5 and after week 14
    for tm in teams['team_id'].values:
        team_matchups = matchups.loc[(matchups['home_team_id'] == tm) |
                                     (matchups['away_team_id'] == tm), 'game_id'].values
        for j in range(NUM_WEEKS):
            if (j < 4) or (j > 13):
                r_eq = r_eq + 1
                for i in team_matchups:
                    for k in range(NUM_SLOTS):
                        A_eq[r_eq, get_index(i, j, k)] = 1
                b_eq[r_eq] = 1
            else:
                r_in = r_in + 1
                for i in team_matchups:
                   for k in range(NUM_SLOTS):
                       A_in[r_in, get_index(i, j, k)] = 1
                b_in[r_in] = 1
    
    # 3. Avoid 3 consecutive road games
    for tm in teams['team_id'].values:
        team_home_matchups = matchups.loc[matchups['home_team_id'] == tm, 'game_id'].values
        for j in range(NUM_WEEKS - 2):
            r_in = r_in + 1
            for k in range(NUM_SLOTS):
                for i in team_home_matchups:
                    A_in[r_in, get_index(i, j, k)] = -1
                    A_in[r_in, get_index(i, j + 1, k)] = -1
                    A_in[r_in, get_index(i, j + 2, k)] = -1
            b_in[r_in] = -1
                
    
    # 4. Stadium conflicts - LAR/LAC and NYG/NYJ cannot be home same weeks
    la_home_matchups = matchups.loc[matchups['home_team_abbr'].isin(['LA','LAC']), 'game_id'].values
    for j in range(NUM_WEEKS):
        r_in = r_in + 1
        for i in la_home_matchups:
            for k in range(NUM_SLOTS):
                A_in[r_in, get_index(i,j,k)] = 1
        b_in[r_in] = 1
    
    ny_home_matchups = matchups.loc[matchups['home_team_abbr'].isin(['NYG','NYJ']), 'game_id'].values
    for j in range(NUM_WEEKS):
        r_in = r_in + 1
        for i in ny_home_matchups:
            for k in range(NUM_SLOTS):
                A_in[r_in, get_index(i,j,k)] = 1
        b_in[r_in] = 1
        
    
        
    # 5. Last game of season must be against divisional opp - to minimize number
    #    of constraints and ease in solve, will code this up by setting many values
    #    equal to 0
    non_divisional_matchups = matchups.loc[matchups['home_division'] != matchups['away_division'],'game_id'].values
    r_eq = r_eq + 1
    for i in non_divisional_matchups:
        for k in range(NUM_SLOTS):
            A_eq[r_eq, get_index(i, 17, k)] = 1
    b_eq[r_eq] = 0
    
    # 6. Can't play back-to-back or semi-repeater
    # First, find the pairs of matchups that are same team home and away
    same_team_matchups = matchups.merge( matchups,
                                        how='inner',
                   left_on=['Home', 'Away'],
                   right_on=['Away','Home'])[["Home_x","Away_x", "game_id_x", "game_id_y"]]
    # To avoid having dups, filter so that game_id_x < game_id_y
    same_team_matchups = same_team_matchups.loc[same_team_matchups['game_id_x'] <
                                                same_team_matchups['game_id_y']].reset_index()
    for stm in range(same_team_matchups.shape[0]):
        game_id_x = same_team_matchups['game_id_x'].iloc[stm]
        game_id_y = same_team_matchups['game_id_y'].iloc[stm]
        for j in range(NUM_WEEKS - 2):
            r_in = r_in + 1
            for k in range(NUM_SLOTS):
                A_in[r_in, get_index(game_id_x, j, k)] = 1
                A_in[r_in, get_index(game_id_x, j + 1, k)] = 1
                A_in[r_in, get_index(game_id_x, j + 2, k)] = 1
                A_in[r_in, get_index(game_id_y, j, k)] = 1
                A_in[r_in, get_index(game_id_y, j + 1, k)] = 1
                A_in[r_in, get_index(game_id_y, j + 2, k)] = 1
            b_in[r_in] = 1
            
    # 7. Games where team must be home/road
    #    NOTE - Actual practitioners would know the full gamut here
    #    DET and DAL must be home for Thanksgiving (Week 12)
    #    PHI must be home to open season on Thursday
    tnf_slot = slots.loc[slots['slot_desc'] == "TNF", 'slot_id'].iloc[0]
    for tm in ['Dallas Cowboys', 'Detroit Lions']:
        team_home_matchups = matchups.loc[matchups['Home'] == tm, 'game_id'].values
        r_eq = r_eq + 1
        for i in team_home_matchups:
            A_eq[r_eq, get_index(i, THANKSGIVING_WEEK, tnf_slot)] = 1
        b_eq[r_eq] = 1
    for tm in ['Philadelphia Eagles']:
        team_home_matchups = matchups.loc[matchups['Home'] == tm, 'game_id'].values
        r_eq = r_eq + 1
        for i in team_home_matchups:
            A_eq[r_eq, get_index(i, 0, tnf_slot)] = 1
        b_eq[r_eq] = 1

    
    # 8. Each week contains 1 SNF game, 1 MNF game, and 1 TNF game, with following
    #   exceptions:
    #    - Thanksgiving, which contains 3 TNF games, and DAL/DET much each be home
    #    - Wk 18, which is scheduled after Wk 17, so for sake of this process
    #       will have 0 MNF, TNF, or SNF games
    if NUM_SLOTS == 4:
        for k in range(NUM_SLOTS):
            slot_desc = slots.loc[slots['slot_id'] == k, 'slot_desc'].iloc[0]
            if slot_desc in ['SNF', 'MNF']:
                for j in range(NUM_WEEKS):
                    if j < 17:
                        r_eq = r_eq + 1
                        for i in range(NUM_MATCHUPS):
                            A_eq[r_eq, get_index(i,j,k)] = 1
                        b_eq[r_eq] = 1  
                    elif j == 17: # if last week of season, no game in this slot
                        r_eq = r_eq + 1
                        for i in range(NUM_MATCHUPS):
                            A_eq[r_eq, get_index(i,j,k)] = 1
                        b_eq[r_eq] = 0
            elif slot_desc == 'TNF':
                for j in range(NUM_WEEKS):
                    if j < 17 and j != THANKSGIVING_WEEK:
                        r_eq = r_eq + 1
                        for i in range(NUM_MATCHUPS):
                            A_eq[r_eq, get_index(i,j,k)] = 1
                        b_eq[r_eq] = 1
                    elif j == THANKSGIVING_WEEK:
                        r_eq = r_eq + 1
                        for i in range(NUM_MATCHUPS):
                            A_eq[r_eq, get_index(i,j,k)] = 1
                        b_eq[r_eq] = 3
                    elif j == 17:
                        r_eq = r_eq + 1
                        for i in range(NUM_MATCHUPS):
                            A_eq[r_eq, get_index(i,j,k)] = 1
                        b_eq[r_eq] = 0

    # 9. Max 2 Thursday games per team, max 1 of those at home
    tnf_slot = slots.loc[slots['slot_desc'] == "TNF", 'slot_id'].iloc[0]
    for tm in range(NUM_TEAMS):
        r_in = r_in + 1
        tm_matchups = matchups.loc[(matchups['away_team_id'] == tm) |
                                   (matchups['home_team_id'] == tm), 'game_id']
        for i in tm_matchups:
            for j in range(NUM_WEEKS):
                A_in[r_in, get_index(i, j, tnf_slot)] = 1
        b_in[r_in] = 2
        
        r_in = r_in + 1
        tm_matchups_home = matchups.loc[matchups['home_team_id'] == tm, 'game_id']
        for i in tm_matchups_home:
            for j in range(NUM_WEEKS):
                A_in[r_in, get_index(i, j, tnf_slot)] = 1
        b_in[r_in] = 1

    
    # 10. Monday restrictions:
    #     - Teams that play a road game on Monday cannot play on the road the following week.
    #     - Any team that plays on Monday can't play on the next Thursday 
    mnf_slot = slots.loc[slots['slot_desc'] == "MNF", 'slot_id'].iloc[0]
    for tm in range(NUM_TEAMS):
        team_away_matchups = matchups.loc[matchups['away_team_id'] == tm, 'game_id'].values
        team_home_matchups = matchups.loc[matchups['home_team_id'] == tm, 'game_id'].values
        for j in range(NUM_WEEKS - 1):
            r_in = r_in + 1
            for i in team_away_matchups:
                A_in[r_in, get_index(i, j, mnf_slot)] = 1
                for k in range(NUM_SLOTS):
                    A_in[r_in, get_index(i, j+1, k)] = 1
            b_in[r_in] = 1
            
            r_in = r_in + 1
            for i in np.concatenate((team_away_matchups, team_home_matchups)):
                A_in[r_in, get_index(i, j, mnf_slot)] = 1
                A_in[r_in, get_index(i, j+1, tnf_slot)] = 1
            b_in[r_in] = 1
                        
    
    # 11. Thursday Rules
    #     - If playing a road Thursday game, cannot have been on road the previous week. 
    #     - If playing  a home Thursday game, must have either played in same geographic division
    #      previous week (i.e. NFC East team must have played in NFC East or AFC East stadium)
    #     - The teams that play on Thursday the week after Thanksgiving must have also played
    #       on Thanksgiving
    #     - Teams that play on Thursday can't have played previous SNF or MNF
    #     - Cannot travel more than two time zones for Thursday game
    tnf_slot = slots.loc[slots['slot_desc'] == "TNF", 'slot_id'].iloc[0]
    mnf_slot = slots.loc[slots['slot_desc'] == "MNF", 'slot_id'].iloc[0]
    snf_slot = slots.loc[slots['slot_desc'] == "SNF", 'slot_id'].iloc[0]
    matchups['home_division_geography'] = matchups['team_division_home'].apply(lambda x: x.split()[1])
    matchups['away_division_geography'] = matchups['team_division_away'].apply(lambda x: x.split()[1])
    for tm in range(NUM_TEAMS):
        team_away_matchups = matchups.loc[matchups['away_team_id'] == tm, 'game_id'].values
        for j in range(1, NUM_WEEKS):
            r_in = r_in + 1
            for i in team_away_matchups:
                A_in[r_in, get_index(i, j, tnf_slot)] = 1
                for j in range(NUM_SLOTS):
                    A_in[r_in, get_index(i, j-1, k)] = 1
            b_in[r_in] = 1
            
        team_home_matchups = matchups.loc[matchups['home_team_id'] == tm, 'game_id'].values
        team_far_away_matchups = matchups.loc[(matchups['away_team_id'] == tm) & 
                                          (matchups['home_division_geography'] != matchups['away_division_geography']), 'game_id'].values
        for j in range(1, NUM_WEEKS):
            r_in = r_in + 1
            for i in team_home_matchups:
                A_in[r_in, get_index(i, j, tnf_slot)] = 1
            for i in team_far_away_matchups:
                for k in range(NUM_SLOTS):
                    A_in[r_in, get_index(i, j, k)] = 1
            b_in[r_in] = 1
    
    # Week after thanksgiving rule - code up by forcing restraint that for each
    # matchup, 2 "points" if it occurs Thurs after thanksgiving, so must
    # get -1 point back for each of the two teams playing on Thanksgiving.
    for i in range(NUM_MATCHUPS):
        home_id = matchups.loc[matchups['game_id'] == i, 'home_team_id'].iloc[0]
        away_id = matchups.loc[matchups['game_id'] == i, 'away_team_id'].iloc[0]
        matchups_involving_teams = matchups.loc[(matchups['home_team_id'].isin([home_id, away_id])) |
                                            (matchups['away_team_id'].isin([home_id, away_id])), 'game_id'].values
        r_in = r_in + 1
        A_in[r_in, get_index(i, THANKSGIVING_WEEK + 1, tnf_slot)] = 2
        for i in matchups_involving_teams:
            A_in[r_in, get_index(i, THANKSGIVING_WEEK, tnf_slot)] = -1.5
        b_in[r_in] = 0
        
    # Cannot play on Thursday if played previous SNF or MNF
    for tm in range(NUM_TEAMS):
        team_matchups = matchups.loc[(matchups['away_team_id'] == tm) |
                                     (matchups['home_team_id'] == tm), 'game_id'].values
        for j in range(1, NUM_WEEKS):            
            r_in = r_in + 1
            for i in team_matchups:
                A_in[r_in, get_index(i, j-1, mnf_slot)] = 1
                A_in[r_in, get_index(i, j-1, snf_slot)] = 1
                A_in[r_in, get_index(i, j, tnf_slot)] = 1
            b_in[r_in] = 1
    
    # Cannot travel more than 2 time zones for Thursday game - in other words
    # no Thursday games can feature 1 time from ET and 1 time from PT
    r_eq = r_eq + 1
    for i in range(NUM_MATCHUPS):
        time_zones = [matchups.at[i,'home_time_zone'], matchups.at[i,'away_time_zone']]
        if sum(x == 'ET' for x in time_zones) == 1 and sum(x == 'PT' for x in time_zones) == 1:
             for j in range(NUM_WEEKS):
                A_eq[r_eq, get_index(i, j, tnf_slot)] = 1
    b_eq[r_eq] = 0
        
    
    # 12. Must be minimum quality of matchup to be in primetime
    #     Historically, 10th percentile of arithmetic_mean_intrigue for games
    #     in MNF has been 93, SNF has been 100, TNF has been 88
    if NUM_SLOTS == 4:
        r_eq = r_eq + 1
        for i in range(NUM_MATCHUPS):
            mean_intrigue = matchups.loc[matchups['game_id'] == i, 'arithmetic_mean_intrigue'].iloc[0]
            for j in range(NUM_WEEKS):
                for k in range(NUM_SLOTS):
                    slot_desc = slots.loc[slots['slot_id'] == k, 'slot_desc'].iloc[0]
                    if slot_desc == 'MNF' and mean_intrigue < 93:
                        A_eq[r_eq, get_index(i,j,k)] = 1
                    elif slot_desc == 'SNF' and mean_intrigue < 100:
                        A_eq[r_eq, get_index(i,j,k)] = 1
                    elif slot_desc == 'TNF' and mean_intrigue < 88:
                        A_eq[r_eq, get_index(i,j,k)] = 1
        b_eq[r_eq] = 0
        
    # 13. Max 5 total primetime games per team. Max 2 primetime games in 
    #     any 5 week stretch.
    primetime_slots = slots.loc[slots['slot_desc'].isin(['SNF', 'MNF', 'TNF']), 'slot_id'].values
    for tm in range(NUM_TEAMS):
        r_in = r_in + 1
        team_matchups = matchups.loc[(matchups['home_team_id'] == tm) | 
                                     (matchups['away_team_id'] == tm), 'game_id'].values
        for i in team_matchups:
            for j in range(NUM_WEEKS):
                for k in primetime_slots:
                    A_in[r_in, get_index(i,j,k)] = 1
        b_in[r_in] = 5
        
        for j in range(NUM_WEEKS - 4):
            r_in = r_in + 1
            for i in team_matchups:
                for k in primetime_slots:
                    A_in[r_in, get_index(i, j, k)] = 1
                    A_in[r_in, get_index(i, j+1, k)] = 1
                    A_in[r_in, get_index(i, j+2, k)] = 1
                    A_in[r_in, get_index(i, j+3, k)] = 1
                    A_in[r_in, get_index(i, j+4, k)] = 1
            b_in[r_in] = 2
        
    # 14. Cannot be away for first two games of season or final two games of season
    for tm in range(NUM_TEAMS):
        team_home_matchups = matchups.loc[matchups['home_team_id'] == tm, 'game_id'].values
        r_in = r_in + 1
        for i in team_home_matchups:
            for k in range(NUM_SLOTS):
                A_in[r_in, get_index(i, 0, k)] = -1
                A_in[r_in, get_index(i, 1, k)] = -1
        b_in[r_in] = -1
        r_in = r_in + 1
        for i in team_home_matchups:
            for k in range(NUM_SLOTS):
                A_in[r_in, get_index(i, NUM_WEEKS - 2, k)] = -1
                A_in[r_in, get_index(i, NUM_WEEKS - 1, k)] = -1
        b_in[r_in] = -1
        
    # 15. In any 5 week stretch, cannot be home more than 3 games
    for tm in range(NUM_TEAMS):
        team_home_matchups = matchups.loc[matchups['home_team_id'] == tm, 'game_id'].values
        for j in range(NUM_WEEKS - 4):
            r_in = r_in + 1
            for i in team_home_matchups:
                for k in range(NUM_SLOTS):
                    A_in[r_in, get_index(i, j, k)] = 1
                    A_in[r_in, get_index(i, j+1, k)] = 1
                    A_in[r_in, get_index(i, j+2, k)] = 1
                    A_in[r_in, get_index(i, j+3, k)] = 1
                    A_in[r_in, get_index(i, j+4, k)] = 1
            b_in[r_in] = 3

    # 16. Cannot play road game against team coming off bye
    for tm in range(NUM_TEAMS):
        team_away_matchups = matchups.loc[matchups['away_team_id'] == tm, 'game_id'].values
        for i in team_away_matchups:
            opposing_team = matchups.loc[matchups['game_id'] == i, 'home_team_id'].iloc[0]
            opposing_team_matchups = matchups.loc[(matchups['away_team_id'] == opposing_team) |
                                                  (matchups['home_team_id'] == opposing_team), 'game_id'].values
            for j in range(1, NUM_WEEKS):
                r_in = r_in + 1
                for k in range(NUM_SLOTS):
                    A_in[r_in, get_index(i, j, k)] = 1
                for i_opp in opposing_team_matchups:
                    for k in range(NUM_SLOTS):
                        A_in[r_in, get_index(i_opp, j-1, k)] = -1
                b_in[r_in] = 0

    
    return A_eq, A_in, b_eq, b_in

    
                            
            

