#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:26:03 2025

@author: neil
"""
import pandas as pd
from scipy.sparse import lil_matrix
import numpy as np
from itertools import chain

def get_index(i, j, k, l,
                   num_teams=32, num_stadiums=32, num_weeks=18, num_slots=1):
    return (i * num_stadiums*num_weeks*num_slots + 
            j * num_weeks*num_slots +
            k * num_slots +
            l)

def flat_to_indices(idx, 
                    num_teams=32, num_stadiums=32, num_weeks=18, num_slots=1):
    total_vars = num_teams*num_stadiums * num_weeks * num_slots
    i, remainder = divmod(idx, num_stadiums * num_weeks * num_slots)
    j, remainder = divmod(remainder, num_weeks * num_slots)
    k, l = divmod(remainder, num_slots)
    
    return i, j, k, l

def create_constraints(teams, full_constraints=False):
    """
    Creates the constraint matrices and RHS for a valid NFL schedule.
    
    The schedule is set up with binary variables x_ijkl, which contain the
    value 1 iff team i is playing in stadium j during week k and time slot l,
    where l = 0 is regular Sunday afternoon game, l = 1 is TNF, l = 2 is SNF,
    l = 3 is MNF.
    
    Some constraints taken from a Gurobi webinar on NFL scheduling:
    https://cdn.gurobi.com/wp-content/uploads/Creating-the-NFL-Schedule.pdf?x58142
    
    The constraints are:
        1. Each game must be a valid game (home and away team)
        2. Each team must play exactly 17 games, with 1 bye week between Wk 5
           and Wk 14
        3. Legal assignment of games for each team:
            - 6 games against divisional opponents
            - 4 games against teams from a division within its conference
            - 4 games against teams from a division in the other conference
            - 2 games against teams from the two remaining divisions in own conference,
               based on previous season divisional ranking
            - 1 game against non-conference opponent from other division,
              based on previous season divisional ranking
        4. Avoid having 3 consecutive road games
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
        11. Each team must have 8 or 9 home games
                - After review, this isn't necessary because the other 
                  constraints will enforce it
        12. Cannot play same team within 3 week span (i.e. not back-to-back
                                                      or semi-repeater)
              

    Parameters
    ----------
    teams : Pandas DataFrame
        Contains the teams participating in the season, their record from
        the previous year, and their division

    Returns
    -------
    A_in, A_eq, b_in, b_eq - the inequality constraint matrix,
     equality constraint matrix, inequality right hand side, and equality
     right hand side, respectively.

    """    
    
    # Define necessary constants
    num_teams = 32
    num_stadiums = num_teams
    num_weeks = 18
    num_slots = 1
    num_vars = num_teams * num_stadiums * num_weeks * num_slots
    
    slots = pd.DataFrame({
        'slot_id': range(num_slots),
        'slot_desc': ['Sun', 'TNF', 'SNF', 'MNF']})
    
    
    # Create divisions df to help later
    divisions = teams['team_division'].unique()
    divisions = pd.DataFrame({'division': divisions,
                             'conference': ["" for _ in range(len(divisions))],
                             'direction': ["" for _ in range(len(divisions))]})
    for d in range(divisions.shape[0]):
        conference, direction = divisions['division'].iloc[d].split(' ', 1)
        divisions.loc[d, 'conference'] = conference
        divisions.loc[d, 'direction'] = direction
    
    # Division pairs - interconference and interconference. These are the same 
    # interconference pairs as league had in 2021 (assuming 4 yr rotation)
    # and same intraconference pairs as league had in 2022 (assuming 3 yr rotation)
    division_pairs = [['NFC West', 'AFC South'],
                      ['NFC North', 'AFC North'],
                      ['NFC East', 'AFC West'],
                      ['NFC South', 'AFC East'],
                      ['NFC South', 'NFC West'],
                      ['NFC East', 'NFC North'],
                      ['AFC North', 'AFC East'],
                      ['AFC West', 'AFC South']]
    division_pairs = pd.DataFrame(division_pairs,columns=['Div1', 'Div2'])
    division_pairs = pd.concat([division_pairs,
              pd.DataFrame({
                  'Div1': division_pairs['Div2'],
                  'Div2': division_pairs['Div1']})
              ]).reset_index()
    
    # Define constraint matrices
    A_eq = lil_matrix((10000, num_vars))
    #A_eq = np.zeros([10000, num_vars])
    A_in = lil_matrix((10000, num_vars))
    #A_in = np.zeros([10000, num_vars])
    b_eq = np.zeros(10000)
    b_in = np.zeros(10000)
    r_eq = -1
    r_in = -1
    
    # 1. Valid Game framework - define mathematically be noting
    #    that the sum of home teams at each stadium must equal
    #    the sum of visiting teams at each stadium
    for j in range(num_stadiums):
        for k in range(num_weeks):
            r_eq = r_eq + 1
            for i in range(num_teams):
                for l in range(num_slots):
                    if i == j:
                        A_eq[r_eq, get_index(i,j,k,l)] = 1
                    else:
                        A_eq[r_eq, get_index(i,j,k,l)] = -1
            b_eq[r_eq] = 0
        
    # 2. Each team plays 17 games, with at most 1 game per week
    #     Bye week is between week 5 and 14

    for i in range(num_teams):
        for k in range(num_weeks):
            if k in list(range(0, 5)) + list(range(15, num_weeks)):
                r_eq = r_eq + 1
                for j in range(num_stadiums):
                    for l in range(num_slots):
                        A_eq[r_eq, get_index(i,j,k,l)] = 1
                b_eq[r_eq] = 1

            else:
                r_in = r_in + 1
                for j in range(num_stadiums):
                    for l in range(num_slots):
                        A_in[r_in, get_index(i,j,k,l)] = 1
                b_in[r_in] = 1
    
    for i in range(num_teams):
        r_eq = r_eq + 1
        for j in range(num_stadiums):
            for k in range(num_weeks):
                for l in range(num_slots):
                    A_eq[r_eq, get_index(i,j,k,l)] = 1
        b_eq[r_eq] = 17
        
        
    # 3. Legal assignment of games for each team
    #   - 1 home game and 1 away game against each team in your division
    for tm1 in range(num_teams):
        for tm2 in range(num_teams):
            if tm1 == tm2:
                continue
            tm1_div = teams.loc[teams['team_id'] == tm1, 'team_division'].iloc[0]
            tm2_div = teams.loc[teams['team_id'] == tm2, 'team_division'].iloc[0]
            if tm1_div == tm2_div:
                #print(f"{tm1}_{tm2}")
                r_eq = r_eq + 1
                for k in range(num_weeks):
                    for l in range(num_slots):
                        A_eq[r_eq, get_index(tm1, tm2, k, l)] = 1
                b_eq[r_eq] = 1
                
    #   - 1 total game against each team in divisions as assigned,
    #     for each of the 2 assigned divisions, must play 2 home games and 2 road games
    for tm1 in range(num_teams):
        tm1_div = teams.loc[teams['team_id'] == tm1, 'team_division'].iloc[0]
        divisions_to_play = division_pairs[division_pairs['Div1'] == tm1_div]['Div2'].values
        for div2 in divisions_to_play:
            team_ids_in_division = teams[teams['team_division'] == div2]['team_id'].values
            
            # constraint that enforces exactly 1 game between each pair of teams
            for tm2 in team_ids_in_division:
                r_eq = r_eq + 1
                for k in range(num_weeks):
                    for l in range(num_slots):
                        A_eq[r_eq, get_index(tm1, tm2, k, l)] = 1
                        A_eq[r_eq, get_index(tm2, tm1, k, l)] = 1
                b_eq[r_eq] = 1
            
            # constraint that enforces exactly 2 home games against division
            r_eq = r_eq + 1
            for tm2 in team_ids_in_division:
                for k in range(num_weeks):
                    for l in range(num_slots):
                        A_eq[r_eq, get_index(tm2, tm1, k, l)] = 1
            b_eq[r_eq] = 2
            
            # constraint that enforces exactly 2 road games against division
            r_eq = r_eq + 1
            for tm2 in team_ids_in_division:
                for k in range(num_weeks):
                    for l in range(num_slots):
                        A_eq[r_eq, get_index(tm1, tm2, k, l)] = 1
            b_eq[r_eq] = 2
    
    #   - 1 game (1 home/1 away) against 2 other teams with same division rank in 
    #     your conference, that aren't in assigned division
    
    for tm1 in range(num_teams):
        tm1_div_rank = teams.loc[teams['team_id'] == tm1, 'division_place'].iloc[0]
        tm1_div = teams.loc[teams['team_id'] == tm1, 'team_division'].iloc[0]
        conf = divisions.loc[divisions['division'] == tm1_div, 'conference'].iloc[0]
        # find 2 divisions to play by iteratively removing divisions that don't work,
        # including those that are already playing and those in other conference
        divisions_to_play = division_pairs[division_pairs['Div1'] == tm1_div]['Div2'].values
        divisions_to_consider = divisions.loc[~(divisions['division'].isin(divisions_to_play)),:]
        divisions_to_consider = divisions_to_consider.loc[~(divisions['division'] == tm1_div), :]
        divisions_to_consider = divisions_to_consider.loc[divisions['conference'] == conf, :]
        
        teams_to_play = []
        for tm2 in range(num_teams):
            tm2_div = teams.loc[teams['team_id'] == tm2, 'team_division'].iloc[0]
            tm2_div_rank = teams.loc[teams['team_id'] == tm2, 'division_place'].iloc[0]
            if (divisions_to_consider['division'].str.contains(tm2_div).any()) and (tm1_div_rank == tm2_div_rank):
                teams_to_play.append(tm2)
                
        # Exactly one game vs each of those teams
        for tm2 in teams_to_play:
            r_eq = r_eq + 1
            for k in range(num_weeks):
                for l in range(num_slots):
                    A_eq[r_eq, get_index(tm1, tm2, k, l)] = 1
                    A_eq[r_eq, get_index(tm2, tm1, k, l)] = 1
            b_eq[r_eq] = 1
            
        # Exactly one road game against those two teams combined
        r_eq = r_eq + 1
        for tm2 in teams_to_play:
            for k in range(num_weeks):
                for l in range(num_slots):
                    A_eq[r_eq, get_index(tm1, tm2, k, l)] = 1
        b_eq[r_eq] = 1
        
        # Exactly one home game against those two teams combined
        r_eq = r_eq + 1
        for tm2 in teams_to_play:
            for k in range(num_weeks):
                for l in range(num_slots):
                    A_eq[r_eq, get_index(tm2, tm1, k, l)] = 1
        b_eq[r_eq] = 1
        
    #   - 1 game against non-confrence opponent from a division that is not
    #     scheduled to play
    for tm1 in range(num_teams):
        tm1_div_rank = teams.loc[teams['team_id'] == tm1, 'division_place'].iloc[0]
        tm1_div = teams.loc[teams['team_id'] == tm1, 'team_division'].iloc[0]
        conf = divisions.loc[divisions['division'] == tm1_div, 'conference'].iloc[0]
        # find the possible divisions by removing those that it isn't
        divisions_to_play = division_pairs[division_pairs['Div1'] == tm1_div]['Div2'].values
        divisions_to_consider = divisions.loc[~(divisions['division'].isin(divisions_to_play)),:]
        divisions_to_consider = divisions_to_consider.loc[~(divisions['division'] == tm1_div), :]
        divisions_to_consider = divisions_to_consider.loc[~(divisions['conference'] == conf), :]
        
        teams_to_play = []
        for tm2 in range(num_teams):
            tm2_div = teams.loc[teams['team_id'] == tm2, 'team_division'].iloc[0]
            tm2_div_rank = teams.loc[teams['team_id'] == tm2, 'division_place'].iloc[0]
            if (divisions_to_consider['division'].str.contains(tm2_div).any()) and (tm1_div_rank == tm2_div_rank):
                teams_to_play.append(tm2)
                
        # Exactly one game vs one of those teams
        r_eq = r_eq + 1
        for tm2 in teams_to_play:
            for k in range(num_weeks):
                for l in range(num_slots):
                    A_eq[r_eq, get_index(tm1, tm2, k, l)] = 1
                    A_eq[r_eq, get_index(tm2, tm1, k, l)] = 1
        b_eq[r_eq] = 1
            

    
    # 4. Avoid 3 consecutive road games - coded as a constraint to say
    #    must play at last 1 home game in every 3 week stretch
        for k in range(num_weeks - 2):
            r_in = r_in + 1
            for k_add in [0, 1, 2]:
                for j in range(num_stadiums):
                    for l in range(num_slots):
                        A_in[r_in, get_index(i, j, k + k_add, l)] = -1
            b_in[r_in] = -1
    
    # 5. Teams that play a road game on Monday night cannot play on the road
    #       the following week
    
    
    # 6. Max 2 Thursday night games per team, max 1 of those games at home
    
    
    
    # 7. Stadium conflicts - LAR/LAC and NYG/NYJ cannot be home same weeks
    lac = teams.loc[teams['team_abbr'] == 'LAC', 'team_id'].iloc[0]
    lar = teams.loc[teams['team_abbr'] == 'LA', 'team_id'].iloc[0]
    nyg = teams.loc[teams['team_abbr'] == 'NYG', 'team_id'].iloc[0]
    nyj = teams.loc[teams['team_abbr'] == 'NYJ', 'team_id'].iloc[0]
    
    for k in range(num_weeks):
        r_in = r_in + 1
        for i in range(num_teams):
            for k in range(num_slots):
                A_in[r_in, get_index(i, lac, k, l)] = 1
                A_in[r_in, get_index(i, lar, k, l)] = 1
        b_in[r_in] = 1
        r_in = r_in + 1
        for i in range(num_teams):
            for k in range(num_slots):
                A_in[r_in, get_index(i, nyg, k, l)] = 1
                A_in[r_in, get_index(i, nyj, k, l)] = 1
        b_in[r_in] = 1


    
    # 8. Each week contains 1 SNF game, 1 MNF game, and 1 TNF game, with following
    #        exceptions:
    #            - Thanksgiving, which contains 3 TNF games, and DAL/DET much each be home
    #            - Wk 18, which is scheduled after Wk 17, so for sake of this process
    #              will have 0 MNF, TNF, or SNF games
    for k in range(num_weeks):
        # Sunday regulartions
        r_eq = r_eq + 1
        snf = 
    
    
    # 9. Last game of season (Wk 18) must be against divisional opponent
    
    # 10. Thursday game constraints:
        # - If play road Thursday game, then need to be home previous week
        #    - All teams playing home Thursday games must play within division
        #      or same division other conference (i.e. AFC East vs NFC East)
        #      during previous week
        
        
    
    # 12. Cannot play same team within 3 week span (i.e. not back-to-back
    #                                                  or semi-repeater)
    '''

    return A_eq, A_in, b_eq, b_in
    
        
        
        
        
        
        
        
        
        