#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:54:12 2025

@author: neil
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from data.load_data import get_teams_and_standings, add_popularity_metrics


# Model viewership
def model_viewership():
    # viewership data manually saved from Sports Media Watch website
    # https://www.sportsmediawatch.com/nfl-tv-ratings-viewership-2023/
    viewership_data = pd.read_csv("data/Viewership_Data.csv")
    
    viewership_data.groupby('Window').agg(
        Viewers=('Viewers', 'mean'),
        PctMarkets=('PctMarkets', 'mean'),
        n = ('Viewers', 'size'))
    
    
    # To create the appropriate data frame for modeling, we also need to know
    # the various data points for each team in the season before each game
    # of data that we have on hand. For example, we have 2023 data, so we 
    # need to know 2022 information, etc.
    list_of_teams = list()
    for yr in viewership_data['Season'].unique() - 1:
        teams_this_year = get_teams_and_standings(yr)
        teams_this_year = add_popularity_metrics(teams_this_year)
        teams_this_year['year'] = yr
        list_of_teams.append(teams_this_year)
        
    team_seasons = pd.concat(list_of_teams)
    team_seasons['join_year'] = team_seasons['year'] + 1
    team_seasons['market_pop'] = pd.to_numeric(team_seasons['market_pop'])
    
    # Firstly, will create a second-team-agnostic team viewership data
    # valuation
    views_away = viewership_data.loc[viewership_data['PctMarkets'] == 100, ['Season', 'Week', 'Window', 'Away', 'Viewers']]
    views_away = views_away.rename(columns={'Away':'Team'})
    views_home = viewership_data.loc[viewership_data['PctMarkets'] == 100, ['Season', 'Week', 'Window', 'Home', 'Viewers']]
    views_home = views_home.rename(columns={'Home':'Team'})
    
    # Join in information that we know about these teams
    team_games = pd.concat([views_away, views_home])
    # Change a couple of abbreviations to match other sources
    team_games['Team']=team_games['Team'].replace({'WSH':'WAS', 'LAR':'LA'})
    
    team_games = team_games.merge(team_seasons.loc[:, ['team_abbr', 'team_name', 'join_year', 'team_division', 'WinPct', 'market_pop', 'twitter_followers']],
                                  how='inner',
                                  left_on=['Team','Season'],
                                  right_on=['team_abbr','join_year'])
    
    
    # Basic linear model to predict number of viewers for any particular team
    # Note that when we try to add market population as a predictor, it has P value
    # of .14, compared to p-value of twitter followers of .003. 
    intrigue_model = smf.ols('Viewers ~ C(Window) + WinPct + twitter_followers', 
                           data=team_games.loc[team_games['Window'].isin(['SNF','MNF','TNF']),:]).fit()
    # We note unusual behavior if include WinPct**2 term, namely that not clear that better
    # win pct lead to more viewers
    
    # To create the intrigue score for a particular team, can remove the coefficients
    # and confounding terms from the model before predicting
    
    team_seasons['intrigue_unscaled'] = intrigue_model.params['WinPct'] * team_seasons['WinPct'] + \
        intrigue_model.params['twitter_followers'] * team_seasons['twitter_followers']
    # For ease of understanding, scale so that mean is 100 and standard deviation is 20. Take
    # care to save the constants for use outside of this process.
    mean_intrigue_unscaled = team_seasons['intrigue_unscaled'].mean()
    std_intrigue_unscaled = team_seasons['intrigue_unscaled'].std()
    team_seasons['intrigue'] = (team_seasons['intrigue_unscaled'] - mean_intrigue_unscaled) / \
        std_intrigue_unscaled * 20 + 100
        
    # Given this information, we now want to predict how many viewers will watch
    # a particular game, given the two teams playing and the window
    viewership_with_team_data = viewership_data.merge(team_seasons.loc[:,['team_abbr', 'join_year','intrigue', 'team_division']], 
                          how='inner',
                          left_on=['Away', 'Season'], 
                          right_on=['team_abbr', 'join_year']).merge(
                        team_seasons.loc[:,['team_abbr', 'join_year','intrigue', 'team_division']], 
                              how='inner',
                              left_on=['Home', 'Season'], 
                              right_on=['team_abbr', 'join_year'],
                              suffixes=("_away", "_home"))
    viewership_with_team_data = viewership_with_team_data.drop(columns=['join_year_away', 'join_year_home', 'team_abbr_away', 'team_abbr_home'])
    # Create a couple of additional features that might be relevant in the model
    viewership_with_team_data['same_division'] = (viewership_with_team_data['team_division_away'] == 
                                                  viewership_with_team_data['team_division_home']).astype(int)
    viewership_with_team_data['arithmetic_mean_intrigue'] = (viewership_with_team_data['intrigue_home'] +
                                                             viewership_with_team_data['intrigue_away']) / 2.0
    viewership_with_team_data['harmonic_mean_intrigue'] = 2.0 / (1 / viewership_with_team_data['intrigue_away'] +
                                                                 1 / viewership_with_team_data['intrigue_home'])
    viewership_with_team_data['max_intrigue'] = np.maximum(viewership_with_team_data['intrigue_away'],
                                                           viewership_with_team_data['intrigue_home'])
    viewership_with_team_data['min_intrigue'] = np.minimum(viewership_with_team_data['intrigue_away'],
                                                           viewership_with_team_data['intrigue_home'])
    viewership_with_team_data['in_flex_window'] = (viewership_with_team_data['Week'] >= 14).astype(int)
    viewership_with_team_data['max_above_average'] = viewership_with_team_data['max_intrigue'].apply(lambda x: 0 if x < 100 else x - 100)
    viewership_with_team_data['min_above_average'] = viewership_with_team_data['min_intrigue'].apply(lambda x: 0 if x < 100 else x - 100)
    viewership_with_team_data['two_elite_teams'] = np.where(viewership_with_team_data['min_intrigue'] >= 120, 1, 0)
    viewership_with_team_data['two_aavg_teams'] = np.where((viewership_with_team_data['min_intrigue'] >= 110) &
                                                           (viewership_with_team_data['min_intrigue'] < 120), 1, 0)

    
    # A particularly interesting finding is that the identity of the identity of the less-important
    # team doesn't seem to matter much. P-value for max_intrigue is 0 (coeff of .11), but
    # p-value for min intrigue team is .39 (coeff of .025)
    game_viewers_model = smf.ols('Viewers ~ C(Window) + max_intrigue   + \
                                 two_elite_teams + two_aavg_teams',
            data = viewership_with_team_data.loc[viewership_with_team_data['Window'].isin(['SNF','MNF','TNF'])]).fit()
    # The variables for same_division and in_flex_window both return insignificant coefficients,
    # so will remove them
    # If try adding an interaction term between max intrigue and min intrigue,
    # we quickly notice that get some undesirable results (i.e. game between two teams 
    # with 60 intrigue would be more highly rated than between a 100 and 140)
    
    # Need to be careful because data sample only has games that have been selected
    # for primetime for some reason, whereas our optimization will include all
    # sorts of games, even those that would never get a second sniff at primetime
    
    # Try Bayesian-like process, where we first create a no-stats model that
    # predicts the number of viewers just based on the window and week,
    # then regress actual viewership back to that number according to variables
    # related to the intrigue level of the game
    
    return intrigue_model, game_viewers_model, mean_intrigue_unscaled, std_intrigue_unscaled

