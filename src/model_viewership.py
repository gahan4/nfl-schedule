#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:54:12 2025

@author: neil
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from data.load_data import get_teams_and_standings, add_popularity_metrics,get_jersey_sales_rankings


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
    
    # Add in jersey sales information
    jersey_url_df = pd.DataFrame({
    "Season": [2021, 2022, 2024],
    "URL": [
        "https://nflpa.com/partners/posts/top-50-nfl-player-sales-list-march-1-2021-february-28-2022",
        "https://nflpa.com/partners/posts/top-50-nfl-player-sales-list-march-1-2022-february-28-2023",
        #"https://nflpa.com/partners/posts/top-50-nfl-player-sales-list-march-1-2023-february-28-2024",
        "https://nflpa.com/partners/posts/top-50-nfl-player-sales-list-march-1-august-31-2024"
        ]
    })
    jersey_sales_df_list = []
    for _, row in jersey_url_df.iterrows():
        season = row["Season"]
        url = row["URL"]
        
        # Get rankings data from the URL
        rankings_df = get_jersey_sales_rankings(url)
        
        # Ensure the function returned a valid DataFrame before adding to list
        if isinstance(rankings_df, pd.DataFrame) and not rankings_df.empty:
            rankings_df["Season"] = season  # Add the season column
            jersey_sales_df_list.append(rankings_df)    
    jersey_sales_df = pd.concat(jersey_sales_df_list, ignore_index=True)
    jersey_sales_df = jersey_sales_df.rename(columns={"Player Name": "Player", "Player Team": "Team"})
    # There are some cases with a (*) in the team name column, indicating player
    # movement between seasons. Identify and address these players and change their team
    # to the next team they were part of the following season.
    jersey_sales_df.loc[(jersey_sales_df["Player"] == "Russell Wilson") & 
                        (jersey_sales_df["Season"] == 2021), "Team"] = "Denver Broncos"
    jersey_sales_df.loc[(jersey_sales_df["Player"] == "Davante Adams") & 
                        (jersey_sales_df["Season"] == 2021), "Team"] = "Las Vegas Raiders"
    jersey_sales_df.loc[(jersey_sales_df["Player"] == "Amari Cooper") & 
                        (jersey_sales_df["Season"] == 2021), "Team"] = "Cleveland Browns"
    # Add in flag for players who retired after a particular season
    jersey_sales_df['RetiredAfterSeason'] = 0
    jersey_sales_df.loc[(jersey_sales_df["Player"] == "Tom Brady") & 
                        (jersey_sales_df["Season"] == 2022), "RetiredAfterSeason"] = 1
    jersey_sales_df.loc[(jersey_sales_df["Player"] == "Rob Gronkowski") & 
                        (jersey_sales_df["Season"] == 2021), "RetiredAfterSeason"] = 1
    # Also add a flag for players who definitely won't be on a team next season
    # but were on a team during the last season with data - i.e. Myles Garrett
    # or Aaron Rodgers
    jersey_sales_df['WillChangeTeams'] = 0
    jersey_sales_df.loc[(jersey_sales_df["Player"] == "Myles Garrett") & 
                        (jersey_sales_df["Season"] == 2024), "WillChangeTeams"] = 1
    jersey_sales_df.loc[(jersey_sales_df["Player"] == "Aaron Rodgers") & 
                        (jersey_sales_df["Season"] == 2024), "WillChangeTeams"] = 1
    
    # Apply exponential decay to "score" each player based on how their popularity
    # might influence fan interest in watching game. By choosing an exponent of -.05,
    # the top player each year gets 1 intrigue, falling by ~.05 points. 50th
    # player gets about .09 intrigue.
    decay_rate = 0.05
    jersey_sales_df['PlayerIntrigue'] = jersey_sales_df['Rank'].apply(lambda x: np.exp(-decay_rate * (x - 1)))
    # Apply penalties to remove players who will leave team after the season
    jersey_sales_df.loc[jersey_sales_df["RetiredAfterSeason"] == 1, "PlayerIntrigue"] = 0  
    jersey_sales_df.loc[jersey_sales_df["WillChangeTeams"] == 1, "PlayerIntrigue"] = 0  
    team_jersey_score_df = jersey_sales_df.groupby(["Team","Season"])["PlayerIntrigue"].sum().reset_index()
    team_jersey_score_df['JoinSeason'] = team_jersey_score_df['Season'] + 1
    team_jersey_score_df = team_jersey_score_df.rename(columns = {"PlayerIntrigue": "WeightedJerseySales"})
    # And join those results in
    team_seasons = team_seasons.merge(team_jersey_score_df[['Team', 'JoinSeason', 'WeightedJerseySales']],
                                      how="left",
                                      left_on=['team_name', 'join_year'],
                                      right_on=['Team', 'JoinSeason']).drop(columns=['Team', 'JoinSeason'])
    team_seasons['WeightedJerseySales'] = team_seasons['WeightedJerseySales'].fillna(0)
    
    
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
    
    
    team_games = team_games.merge(team_seasons.loc[:, ['team_abbr', 'team_name', 'join_year', 
                                                       'team_division', 'WinPct', 'market_pop', 
                                                       'WeightedJerseySales','twitter_followers']],
                                  how='inner',
                                  left_on=['Team','Season'],
                                  right_on=['team_abbr','join_year'])
    
    
    # Basic linear model to predict number of viewers for any particular team
    # Note that when we try to add market population as a predictor, it has P value
    # of .14, compared to p-value of twitter followers of .003. 
    team_games_for_model = team_games.loc[team_games['Window'].isin(['SNF','MNF','TNF']),:]
    intrigue_model = smf.ols('Viewers ~ C(Window) + WinPct + twitter_followers + WeightedJerseySales', 
                           data=team_games_for_model).fit()
    # We note unusual behavior if include WinPct**2 term, namely that not clear that better
    # win pct lead to more viewers
    
    # residual anlysis
    # Get predictions
    team_games_for_model["predicted_viewers"] = intrigue_model.predict(team_games_for_model)
    
    # Compute residuals (actual - predicted)
    team_games_for_model["residuals"] = team_games_for_model["Viewers"] - team_games_for_model["predicted_viewers"]

    
    
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
    data_for_model = viewership_with_team_data.loc[viewership_with_team_data['Window'].isin(['SNF','MNF','TNF'])]
    game_viewers_model = smf.ols('Viewers ~ C(Window) + max_intrigue + \
                                 two_elite_teams + ',
            data = data_for_model).fit()
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

