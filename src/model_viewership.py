#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:54:12 2025

@author: neil
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from data.load_data import get_teams_and_standings, add_popularity_metrics, \
    get_jersey_sales_rankings, get_draft_history, add_jersey_sales_metrics, add_draft_intrigue_metrics, \
        add_high_value_qb_metrics


# Model viewership
def model_viewership(show_plots=False):
    # viewership data manually saved from Sports Media Watch website
    # https://www.sportsmediawatch.com/nfl-tv-ratings-viewership-2023/
    viewership_data = pd.read_csv("data/Viewership_Data.csv")
    
    viewership_data.groupby('Window').agg(
        Viewers=('Viewers', 'mean'),
        PctMarkets=('PctMarkets', 'mean'),
        n = ('Viewers', 'size'))
    
    # Change a couple of abbreviations to match other sources
    viewership_data['Away']=viewership_data['Away'].replace({'WSH':'WAS', 'LAR':'LA'})
    viewership_data['Home']=viewership_data['Home'].replace({'WSH':'WAS', 'LAR':'LA'})

    
    
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
    
    # Going to add in binary column for case where team acquired notable
    # high-value QB after the season, which could be helpful for understanding
    # viewership
    team_seasons = add_high_value_qb_metrics(team_seasons)
    
    # Add in jersey sales information
    team_seasons = add_jersey_sales_metrics(team_seasons)
    
    
    # Add in information related to draft. Expect that teams that pick highly
    # in a particular season's draft might be interesting.
    team_seasons = add_draft_intrigue_metrics(team_seasons)
    
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
    
    
    team_games = team_games.merge(team_seasons.loc[:, ['team_abbr', 'team_name', 'join_year', 'time_zone',
                                                       'team_division', 'WinPct', 'market_pop', 
                                                       'WeightedJerseySales','twitter_followers',
                                                       'PickIntrigue', 'new_high_value_qb']],
                                  how='inner',
                                  left_on=['Team','Season'],
                                  right_on=['team_abbr','join_year'])
    
    # League gets right to flex games starting from week 12 full-time, though
    # there are some other rules regarding SNF (can do limited number of earlier
    # flexes) and TNF (max 2 flex games)
    team_games['InFlexWindow'] = team_games['Week'].apply(lambda x: 1 if x >= 12 else 0)
    
    # Add in column that references whether a game is the only one in its window
    # Looks like ESPN did some MNF doubleheaders early in some seasons
    mnf_counts = team_games[team_games['Window'] == 'MNF'].groupby(["Week", "Season"]).size().reset_index(name="num_mnf_games")
    team_games = team_games.merge(mnf_counts, on=["Week", "Season"], how="left").fillna(0)
    # Create the binary flag: 1 if more than one MNF game exists, 0 otherwise
    team_games["SharedMNFWindow"] = ((team_games["num_mnf_games"] > 2) & (team_games['Window'] == 'MNF')).astype(int)
    team_games = team_games.drop(columns='num_mnf_games')
    
    # Basic linear model to predict number of viewers for any particular team
    # Note that when we try to add market population as a predictor, it has P value
    # of .14, compared to p-value of twitter followers of .003. 
    team_games_for_model = team_games.loc[team_games['Window'].isin(['SNF','MNF','TNF']),:]
    intrigue_model = smf.ols('Viewers ~ C(Window) + SharedMNFWindow + \
                                         WinPct + twitter_followers + \
                                        WeightedJerseySales ', 
                           data=team_games_for_model).fit()
    # We note unusual behavior if include WinPct**2 term, namely that not clear that better
    # win pct lead to more viewers.
    # Other terms that don't prove significant: InFlexWindow, PickIntrigue (possibly because
    #   of JAX's #1 pick going towards DE), time_zone of team
    # Challenging to include variables related to player movement (e.g. Aaron Rodgers first
    #  game with NYJ more popular than expected) because that movement hasn't happened yet
    # The new_high_value_qb coefficient was .71, but with only 13 total primetime games
    #  for the 6 new high-value qb's, there wasn't enough for the model to feel certain
    
    # Try a lasso approach as well
    lasso_df = team_games_for_model.copy()

    # Select features for the model
    # Define feature columns
    categorical_features = ['Window']
    numeric_features = ['WinPct', 'twitter_followers', 'WeightedJerseySales',
                           'new_high_value_qb', 'SharedMNFWindow', 'market_pop']
    
    # Create transformations
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),  # Scale numeric features
        ("cat", Pipeline([
            ("onehot", OneHotEncoder(drop="first", sparse_output=False)),  # One-hot encode categorical variables
            ("scaler", StandardScaler())  # Scale the one-hot encoded features
        ]), categorical_features)
    ])
    
    # Define full pipeline
    intrigue_model_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', LassoCV(cv=10))  # Lasso with cross-validation
    ])
    
    # Fit the pipeline
    X = lasso_df[categorical_features + numeric_features]
    y = lasso_df['Viewers']
    intrigue_model_pipeline.fit(X, y)    
    # Print the best alpha found
    coefficients = intrigue_model_pipeline.named_steps['model'].coef_
    feature_names = intrigue_model_pipeline.named_steps['preprocessing'].get_feature_names_out()
    
    lasso_results = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Analyze coefficient standard deviation across folds
    if show_plots:
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import KFold
        
        # Get feature matrix and target variable
        X_transformed = intrigue_model_pipeline.named_steps['preprocessing'].fit_transform(X)
        y_values = y.values  # Convert y to NumPy array for indexing
        
        # Define cross-validation strategy
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        coefs_list = []  # Store coefficients for each fold
        
        # Perform manual cross-validation
        for train_idx, test_idx in kf.split(X_transformed):
            X_train, X_test = X_transformed[train_idx], X_transformed[test_idx]
            y_train, y_test = y_values[train_idx], y_values[test_idx]
        
            # Fit Lasso with the best alpha from LassoCV
            lasso = Lasso(alpha=intrigue_model_pipeline.named_steps['model'].alpha_)
            lasso.fit(X_train, y_train)
        
            # Store coefficients
            coefs_list.append(lasso.coef_)
        
        # Convert to DataFrame
        coefs_df = pd.DataFrame(coefs_list, columns=numeric_features + 
                                list(intrigue_model_pipeline.named_steps['preprocessing'].named_transformers_['cat']
                                     .named_steps['onehot'].get_feature_names_out()))
        
        # Compute standard deviation of coefficients across folds
        coef_variability = coefs_df.std(axis=0).sort_values(ascending=False)
        
        # Display coefficient variability
        print("Coefficient Standard Deviations Across Folds:")
        print(coef_variability)
    
        plt.figure(figsize=(10, 5))
        coef_variability.plot(kind="bar", color="blue", alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel("Feature")
        plt.ylabel("Standard Deviation of Coefficient")
        plt.title("Coefficient Stability Across CV Folds")
        plt.show()
        
    
    # Add a dummy 'Window' column with a fixed value, e.g., 'SNF'
    team_seasons["Window"] = "SNF"

    for col in team_games_for_model.columns:
        if col not in team_seasons.columns:
            team_seasons[col] = 0  # Assign a neutral placeholder

    # Run through pipeline
    team_seasons['intrigue_raw'] = intrigue_model_pipeline.predict(team_seasons)
    
    team_seasons = team_seasons.drop(columns=['Window', 'SharedMNFWindow'])
    
    # Standardize intrigue scores to 100-based scale
    mean_intrigue = team_seasons["intrigue_raw"].mean()
    std_intrigue = team_seasons["intrigue_raw"].std()
    team_seasons["intrigue"] = 100 + 20 * (team_seasons["intrigue_raw"] - mean_intrigue) / std_intrigue
    
    # Display the top teams by intrigue
    if show_plots:
        team_seasons[["team_abbr", 'join_year', "intrigue_scaled"]].sort_values(by="intrigue_scaled", ascending=False).head(10)

    # Given this information, we now want to predict how many viewers will watch
    # a particular game, given the two teams playing and the window
    viewership_with_team_data = viewership_data.merge(team_seasons.loc[:,['team_abbr', 'join_year','intrigue', 'team_division', 'WinPct']], 
                          how='inner',
                          left_on=['Away', 'Season'], 
                          right_on=['team_abbr', 'join_year']).merge(
                        team_seasons.loc[:,['team_abbr', 'join_year','intrigue', 'team_division', 'WinPct']], 
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
    viewership_with_team_data['min_intrigue'] = np.minimum(viewership_with_team_data['intrigue_away'],
                                                           viewership_with_team_data['intrigue_home'])

    viewership_with_team_data['in_flex_window'] = (viewership_with_team_data['Week'] >= 14).astype(int)
    viewership_with_team_data['max_above_average'] = viewership_with_team_data['max_intrigue'].apply(lambda x: 0 if x < 100 else x - 100)
    viewership_with_team_data['min_above_average'] = viewership_with_team_data['min_intrigue'].apply(lambda x: 0 if x < 100 else x - 100)
    viewership_with_team_data['max_below_average'] = viewership_with_team_data['max_intrigue'].apply(lambda x: 0 if x > 100 else 100 - x)
    viewership_with_team_data['min_below_average'] = viewership_with_team_data['min_intrigue'].apply(lambda x: 0 if x > 100 else 100 - x)
    viewership_with_team_data['two_elite_teams'] = np.where(viewership_with_team_data['min_intrigue'] >= 120, 1, 0)
    viewership_with_team_data['two_aavg_teams'] = np.where((viewership_with_team_data['min_intrigue'] >= 110) &
                                                           (viewership_with_team_data['min_intrigue'] < 120), 1, 0)
    viewership_with_team_data['WinPct_mean'] = (viewership_with_team_data['WinPct_home'] + 
                                                  viewership_with_team_data['WinPct_away']) / 2.0
    viewership_with_team_data['WinPct_multiply'] = (viewership_with_team_data['WinPct_home'] * 
                                                  viewership_with_team_data['WinPct_away']) 
    viewership_with_team_data['AboveAverage_multiply'] = (viewership_with_team_data['max_above_average'] * 
                                                  viewership_with_team_data['min_above_average']) 


    
    # Flag for repeated MNF games
    mnf_counts = viewership_with_team_data[viewership_with_team_data['Window'] == 'MNF'].groupby(["Week", "Season"]).size().reset_index(name="num_mnf_games")
    viewership_with_team_data = viewership_with_team_data.merge(mnf_counts, on=["Week", "Season"], how="left").fillna(0)
    # Create the binary flag: 1 if more than one MNF game exists, 0 otherwise
    viewership_with_team_data["SharedMNFWindow"] = ((viewership_with_team_data["num_mnf_games"] > 1) & (viewership_with_team_data['Window'] == 'MNF')).astype(int)
    viewership_with_team_data = viewership_with_team_data.drop(columns='num_mnf_games')

    # Create simple model to predict number of viewers given solely the window 
    data_for_model = viewership_with_team_data.loc[viewership_with_team_data['Window'].isin(['SNF','MNF','TNF'])]

    team_agnostic_model = smf.ols('Viewers ~ C(Window) + SharedMNFWindow ',
            data = data_for_model).fit()
    
    data_for_model['team_agnostic_viewership_pred'] = team_agnostic_model.predict(data_for_model)
    data_for_model['viewership_over_expected'] = data_for_model['Viewers'] - data_for_model['team_agnostic_viewership_pred']
    
    if show_plots:
        # Key visualization - create a plot that shows intrigue of teams and number
        # of viewers over expectation
        # Set up the figure and axis
        plt.figure(figsize=(8, 6))
        
        # Create scatter plot
        sc = plt.scatter(
            data_for_model["min_intrigue"],
            data_for_model["max_intrigue"],
            c=data_for_model["viewership_over_expected"],  # Color based on viewership difference
            cmap="coolwarm",  # Red for high, blue for low
            edgecolor="black",
            alpha=0.75,
            vmin=-5,
            vmax=5
        )
        
        # Add color bar
        cbar = plt.colorbar(sc)
        cbar.set_label("Viewers Over Expected (Millions)")
        
        # Labels and title
        plt.xlabel("Min Intrigue")
        plt.ylabel("Max Intrigue")
        plt.title("Viewership Over Expected by Team Intrigue Scores")
        
        # Show the plot
        plt.show()
    
    # A particularly interesting finding is that the identity of the identity of the less-important
    # team doesn't seem to matter much. P-value for max_intrigue is 0 (coeff of .11), but
    # p-value for min intrigue team is .39 (coeff of .025)
    game_viewers_model = smf.ols('Viewers ~ C(Window) + SharedMNFWindow + \
                                  max_above_average  + max_below_average + \
                                      min_above_average',
            data = data_for_model).fit()
    # The variables for same_division and in_flex_window both return insignificant coefficients,
    # so will remove them
    # If try adding an interaction term between max intrigue and min intrigue,
    # we quickly notice that get some undesirable results (i.e. game between two teams 
    # with 60 intrigue would be more highly rated than between a 100 and 140)
    
    # Need to be careful because data sample only has games that have been selected
    # for primetime for some reason, whereas our optimization will include all
    # sorts of games, even those that would never get a second sniff at primetime
    
    # What about lassoing that?
    # Make a copy to avoid modifying original data
    lasso_game_df = data_for_model.copy()
    
    # Define custom transformation function
    def add_intrigue_features(X):
        X = X.copy()  # Avoid modifying the original data

        # Compute max and min intrigue
        X["max_intrigue"] = X[["intrigue_home", "intrigue_away"]].max(axis=1)
        X["min_intrigue"] = X[["intrigue_home", "intrigue_away"]].min(axis=1)
        X['max_above_average'] = X['max_intrigue'].apply(lambda x: 0 if x < 100 else x - 100)
        X['min_above_average'] = X['min_intrigue'].apply(lambda x: 0 if x < 100 else x - 100)
        
        # Compute the multiplication term
        X["AboveAverage_multiply"] = X["max_above_average"] * X["min_above_average"]
        
        return X

    
    # Define feature sets
    intrigue_transformer = FunctionTransformer(add_intrigue_features, validate=False)
    categorical_features = ["Window"]
    numeric_features = ["SharedMNFWindow", 'intrigue_home', 'intrigue_away',
                        'same_division']
    created_features = ["max_intrigue", "min_intrigue", "max_above_average", 
            "min_above_average", "AboveAverage_multiply"]
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features + created_features),  # Scale numeric features
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)  # One-hot encode categorical
    ])
    
    # Define full pipeline
    game_viewers_model_pipeline = Pipeline([
        ('feature_engineering', intrigue_transformer),  # Add intrigue-based features
        ('preprocessing', preprocessor), # Standard scaling & encoding
        ('model', LassoCV(cv=10))  # Lasso with cross-validation
    ])
    
    # Define input (X) and target (y)
    X = lasso_game_df[numeric_features + categorical_features + created_features]
    y = lasso_game_df["Viewers"]
    
    # Fit pipeline
    game_viewers_model_pipeline.fit(X, y)
    
    # View coefficients
    coef_df = pd.DataFrame(
        {"Feature": preprocessor.get_feature_names_out(), 
         "Coefficient": game_viewers_model_pipeline.named_steps["model"].coef_}
    )
    
    # Bias element of min intrigue: If a team was chosen to play in a primetime
    # game despite being unpopular and bad record, they probably have something
    # going for them that our model doesn't account for. Also, just not many very
    # bad teams playing in primetime. So, going to make sure that there is a monotonic
    # increase in min_intrigue and no way for model to favor a bad team. When include
    # min_above_average and min_below average, get undesirable results.
        
    
    return intrigue_model_pipeline, game_viewers_model_pipeline, mean_intrigue, std_intrigue

