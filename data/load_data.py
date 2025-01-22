#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:05:45 2025

@author: neil
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re

def get_teams_and_standings(season):
    """
    Finds the teams participating in the NFL that season and their record
    during the year. Uses the nfl_data_py package to find the information.

    Parameters
    ----------
    season : int
        The season that you want to find the teams and standings for.

    Returns
    -------
    Pandas data frame that contains information about the team, their record,
    their division ranking, etc.

    """
    teams = nfl.import_team_desc()
    
    games = nfl.import_schedules([season])
    # filter to only include regular season games
    games = games[games['game_type'] == 'REG']
    
    # From these game-level results, create the league standings for the season
    games['home_win'] = np.where(games['home_score'] > games['away_score'], 1, 0)
    games['home_loss'] = np.where(games['home_score'] < games['away_score'], 1, 0)
    games['tie'] = np.where(games['home_score'] == games['away_score'], 1, 0)
    
    # Find number of wins, losses, and ties for each team, as both the home and away team
    home_results = games.groupby('home_team').agg(
        home_W = ('home_win', 'sum'),
        home_L = ('home_loss', 'sum'),
        home_T = ('tie', 'sum')).reset_index().rename(columns={'home_team':'team'})
    away_results = games.groupby('away_team').agg(
        away_W = ('home_loss', 'sum'),
        away_L = ('home_win', 'sum'),
        away_T = ('tie', 'sum')).reset_index().rename(columns={'away_team':'team'})
    
    standings = home_results.merge(away_results,how='inner', on='team')
    standings['W'] = standings['home_W'] + standings['away_W']
    standings['L'] = standings['home_L'] + standings['away_L']
    standings['T'] = standings['home_T'] + standings['away_T']
    standings['WinPct'] = (standings['W'] + 0.5*standings['T']) / \
        (standings['W'] + standings['L'] + standings['T'])
    
    # For each team, join in the standings information, and create variable
    # that contains info relevant to how each team did in their division
    teams = teams.merge(standings, how='inner', left_on='team_abbr', right_on='team')
    teams['division_place'] = teams.groupby('team_division')['WinPct'].rank(method='first', ascending=False).astype(int)
    
    # For each team, make a note if they are in the 
    
    # Add an ID for help down the line
    # Create appropriate data frames to be able to understand Id's
    teams = teams.sort_values(by="team_abbr")
    teams['team_id'] = range(0, len(teams))
    
    return(teams)


def add_popularity_metrics(teams):
    """
    Given a data frame containing the teams in the NFL, searches the web
    to find their metro area population
    Parameters
    ----------
    teams : Pandas Data Frame
        Must contain a column called 'team_nick' that contains the team's nicknae.

    Returns
    -------
    Pandas data frame - the same as inputted, but containing additional columns
    with the name of the TV market the team is in and the size of that market.

    """
    url = 'https://www.sportsmediawatch.com/nba-market-size-nfl-mlb-nhl-nielsen-ratings/'
    metro_area_population = pd.read_html(url)[0]
    
    # for each team, find the market it is located in
    teams['market'] = None
    teams['market_pop'] = None
    for r in range(teams.shape[0]):
        metro_area_population['has_team'] = metro_area_population['NFL'].str.contains(teams['team_nick'][r])
        index_of_team = metro_area_population[metro_area_population['has_team']].index.tolist()
        teams.loc[r, 'market'] = metro_area_population['Market'][index_of_team[0]]
        teams.loc[r, 'market_pop'] = metro_area_population['Homes (000)'][index_of_team[0]] * 1000
        
    # Also, find the number of twitter followers per team. Might better capture subtleties
    # like popular teams in small cities (i.e. Packers) and unpopular teams in
    # larger markets (i.e. LA teams)
    url_twitter_followers = 'https://www.sportsmillions.com/picks/nfl/which-nfl-team-has-the-most-x-twitter-followers'
    response = requests.get(url_twitter_followers)
    response.raise_for_status()
    
    # Use beautifulsoup to read the data, use regular expression to convert the followers
    # string into a float, then merge with the current teams data
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table') 
    followers = pd.read_html(str(table))[0]
    followers['X/Twitter Followers'] = 1000000 * followers['X/Twitter Followers'].apply(lambda x: float(re.search(r"[\d.]+", x).group()))
    followers.loc[followers['NFL Team'] == 'LA Chargers', 'NFL Team'] = 'Los Angeles Chargers'    
    followers = followers.loc[:, ['NFL Team', 'X/Twitter Followers']].rename(columns={'NFL Team':'team_name',
                                                                                      'X/Twitter Followers': 'twitter_followers'})
    
    teams = teams.merge(followers, how='left', on='team_name')
    
    return(teams)
