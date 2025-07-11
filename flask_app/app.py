#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL Schedule App - Flask Version

A modern web application for viewing and analyzing NFL schedules.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Load data
def load_data():
    """Load all required data files."""
    teams = pd.read_csv("results/teams.csv", index_col=False)
    scheduled_games = pd.read_csv("results/scheduled_games.csv", index_col=False)
    intrigue_model_pipeline = joblib.load('results/intrigue_model_pipeline.pkl')
    
    # Calculate intrigue percentile
    scheduled_games['Intrigue_Percentile'] = scheduled_games['SNF_Viewers'].rank(pct=True) * 100
    
    return teams, scheduled_games, intrigue_model_pipeline

# Load data globally
teams, scheduled_games, intrigue_model_pipeline = load_data()

# Generate coefficient plots at startup to ensure they're always current
print("Generating coefficient plots...")
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_coefficient_plots import generate_coefficient_plots
generate_coefficient_plots()

# Helper functions
def format_opponent_text(opponent, slot, home):
    """Format opponent text for display."""
    color_map = {
        'MNF': '#B59410',  # dark gold
        'SNF': '#32CD32',  # green
        'TNF': '#800080',  # purple
        'Sun': '#808080'   # gray
    }
    
    text_color = 'white' if home else color_map[slot]
    background_color = color_map[slot] if home else 'white'
    
    return {
        'opponent': opponent,
        'slot': slot,
        'home': home,
        'text_color': text_color,
        'background_color': background_color
    }

def format_projected_viewers(row):
    """Format projected viewers for display."""
    if row['Slot'] == 'TNF':
        return round(row['TNF_Viewers'], 1)
    elif row['Slot'] == 'MNF':
        return round(row['MNF_Viewers'], 1)
    elif row['Slot'] == 'SNF':
        return round(row['SNF_Viewers'], 1)
    else:
        return '--'

def get_intrigue_color(percentile):
    """Get color for intrigue percentile display."""
    if percentile <= 25:
        return '#d42449'  # red
    elif percentile <= 50:
        return '#ff6b6b'  # light red
    elif percentile <= 75:
        return '#4ecdc4'  # light green
    else:
        return '#118041'  # green

@app.route('/')
def home():
    """Home page."""
    return render_template('home.html')

@app.route('/schedule')
def schedule():
    """League schedule page."""
    # Create schedule matrix
    week_names = sorted(scheduled_games['Week'].unique())
    team_names = sorted(teams['team_abbr'].unique())
    
    # Create schedule data for template
    schedule_data = []
    for week in week_names:
        week_data = {'week': week, 'games': []}
        for team in team_names:
            # Find game for this team in this week
            game = scheduled_games[
                (scheduled_games['Week'] == week) & 
                ((scheduled_games['home_team_abbr'] == team) | 
                 (scheduled_games['away_team_abbr'] == team))
            ]
            
            if game.empty:
                # Bye week
                week_data['games'].append({
                    'team': team,
                    'opponent': None,
                    'slot': None,
                    'home': None,
                    'bye': True
                })
            else:
                game = game.iloc[0]
                home_team = game['home_team_abbr']
                away_team = game['away_team_abbr']
                slot = game['Slot']
                
                if team == home_team:
                    opponent = away_team
                    home = True
                else:
                    opponent = home_team
                    home = False
                
                week_data['games'].append({
                    'team': team,
                    'opponent': opponent,
                    'slot': slot,
                    'home': home,
                    'bye': False
                })
        
        schedule_data.append(week_data)
    
    return render_template('schedule.html', 
                         schedule_data=schedule_data, 
                         teams=team_names)

@app.route('/team/<team_abbr>')
def team_analysis(team_abbr):
    """Individual team analysis page."""
    team_info = teams[teams['team_abbr'] == team_abbr].iloc[0]
    
    # Get team schedule
    team_schedule = scheduled_games[
        (scheduled_games['home_team_abbr'] == team_abbr) |
        (scheduled_games['away_team_abbr'] == team_abbr)
    ].sort_values(by='Week')
    
    # Process schedule data
    schedule_data = []
    for _, game in team_schedule.iterrows():
        if team_abbr == game['away_team_abbr']:
            opponent = f"@ {game['home_team_abbr']}"
            opponent_intrigue = game['intrigue_home']
            home = False
        else:
            opponent = f"vs {game['away_team_abbr']}"
            opponent_intrigue = game['intrigue_away']
            home = True
        
        date_obj = datetime.strptime(game['Date'], "%Y-%m-%d")
        formatted_date = date_obj.strftime('%-m/%-d')
        
        projected_viewers = format_projected_viewers(game)
        intrigue_color = get_intrigue_color(game['Intrigue_Percentile'])
        
        schedule_data.append({
            'week': game['Week'],
            'date': formatted_date,
            'slot': game['Slot'],
            'opponent': opponent,
            'opponent_intrigue': round(opponent_intrigue, 0),
            'projected_viewers': projected_viewers,
            'intrigue_percentile': round(game['Intrigue_Percentile'], 0),
            'intrigue_color': intrigue_color
        })
    
    # Add bye weeks
    all_weeks = sorted(scheduled_games['Week'].unique())
    scheduled_weeks = set(game['week'] for game in schedule_data)
    bye_weeks = [week for week in all_weeks if week not in scheduled_weeks]
    
    for bye_week in bye_weeks:
        schedule_data.append({
            'week': bye_week,
            'date': '',
            'slot': 'BYE',
            'opponent': 'BYE WEEK',
            'opponent_intrigue': '',
            'projected_viewers': '',
            'intrigue_percentile': '',
            'intrigue_color': '',
            'bye': True
        })
    
    # Sort by week
    schedule_data.sort(key=lambda x: x['week'])
    
    # Team stats
    team_stats = {
        'name': team_info['team_name'],
        'abbr': team_abbr,
        'record': f"{team_info['W']}-{team_info['L']}",
        'win_pct': f"{team_info['WinPct']:.3f}".lstrip("0"),
        'win_pct_rank': int(teams['WinPct'].rank(ascending=False, method='min')[teams['team_abbr'] == team_abbr].values[0]),
        'twitter_followers': team_info['twitter_followers'] / 1000000,
        'twitter_rank': int(teams['twitter_followers'].rank(ascending=False, method='min')[teams['team_abbr'] == team_abbr].values[0]),
        'jersey_sales': team_info['WeightedJerseySales'],
        'jersey_rank': int(teams['WeightedJerseySales'].rank(ascending=False, method='min')[teams['team_abbr'] == team_abbr].values[0]),
        'market_pop': f"{team_info['market_pop'] / 1000000:.1f} M",
        'market_rank': int(teams['market_pop'].rank(ascending=False, method='min')[teams['team_abbr'] == team_abbr].values[0]),
        'intrigue': round(team_info['intrigue'], 0),
        'intrigue_rank': int(teams['intrigue'].rank(ascending=False, method='min')[teams['team_abbr'] == team_abbr].values[0])
    }
    
    return render_template('team_analysis.html', 
                         team_stats=team_stats, 
                         schedule_data=schedule_data)

@app.route('/team', methods=['GET'])
def team_analysis_dropdown():
    """Team analysis page with dropdown selection."""
    team_names = sorted(teams['team_abbr'].unique())
    selected_team = team_names[0]  # Default to first team
    
    # Render the team analysis for the selected team
    team_info = teams[teams['team_abbr'] == selected_team].iloc[0]
    team_schedule = scheduled_games[(scheduled_games['home_team_abbr'] == selected_team) | (scheduled_games['away_team_abbr'] == selected_team)].sort_values(by='Week')
    schedule_data = []
    for _, game in team_schedule.iterrows():
        if selected_team == game['away_team_abbr']:
            opponent = f"@ {game['home_team_abbr']}"
            opponent_intrigue = game['intrigue_home']
            home = False
        else:
            opponent = f"vs {game['away_team_abbr']}"
            opponent_intrigue = game['intrigue_away']
            home = True
        date_obj = datetime.strptime(game['Date'], "%Y-%m-%d")
        formatted_date = date_obj.strftime('%-m/%-d')
        projected_viewers = format_projected_viewers(game)
        intrigue_color = get_intrigue_color(game['Intrigue_Percentile'])
        schedule_data.append({
            'week': game['Week'],
            'date': formatted_date,
            'slot': game['Slot'],
            'opponent': opponent,
            'opponent_intrigue': round(opponent_intrigue, 0),
            'projected_viewers': projected_viewers,
            'intrigue_percentile': round(game['Intrigue_Percentile'], 0),
            'intrigue_color': intrigue_color
        })
    all_weeks = sorted(scheduled_games['Week'].unique())
    scheduled_weeks = set(game['week'] for game in schedule_data)
    bye_weeks = [week for week in all_weeks if week not in scheduled_weeks]
    for bye_week in bye_weeks:
        schedule_data.append({
            'week': bye_week,
            'date': '',
            'slot': 'BYE',
            'opponent': 'BYE WEEK',
            'opponent_intrigue': '',
            'projected_viewers': '',
            'intrigue_percentile': '',
            'intrigue_color': '',
            'bye': True
        })
    schedule_data.sort(key=lambda x: x['week'])
    team_stats = {
        'name': team_info['team_name'],
        'abbr': selected_team,
        'record': f"{team_info['W']}-{team_info['L']}",
        'win_pct': f"{team_info['WinPct']:.3f}".lstrip("0"),
        'win_pct_rank': int(teams['WinPct'].rank(ascending=False, method='min')[teams['team_abbr'] == selected_team].values[0]),
        'twitter_followers': team_info['twitter_followers'] / 1000000,
        'twitter_rank': int(teams['twitter_followers'].rank(ascending=False, method='min')[teams['team_abbr'] == selected_team].values[0]),
        'jersey_sales': team_info['WeightedJerseySales'],
        'jersey_rank': int(teams['WeightedJerseySales'].rank(ascending=False, method='min')[teams['team_abbr'] == selected_team].values[0]),
        'market_pop': f"{team_info['market_pop'] / 1000000:.1f} M",
        'market_rank': int(teams['market_pop'].rank(ascending=False, method='min')[teams['team_abbr'] == selected_team].values[0]),
        'intrigue': round(team_info['intrigue'], 0),
        'intrigue_rank': int(teams['intrigue'].rank(ascending=False, method='min')[teams['team_abbr'] == selected_team].values[0])
    }
    return render_template('team_analysis_dropdown.html', team_stats=team_stats, schedule_data=schedule_data, team_names=team_names, selected_team=selected_team)

@app.route('/analysis')
def analysis():
    """Analysis page."""
    # Prepare teams data for the analysis table
    teams_data = []
    for _, team in teams.iterrows():
        # Calculate intrigue color and text color (same logic as team analysis)
        # INVERT the percentile so high scores = green, low scores = red
        intrigue_percentile = (teams['intrigue'].rank(ascending=True, method='min')[teams['team_abbr'] == team['team_abbr']].values[0] - 1) / (len(teams) - 1) * 100
        
        # Color interpolation logic (same as team analysis)
        rgb_low = (212, 36, 73)     # Red
        rgb_mid = (255, 255, 255)   # White  
        rgb_high = (17, 128, 65)    # Green
        
        if intrigue_percentile <= 50:
            color_int = [
                int((255 - rgb_low[0]) / 50.0 * intrigue_percentile + rgb_low[0]),
                int((255 - rgb_low[1]) / 50.0 * intrigue_percentile + rgb_low[1]),
                int((255 - rgb_low[2]) / 50.0 * intrigue_percentile + rgb_low[2])
            ]
        else:
            color_int = [
                int((rgb_high[0] - 255) / 50 * intrigue_percentile + 2 * 255 - rgb_high[0]),
                int((rgb_high[1] - 255) / 50 * intrigue_percentile + 2 * 255 - rgb_high[1]),
                int((rgb_high[2] - 255) / 50 * intrigue_percentile + 2 * 255 - rgb_high[2])
            ]
        
        intrigue_color = f"rgb({color_int[0]}, {color_int[1]}, {color_int[2]})"
        intrigue_text_color = 'black' if intrigue_percentile >= 25 and intrigue_percentile <= 75 else 'white'
        
        teams_data.append({
            'team_abbr': team['team_abbr'],
            'WinPct': team['WinPct'],
            'market_pop': team['market_pop'],
            'twitter_followers': team['twitter_followers'],
            'new_high_value_qb': team['new_high_value_qb'],
            'WeightedJerseySales': team['WeightedJerseySales'],
            'intrigue': team['intrigue'],
            'intrigue_color': intrigue_color,
            'intrigue_text_color': intrigue_text_color
        })
    
    return render_template('analysis.html', teams_data=teams_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 