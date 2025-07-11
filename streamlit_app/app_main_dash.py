#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 10:18:39 2025

@author: neil
"""

import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd

# Load datasets
scheduled_games = pd.read_csv("results/scheduled_games.csv")
teams = pd.read_csv("results/teams.csv")

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("NFL Schedule Viewer"),
    dash_table.DataTable(
        id='schedule-table',
        columns=[
            {'name': 'Week', 'id': 'Week'},
            {'name': 'Home', 'id': 'Home'},
            {'name': 'Away', 'id': 'Away'},
            {'name': 'Slot', 'id': 'Slot'}
        ],
        data=scheduled_games[['Week', 'Home', 'Away', 'Slot']].to_dict('records'),
        row_selectable='single',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'}
    ),
    html.Div(id='game-details')
])

# Callback for game details
@app.callback(
    Output('game-details', 'children'),
    Input('schedule-table', 'selected_rows')
)
def display_game_details(selected_rows):
    if selected_rows is None or len(selected_rows) == 0:
        return "Select a game to see details."
    
    row = scheduled_games.iloc[selected_rows[0]]
    return html.Div([
        html.H3(f"{row['Away']} at {row['Home']} (Week {row['Week']})"),
        html.P(f"Slot: {row['Slot']}")
    ])

# Run app
app.server.config["SERVER_NAME"] = "localhost:8050"
if __name__ == '__main__':
    app.run_server(debug=True, host="127.0.0.1", port=8080)
