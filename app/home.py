#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:46:08 2025

@author: neil
"""

import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import sys
import sklearn
from datetime import datetime, date, timedelta

# Comment - yellow background for MNF, green for SNF, purple for TNF
#     appears that color highlight with white text means home, opposite for road

st.set_page_config(page_title="NFL Schedule App", layout="wide")

st.markdown(
    """
    <style>
    body {
            zoom: 0.7;
       }
    div.stTabs {
        display: flex;
        padding-top: 0px; /* Move tabs higher */
        margin-bottom: 0px;
        margin-top: -50px; /* Move tabs even higher */
    }
    /* New styles for wider text display */
    h1, h2, h3, h4, h5, h6, p {
        white-space: nowrap;
    }
    th {
        font-size: 18px;
        background-color: black;
        color: white;
    }
    .week-highlight {
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    .bye-week {
        text-align: center;
        font-weight: bold;
        background-color: #71797E;
    }
    .stApp {
        background-color: #0e1117;
        color: white
    }
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        font-weight: bold;
    }
    .stRadio label {
          color: white !important;
    }
   </style>
    """,
    unsafe_allow_html=True
)


teams = pd.read_csv("results/teams.csv", index_col=False)
scheduled_games = pd.read_csv("results/scheduled_games.csv", index_col=False)
# intrigue percentile to be displayed later
scheduled_games['Intrigue_Percentile'] = scheduled_games['SNF_Viewers'].rank(pct=True) * 100
intrigue_model_pipeline = joblib.load('results/intrigue_model_pipeline.pkl')

week_names = sorted(scheduled_games['Week'].unique())
team_names = sorted(teams['team_abbr'].unique())

st.session_state['teams'] = teams
st.session_state['scheduled_games'] = scheduled_games

# Create an empty DataFrame for the schedule table
schedule_df = pd.DataFrame('', index=week_names, columns=team_names)

# Populate the schedule table
for _, row in scheduled_games.iterrows():
    week, home_team, away_team, slot = row['Week'], row['team_abbr_home'], row['team_abbr_away'], row['Slot']
    schedule_df.loc[week, home_team] = f'{away_team}'
    schedule_df.loc[week, away_team] = f'{home_team}'

# Function to get colored text based on slot
def format_opponent_text(opponent, slot, home):
    color_map = {
        'MNF': '#B59410', # this is a dark gold
        'SNF': '#32CD32',
        'TNF': '#800080',
        'Sun': 'gray'
    }
    text_color = 'white' if home else color_map[slot]
    background_color = color_map[slot] if home else 'white'

    cell_style = (
        f"background-color:{background_color}; color:{text_color};"
        f"padding:6px; border-radius:4px; border: 1px solid black;" 
        f"text-align: center; vertical-align: middle;"
        )
    return f"<td style='{cell_style}'><b>{opponent}</b></td>"

# Function to format the color for the intrigue perentile
def get_intrigue_color(percentile):
    '''
    Function to create the HTML style for the intrigue percentile column. Green
    background is very high ranked game, red background is very low ranked game.
    
    Parameters
    ----------
    percentile : Int
        The percentile (from 0 to 100) that the game's intrigue falls in

    Returns
    -------
    str
        HTML style string.

    '''
    rgb_high = (17,128,65) # the color for 100th percentile
    rgb_low = (212, 36, 73) # color for 0th percentils
    # Convert percentile to a color scale from red (low) to white (mid) to green (high)
    color_int = [int((255 - rgb_low[x])/50.0 * percentile + rgb_low[x]) if percentile <= 50 else int((rgb_high[x] - 255) / 50 * percentile + 2 * 255 - rgb_high[x]) for x in range(3)]
    color_hex = [hex(color_int[x])[2:] for x in range(3)]
    color_hex = [f"0{color_hex[x]}" if len(color_hex[x]) == 1 else color_hex[x] for x in range(3)]
    color_string = "".join(color_hex).upper()
    
    text_color = 'black' if percentile >= 25 and percentile <= 75 else 'white'
    
    cell_style = (
        f"background-color: #{color_string}; color:{text_color};"
        "width: 150px;"
        f"padding:6px; border-radius:4px; border: 1px solid black;" 
        "font-weight: bold;"
        "text-align: center;"
    )
    
    return f"<td style='{cell_style}'><b>{percentile:.0f}</b></td>"

# Simple function to format the projected number of viewers
def format_projected_viewers(row):
    if row['Slot'] == 'TNF':
        return round(row['TNF_Viewers'], 1)
    elif row['Slot'] == 'MNF':
        return round(row['MNF_Viewers'], 1)
    elif row['Slot'] == 'SNF':
        return round(row['SNF_Viewers'], 1)
    else:
        return '--'


# Place radio button in each column to allow user to select page
page_options = ["Home", "League Schedule", "Individual Team Analysis", "Analysis"]

# Horizontal page selector
selected_page = st.radio(
    "Navigation:", 
    page_options,
    horizontal=True
)


# =============================================================================
# button0, button1, button2, button3 = st.columns(4)
# with button0:
#     if st.button(page_options[0]):
#         st.session_state.selected_page = page_options[0]
# with button1:
#     if st.button(page_options[1]):
#         st.session_state.selected_page = page_options[1]
# with button2:
#     if st.button(page_options[2]):
#         st.session_state.selected_page = page_options[2]
# with button0:
#     if st.button(page_options[3]):
#         st.session_state.selected_page = page_options[3]
# 
# =============================================================================


# Landing/Home page
if selected_page == page_options[0]:
    # Page Title
    st.title("🏈 NFL Scheduling App")

    # Introduction
    st.write("""
    Welcome to the **NFL Scheduling App**! This platform presents a prototype schedule for the 2025 NFL season. The schedule was created to maximize primetime television viewership across the season,
    while respecting leaguewide constraints related to competitive balance, travel, etc.
    """)
    
    # Overview of App Sections
    st.header("App Sections")
    st.write("""
    - **League Schedule**: View the schedule grid for the entire league, with all 18 weeks and 32 teams.
    - **Individual Team Analysis**: View the schedule for a selected team, as well as the variables driving viewership projections for that team.
    - **Analysis**: Cover a deeper-dive into the math behind schedule creation.
    """)
    
    # Intrigue Score Explanation
    st.header("How Does It Work?")
    st.markdown("""
    Creating the schedule breaks down into 4 main phases: 
    - **Data Collection**: Find information from the web relevant to understanding the state of primetime television viewership and the parameters of the 2025 NFL schedule.
    - **Viewership Modeling**: Using historical data, determine the variables that lead to individual teams attracting more viewership, and then to games more broadly, to project the number of viewers for each game in each of the different primetime windows.
    - **Scheduling**: Applying the viewership model to all 272 matchups in the 2025 NFL season, and using mathematical optimization techniques
            to find the schedule that respects all mandated scheduling constraints (related to )
    - **App Creation**: Making the app that you are looking at right now!
    """)

    # Intrigue Score Explanation
    st.header("Methods Used")
    st.markdown("""
        All code for this project was written in Python.
        1. Data collection involved web scraping using BeautifulSoup, as well as some manual entry of data from trickier sources.
        2. The viewership model involved a two-step process. Firstly, a "team intrigue" model was fit to determine the 
           expected number of viewers who would watch a game given just one individual team. Secondly, a model was fit to predict
           the number of viewers given the intrigue scores of the teams (among other factors). These models
           were lasso models fit using 5-fold cross-validation in the scikit-learn package. A data pipeline was created
           to handle feature engineering and preprocessing. 
        3. Scheduling was done using integer linear optimization techniques (ILP), achieved with Google's OR-TOOLS package
            in Python. 
    """)

    st.header("Limitations")
    st.markdown("""
        This schedule probably isn't ready for the prime time. Here are some areas where it falls short, relative to what would be required for a real NFL schedule:
- Viewership data was collected from public sources from just 2 seasons of games (2022-23), and only for games in the traditional primetime windows. Real practitioners would hopefully have a much more robust viewership dataset. 
- Only a small number of variables were tested to create the viewership model, and just 2 were included in the final model. Real practitioners would probably spend more time collecting possible factors for their viewership model and testing different model architectures with their more robust dataset.
- To solve for the optimal schedule, a free solver (called CBC) was run on a personal laptop. Real practitioners would have access to better solvers and bigger machines.
- As a result of the limited computational power available, not every constraint that the league might consider was included. For example, this schedule does not account for international games or dates when a team's stadium might be used by other uses (e.g. concerts). Additionally, certain competition constraints, like restrictions on instances of playing a team coming off its bye, were not used in this process.
        """)

    

    # Intrigue Score Explanation
    st.header("What is the Intrigue Score?")
    st.markdown("""
    The **Intrigue Score** is a metric designed to quantify a team's appeal to viewers. It is the basis for the viewership model. It's calculated using several factors:
    - **Win Percentage**: Teams with higher recent success tend to attract more viewers.
    - **Twitter Followers**: A larger social media following indicates greater fan engagement.
    - **Jersey Sales Leaders**: Popular players often boost a team's attractiveness.
    - **Market Size**: Teams from larger markets typically draw more attention.
    """)

    # Game Viewership Model
    st.header("Game Viewership Model")
    st.markdown("""
    To project the number of viewers for each game, we've developed a model that considers:
    - **Team Intrigue Scores**: Both participating teams' scores influence expected viewership.
    - **Game Slot**: Prime-time slots like Thursday Night Football (TNF), Sunday Night Football (SNF), and Monday Night Football (MNF) generally attract more viewers.
    """)

    # Schedule Optimization
    #st.header("Schedule Optimization")
    #st.markdown("""
    #Creating an optimal NFL schedule is a complex task that balances various constraints:
    #- **Logistical Constraints**: Ensuring teams have appropriate balance between home/road games, travel considerations, etc.
    #- **Maximizing Viewership**: Strategically placing games in slots that maximize audience engagement.
    #""")
    #st.markdown("""
    #To achieve this, mathematical optimization techniques were used. Learn more about NFL scheduling optimization](https://www.gurobi.com/events/creating-the-nfl-schedule-with-mathematical-optimization/).
    #""")

    # Footer
    st.markdown("---")
    st.markdown("Explore the app using the navigation buttons above to gain deeper insights into the NFL schedule and its various components.")
    st.markdown("Code is on Github [here](https://github.com/gahan4/nfl-schedule/).")

# League Schedule page
elif selected_page == page_options[1]:

    # Write info to app
    st.write("### NFL Schedule Table")
    
    # Write schedule df
    html_table = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    html_table += "<tr><th>Week</th>" + "".join(f"<th>{team}</th>" for team in schedule_df.columns) + "</tr>"
    
    for week in week_names:
        html_table += f"<tr><td class = 'week-highlight'>{week}</td>"
        for team in team_names:
            opponent = schedule_df.loc[week, team]
            relevant_game_id = scheduled_games.loc[(scheduled_games['Week'] == week) & 
                                       ((scheduled_games['home_team_abbr'] == team) | 
                                        (scheduled_games['away_team_abbr'] == team)), 'game_id']
            if relevant_game_id.empty: # if team has bye this week
                html_table += "<td style=border: 1px solid black;> </td>"
                continue
            else:
                relevant_game_id = relevant_game_id.iloc[0]
            slot = scheduled_games.loc[scheduled_games['game_id'] == relevant_game_id, 'Slot']
            slot = slot.iloc[0] if not slot.empty else 'N/A'
            home_team_abbr = scheduled_games.loc[scheduled_games['game_id'] == relevant_game_id, 'home_team_abbr'].iloc[0]
            home = 1 if home_team_abbr == team else 0
            formatted_opponent = format_opponent_text(opponent, slot, home) if opponent else ''
            html_table += formatted_opponent
        html_table += "</tr>"
    
    html_table += "</table>"
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.write("### Legend")
    st.markdown("🟨 **Gold:** MNF &nbsp;&nbsp;&nbsp;&nbsp;🟩 **Green:** SNF&nbsp;&nbsp;&nbsp;&nbsp;🟪 **Purple:** TNF&nbsp;&nbsp;&nbsp;&nbsp;⬜ **Gray:** Sun Afternoon")
    st.markdown("**Colored Background:** Home Game, **White Background:** Away Game")
        
# Individual team page
elif selected_page == page_options[2]:
    team_choice = st.selectbox("Select a team:", team_names)

    col1, col2 = st.columns([1, 1]) 
    # In the left column, display 
    with col1:  
        team_schedule = scheduled_games[(scheduled_games['home_team_abbr'] == team_choice) |
                                        (scheduled_games['away_team_abbr'] == team_choice)]
        team_schedule = team_schedule.sort_values(by='Week')
        team_schedule['Opponent'] = team_schedule.apply(lambda row: f"@ {row['home_team_abbr']}" if team_choice == row['away_team_abbr'] else f"vs {row['away_team_abbr']}", axis=1)
        team_schedule['Opponent_Intrigue'] = team_schedule.apply(lambda row: row['intrigue_home'] if team_choice == row['away_team_abbr'] else row['intrigue_away'], axis=1)
        team_schedule['formatted_date'] = team_schedule.apply(lambda row: datetime.strptime(row['Date'], "%Y-%m-%d").strftime('%-m/%-d'), axis=1)
        
        
        team_schedule['projected_viewers'] = team_schedule.apply(format_projected_viewers, axis=1)
    
        st.write(f"### Schedule for {team_choice}")
        
        html_schedule = "<table>"
        html_schedule += "<tr><th style='text-align: center;'>Week</th><th style='text-align: center;'>Date</th><th style='text-align: center;'>Slot</th><th style='text-align: center;'>Opponent</th><th style='text-align: center;'>Opponent Intrigue</th><th style='text-align:center;'>Projected Viewers (M)</th><th style='min-width: 150px; text-align: center;'>Game Intrigue Percentile</th></tr>"
        
        for wk in week_names:
            if wk in team_schedule['Week'].values:
                row = team_schedule[team_schedule['Week'] == wk].iloc[0]
                html_schedule += f"<tr><td style='text-align: center;'>{row['Week']}</td><td style='text-align: center;'>{row['formatted_date']}</td><td style='text-align: center;'>{row['Slot']}</td><td style='text-align: center;'>{row['Opponent']}</td><td style='text-align: center;'>{row['Opponent_Intrigue']:.0f}</td><td style='text-align: center;'>{row['projected_viewers']}</td>"
                html_schedule += get_intrigue_color(row['Intrigue_Percentile'])
                html_schedule += "</tr>"
            else: # bye week
                html_schedule += f"<tr><td style='text-align:center;'>{wk}</td><td colspan='6' class='bye-week'>BYE WEEK</td></tr>"

        html_schedule += "</table>"
        st.markdown(html_schedule, unsafe_allow_html=True)
        
        # Add a key or context explanation below the table
        st.markdown("""
        ### Key
        
        - **Opponent Intrigue**: Intrigue score of the opposing team, based on factors like  popularity, performance, and market size. 100 is average, higher is better.
        - **Projected Viewers (M)**: Projected number of viewers for the game (in millions). Estimate is based on historical data, team popularity, and game slot. No projections for traditional Sunday afternoon games.
        - **Game Intrigue Percentile**: Ranks the game based on projected number of slot-agnostic viewers, relative to all other 2025 matchups. 0th percentile is worst game, 100th percentile is best game.
        """)
            
    
    with col2:
        
        st.write(f"### Analysis of {team_choice} Viewership Projections")

        st.markdown(f"""
                    <div style="width: 550px; word-wrap: break-word;">
                    The projection for the number of viewers of any particular game is primarily
                 based off the "Intrigue Score" of the two teams involved. The plot below shows how
                 {team_choice}'s Intrigue Score was compiled. In the plot, the gray bars represent the
                 spread of how much each factor contributes to Intrigue Score for the 32 teams, and the
                 red lines show where {team_choice}'s values stand in the distribution. 
                 <br> <br> 
                 </div>
                 """ ,unsafe_allow_html=True)

        # Want to show a bar plot containing the contribution to intrigue
        # of each feature
        preprocessing = intrigue_model_pipeline.named_steps['preprocessing']
        # Get the names of the features handled by StandardScaler (numeric features)
        num_features = preprocessing.transformers_[0][2]  # StandardScaler is at index 0 in transformers_
        
        # Get the names of the features handled by OneHotEncoder (categorical features)
        #cat_features = preprocessing.transformers_[1][1].named_steps['onehot'].get_feature_names_out()
        #cat_features = preprocessing.transformers_[1][1].get_feature_names_out(['Window'])
        cat_features = ['Window_SNF', 'Window_TNF']
        # Apply preprocessing (scaling, one-hot encoding) - give SNF values for ease of calculation
        team_row = teams.loc[teams['team_abbr'] == team_choice]
        team_rows = teams.copy()
        
        team_rows['Window'] = 'SNF'
        team_rows['SharedMNFWindow'] = 0
        scaled_data = intrigue_model_pipeline.named_steps['preprocessing'].transform(team_rows)
        
        # Get the model coefficients
        model_coefficients = intrigue_model_pipeline.named_steps['model'].coef_
        
        
        # Get the feature names after preprocessing (scaled numerical and one-hot encoded categorical)
        all_feature_names = num_features + list(cat_features)
        
        # Calculate feature contributions for each feature
        #contributions = scaled_data.flatten() * model_coefficients.flatten()
        contributions = scaled_data * model_coefficients.T
        
        contribution_df = pd.DataFrame(contributions, columns=all_feature_names)

        # Reshape the result_df to long format
        contribution_df = contribution_df.melt(var_name='Feature', value_name='Contribution')
        
        # Add the team names
        contribution_df['team'] = np.tile(team_rows['team_abbr'], len(all_feature_names))
        std_intrigue_unscaled = 1.1693118183117757
        contribution_df = contribution_df[~contribution_df['Feature'].isin(['SharedMNFWindow', 'new_high_value_qb'] + list(cat_features))]
        contribution_df['IntriguePointsAdded'] = contribution_df['Contribution'] * 20 / std_intrigue_unscaled

        nicer_contribution_names = {
            'market_pop': 'Market Population',
            'new_high_value_qb': 'New High-Value QB',
            'WeightedJerseySales': 'Jersey Sales',
            'twitter_followers': 'Twitter Followers',
            'WinPct': 'Win Pct'}
        contribution_df['Feature'] = contribution_df['Feature'].map(nicer_contribution_names)
        
        # Prepare data for tornado plot
        # Calculate min, max, and team contribution
        feature_stats = contribution_df.groupby('Feature').agg(
            min_points=('IntriguePointsAdded', 'min'),
            max_points=('IntriguePointsAdded', 'max'),
        ).reset_index()
        
        # Now add the team contribution by merging with the filtered team dataframe
        team_contrib = contribution_df.loc[contribution_df['team'] == team_choice].groupby('Feature')['IntriguePointsAdded'].first().reset_index()
        team_contrib = team_contrib.rename(columns={'IntriguePointsAdded': 'team_contrib'})
        
        # Merge the team_contrib with feature_stats
        feature_stats = feature_stats.merge(team_contrib, on='Feature', how='left')
        
        # Calculate the spread for each feature (max - min)
        feature_stats['spread'] = feature_stats['max_points'] - feature_stats['min_points']
        
        # Sort by spread (largest spread at the top)
        feature_stats = feature_stats.sort_values('spread', ascending=False)

        # Create the tornado plot
        plt.figure(figsize=(7, 5))
        
        # Plot each feature
        num_feature_stats = feature_stats.shape[0]
        for i in range(num_feature_stats):
            y_val_to_plot = num_feature_stats - i - 1
            # Plot the range of intrigue points (from min to max)
            row = feature_stats.iloc[i]
            plt.plot([row['min_points'], row['max_points']], 
                     [y_val_to_plot, y_val_to_plot], 
                     color='grey', linewidth=48)
            
            # Plot the team contribution within the range (using a red vertical line)
            plt.plot([row['team_contrib'], row['team_contrib']], 
                     [y_val_to_plot-0.25, y_val_to_plot+0.25], color='red', 
                     linewidth=4, label='Team Contribution' if i == 3 else "")
            
        
        # Set labels, title, and adjust layout
        plt.yticks(list(reversed(range(num_feature_stats))), feature_stats['Feature'])
        plt.xlabel('Intrigue Points Added')
        plt.title('Tornado Plot: Intrigue Points by Feature')
        
        plt.xlim(-30, 30)
        
        # Add legend
        plt.legend()
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Show the plot
        st.pyplot(plt, use_container_width=True)
        
        
        #st.dataframe(contribution_df)

        # Optionally, plot the changes in intrigue
        # plt.figure(figsize=(6, 4))
        # plt.barh(contribution_df['Feature'],
        #          contribution_df['IntriguePointsAdded'])
        # plt.xlabel('Change in Intrigue Score')
        # plt.title('Change in Intrigue by Feature')
        # plt.xlim(-20, 20)

        # #plt.tight_layout()
        
        # st.markdown('''
        #             The plot below shows how each factor influenced the intrigue score.
        #             ''')
        
        # st.pyplot(plt, use_container_width=True)

        
        # Display additional team information in an HTML table        
        team_info = teams[teams['team_abbr'] == team_choice].iloc[0]
        
        record = f"{team_info['W']}-{team_info['L']}"
        win_pct =  f"{team_info['WinPct']:.3f}".lstrip("0")
        record_rank = teams['WinPct'].rank(ascending=False, method='min')[teams['team_abbr'] == team_choice].values[0]
        twitter_followers = team_info['twitter_followers'] / 1000000
        if twitter_followers <= 1:
            twitter_followers = int(twitter_followers * 1000)
            twitter_followers_string = f"{twitter_followers} K"
        else:
            twitter_followers_string = f"{twitter_followers:.1f} M"
        twitter_rank = teams['twitter_followers'].rank(ascending=False, method='min')[teams['team_abbr'] == team_choice].values[0]
        jersey_sales = team_info['WeightedJerseySales']
        jersey_rank = teams['WeightedJerseySales'].rank(ascending=False, method='min')[teams['team_abbr'] == team_choice].values[0]
        market_pop = f"{team_info['market_pop'] / 1000000:.1f} M"
        market_rank = teams['market_pop'].rank(ascending=False, method='min')[teams['team_abbr'] == team_choice].values[0]

        intrigue = team_info['intrigue']
        intrigue_rank = teams['intrigue'].rank(ascending=False, method='min')[teams['team_abbr'] == team_choice].values[0]
        
        team_info_df = pd.DataFrame({
            "Metric": ["Win Pct", "Win Pct Rank", "Twitter Followers", "Twitter Rank", "Weighted Jersey Sales", "Weighted Jersey Sales Rank",
                       "Market Population", "Market Population Rank",
                       "Intrigue Score", "Intrigue Rank"],
            "Value": [win_pct, record_rank, twitter_followers_string, twitter_rank, 
                      jersey_sales, jersey_rank, market_pop, market_rank,
                      intrigue_rank, intrigue_rank]
        })        
        
        html_team_info = f"""
        <table>
            <tr>
                <th>Category</th>
                <th>Value</th>
                <th>Rank</th>
            </tr>
            <tr>
                <td>Win Pct</td>
                <td>{win_pct}</td>
                <td>{int(record_rank)}</td>
            </tr>
            <tr>
                <td>Twitter Followers</td>
                <td>{twitter_followers_string}</td>
                <td>{int(twitter_rank)}</td>
            </tr>
            <tr>
                <td>Weighted Jersey Sales</td>
                <td>{jersey_sales:.1f}</td>
                <td>{int(jersey_rank)}</td>
            </tr>
            <tr>
                <td>Market Population</td>
                <td>{market_pop}</td>
                <td>{int(market_rank)}</td>
            </tr>
            <tr>
                <td>Intrigue Score</td>
                <td>{int(intrigue)}</td>
                <td>{int(intrigue_rank)}</td>
            </tr>
        </table>
        """
        
        # Want to 
        
        st.subheader(f"Overview of {team_choice} Metrics")
        st.markdown(html_team_info, unsafe_allow_html=True)
        st.markdown(f"""
        ### Key
        - **Win Pct** Team's win percentage during the 2024 regular season.
        - **Twitter Followers**: Team's number of twitter followers (in Nov 2024)
        - **Weighted Jersey Sales**: Each player who finished in Top 50 of NFL apparel sales (according to NFLPA) was
          given a score (1 for highest-seller, down to ~.1 for 50th highest seller), and Weighted Jersey Sales variable
          takes sum of scores for all players expected to be on team in 2025.
        - **Market Population**: Number of people who live in the team's home TV market.
        - **Intrigue Score**: Model's prediction of how "intriguing" the team will be to watch, with 100 being average,
             and higher values being better. As a frame of reference, an intrigue of 120 would indicate the team is
             one standard deviation more intriguing than league average, 80 is one standard deviation less intriguing
             than league average.

        """)

    
