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
import io
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
    .wrapped-text {
            max-width: 1200px;  /* Limit the width to 800px (adjust as needed) */
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
#viewership_model_pipeline = joblib.load('results/viewership_model_pipeline.pkl')


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
    st.markdown("""
    <div style="width: 1600px; word-wrap: break-word;">
    Welcome to the <b>NFL Scheduling App</b>! This platform presents a prototype schedule for the 2025 NFL season. 
    The schedule was created to maximize primetime television viewership across the season,
    while respecting leaguewide constraints related to competitive balance, travel, etc.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""\n \n
                All code for this project is on Github [here](https://github.com/gahan4/nfl-schedule/).""")

    
    # Overview of App Sections
    st.header("App Sections")
    st.write("""
    - **League Schedule**: View the schedule grid for the entire league, with all 18 weeks and 32 teams.
    - **Individual Team Analysis**: View the schedule for a selected team, as well as the variables driving viewership projections for that team.
    - **Analysis**: Cover a deeper-dive into the process behind schedule creation.
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
- Viewership data was collected from public sources for just 2 seasons of games (2022-23), and only for games in the traditional primetime windows. Real practitioners would hopefully have a much more robust viewership dataset. 
- Only a small number of variables were tested to create the viewership model, and just 2 were included in the final model. Real practitioners would probably spend more time collecting possible factors for their viewership model and testing different model architectures with their more robust dataset.
- To solve for the optimal schedule, a free solver (called CBC) was run on a personal laptop. Real practitioners would have access to better solvers and bigger machines.
- As a result of the limited computational power available, not every constraint that the league might consider was included. For example, this schedule does not account for international games or dates when a team's stadium might be used by other uses (e.g. concerts). Additionally, certain competition constraints, like restrictions on instances of playing a team coming off its bye, were not used in this process.
        """)

    

    # Intrigue Score Explanation
    st.header("Modeling Methodology")
    st.markdown("""
    The **Intrigue Score** is a metric designed to quantify a team's appeal to viewers. It is the basis for the viewership model. It's calculated using several factors:
    - **Win Percentage**: Teams with higher recent success tend to attract more viewers.
    - **Twitter Followers**: A larger social media following indicates greater fan engagement.
    - **Jersey Sales Leaders**: Popular players often boost a team's attractiveness.
    - **Market Size**: Teams from larger markets typically draw more attention.
    
    A second model predicts the number of viewers for each game, based off the following factors:
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
    st.markdown("Explore the app using the navigation buttons at the top of the page.")

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
        - **Win Pct**: Team's win percentage during the 2024 regular season.
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
        
        st.write(f"### Analysis of {team_choice} Viewership Projections")

        st.markdown(f"""
                    <div style="width: 550px; word-wrap: break-word;">
                    The projection for the number of viewers of any particular game is primarily
                 based on the "Intrigue Score" of the two teams involved. The plot below shows how
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

elif selected_page == page_options[3]:
    # Page Layout
    st.title("Analysis")
    
    # Introduction Section
    st.markdown("""
        ## Introduction       
        <div class="wrapped-text">
        The goal of the NFL schedule optimization model is to maximize viewership while respecting certain constraints.  This
        page reviews more specifics of the process used to create the schedule.
        </div>
        
        
        <div class="wrapped-text">
        At a high level, the schedule was created by creating a model to predict
        the likely number of viewers for any game in any slot and then optimizing
        to create a schedule that maximized the projected number of primetime viewers while adhering
        to various league constraints. The modeling step was divided into two parts,
        with a first-level model predicting the "intrigue" of each individual team as compared
        to the other teams in the league, and the second-level model predicting the number
        of viewers for a particular game given the intrigue of the teams involved.
        </div>
            """, unsafe_allow_html=True)
    
    # Data Sources Section
    st.markdown("""
        ## Data Sources
        <div class="wrapped-text">
        In order to create the viewership models, we needed training data.
        The following data sources were used to collect information that was tested in the viewership model. 
        This information was collected for each team and season. Ideally, information
        tested in the model was required to have been known at the time of schedule creation, around 6 months prior to the first game of the season. 
        However, some information, such as market size and number of twitter followers, was taken 
        from the present day (Feb 2025).
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
            - Viewership data was manually taken from SportsMediaWatch. An example link is [here](https://www.sportsmediawatch.com/nfl-tv-ratings-viewership-2023/). Data was uploaded to this website
              in the form of a picture, presenting the number of viewers in every game slot. Data was only acquired for the 2022 and 2023 seasons. Note that data was not available for each game that was
              not independently rated - most relevant, there was no viewership figure for each specific Sunday afternoon game.
            - General information about teams was acquired via the nfl_data_py package. This included their name, division, city, etc.
            - The population of each team's home market was scraped from a SportsMediaWatch article ([Link](https://www.sportsmediawatch.com/nba-market-size-nfl-mlb-nhl-nielsen-ratings/)) using the pandas package.
            - The number of twitter followers for each team was scraped from an article on [SportsMillions](https://www.sportsmillions.com/picks/nfl/which-nfl-team-has-the-most-x-twitter-followers) using the BeautifulSoup package.
              Data is current as of Nov 2024.
            - The jersey sales rankings were scraped from the NFLPA website using BeautifulSoup. An example of such a website is [here](https://nflpa.com/partners/posts/top-50-nfl-player-sales-list-march-1-2021-february-28-2022). The NFLPA publishes 
              a list of the top 50 players in apparel rankings over each league-season. For purposes here, all players were assigned to a specific team (the team they would be on the upcoming season), assigned a value using a decay
              function that assigned weight of 1 to the top-selling player and decayed by e^{-.05 * (Rank - 1)} for each subsequent player (so the 2nd highest seller would get weight ~.95, the 50th highest around ~.09), and then 
              the total score for each team was calculated.
            - A draft intrigue metric was created to attempt to understand the impact of a team having high draft picks on viewership. Draft slot information was scraped from Wikipedia using BeautifulSoup. The idea was to
              assign a value to each draft slot in a highly decayed manner, overwhelmingly upweighting top picks. However, this variable did not prove predictive in the Team Intrigue model, potentially because
              not all top picks were used on offensive players (especially QB's) who may have moved viewership.
            - The required matchups for the 2025 season were scraped from a league press release ([link](https://operations.nfl.com/updates/the-game/2025-opponents-determined/)) using BeautifulSoup.
    """, unsafe_allow_html=True)
    

        

    # Model Explanation Section
    st.markdown("""
                ## Projecting Viewership
        We used two primary models in the scheduling process: the **Team Intrigue Model** and the **Viewership Model**.
    
        ### Team Intrigue Model
        The **Team Intrigue Model** was built to predict the intrigue score of a team based on multiple features, including:
        - Team performance (Win Percentage from previous season)
        - Number of twitter followers
        - Population of home market
        - Popularity of individual players ont eam (measured using jersey sales)
        - Key team changes such as the introduction of a new quarterback or the draft position (e.g., the team with the 1st overall pick could have more intrigue).
        - Nuisance variables were added to account for game slot (i.e. SNF gets more viewership than TNF,
                                                                  games played as part of an MNF doubleheader get fewer viewers than standalone games)
        
    """)
    
    st.markdown("""
    <div class="wrapped-text">
    A Lasso regression model was selected for feature selection because it helps in determining the most influential variables while avoiding overfitting by applying L1 regularization.
    The bar plot below shows the coefficients that the model chose - note that
    all variables were normally scaled, and the response variable is "number of viewers". Aside from the nuisance variables,
    we see that the most important features are the previous season's win percentage, twitter followers, and weighted jersey sales,
    with market population and the presence of a new high-value QB providing some value.
    </div>
    """, unsafe_allow_html=True)
       
       # Plot Lasso Coefficients
    preprocessing = intrigue_model_pipeline.named_steps['preprocessing']
    # Get the names of the features handled by StandardScaler (numeric features)
    num_features = preprocessing.transformers_[0][2]  # StandardScaler is at index 0 in transformers_
        
    # Get the names of the features handled by OneHotEncoder (categorical features)
    cat_features = ['Window_SNF', 'Window_TNF']
    # Apply preprocessing (scaling, one-hot encoding) - give SNF values for ease of calculation
                
    # Get the model coefficients
    model_coefficients = intrigue_model_pipeline.named_steps['model'].coef_
          
    # Get the feature names after preprocessing (scaled numerical and one-hot encoded categorical)
    all_feature_names = num_features + list(cat_features)
    coef_df = pd.DataFrame({'Feature': all_feature_names,
                 'Coef': model_coefficients})
    
    feature_name_map = {
        'market_pop': 'Market Population',
        'WeightedJerseySales': 'Weighted Jersey Sales',
        'twitter_followers': 'Twitter Followers',
        'WinPct': 'Prev Season Win Pct',
        'new_high_value_qb': 'New High Value QB',
        'SharedMNFWindow': 'Shared MNF Window',
        'Window_TNF': 'TNF Slot',
        'Window_SNF': 'SNF Slot'
    }
    coef_df['BetterFeatureName'] = coef_df['Feature'].map(feature_name_map)
    coef_df = coef_df.sort_values(by = 'Coef', ascending=True, key = abs)
        
    plt.figure(figsize=(10, 6))
    plt.barh(coef_df['BetterFeatureName'], coef_df['Coef'])
    plt.xlabel("Coefficient Value (All Variables Scaled)")
    plt.title("Intrigue Model Feature Coefficients")
    
    # Save to BytesIO buffer in order to help with sizing
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=200)
    buf.seek(0)
    
    # Display with reduced visual width
    st.image(buf, width=1000)  # Shrink display width

       
    st.markdown("""
    ### Viewership Model
    The **Viewership Model** predicts the viewership of a game based on the intrigue scores of the two teams involved. Factors included in this model:
    - **Intrigue Scores of Both Teams**: Based on the **Team Intrigue Model**.
    - **Additional Factors**: Whether the game is a divisional matchup, which could increase interest.
    - **Challenges**: A major challenge in building this model was the relatively small number of games with low-ranked teams, leading to a risk of overfitting.
   """)

       
    st.markdown("""
        <div class="wrapped-text">
        A Lasso model was also chosen here. In our particular case, we are particularly worried about overfitting because of limitations
        in the data sample. Notably, we only had 2 seasons worth of data, and only had data available for
        primetime games. As a result, model structures that featured more complex interactions between the two teams caused non-intuitive
        behavior. As an example, there have not been many games where two "non-intriguing" teams have played in primetime. As
        a result, these sorts of games were not in the training data, though there are certainly many games on the schedule
        between two non-intriguing teams. But, many complex model structures were unable to learn that these games are likely
        to be very unpopular, so overweighted on the few such primetime games (which were probably put in primetime because
                                                                               of a factor not considered in the intrigue model),
        causing non-desired behavior. 
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="wrapped-text">
        As an example, the plot below shows the number of "viewers over expected" based on the intrigue score of the two
        teams. Here, expectation is determined simply by meta-factors such as the game's slot. 
        </div>
    """, unsafe_allow_html=True)

    
    st.image("results/viewership_over_expected.png")
    
    
    # Also plot viewership model coefficients
    
    #preprocessing = viewership_model_pipeline.named_steps['preprocessing']
    # Get the names of the features handled by StandardScaler (numeric features)
    #num_features = preprocessing.transformers_[0][2]  # StandardScaler is at index 0 in transformers_
        
    # Get the names of the features handled by OneHotEncoder (categorical features)
    #cat_features = ['Window_SNF', 'Window_TNF']
    # Apply preprocessing (scaling, one-hot encoding) - give SNF values for ease of calculation
                
    # Get the model coefficients
    # model_coefficients = viewership_model_pipeline.named_steps['model'].coef_
          
    # # Get the feature names after preprocessing (scaled numerical and one-hot encoded categorical)
    # all_feature_names = num_features + list(cat_features)
    
    # plt.figure(figsize=(10, 6))
    # plt.barh(all_feature_names, model_coefficients)
    # plt.xlabel("Coefficient Value (All Variables Scaled)")
    # plt.title("Intrigue Model Feature Coefficients")
    # st.pyplot(plt)
    

    
    
    # Scheduling Constraints Section
    st.markdown("""
        ## Scheduling Constraints
        The following constraints were incorporated into the scheduling process:
        - **Basic Scheduling Considerations**: All 272 games prescribed by the league must be played exactly once. Max 1 game per team per week.
        - **Number of Primetime Games**: Exactly one game must be scheduled in each of the 3 primetime windows (TNF, SNF, and MNF) in each week,
           with the exception of Week 18 (no primetime games) and Thanksgiving (3 primetime games).
        - **Bye Week**: Each team must have one bye week between Weeks 5-14.
        - **Stadium Conflicts**: The NY and LA teams cannot both be home during the same week.
        - **Week 18**: Last game of season must be against divisional opponent.
        - **Spacing**: Two teams cannot play 2 games within 2 weeks of each other (i.e. if they play Week X, cannot play again until Week X+3)
        - **Home/Road Balance**: Teams must have at least 1 home game every 3 weeks. Cannot play 4 consecutive home games.
        - **Beginning/End**: Each team must have 1 home game during Weeks 1-2 and 1 home game during Weeks 17-18.
        - **Restricted Dates**: Dallas and Detroit must be home on Thanksgiving.
        - **Thursday Restrictions**
            - Max 2 TNF games per team, with a max of 1 of those at home.
            - If play road Thursday game, then need to be home previous week
            - All teams playing home Thursday games must play within division
              or same division other conference (i.e. AFC East vs NFC East)
              during previous week
            - Teams that play Thursday after Thanksgiving must have played
                on Thanksgiving
            - Teams that play on Thursday can't have played previous SNF or MNF
            - Cannot travel more than two time zones for Thursday game
        - **Primetime Restrictions** 
            - Minimum quality of primetime game required (mean intrigue of 88 for TNF, 93 for MNF, 100 for SNF).
            - Max 5 total primetime games per team
    """)
    
        
    # Problem Setup in Solver Section
    st.markdown("""
        ## Solver Setup and Problem Formulation
        The scheduling problem was formulated as an **integer programming** problem using Google's OR-Tools. A binary variable \(x_{ijk}\) was introduced to represent whether **matchup i** occurs in **week j** at **slot k**.
    
        The problem is set up as follows:
        - **Variables**: Binary variable $x_{ijk}$ was defined for each matchup i, week j, and slot k. With 272 matchups, 18 weeks,
            and 4 slots per week, this created 19,584 binary variables.
        - **Objective Function**: Created by projecting the number of viewers for each matchup in each slot. Assume
           that all games not in a primetime slot would have 0 viewers.
        
        In total, the final model had:
        - **588 equality constraints**
        - **5748 inequality constraints**
        
        The final problem was solved using a CBC solver through Google's OR-Tools, which efficiently handles large constraint sets.
        The solver was allowed to run for 2 hours on a personal laptop.
 
    """)
    
   
    
    
