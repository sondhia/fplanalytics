import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpl_data.transform import FplApiDataTransformed
import requests
from datetime import datetime
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import time

# Configuration and Setup
st.set_page_config(
    page_title="FPL Data Explorer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for dark mode support
st.markdown("""
    <style>
    .dark-mode {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .metric-card {
        background-color: #2C2C2C;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        # First try to get the raw API data
        response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
        if response.status_code != 200:
            st.error(f"Failed to fetch FPL API data. Status code: {response.status_code}")
            return None, pd.DataFrame(), 1
            
        # Try to parse the JSON response
        try:
            raw_data = response.json()
        except Exception as e:
            st.error(f"Failed to parse FPL API response: {str(e)}")
            return None, pd.DataFrame(), 1
            
        # Initialize FplApiDataTransformed with error handling
        try:
            data = FplApiDataTransformed()
        except Exception as e:
            st.error(f"Failed to initialize FplApiDataTransformed: {str(e)}")
            
            # Fallback to manual data transformation
            try:
                players_df = pd.DataFrame(raw_data['elements'])
                gameweeks_df = pd.DataFrame(raw_data['events'])
                
                # Basic data cleaning
                players_df = players_df.rename(columns={
                    'web_name': 'player_name',
                    'element_type': 'position',
                    'now_cost': '£',
                    'total_points': 'Pts',
                    'selected_by_percent': 'selected_by_percent',
                    'minutes': 'MP',
                    'team_code': 'team'
                })
                players_df['£'] = players_df['£'] / 10  # Convert to actual cost
                players_df['selected_by_percent'] = players_df['selected_by_percent'].astype(float)
                
                # Get current gameweek
                current_gameweek = 1
                if not gameweeks_df.empty:
                    # First try to find current gameweek
                    current_gw_mask = gameweeks_df['is_current'] == True
                    if current_gw_mask.any():
                        current_gameweek = gameweeks_df[current_gw_mask].index[0] + 1
                    # If no current gameweek, try to find next gameweek
                    elif (gameweeks_df['is_next'] == True).any():
                        next_gw_mask = gameweeks_df['is_next'] == True
                        current_gameweek = gameweeks_df[next_gw_mask].index[0]
                    # If still no gameweek found, find the last finished gameweek
                    elif (gameweeks_df['finished'] == True).any():
                        finished_gws = gameweeks_df[gameweeks_df['finished'] == True]
                        current_gameweek = finished_gws.index.max() + 1
                    else:
                        st.warning("Could not determine current gameweek. Defaulting to Gameweek 1.")
                
                return None, players_df, current_gameweek
            except Exception as e:
                st.error(f"Failed to perform manual data transformation: {str(e)}")
                return None, pd.DataFrame(), 1
        
        # If FplApiDataTransformed initialized successfully
        players_df = data.players_df.reset_index()
        gameweeks_df = data.gameweeks_df
        
        # Get current gameweek with improved error handling
        current_gameweek = 1
        try:
            # First try to find current gameweek
            current_gw_mask = gameweeks_df['is_current'] == True
            if current_gw_mask.any():
                current_gameweek = gameweeks_df[current_gw_mask].index[0] + 1
            # If no current gameweek, try to find next gameweek
            elif (gameweeks_df['is_next'] == True).any():
                next_gw_mask = gameweeks_df['is_next'] == True
                current_gameweek = gameweeks_df[next_gw_mask].index[0]
            # If still no gameweek found, find the last finished gameweek
            elif (gameweeks_df['finished'] == True).any():
                finished_gws = gameweeks_df[gameweeks_df['finished'] == True]
                current_gameweek = finished_gws.index.max() + 1
            else:
                st.warning("Could not determine current gameweek. Defaulting to Gameweek 1.")
        except Exception as e:
            st.warning(f"Error determining current gameweek: {str(e)}. Defaulting to Gameweek 1.")
        
        return data, players_df, current_gameweek
        
    except Exception as e:
        st.error(f"Unexpected error loading FPL data: {str(e)}")
        return None, pd.DataFrame(), 1

# Load data with retry mechanism
MAX_RETRIES = 3
retry_count = 0
data, players_df, current_gameweek = None, pd.DataFrame(), 1

while retry_count < MAX_RETRIES:
    data, players_df, current_gameweek = load_data()
    if data is not None or not players_df.empty:
        break
    retry_count += 1
    if retry_count < MAX_RETRIES:
        st.warning(f"Retrying data load... (Attempt {retry_count + 1}/{MAX_RETRIES})")
        time.sleep(2)  # Wait 2 seconds before retrying

# Check if data loaded successfully
if data is None and players_df.empty:
    st.error("Failed to load FPL data after multiple attempts. Please try refreshing the page.")
    st.stop()

# Add data validation
if not players_df.empty:
    required_columns = ['player_name', 'Pts', 'pos', '£']
    missing_columns = [col for col in required_columns if col not in players_df.columns]
    if missing_columns:
        st.error(f"Missing required columns in player data: {', '.join(missing_columns)}")
        st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Player Explorer", "League Analysis", "Predictions", "Team Optimizer"],
    index=0
)

# Home Page
if page == "Home":
    st.title("Fantasy Premier League Data Explorer 🎯")
    
    # Introduction
    st.markdown("""
    Welcome to the FPL Data Explorer! This tool helps you make data-driven decisions 
    for your Fantasy Premier League team using advanced analytics and machine learning.
    """)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Gameweek",
            current_gameweek,
            delta=None,
            help="Current active gameweek"
        )
    with col2:
        avg_points = round(players_df['Pts'].mean(), 1)
        st.metric(
            "Average Points",
            avg_points,
            help="Average points per player"
        )
    with col3:
        total_players = len(players_df)
        st.metric(
            "Players",
            f"{total_players:,}",
            help="Total number of players in FPL"
        )
    with col4:
        avg_price = round(players_df['£'].mean(), 1)
        st.metric(
            "Average Price",
            f"£{avg_price}M",
            help="Average player price"
        )
    
    # Create two columns for the main content
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Top Performers Section
        st.subheader("🌟 Top Performers")
        
        # Get top 5 performers with basic stats
        display_cols = ['player_name', 'team', 'pos', '£', 'Pts']
        if 'selected_by_percent' in players_df.columns:
            display_cols.append('selected_by_percent')
        
        top_performers = players_df.nlargest(5, 'Pts')[display_cols].copy()
        
        # Format the dataframe
        column_config = {
            'player_name': 'Player',
            'team': 'Team',
            'pos': 'Position',
            '£': st.column_config.NumberColumn('Price (£M)', format="%.1f"),
            'Pts': st.column_config.NumberColumn('Points', format="%d"),
        }
        if 'selected_by_percent' in top_performers.columns:
            column_config['selected_by_percent'] = st.column_config.NumberColumn('Selected By %', format="%.1f")
        
        st.dataframe(
            top_performers,
            column_config=column_config,
            hide_index=True
        )
        
        # Points Distribution by Position
        st.subheader("📊 Points Distribution by Position")
        fig = px.box(players_df,
                    x='pos',
                    y='Pts',
                    color='pos',
                    title='Points Distribution by Position',
                    labels={'pos': 'Position', 'Pts': 'Points'})
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Price vs Points Scatter Plot
        st.subheader("💰 Value Analysis")
        fig = px.scatter(players_df,
                        x='£',
                        y='Pts',
                        color='pos',
                        hover_data=['player_name', 'team'],
                        title='Points vs Price',
                        labels={'£': 'Price (£M)', 'Pts': 'Points', 'pos': 'Position'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        # Best Value Players
        st.subheader("💎 Best Value Players")
        
        # Calculate value score (points per million)
        value_players = players_df.copy()
        value_players['value_score'] = value_players['Pts'] / value_players['£']
        
        # Get top 5 value players with minimum 3 games played
        best_value = value_players[value_players['MP'] >= 270].nlargest(5, 'value_score')[
            ['player_name', 'pos', '£', 'Pts', 'value_score']
        ].copy()
        
        # Format value score
        best_value['value_score'] = best_value['value_score'].round(1)
        
        st.dataframe(
            best_value,
            column_config={
                'player_name': 'Player',
                'pos': 'Position',
                '£': st.column_config.NumberColumn('Price (£M)', format="%.1f"),
                'Pts': st.column_config.NumberColumn('Points', format="%d"),
                'value_score': st.column_config.NumberColumn('Points/£M', format="%.1f")
            },
            hide_index=True
        )
        
        # # Most Selected Players (only show if data is available)
        # if 'selected_by_percent' in players_df.columns:
        #     st.subheader("👥 Most Selected")
        #     popular_players = players_df.nlargest(5, 'selected_by_percent')[
        #         ['player_name', 'pos', 'selected_by_percent', 'Pts']
        #     ]
            
        #     # Create bar chart for ownership
        #     fig = px.bar(popular_players,
        #                 x='player_name',
        #                 y='selected_by_percent',
        #                 color='Pts',
        #                 title='Most Selected Players',
        #                 labels={
        #                     'player_name': 'Player',
        #                     'selected_by_percent': 'Ownership %',
        #                     'Pts': 'Points'
        #                 })
            
        #     fig.update_layout(xaxis_tickangle=-45)
        #     st.plotly_chart(fig, use_container_width=True)
        # else:
        #     st.subheader("👥 Player Selection")
        #     st.info("Player selection data is not available at the moment.")
        
        # Quick Tips
        st.subheader("💡 Quick Tips")
        st.markdown("""
        - **Best Formation**: Based on current top performers, 3-4-3 is popular
        - **Value Picks**: Look for players with high Points/£M ratio
        - **Form Players**: Check the Player Explorer for detailed form analysis
        - **Team Structure**: Diversify your team across multiple clubs
        """)
    
    # Add feature showcase at the bottom
    st.markdown("---")
    st.subheader("🚀 Explore More Features")
    
    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    
    with feature_col1:
        st.markdown("""
        #### 🔍 Player Explorer
        Deep dive into individual player statistics and performance trends
        """)
    
    with feature_col2:
        st.markdown("""
        #### 📊 League Analysis
        Analyze league-wide trends and statistical patterns
        """)
    
    with feature_col3:
        st.markdown("""
        #### 🎯 Predictions
        ML-powered predictions for player performance
        """)
    
    with feature_col4:
        st.markdown("""
        #### 🔄 Team Optimizer
        Build the optimal team within your budget
        """)

# Player Explorer
elif page == "Player Explorer":
    st.title("Player Explorer 🔍")
    
    # Create two columns for filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Convert text input to dropdown
        selected_player = st.selectbox(
            "Select Player",
            options=players_df['player_name'].sort_values(),
            index=None,
            placeholder="Choose a player..."
        )
    with col2:
        position_filter = st.multiselect("Filter by Position", players_df['pos'].unique())
    with col3:
        price_range = st.slider("Price Range", 
                              float(players_df['£'].min()), 
                              float(players_df['£'].max()), 
                              (4.0, 13.0))
    
    # Apply filters to create filtered_df
    filtered_df = players_df.copy()
    if position_filter:
        filtered_df = filtered_df[filtered_df['pos'].isin(position_filter)]
    filtered_df = filtered_df[
        (filtered_df['£'] >= price_range[0]) & 
        (filtered_df['£'] <= price_range[1])
    ]
    
    if selected_player:
        # Get player's data
        player_data = players_df[players_df['player_name'] == selected_player].iloc[0]
        
        # Create tabs for different analyses
        tabs = st.tabs(["Season Overview", "Form Analysis", "Fixtures & History", "Comparison", "Advanced Stats"])
        
        with tabs[0]:  # Season Overview
            # Create three columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Points", int(player_data['Pts']))
                st.metric("Price", f"£{player_data['£']}M")
                if 'selected_by_percent' in player_data:
                    st.metric("Selected By", f"{player_data['selected_by_percent']}%")
            
            with col2:
                ppg = player_data['Pts'] / (current_gameweek - 1)
                st.metric("Points per Game", f"{ppg:.2f}")
                if 'form' in player_data:
                    st.metric("Form", player_data['form'])
                if 'MP' in player_data:
                    st.metric("Minutes Played", int(player_data['MP']))
            
            with col3:
                if 'GS' in player_data:
                    st.metric("Goals", int(player_data.get('GS', 0)))
                if 'A' in player_data:
                    st.metric("Assists", int(player_data.get('A', 0)))
                if 'CS' in player_data:
                    st.metric("Clean Sheets", int(player_data.get('CS', 0)))
            
            # Season Progress
            st.subheader("Season Progress")
            progress = (current_gameweek - 1) / 38 * 100
            st.progress(progress / 100, text=f"Season Progress: {progress:.1f}%")
            
            # Performance Breakdown
            st.subheader("Performance Breakdown")
            
            # Create columns for detailed stats
            col1, col2 = st.columns(2)
            
            with col1:
                # Attacking Stats
                st.markdown("**Attacking**")
                attack_stats = {
                    'Goals': int(player_data.get('GS', 0)),
                    'Assists': int(player_data.get('A', 0)),
                    'xG': float(player_data.get('xG', 0)),
                    'xA': float(player_data.get('xA', 0)),
                    'Shots': int(player_data.get('S', 0)),
                    'Key Passes': int(player_data.get('KP', 0))
                }
                
                # Create a DataFrame for better display
                attack_df = pd.DataFrame(list(attack_stats.items()), columns=['Metric', 'Value'])
                st.dataframe(attack_df, hide_index=True)
            
            with col2:
                # Defensive Stats
                st.markdown("**Defensive**")
                defense_stats = {
                    'Clean Sheets': int(player_data.get('CS', 0)),
                    'Goals Conceded': int(player_data.get('GC', 0)),
                    'Own Goals': int(player_data.get('OG', 0)),
                    'Yellow Cards': int(player_data.get('YC', 0)),
                    'Red Cards': int(player_data.get('RC', 0)),
                    'Saves': int(player_data.get('S', 0)) if player_data['pos'] == 'GKP' else '-'
                }
                
                # Create a DataFrame for better display
                defense_df = pd.DataFrame(list(defense_stats.items()), columns=['Metric', 'Value'])
                st.dataframe(defense_df, hide_index=True)
            
            # Value Analysis
            st.subheader("Value Analysis")
            points_per_million = player_data['Pts'] / player_data['£']
            minutes_per_point = player_data['MP'] / player_data['Pts'] if player_data['Pts'] > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Points per £M", f"{points_per_million:.2f}")
            with col2:
                st.metric("Minutes per Point", f"{minutes_per_point:.1f}")
        
        with tabs[1]:  # Form Analysis
            st.subheader("Recent Form")
            
            # Form metrics
            col1, col2 = st.columns(2)
            with col1:
                if 'form' in player_data:
                    st.metric("Current Form", player_data['form'])
                if 'BPS' in player_data:
                    st.metric("Bonus Points System", int(player_data['BPS']))
            with col2:
                if 'II' in player_data:
                    st.metric("ICT Index", player_data['II'])
                if 'T' in player_data:
                    st.metric("Threat", player_data['T'])
            
            # Form components if available
            if all(stat in player_data for stat in ['I', 'C', 'T']):
                st.subheader("ICT Breakdown")
                
                # Create ICT breakdown chart
                ict_data = {
                    'Component': ['Influence', 'Creativity', 'Threat'],
                    'Value': [float(player_data['I']), float(player_data['C']), float(player_data['T'])]
                }
                ict_df = pd.DataFrame(ict_data)
                
                fig = px.bar(ict_df, x='Component', y='Value',
                           title='ICT Components',
                           color='Component')
                st.plotly_chart(fig)
        
        with tabs[2]:  # Fixtures & History
            st.subheader("Upcoming Fixtures")
            
            # Get fixtures if available through the data object
            if data and hasattr(data, 'get_player_summary'):
                fixtures = data.get_player_summary(player_data['player_id'], "fixtures")
                if not fixtures.empty:
                    # Create a DataFrame for fixtures display
                    display_df = fixtures.copy()
                    
                    # Convert difficulty to stars
                    display_df['Difficulty'] = display_df['difficulty'].apply(lambda x: '⭐' * int(x))
                    
                    # Rename columns for display
                    display_df = display_df[['team', 'difficulty', 'Difficulty']]
                    display_df.columns = ['Opponent', 'Difficulty Rating', 'Difficulty']
                    
                    # Display fixtures using native Streamlit
                    st.dataframe(
                        display_df,
                        column_config={
                            'Opponent': 'Opponent',
                            'Difficulty Rating': st.column_config.NumberColumn(
                                'Difficulty Rating',
                                help='Fixture difficulty rating from 1 (easiest) to 5 (hardest)',
                                format='%d'
                            ),
                            'Difficulty': st.column_config.Column(
                                'Difficulty',
                                help='Visual representation of fixture difficulty'
                            )
                        },
                        hide_index=True
                    )
                    
                    # Add fixture difficulty explanation
                    st.markdown("""
                    **Fixture Difficulty Rating:**
                    - ⭐ : Very Easy
                    - ⭐⭐ : Easy
                    - ⭐⭐⭐ : Medium
                    - ⭐⭐⭐⭐ : Hard
                    - ⭐⭐⭐⭐⭐ : Very Hard
                    """)
                else:
                    st.info("No upcoming fixtures available.")
            
            # Historical performance
            st.subheader("Season History")
            if data and hasattr(data, 'get_player_summary'):
                history = data.get_player_summary(player_data['player_id'], "history")
                if not history.empty:
                    # Points trend
                    fig = px.line(history, y='Pts',
                                title='Points History',
                                labels={'Pts': 'Points'})
                    st.plotly_chart(fig)
        
        with tabs[3]:  # Comparison
            st.subheader("Position Comparison")
            
            # Compare with similar priced players in same position
            similar_players = players_df[
                (players_df['pos'] == player_data['pos']) &
                (abs(players_df['£'] - player_data['£']) <= 1)
            ].copy()
            
            # Create comparison metrics
            similar_players['value_score'] = similar_players['Pts'] / similar_players['£']
            
            # Scatter plot of similar players
            fig = px.scatter(similar_players,
                           x='£', y='Pts',
                           hover_data=['player_name', 'value_score'],
                           title=f'Comparison with other {player_data["pos"]}s',
                           labels={'£': 'Price (£M)', 'Pts': 'Points'})
            
            # Add point for selected player
            fig.add_scatter(x=[player_data['£']], y=[player_data['Pts']],
                          mode='markers',
                          marker=dict(size=15, color='red'),
                          name=player_data['player_name'])
            
            st.plotly_chart(fig)
            
            # Show top 5 similar players
            st.subheader("Similar Players Comparison")
            similar_players = similar_players.nlargest(5, 'Pts')[
                ['player_name', '£', 'Pts', 'value_score']
            ]
            st.dataframe(similar_players)
        
        with tabs[4]:  # Advanced Stats
            st.subheader("Advanced Statistics")
            
            # Expected Goals vs Actual
            if all(stat in player_data for stat in ['xG', 'GS']):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Expected Goals (xG)", f"{player_data['xG']:.2f}")
                    st.metric("Goals Scored", int(player_data['GS']))
                with col2:
                    xg_diff = float(player_data['GS']) - float(player_data['xG'])
                    st.metric("Goals vs xG", f"{xg_diff:+.2f}")
            
            # Expected Assists vs Actual
            if all(stat in player_data for stat in ['xA', 'A']):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Expected Assists (xA)", f"{player_data['xA']:.2f}")
                    st.metric("Assists", int(player_data['A']))
                with col2:
                    xa_diff = float(player_data['A']) - float(player_data['xA'])
                    st.metric("Assists vs xA", f"{xa_diff:+.2f}")
            
            # Bonus Points Analysis if available
            if 'BPS' in player_data:
                st.subheader("Bonus Points Analysis")
                bonus_stats = {
                    'Total Bonus Points': int(player_data.get('B', 0)),
                    'Bonus Point System Score': int(player_data['BPS']),
                    'Average BPS per Game': int(player_data['BPS']) / (current_gameweek - 1)
                }
                
                # Display bonus stats
                for stat, value in bonus_stats.items():
                    st.metric(stat, f"{value:.1f}")
    else:
        # Display filtered players table when no player is selected
        st.dataframe(
            filtered_df,
            column_config={
                'player_name': 'Player',
                'team': 'Team',
                'pos': 'Position',
                '£': st.column_config.NumberColumn('Price (£M)', format="%.1f"),
                'Pts': st.column_config.NumberColumn('Points', format="%d"),
                'form': st.column_config.NumberColumn('Form', format="%.2f") if 'form' in filtered_df.columns else None,
                'selected_by_percent': st.column_config.NumberColumn('Selected By %', format="%.1f") if 'selected_by_percent' in filtered_df.columns else None
            },
            hide_index=True
        )

# League Analysis
elif page == "League Analysis":
    st.title("League Analysis 📊")
    
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Points Distribution", "Expected Stats", "Value Analysis", "Team Performance", "Team Network"]
    )
    
    if analysis_type == "Points Distribution":
        # Enhanced points distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(players_df, x='Pts',
                             title='Points Distribution',
                             nbins=30)
            st.plotly_chart(fig)
        
        with col2:
            fig = px.box(players_df, x='pos', y='Pts',
                        title='Points by Position')
            st.plotly_chart(fig)
            
    elif analysis_type == "Expected Stats":
        # Correlation analysis
        st.subheader("Expected Stats Analysis")
        
        # Define the columns we know exist in our DataFrame
        available_stats = []
        potential_stats = [
            'Pts',      # Points
            'MP',       # Minutes played
            'GS',       # Goals scored
            'A',        # Assists
            'CS',       # Clean sheets
            'GC',       # Goals conceded
            'OG',       # Own goals
            'YC',       # Yellow cards
            'RC',       # Red cards
            'S',        # Saves
            'BPS',      # Bonus points system
            'xG',       # Expected goals
            'xA',       # Expected assists
            '£'         # Price
        ]
        
        # Check which stats are available
        for stat in potential_stats:
            if stat in players_df.columns:
                available_stats.append(stat)
        
        if len(available_stats) < 2:
            st.warning("Not enough statistical columns available for correlation analysis.")
        else:
            # Create correlation matrix with available columns
            corr_matrix = players_df[available_stats].corr()
            
            # Create heatmap
            fig = px.imshow(corr_matrix,
                          title='Statistical Correlation Analysis',
                          labels=dict(color="Correlation"),
                          color_continuous_scale="RdBu",
                          aspect="auto")  # Make the heatmap square
            
            # Update layout for better readability
            fig.update_layout(
                width=800,
                height=800,
                xaxis_title="",
                yaxis_title="",
                xaxis={'side': 'bottom'}
            )
            
            # Update axis labels to be more readable
            stat_labels = {
                'Pts': 'Points',
                'MP': 'Minutes Played',
                'GS': 'Goals',
                'A': 'Assists',
                'CS': 'Clean Sheets',
                'GC': 'Goals Conceded',
                'OG': 'Own Goals',
                'YC': 'Yellow Cards',
                'RC': 'Red Cards',
                'S': 'Saves',
                'BPS': 'Bonus Points',
                'xG': 'Expected Goals',
                'xA': 'Expected Assists',
                '£': 'Price'
            }
            
            fig.update_xaxes(ticktext=[stat_labels.get(stat, stat) for stat in available_stats],
                           tickvals=list(range(len(available_stats))))
            fig.update_yaxes(ticktext=[stat_labels.get(stat, stat) for stat in available_stats],
                           tickvals=list(range(len(available_stats))))
            
            st.plotly_chart(fig)
            
            # Add explanation
            st.markdown("""
            ### Understanding the Correlation Matrix
            
            This heatmap shows how different statistics relate to each other:
            - **Dark Red (1.0)**: Perfect positive correlation
            - **Dark Blue (-1.0)**: Perfect negative correlation
            - **White (0)**: No correlation
            
            Key Insights:
            - Look for dark red/blue squares to find strong relationships
            - Points (Pts) correlations show what contributes most to scoring
            - Price (£) correlations indicate value for money
            - Expected stats (xG, xA) vs actual performance show over/underperformance
            """)
            
            # Show strongest correlations
            st.subheader("Key Statistical Relationships")
            
            # Get all correlations in a flat format
            correlations = []
            for i in range(len(available_stats)):
                for j in range(i+1, len(available_stats)):
                    stat1 = available_stats[i]
                    stat2 = available_stats[j]
                    corr_value = corr_matrix.iloc[i, j]
                    correlations.append({
                        'Stat 1': stat_labels.get(stat1, stat1),
                        'Stat 2': stat_labels.get(stat2, stat2),
                        'Correlation': corr_value
                    })
            
            # Convert to DataFrame and sort by absolute correlation
            corr_df = pd.DataFrame(correlations)
            corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
            corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).head(10)
            corr_df = corr_df.drop('Abs_Correlation', axis=1)
            
            # Display the strongest correlations
            st.dataframe(
                corr_df,
                column_config={
                    'Stat 1': 'Statistic 1',
                    'Stat 2': 'Statistic 2',
                    'Correlation': st.column_config.NumberColumn(
                        'Correlation',
                        help='Correlation coefficient between -1 and 1',
                        format="%.3f"
                    )
                },
                hide_index=True
            )
            
    elif analysis_type == "Value Analysis":
        st.subheader("Value for Money Analysis")
        
        # Calculate points per million and ensure positive values for marker size
        players_df['value_score'] = players_df['Pts'] / players_df['£']
        players_df['marker_size'] = np.maximum(players_df['value_score'], 0.1) * 5  # Ensure positive values
        
        # Create scatter plot
        fig = px.scatter(players_df,
                        x='£',
                        y='Pts',
                        color='pos',
                        size='marker_size',
                        hover_data=['player_name', 'value_score'],
                        labels={
                            '£': 'Price (£M)',
                            'Pts': 'Total Points',
                            'pos': 'Position',
                            'value_score': 'Points per £M'
                        },
                        title='Points vs Price by Position')
        
        # Update layout for better readability
        fig.update_layout(
            legend_title="Position",
            hovermode='closest',
            width=800,
            height=600
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Price: £%{x}M<br>" +
                         "Points: %{y}<br>" +
                         "Points per £M: %{customdata[1]:.1f}<br>" +
                         "<extra></extra>"
        )
        
        st.plotly_chart(fig)
        
        # Add explanation
        st.markdown("""
        ### Understanding the Value Analysis
        - **Bubble Size**: Larger bubbles indicate better value (more points per million spent)
        - **Position Colors**: Different colors represent different playing positions
        - **Hover**: Mouse over points to see detailed player information
        """)
        
        # Show top value players
        st.subheader("Top Value Players")
        top_value = players_df.nlargest(10, 'value_score')[
            ['player_name', 'pos', '£', 'Pts', 'value_score']
        ].copy()
        top_value['value_score'] = top_value['value_score'].round(2)
        st.dataframe(
            top_value,
            column_config={
                'player_name': 'Player',
                'pos': 'Position',
                '£': 'Price (£M)',
                'Pts': 'Points',
                'value_score': 'Points per £M'
            },
            hide_index=True
        )
        
    elif analysis_type == "Team Performance":
        st.subheader("Team Performance Analysis")
        
        # Team-wise analysis
        team_stats = players_df.groupby('team').agg({
            'Pts': ['mean', 'sum'],
            'MP': 'sum'  # Changed from 'minutes' to 'MP'
        }).reset_index()
        
        team_stats.columns = ['team', 'avg_points', 'total_points', 'total_minutes']
        
        # Create two separate visualizations for better clarity
        col1, col2 = st.columns(2)
        
        with col1:
            # Total points by team
            fig_total = px.bar(team_stats,
                        x='team',
                        y='total_points',
                        title='Total Team Points',
                        labels={
                            'team': 'Team',
                            'total_points': 'Total Points'
                        })
            fig_total.update_layout(showlegend=False)
            st.plotly_chart(fig_total, use_container_width=True)
        
        with col2:
            # Average points by team
            fig_avg = px.bar(team_stats,
                        x='team',
                        y='avg_points',
                        title='Average Points per Player',
                        labels={
                            'team': 'Team',
                            'avg_points': 'Avg Points/Player'
                        })
            fig_avg.update_layout(showlegend=False)
            st.plotly_chart(fig_avg, use_container_width=True)
        
        # Add team performance metrics table
        st.subheader("Team Performance Metrics")
        st.dataframe(
            team_stats,
            column_config={
                'team': 'Team',
                'avg_points': st.column_config.NumberColumn('Avg Points/Player', format="%.1f"),
                'total_points': st.column_config.NumberColumn('Total Points', format="%d"),
                'total_minutes': st.column_config.NumberColumn('Total Minutes', format="%d")
            },
            hide_index=True
        )
    elif analysis_type == "Team Network":
        st.subheader("Team Network Analysis 🕸️")
        
        # Get unique teams
        teams = sorted(players_df['team'].unique())
        
        # Team selection
        selected_team = st.selectbox("Select Team", teams)
        
        if selected_team:
            # Add filters
            st.sidebar.subheader("Network Filters")
            min_goals = st.sidebar.number_input(
                "Minimum Goals for Players",
                min_value=0,
                max_value=50,
                value=1,
                help="Only show players with at least this many goals"
            )
            min_assists = st.sidebar.number_input(
                "Minimum Assists for Players",
                min_value=0,
                max_value=50,
                value=1,
                help="Only show players with at least this many assists"
            )
            min_combinations = st.sidebar.number_input(
                "Minimum Goal/Assist Combinations",
                min_value=1,
                max_value=20,
                value=1,
                help="Minimum number of goal/assist combinations between players to show connection"
            )
            show_all_players = st.sidebar.checkbox(
                "Show All Players",
                value=False,
                help="Show players even if they don't meet the minimum goals/assists criteria"
            )
            
            # Filter players for selected team
            team_players = players_df[players_df['team'] == selected_team].copy()
            
            # Convert goals and assists to numeric, replacing NaN with 0
            team_players['GS'] = pd.to_numeric(team_players['GS'], errors='coerce').fillna(0)
            team_players['A'] = pd.to_numeric(team_players['A'], errors='coerce').fillna(0)
            
            # Filter players based on criteria
            if not show_all_players:
                significant_players = team_players[
                    (team_players['GS'] >= min_goals) | 
                    (team_players['A'] >= min_assists)
                ].copy()
            else:
                significant_players = team_players.copy()
            
            if len(significant_players) > 0:
                # Create nodes (players)
                nodes = significant_players[['player_name', 'pos', 'Pts', 'GS', 'A']].values.tolist()
                
                # Create edges based on goals and assists
                edges = []
                edge_weights = []
                edge_labels = []
                
                # Create connections based on goals and assists
                for i, player1 in enumerate(nodes):
                    for j, player2 in enumerate(nodes):
                        if i < j:  # Avoid duplicate connections
                            # Get player stats
                            p1_goals = float(player1[3])
                            p1_assists = float(player1[4])
                            p2_goals = float(player2[3])
                            p2_assists = float(player2[4])
                            
                            # Calculate connection strength based on goals and assists
                            connection_strength = 0
                            connection_label = []
                            
                            # Player 1 assists Player 2's goals
                            if p1_assists > 0 and p2_goals > 0:
                                combinations = min(p1_assists, p2_goals)
                                if combinations >= min_combinations:
                                    connection_strength += combinations
                                    connection_label.append(f"{player1[0]} → {player2[0]}: {combinations:.0f} G/A")
                            
                            # Player 2 assists Player 1's goals
                            if p2_assists > 0 and p1_goals > 0:
                                combinations = min(p2_assists, p1_goals)
                                if combinations >= min_combinations:
                                    connection_strength += combinations
                                    connection_label.append(f"{player2[0]} → {player1[0]}: {combinations:.0f} G/A")
                            
                            # Only create edge if there's a significant connection
                            if connection_strength >= min_combinations:
                                edges.append((player1[0], player2[0]))
                                edge_weights.append(connection_strength)
                                edge_labels.append("<br>".join(connection_label))
                
                if len(edges) > 0:
                    # Display team statistics with dark theme first
                    st.markdown("<div style='background-color: rgba(17, 17, 17, 0.7); padding: 15px; border-radius: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                    st.subheader("Team Statistics")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Active Players", len(significant_players))
                    with col2:
                        total_goals = significant_players['GS'].sum()
                        st.metric("Total Goals", f"{total_goals:.0f}")
                    with col3:
                        total_assists = significant_players['A'].sum()
                        st.metric("Total Assists", f"{total_assists:.0f}")
                    with col4:
                        avg_pts = significant_players['Pts'].mean()
                        st.metric("Average Points", f"{avg_pts:.1f}")
                    with col5:
                        total_combinations = sum(edge_weights)
                        st.metric("Total Combinations", f"{total_combinations:.0f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Create network graph
                    fig = go.Figure()
                    
                    # Calculate node positions using a circular layout
                    num_nodes = len(nodes)
                    angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
                    radius = 1
                    node_x = radius * np.cos(angles)
                    node_y = radius * np.sin(angles)
                    
                    # Add edges (connections between players)
                    max_weight = max(edge_weights)
                    min_weight = min(edge_weights)
                    weight_range = max_weight - min_weight
                    
                    # Define gradient colors for edges
                    edge_color_scale = [
                        [0, 'rgba(70, 130, 180, 0.3)'],    # Steel Blue
                        [0.5, 'rgba(135, 206, 235, 0.5)'], # Sky Blue
                        [1, 'rgba(173, 216, 230, 0.7)']    # Light Blue
                    ]
                    
                    for idx, (player1, player2) in enumerate(edges):
                        p1_idx = [n[0] for n in nodes].index(player1)
                        p2_idx = [n[0] for n in nodes].index(player2)
                        
                        # Get positions for both players
                        x0, y0 = node_x[p1_idx], node_y[p1_idx]
                        x1, y1 = node_x[p2_idx], node_y[p2_idx]
                        
                        # Normalize edge width between 2 and 8
                        if weight_range > 0:
                            width = 2 + 6 * (edge_weights[idx] - min_weight) / weight_range
                            color_idx = (edge_weights[idx] - min_weight) / weight_range
                        else:
                            width = 4
                            color_idx = 0.5
                        
                        # Calculate edge color based on weight
                        edge_color = f'rgba(135, 206, 250, {0.3 + 0.4 * color_idx})'  # Light Blue with varying opacity
                        
                        # Add edge with weight-based width and color
                        fig.add_trace(go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(
                                width=width,
                                color=edge_color
                            ),
                            hovertext=edge_labels[idx],
                            hoverinfo='text'
                        ))
                    
                    # Add nodes (players)
                    position_colors = {
                        'GKP': '#FFD700',  # Gold
                        'DEF': '#4169E1',  # Royal Blue
                        'MID': '#32CD32',  # Lime Green
                        'FWD': '#DC143C'   # Crimson
                    }
                    
                    # Add nodes with position-based colors
                    for idx, (name, pos, pts, goals, assists) in enumerate(nodes):
                        # Calculate node size based on goals + assists
                        node_size = 20 + 5 * (goals + assists)  # Base size + bonus for contributions
                        
                        fig.add_trace(go.Scatter(
                            x=[node_x[idx]],
                            y=[node_y[idx]],
                            mode='markers+text',
                            marker=dict(
                                size=node_size,
                                color=position_colors.get(pos, 'gray'),
                                line=dict(width=2, color='rgba(255, 255, 255, 0.8)'),
                                symbol='circle'
                            ),
                            text=name,
                            textposition='bottom center',
                            textfont=dict(
                                size=12,
                                color='white'
                            ),
                            name=f"{name} ({pos})",
                            hovertemplate=(
                                f"<b>{name}</b><br>" +
                                f"Position: {pos}<br>" +
                                f"Points: {pts:.0f}<br>" +
                                f"Goals: {goals:.0f}<br>" +
                                f"Assists: {assists:.0f}<br>" +
                                f"Goal Involvements: {goals + assists:.0f}" +
                                "<extra></extra>"
                            )
                        ))
                    
                    # Update layout with dark theme
                    fig.update_layout(
                        title=dict(
                            text=f"{selected_team} Player Network",
                            font=dict(color='white', size=24)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        plot_bgcolor='rgba(17, 17, 17, 1)',  # Dark background
                        paper_bgcolor='rgba(17, 17, 17, 1)',  # Dark background
                        width=800,
                        height=800,
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            range=[-1.5, 1.5],
                            showline=False
                        ),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            range=[-1.5, 1.5],
                            showline=False
                        ),
                        hoverlabel=dict(
                            bgcolor='rgba(50, 50, 50, 0.9)',
                            font=dict(color='white')
                        )
                    )
                    
                    # Create two columns for the graph and legend
                    graph_col, legend_col = st.columns([0.7, 0.3])
                    
                    with graph_col:
                        # Display the network graph
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with legend_col:
                        # Add some vertical spacing to align with graph
                        st.markdown("<div style='margin-top: 250px;'></div>", unsafe_allow_html=True)
                        
                        # Add legend for position colors with dark theme styling
                        st.markdown("""
                        <div style='background-color: rgba(17, 17, 17, 0.7); padding: 15px; border-radius: 5px;'>
                        <span style='color: white;'><strong>Position Colors:</strong></span><br>
                        <span style='color: #FFD700'>⬤</span> <span style='color: white'>Goalkeeper (GKP)</span><br>
                        <span style='color: #4169E1'>⬤</span> <span style='color: white'>Defender (DEF)</span><br>
                        <span style='color: #32CD32'>⬤</span> <span style='color: white'>Midfielder (MID)</span><br>
                        <span style='color: #DC143C'>⬤</span> <span style='color: white'>Forward (FWD)</span><br>
                        <br>
                        <span style='color: white;'><strong>Visual Elements:</strong></span><br>
                        <span style='color: white'>• Node Size: Larger nodes indicate more goal involvements (goals + assists)</span><br>
                        <span style='color: white'>• Edge Thickness & Color: Thicker and brighter lines indicate more goal/assist combinations between players</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add explanation with dark theme
                    st.markdown("""
                    <div style='background-color: rgba(17, 17, 17, 0.7); padding: 15px; border-radius: 5px; margin-top: 20px;'>
                    <span style='color: white;'><h3>Understanding the Network Graph</h3>
                    
                    This network visualization shows significant goal/assist combinations between players:
                    
                    <ul>
                    <li><strong>Nodes:</strong> Players with meaningful goal or assist contributions
                        <ul>
                        <li>Size indicates total goal involvements (goals + assists)</li>
                        <li>Color represents position</li>
                        </ul>
                    </li>
                    <li><strong>Edges:</strong> Direct goal/assist combinations between players
                        <ul>
                        <li>Thickness and brightness show number of combinations</li>
                        <li>Hover to see exact combination details</li>
                        </ul>
                    </li>
                    </ul>
                    
                    Use the filters in the sidebar to:
                    <ul>
                    <li>Set minimum goals/assists for players to appear</li>
                    <li>Set minimum combinations for connections to show</li>
                    <li>Show/hide players with fewer contributions</li>
                    </ul></span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"No significant goal/assist combinations found with current thresholds (min {min_combinations} combinations required)")
            else:
                st.warning(f"No players found with at least {min_goals} goals or {min_assists} assists")

# Predictions
elif page == "Predictions":
    st.title("Performance Predictions 🎯")
    
    # Create tabs for different prediction types
    pred_tab = st.tabs(["Feature Selection", "Future Predictions"])
    
    with pred_tab[0]:  # Feature Selection tab
        # Define all possible features with descriptions
        all_features = {
            '£': 'Player Price',
            'MP': 'Minutes Played',
            'GS': 'Goals Scored',
            'A': 'Assists',
            'CS': 'Clean Sheets',
            'GC': 'Goals Conceded',
            'OG': 'Own Goals',
            'PS': 'Penalties Saved',
            'PM': 'Penalties Missed',
            'YC': 'Yellow Cards',
            'RC': 'Red Cards',
            'S': 'Saves',
            'B': 'Bonus Points',
            'BPS': 'Bonus Points System Score',
            'I': 'Influence',
            'C': 'Creativity',
            'T': 'Threat',
            'II': 'ICT Index',
            'xG': 'Expected Goals',
            'xA': 'Expected Assists',
            'xGI': 'Expected Goal Involvements',
            'xGC': 'Expected Goals Conceded'
        }
        
        # Find which features are actually available in the data
        available_features = {k: v for k, v in all_features.items() 
                            if k in players_df.columns and 
                            np.issubdtype(players_df[k].dtype, np.number)}
        
        if len(available_features) < 2:
            st.error("Not enough numeric features available for prediction.")
            st.stop()
        
        # Feature selection UI
        st.subheader("Select Features for Prediction")
        st.markdown("Choose which statistics to include in the prediction model:")
        
        # Create columns for better checkbox layout
        num_cols = 3
        cols = st.columns(num_cols)
        
        # Organize features into categories for better UX
        feature_categories = {
            "Basic Stats": ['MP', 'GS', 'A', 'CS', '£'],
            "Advanced Stats": ['BPS', 'I', 'C', 'T', 'II'],
            "Expected Stats": ['xG', 'xA', 'xGI', 'xGC'],
            "Other Stats": ['GC', 'OG', 'PS', 'PM', 'YC', 'RC', 'S', 'B']
        }
        
        # Create a dictionary to store selected features
        selected_features = {}
        
        # Display features by category
        for category, features in feature_categories.items():
            st.markdown(f"**{category}**")
            category_cols = st.columns(num_cols)
            for i, feature in enumerate(features):
                if feature in available_features:
                    col_idx = i % num_cols
                    with category_cols[col_idx]:
                        selected_features[feature] = st.checkbox(
                            f"{feature} ({available_features[feature]})",
                            value=True  # Default to selected
                        )
        
        # Get list of selected features
        features_to_use = [f for f, selected in selected_features.items() if selected]
        
        if len(features_to_use) < 2:
            st.error("Please select at least 2 features for prediction.")
            st.stop()
        
        # Simple prediction model
        def train_prediction_model(selected_features):
            try:
                # Prepare features
                X = players_df[selected_features].copy()
                
                # Handle any missing values
                X = X.fillna(0)
                
                # Target variable
                y = players_df['Pts']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                return model, X_test, y_test, selected_features
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                return None, None, None, None
        
        # Add a train button
        if st.button("Train Model", type="primary"):
            st.info(f"Training model with selected features: {', '.join(features_to_use)}")
            
            # Train model
            model_result = train_prediction_model(features_to_use)
            
            if model_result[0] is not None:
                model, X_test, y_test, features = model_result
                
                # Store the trained model in session state for use in Future Predictions tab
                st.session_state.trained_model = model
                st.session_state.model_features = features
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Show model performance
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    r2 = r2_score(y_test, predictions)
                    st.metric("R² Score", f"{r2:.3f}")
                with col2:
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    st.metric("RMSE", f"{rmse:.1f} points")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.subheader("Feature Importance")
                fig = px.bar(feature_importance,
                            x='feature',
                            y='importance',
                            title='Feature Importance in Predictions',
                            labels={
                                'feature': 'Player Statistics',
                                'importance': 'Importance Score'
                            })
                
                # Update layout for better readability
                fig.update_layout(
                    xaxis_tickangle=-45,
                    width=800,
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig)
                
                # Show example predictions
                st.subheader("Example Predictions")
                example_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
                
                # Get player names for examples
                example_players = players_df.iloc[y_test.index[example_indices]]['player_name'].values
                
                example_predictions = pd.DataFrame({
                    'Player': example_players,
                    'Actual Points': y_test.iloc[example_indices].values.round(1),
                    'Predicted Points': predictions[example_indices].round(1)
                })
                example_predictions['Difference'] = (example_predictions['Predicted Points'] - example_predictions['Actual Points']).round(1)
                
                st.dataframe(
                    example_predictions,
                    column_config={
                        'Player': 'Player Name',
                        'Actual Points': st.column_config.NumberColumn('Actual Points', format="%.1f"),
                        'Predicted Points': st.column_config.NumberColumn('Predicted Points', format="%.1f"),
                        'Difference': st.column_config.NumberColumn('Difference', format="%.1f")
                    },
                    hide_index=True
                )
                
                # Add prediction explanation
                st.markdown("""
                ### Understanding the Predictions
                - The model uses your selected player statistics to predict FPL points
                - **R² Score**: Shows how well the model explains point variations (higher is better, max is 1.0)
                - **RMSE**: Average prediction error in points (lower is better)
                - The Feature Importance chart shows which of your selected statistics are most crucial for predictions
                - Example predictions show how the model performs on random players
                """)
            else:
                st.warning("Could not train prediction model with selected features.")
    
    with pred_tab[1]:  # Future Predictions tab
        st.subheader("Player Points Projection")
        
        # Check if model has been trained
        if not hasattr(st.session_state, 'trained_model'):
            st.warning("Please train a model in the Feature Selection tab first.")
            st.stop()
        
        # Player selection
        selected_player = st.selectbox(
            "Select a player",
            players_df['player_name'].sort_values(),
            index=None,
            placeholder="Choose a player..."
        )
        
        if selected_player:
            # Get player's current stats
            player_data = players_df[players_df['player_name'] == selected_player].iloc[0]
            
            # Get remaining gameweeks
            remaining_gws = 38 - current_gameweek + 1
            
            if remaining_gws > 0:
                # Create future gameweek predictions
                future_gws = list(range(current_gameweek, 39))
                
                # Get player's feature values
                X_player = player_data[st.session_state.model_features].values.reshape(1, -1)
                
                # Predict points for remaining gameweeks with more realistic values
                base_prediction = st.session_state.trained_model.predict(X_player)[0]
                
                # Scale base prediction to be more realistic (typical FPL points range)
                # Most players score between 1-15 points per game
                scaled_base = min(max(base_prediction / (current_gameweek - 1), 1), 15)
                
                predictions = []
                for _ in range(remaining_gws):
                    # Add random variation of ±30% to the scaled base prediction
                    variation = np.random.uniform(-0.3, 0.3)
                    # Ensure prediction is between 0 and 20 points
                    prediction = max(min(scaled_base * (1 + variation), 20), 0)
                    predictions.append(prediction)
                
                # Create DataFrame for visualization
                projection_df = pd.DataFrame({
                    'Gameweek': future_gws,
                    'Predicted Points': predictions
                })
                
                # Calculate cumulative points starting from current total
                current_total = player_data['Pts']
                cumulative_points = [current_total]
                running_total = current_total
                for pred in predictions:
                    running_total += pred
                    cumulative_points.append(running_total)
                
                # Show current season stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Points", int(current_total))
                with col2:
                    ppg = current_total / (current_gameweek - 1)
                    st.metric("Points per Game", f"{ppg:.1f}")
                with col3:
                    projected_total = cumulative_points[-1]
                    st.metric("Projected End Season Points", f"{projected_total:.0f}")
                
                # Weekly points chart
                st.subheader("Weekly Points Prediction")
                fig_weekly = go.Figure()
                
                # Add historical average line
                fig_weekly.add_hline(
                    y=ppg,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Season Average: {ppg:.1f}",
                    annotation_position="bottom right"
                )
                
                fig_weekly.add_trace(go.Scatter(
                    x=projection_df['Gameweek'],
                    y=projection_df['Predicted Points'],
                    name='Predicted Points',
                    line=dict(color='blue'),
                    mode='lines+markers'
                ))
                
                # Update layout for weekly points
                fig_weekly.update_layout(
                    title=f"{selected_player}'s Predicted Points per Gameweek",
                    xaxis=dict(
                        title="Gameweek",
                        tickmode='linear',
                        tick0=current_gameweek,
                        dtick=1
                    ),
                    yaxis=dict(
                        title="Points",
                        range=[0, max(20, max(projection_df['Predicted Points']) * 1.1)],
                        tickformat='.1f',
                        gridcolor='lightgrey'
                    ),
                    hovermode='x unified',
                    showlegend=True,
                    height=400
                )
                
                # Update hover template for weekly points
                fig_weekly.update_traces(
                    hovertemplate="<br>".join([
                        "Gameweek %{x}",
                        "Predicted: %{y:.1f} points",
                        "<extra></extra>"
                    ])
                )
                
                st.plotly_chart(fig_weekly, use_container_width=True)
                
                # Cumulative points chart
                st.subheader("Cumulative Points Projection")
                
                # Create gameweek array including current gameweek
                all_gameweeks = [current_gameweek - 1] + list(projection_df['Gameweek'])
                
                fig_cumulative = go.Figure()
                fig_cumulative.add_trace(go.Scatter(
                    x=all_gameweeks,
                    y=cumulative_points,
                    name='Cumulative Points',
                    line=dict(color='green'),
                    mode='lines+markers'
                ))
                
                # Add marker for current total
                fig_cumulative.add_trace(go.Scatter(
                    x=[current_gameweek - 1],
                    y=[current_total],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='star'
                    ),
                    name='Current Total',
                    showlegend=True
                ))
                
                # Update layout for cumulative points
                fig_cumulative.update_layout(
                    title=f"{selected_player}'s Cumulative Points Projection",
                    xaxis=dict(
                        title="Gameweek",
                        tickmode='linear',
                        tick0=current_gameweek - 1,
                        dtick=1
                    ),
                    yaxis=dict(
                        title="Total Points",
                        tickformat='.0f',
                        gridcolor='lightgrey'
                    ),
                    hovermode='x unified',
                    height=400
                )
                
                # Update hover template for cumulative points
                fig_cumulative.update_traces(
                    hovertemplate="<br>".join([
                        "Gameweek %{x}",
                        "Total Points: %{y:.0f}",
                        "<extra></extra>"
                    ])
                )
                
                st.plotly_chart(fig_cumulative, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                ### Understanding the Projections
                
                #### Weekly Points Prediction (Blue Chart)
                - Shows predicted points for each remaining gameweek
                - Gray dashed line shows player's season average
                - Predictions include form variation (±30%)
                - Points typically range from 0-20 per game
                
                #### Cumulative Points Projection (Green Chart)
                - Red star shows current total points
                - Green line shows projected point accumulation
                - Final point shows projected end-of-season total
                - Based on current form and historical performance
                
                These projections combine the player's season performance with selected model features to estimate future points.
                """)
                
                # Show detailed predictions table
                st.subheader("Detailed Predictions")
                detailed_predictions = projection_df.copy()
                detailed_predictions['Gameweek'] = detailed_predictions['Gameweek'].astype(int)
                detailed_predictions['Total Points'] = cumulative_points[1:]  # Exclude current total
                detailed_predictions = detailed_predictions.round(1)
                st.dataframe(
                    detailed_predictions,
                    column_config={
                        'Gameweek': st.column_config.NumberColumn('Gameweek', format="%d"),
                        'Predicted Points': st.column_config.NumberColumn('Weekly Points', format="%.1f"),
                        'Total Points': st.column_config.NumberColumn('Cumulative Points', format="%.0f")
                    },
                    hide_index=True
                )
            else:
                st.info("Season has ended - no future predictions available.")

# Team Optimizer
elif page == "Team Optimizer":
    st.title("Team Optimizer 🔄")
    
    # Budget and team value inputs
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Available Budget (£M)", min_value=80.0, max_value=100.0, value=100.0, step=0.5)
    with col2:
        num_transfers = st.number_input("Number of Transfers", min_value=1, max_value=15, value=15)
    
    # Formation selection
    formations = {
        "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
        "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
        "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
        "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
        "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2}
    }
    
    formation = st.selectbox("Formation", list(formations.keys()))
    
    # Optimization preferences
    st.subheader("Optimization Preferences")
    col1, col2 = st.columns(2)
    with col1:
        optimization_metric = st.selectbox(
            "Optimize for",
            ["Total Points", "Form", "Expected Points", "Value"]
        )
    with col2:
        include_bench = st.checkbox("Include Bench Players", value=True)
    
    # Team constraints
    st.subheader("Team Constraints")
    col1, col2 = st.columns(2)
    with col1:
        max_team_players = st.number_input("Max players from same team", min_value=1, max_value=3, value=3)
    with col2:
        min_games_played = st.number_input("Min games played", min_value=0, max_value=10, value=3)
    
    def optimize_team(players_df, formation_req, budget, optimization_metric, max_per_team=3, min_games=3):
        # Create position mapping (update this to match the data)
        pos_map = {
            'GK': 'GKP',  # Update goalkeeper position code
            'DEF': 'DEF',
            'MID': 'MID',
            'FWD': 'FWD'
        }
        
        # Filter players based on minimum games played
        players_df = players_df[players_df['MP'] >= (min_games * 90)]
        
        # Calculate value score based on optimization metric
        if optimization_metric == "Total Points":
            players_df['value_score'] = players_df['Pts']
        elif optimization_metric == "Form":
            players_df['value_score'] = players_df['form'].astype(float)
        elif optimization_metric == "Expected Points":
            players_df['value_score'] = players_df['Pts'] / (current_gameweek - 1)  # Points per game
        else:  # Value
            players_df['value_score'] = players_df['Pts'] / players_df['£']
        
        # Initialize selected players dictionary
        selected_players = {pos: [] for pos in ['GK', 'DEF', 'MID', 'FWD']}
        total_cost = 0
        
        # Select players for each position
        for pos, count in formation_req.items():
            # Get the corresponding position code from the mapping
            pos_code = pos_map.get(pos, pos)
            
            # Filter players for this position
            pos_players = players_df[players_df['pos'] == pos_code].copy()
            
            # Add debug information
            st.write(f"Selecting {count} players for position {pos} (code: {pos_code})")
            st.write(f"Found {len(pos_players)} available players for this position")
            
            # Sort by value score
            pos_players = pos_players.sort_values('value_score', ascending=False)
            
            # Select players considering team constraints
            selected_count = 0
            team_counts = {}
            
            for _, player in pos_players.iterrows():
                if selected_count >= count:
                    break
                    
                # Check team constraint
                team = player['team']
                if team_counts.get(team, 0) >= max_per_team:
                    continue
                
                # Check budget constraint
                if total_cost + player['£'] > budget:
                    continue
                
                # Add player
                selected_players[pos].append(player)
                total_cost += player['£']
                selected_count += 1
                team_counts[team] = team_counts.get(team, 0) + 1
            
            # Check if we found enough players for this position
            if selected_count < count:
                st.warning(f"Could only find {selected_count} of {count} required players for position {pos}")
        
        return selected_players, total_cost
    
    if st.button("Optimize Team", type="primary"):
        with st.spinner("Optimizing your team..."):
            # Map formation positions to actual position names
            formation_req = {
                'GK': formations[formation]['GK'],
                'DEF': formations[formation]['DEF'],
                'MID': formations[formation]['MID'],
                'FWD': formations[formation]['FWD']
            }
            
            # Run optimization
            selected_players, total_cost = optimize_team(
                players_df,
                formation_req,
                budget,
                optimization_metric,
                max_per_team=max_team_players,
                min_games=min_games_played
            )
            
            # Display results
            st.subheader("Optimized Team")
            
            # Display formation and cost
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Formation", formation)
            with col2:
                st.metric("Total Cost", f"£{total_cost:.1f}M")
            
            # Create tabs for different views
            team_tabs = st.tabs(["Starting XI", "Statistics", "Expected Points"])
            
            with team_tabs[0]:
                # Display players by position
                for pos in ['GK', 'DEF', 'MID', 'FWD']:
                    st.subheader(f"{pos}")
                    
                    # Check if we have players for this position
                    if not selected_players[pos]:
                        st.info(f"No {pos} players selected yet.")
                        continue
                    
                    # Create columns for players (minimum 1 column)
                    num_players = len(selected_players[pos])
                    if num_players > 0:
                        cols = st.columns(num_players)
                        
                        for idx, player in enumerate(selected_players[pos]):
                            with cols[idx]:
                                # Create player card
                                st.markdown(
                                    f"""
                                    <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; text-align: center;'>
                                        <h4>{player['player_name']}</h4>
                                        <p>£{player['£']}M</p>
                                        <p>{player['team']}</p>
                                        <p><b>{int(player['Pts'])} pts</b></p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
            
            with team_tabs[1]:
                # Create statistics table
                stats_df = pd.DataFrame()
                for pos in selected_players:
                    if selected_players[pos]:  # Only include positions with players
                        stats_df = pd.concat([stats_df, pd.DataFrame(selected_players[pos])])
                
                if not stats_df.empty:
                    # Select relevant columns
                    display_cols = ['player_name', 'team', 'pos', '£', 'Pts']
                    if 'form' in stats_df.columns:
                        display_cols.append('form')
                    if 'selected_by_percent' in stats_df.columns:
                        display_cols.append('selected_by_percent')
                    
                    stats_df = stats_df[display_cols].copy()
                    
                    # Format the dataframe
                    column_config = {
                        'player_name': 'Player',
                        'team': 'Team',
                        'pos': 'Position',
                        '£': st.column_config.NumberColumn('Price (£M)', format="%.1f"),
                        'Pts': st.column_config.NumberColumn('Points', format="%d")
                    }
                    
                    if 'form' in stats_df.columns:
                        column_config['form'] = st.column_config.NumberColumn('Form', format="%.1f")
                    if 'selected_by_percent' in stats_df.columns:
                        column_config['selected_by_percent'] = st.column_config.NumberColumn('Selected By %', format="%.1f")
                    
                    st.dataframe(
                        stats_df,
                        column_config=column_config,
                        hide_index=True
                    )
                else:
                    st.info("No players selected yet.")
            
            with team_tabs[2]:
                if not stats_df.empty:
                    # Calculate expected points
                    total_expected = 0
                    expected_points = []
                    
                    for pos in selected_players:
                        for player in selected_players[pos]:
                            # Calculate expected points based on form and fixtures
                            points_per_game = player['Pts'] / (current_gameweek - 1)
                            expected_points.append({
                                'player_name': player['player_name'],
                                'position': pos,
                                'points_per_game': points_per_game,
                                'expected_next': points_per_game * 1.1  # Simple projection
                            })
                            total_expected += points_per_game
                    
                    # Display expected points
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Points per Week", f"{total_expected:.1f}")
                    with col2:
                        st.metric("Projected Next GW", f"{total_expected * 1.1:.1f}")
                    
                    # Show expected points table
                    expected_df = pd.DataFrame(expected_points)
                    st.dataframe(
                        expected_df,
                        column_config={
                            'player_name': 'Player',
                            'position': 'Position',
                            'points_per_game': st.column_config.NumberColumn('Points per Game', format="%.1f"),
                            'expected_next': st.column_config.NumberColumn('Expected Next GW', format="%.1f")
                        },
                        hide_index=True
                    )
                else:
                    st.info("No players selected yet.")
            
            # Add explanation
            st.markdown("""
            ### Understanding Your Optimized Team
            
            - **Starting XI**: Visual representation of your team with key stats
            - **Statistics**: Detailed performance metrics for each player
            - **Expected Points**: Projected points based on form and fixtures
            
            The optimization algorithm considers:
            - Formation requirements
            - Budget constraints
            - Team limits
            - Player performance metrics
            - Recent form
            
            To get different results, try:
            - Adjusting the budget
            - Changing the formation
            - Modifying team constraints
            - Selecting different optimization metrics
            """)

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Data updates live! | Last update: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
    unsafe_allow_html=True
)
