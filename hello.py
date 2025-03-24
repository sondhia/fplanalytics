from fpl_data.transform import FplApiDataTransformed
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="FPL Data Explorer",
    page_icon="⚽",
    layout="wide"
)

st.title("Fantasy Premier League Data Explorer")

data = FplApiDataTransformed()
# Add debug statements
st.write("Data loaded successfully")
st.write("Number of players in dataset:", len(data.players_df))
# Reset index and make it a column in players_df
data.players_df = data.players_df.reset_index()

# Add tabs at the top level
tab_player_view, tab_analysis = st.tabs(["Player Explorer", "League Analysis"])

def style_background_player_fdr(cell_value):
    bg = "background-color:"
    
    if cell_value == 1:
        return f"{bg} darkgreen; color: white;"
    elif cell_value == 2:
        return f"{bg} green; color: white;"
    elif cell_value == 3:
        return f"{bg} grey; color: white;"
    elif cell_value == 4:
        return f"{bg} orange; color: white;"
    elif cell_value == 5:
        return f"{bg} darkred; color: white;"
    else:
        return ""

with tab_player_view:
    # Creating two columns for the player view layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Player List")
        search = st.text_input("Search Players", "")
        filtered_df = data.players_df[data.players_df['player_name'].str.contains(search, case=False)] if search else data.players_df
        
        # Display interactive dataframe with row selection
        st.dataframe(
            filtered_df,
            key="player_table",
            on_select="rerun",
            selection_mode="single-row"
        )

        # Get selected player ID
        selected_player_id = None
        if st.session_state.get("player_table"):
            selections = st.session_state.player_table.selection.rows
            selected_player_id = filtered_df.iloc[selections[0]]["player_id"] if selections else None

    with col2:
        if selected_player_id:
            st.subheader(f"Player Details: {data.players_df[data.players_df['player_id'] == selected_player_id]['player_name'].iloc[0]}")
            
            # Creating tabs for fixtures and history
            tab1, tab2 = st.tabs(["Upcoming Fixtures", "Player History"])
            
            with tab1:
                st.subheader("Upcoming Fixtures")
                try:
                    player_fixtures = data.get_player_summary(selected_player_id, "fixtures")
                    if not player_fixtures.empty:
                        # Rename columns
                        player_fixtures = player_fixtures.rename(columns={
                            'team': 'Opponent',
                            'difficulty': 'FDR'
                        })

                        # Create a Plotly table
                        fig = go.Figure(data=[go.Table(
                            header=dict(
                                values=['Opponent', 'FDR'],
                                align='center',
                                fill_color='blue',
                                font=dict(size=20, color='black'),
                                height=40
                            ),
                            cells=dict(
                                values=[player_fixtures['Opponent'], player_fixtures['FDR']],
                                align='center',
                                fill_color=[[
                                    'darkgreen' if x == 1 else
                                    'green' if x == 2 else
                                    'grey' if x == 3 else
                                    'orange' if x == 4 else
                                    'darkred' if x == 5 else
                                    'white' for x in player_fixtures['FDR']
                                ]],
                                font=dict(color=['black', 'white']),
                                height=30
                            ),
                            columnwidth=[25, 25]
                        )])

                        fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=len(player_fixtures) * 35 + 40, 
                            width=50 
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No upcoming fixtures found for this player.")
                except Exception as e:
                    st.error(f"Error loading fixtures: {str(e)}")
            
            with tab2:
                st.subheader("Player History")
                try:
                    player_history = data.get_player_summary(selected_player_id, "history")
                    if not player_history.empty:
                        # Display basic stats
                        st.dataframe(player_history)
                        
                        # Add visualizations
                        if len(player_history) > 0:
                            # Existing charts
                            st.subheader("Points History")
                            st.line_chart(player_history['Pts'])
                            
                            st.subheader("Cumulative Points")
                            player_history['total_points'] = player_history['Pts'].cumsum()
                            st.line_chart(player_history['total_points'])
                            
                            # New expected stats analysis
                            st.subheader("Expected vs Actual Performance")
                            fig_expected = px.line(
                                player_history,
                                y=['xG', 'xA', 'GS', 'A'],
                                title='Expected vs Actual G/A',
                                labels={'value': 'Count', 'variable': 'Metric'}
                            )
                            st.plotly_chart(fig_expected, use_container_width=True)
                            
                            st.subheader("Minutes Played")
                            st.line_chart(player_history['MP'])
                    else:
                        st.info("No historical data found for this player.")
                except Exception as e:
                    st.error(f"Error loading history: {str(e)}")

with tab_analysis:
    st.subheader("League-wide Analysis")
    
    # Create tabs for different analyses
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "Points Distribution", 
        "Expected Stats Analysis",
        "Performance vs Cost"
    ])
    
    with analysis_tab1:
        st.subheader("Points Distribution")
        # Overall points distribution
        fig_points = px.histogram(
            data.players_df, 
            x='Pts',
            title='Distribution of Total Points',
            labels={'total_points': 'Total Points', 'count': 'Number of Players'}
        )
        st.plotly_chart(fig_points, use_container_width=True)
        
        # Points by position
        fig_pos = px.box(
            data.players_df,
            x='pos',
            y='Pts',
            title='Points Distribution by Position'
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    with analysis_tab2:
        st.subheader("Expected Stats Analysis")
        
        # Aggregate player history for overall expected stats
        player_stats = data.players_df.copy()
        
        # Create correlation matrix
        corr_cols = ['Pts', 'xG', 'xA', 'xGC']
        corr_matrix = player_stats[corr_cols].corr()
        
        # Plot correlation heatmap
        fig_corr = px.imshow(
            corr_matrix,
            title='Correlation between Points and Expected Stats',
            labels=dict(color="Correlation")
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plots
        col_a, col_b = st.columns(2)
        with col_a:
            fig_xg = px.scatter(
                player_stats,
                x='xG',
                y='Pts',
                title='Points vs Expected Goals',
                hover_data=['player_name']
            )
            st.plotly_chart(fig_xg, use_container_width=True)
        
        with col_b:
            fig_xa = px.scatter(
                player_stats,
                x='xA',
                y='Pts',
                title='Points vs Expected Assists',
                hover_data=['player_name']
            )
            st.plotly_chart(fig_xa, use_container_width=True)

    with analysis_tab3:
        st.subheader("Performance vs Cost")
        
        # Value (points per million) analysis
        player_stats['value'] = player_stats['Pts'] / player_stats['£']
        
        fig_value = px.scatter(
            player_stats,
            x='£',
            y='Pts',
            color='pos',
            title='Points vs Cost by Position',
            hover_data=['player_name', 'value']
        )
        st.plotly_chart(fig_value, use_container_width=True)