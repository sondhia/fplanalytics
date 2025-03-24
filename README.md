**FPL Data Explorer**

## Overview

The FPL Data Explorer is an interactive web application built with Streamlit that allows users to explore, analyze, and visualize Fantasy Premier League (FPL) data. The project integrates real-time data transformation via the custom FplApiDataTransformed module and leverages powerful visualization libraries such as Plotly and Streamlit’s native charting capabilities.

## Current Features

### Data Integration & Transformation:
Utilizes a custom module (FplApiDataTransformed) to fetch and transform FPL data for seamless integration within the application.

### Player Explorer:
- Player List & Search: Users can search and filter the player list interactively.
- Detailed Player View: On selecting a player, the app displays detailed stats, including upcoming fixtures and historical performance.
- Custom Styling: Dynamic styling (e.g., fixture difficulty color coding) enhances the user experience.

### League Analysis:
Provides multiple analytical perspectives such as:
Points Distribution: Visualizes overall points distribution and breakdowns by player positions.
Expected Stats Analysis: Correlates traditional points with advanced metrics like xG (expected goals) and xA (expected assists) using scatter plots and heatmaps.
Performance vs Cost: Assesses player value by comparing total points relative to their cost.
Interactive Visualizations:
Leverages Plotly for creating interactive charts and tables, facilitating dynamic data exploration.

### Upcoming Features

### Machine Learning Prediction Model:
- Objective: Develop a predictive model that estimates future FPL points for each player based on historical performance and advanced metrics such as xG, xA, and more.
- Model Development:
  - Training Data: Use historical player performance data including cumulative points, minutes played, fixture difficulty, etc.
  - Feature Engineering: Extract and create features like player form, fixture congestion, and expected contributions to refine prediction accuracy.
  - Integration: Seamlessly integrate model predictions into the dashboard to display forecasted points for upcoming game weeks.

### Enhanced Visualizations & Analytics:
- Model Performance Dashboard: Display metrics and visual insights regarding the prediction model's performance (e.g., error metrics, prediction intervals).
- Scenario Analysis: Allow users to simulate various scenarios (e.g., changes in fixture difficulty or player form) to see how predictions may vary.
- User Customization & Filtering:
  - Extend current filtering capabilities with advanced options such as:
    - Filtering by predicted performance metrics.
    - Advanced sorting options based on cost-effectiveness and future potential.

### Real-time Data Updates:
Integrate with live FPL data sources to keep the application updated with the latest statistics, ensuring the prediction model uses the most recent data.

### Improved Documentation & User Guide:
Comprehensive user documentation detailing:
- Data sources and transformation processes.
- An explanation of the predictive model and its features.
Guidelines on interpreting various visualizations and metrics.


Setup & Installation

Prerequisites
Python 3.x
Required libraries: Streamlit, Pandas, Plotly, NumPy, Requests, etc.
(All dependencies are listed in the requirements.txt file.)
Installation
Clone the repository: 

    git clone https://github.com/yourusername/fpl-data-explorer.git
    cd fpl-data-explorer

Install the dependencies:

    pip install -r requirements.txt

Running the Application
Launch the app using Streamlit:

    streamlit run hello.py
