# Prophet Challenge - MercadoLibre Financial Analysis

## Project Overview
This project performs a comprehensive financial analysis for MercadoLibre, Latin America's most popular e-commerce site. Using Facebook's Prophet model, we analyze the company's financial and user data to identify growth patterns and make predictions that could inform trading strategies.

## Main Objectives
The analysis is divided into four key steps:
1. Find unusual patterns in hourly Google search traffic
2. Mine the search traffic data for seasonality
3. Relate the search traffic to stock price patterns
4. Create a time series model with Prophet

## Features
- Analysis of Google search traffic patterns
- Seasonal trend analysis of user interest
- Stock price correlation studies
- Time series forecasting using Prophet
- Visualization of key metrics and trends

## Requirements
- Python 3.8+
- Facebook Prophet
- Pandas
- Numpy
- Matplotlib
- Required Python packages (see requirements.txt)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Ryan-ai-champ/prophet-challenge.git
cd prophet-challenge

## Usage
1. Open the Jupyter Notebook in Google Colab or your local environment
2. Run the cells sequentially to perform the analysis
3. Follow the comments and markdown cells for detailed explanations

The analysis will:
- Analyze Google search traffic patterns
- Identify seasonal trends
- Study correlations with stock price
- Create and evaluate Prophet forecasting models

## Project Structure
```
prophet-challenge/
├── forecasting_net_prophet.ipynb
├── Resources/
│   ├── google_hourly_search_trends.csv
│   └── mercado_stock_price.csv
├── README.md
└── requirements.txt
```

## Data Sources
The analysis uses two main data sources:
- Google hourly search trends data for MercadoLibre
- MercadoLibre stock price data

## Acknowledgments
- MercadoLibre for providing the business context
- Facebook Prophet team for the forecasting tool
