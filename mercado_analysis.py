import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def load_search_trends():
    """Load Google search trends data from CSV file"""
    try:
        search_trends_df = pd.read_csv(
            Path("Resources/google_hourly_search_trends.csv"),
            index_col=0,
            parse_dates=True
        )
        return search_trends_df
    except FileNotFoundError:
        print("Error: Google search trends data file not found in Resources directory")
        return None
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None

def analyze_may_2020(df):
    """Analyze search patterns for May 2020"""
    if df is None:
        return
    
    # Slice data for May 2020
    may_2020 = df.loc['2020-05']
    
    # Calculate statistics
    total_may_2020 = may_2020['Search Trends'].sum()
    monthly_median = df.resample('M')['Search Trends'].sum().median()
    
    # Create visualization
    plt.figure(figsize=(15, 7))
    plt.plot(may_2020.index, may_2020['Search Trends'])
    plt.title('Google Search Traffic - May 2020')
    plt.xlabel('Date')
    plt.ylabel('Search Trends')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('may_2020_search_trends.png')
    plt.close()
    
    # Print findings
    print(f"\nAnalysis Results for May 2020:")
    print(f"Total Search Traffic: {total_may_2020:,.0f}")
    print(f"Median Monthly Traffic: {monthly_median:,.0f}")
    print(f"Percent Difference: {((total_may_2020 - monthly_median) / monthly_median * 100):.1f}%")

def analyze_seasonality(df):
    """Analyze hourly, daily, and weekly patterns in search traffic"""
    if df is None:
        return
    
    # Hourly patterns
    hourly_avg = df.groupby(df.index.hour)['Search Trends'].mean()
    plt.figure(figsize=(10, 6))
    hourly_avg.plot(kind='bar')
    plt.title('Average Search Traffic by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Average Search Trends')
    plt.tight_layout()
    plt.savefig('hourly_patterns.png')
    plt.close()
    
    # Daily patterns
    daily_avg = df.groupby(df.index.dayofweek)['Search Trends'].mean()
    plt.figure(figsize=(10, 6))
    daily_avg.plot(kind='bar')
    plt.title('Average Search Traffic by Day of Week')
    plt.xlabel('Day of Week (0=Monday)')
    plt.ylabel('Average Search Trends')
    plt.tight_layout()
    plt.savefig('daily_patterns.png')
    plt.close()
    
    # Weekly patterns
    weekly_avg = df.groupby(df.index.isocalendar().week)['Search Trends'].mean()
    plt.figure(figsize=(15, 6))
    weekly_avg.plot()
    plt.title('Average Search Traffic by Week of Year')
    plt.xlabel('Week')
    plt.ylabel('Average Search Trends')
    plt.tight_layout()
    plt.savefig('weekly_patterns.png')
    plt.close()

def load_stock_data():
    """Load stock price data from CSV file"""
    try:
        stock_df = pd.read_csv(
            Path("Resources/mercado_stock_price.csv"),
            index_col=0,
            parse_dates=True
        )
        # Verify required columns are present
        if 'close' not in stock_df.columns:
            print("Error: 'close' column not found in stock price data")
            return None
        return stock_df
    except FileNotFoundError:
        print("Error: Stock price data file not found in Resources directory")
        return None
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None

def analyze_stock_relationship(search_df, stock_df):
    """Analyze relationship between search trends and stock prices"""
    if search_df is None or stock_df is None:
        return
    
    # Merge search and stock data
    combined_df = pd.merge(
        search_df,
        stock_df,
        how='inner',
        left_index=True,
        right_index=True
    )
    
    # Create lagged and derived columns
    combined_df['Lagged Search Trends'] = combined_df['Search Trends'].shift(1)
    combined_df['Stock Volatility'] = combined_df['close'].pct_change().rolling(
        window=4
    ).std() * np.sqrt(4)
    combined_df['Hourly Stock Return'] = combined_df['close'].pct_change()
    
    # Visualize relationships
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.scatter(combined_df['Lagged Search Trends'], combined_df['Stock Volatility'])
    plt.title('Lagged Search Trends vs Stock Volatility')
    plt.xlabel('Lagged Search Trends')
    plt.ylabel('Stock Volatility')
    
    plt.subplot(2, 1, 2)
    plt.scatter(combined_df['Lagged Search Trends'], combined_df['Hourly Stock Return'])
    plt.title('Lagged Search Trends vs Hourly Stock Return')
    plt.xlabel('Lagged Search Trends')
    plt.ylabel('Hourly Stock Return')
    
    plt.tight_layout()
    plt.savefig('stock_relationships.png')
    plt.close()

def create_prophet_forecast(df):
    """Create and analyze Prophet forecast"""
    if df is None:
        return
    
    # Prepare data for Prophet
    prophet_df = df.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    model.fit(prophet_df)
    
    # Create forecast
    future = model.make_future_dataframe(periods=80, freq='H')
    forecast = model.predict(future)
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title('Prophet Forecast of Search Trends')
    plt.tight_layout()
    plt.savefig('prophet_forecast.png')
    plt.close()
    
    # Plot components
    fig = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('prophet_components.png')
    plt.close()
    
    # Print key findings
    print("\nProphet Forecast Analysis:")
    print(f"Trend direction: {'Increasing' if forecast['trend'].diff().mean() > 0 else 'Decreasing'}")
    print(f"Average forecasted value: {forecast['yhat'].mean():.2f}")
    
def main():
    """Main execution function"""
    # Load the search trends data
    print("Loading Google search trends data...")
    search_trends_df = load_search_trends()

    # Analyze May 2020 patterns
    print("\nAnalyzing May 2020 search patterns...")
    analyze_may_2020(search_trends_df)

    # Analyze seasonality patterns
    print("\nAnalyzing seasonality patterns...")
    analyze_seasonality(search_trends_df)

    # Load and analyze stock data
    print("\nLoading and analyzing stock data...")
    stock_df = load_stock_data()
    analyze_stock_relationship(search_trends_df, stock_df)

    # Create Prophet forecast
    print("\nCreating Prophet forecast...")
    create_prophet_forecast(search_trends_df)

    print("\nAnalysis complete. Check the generated visualization files for results.")

if __name__ == "__main__":
    main()

