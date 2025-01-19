#!/usr/bin/env python3
"""
MercadoLibre Analysis Script
===========================

This script performs a comprehensive analysis of MercadoLibre search trends and stock data.

The analysis includes:
1. Analysis of unusual patterns in hourly Google search traffic
2. Mining of search traffic data for seasonality patterns
3. Analysis of relationship between search traffic and stock price
4. Time series forecasting using Prophet

Required packages:
- pandas: for data manipulation
- holoviews: for interactive visualization 
- prophet: for time series forecasting
- numpy: for numerical computations
"""
import pandas as pd
import numpy as np
import holoviews as hv
from prophet import Prophet
from datetime import datetime

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
hv.extension('bokeh')

def load_search_trends():
    """
    Load and process the Google search trends dataset.
    
    Returns:
        pandas.DataFrame: Search trends dataset with datetime index
    """
    try:
        print("\nLoading Google search trends data...")
        print("="*80)
        
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    index_col='Date', 
                    parse_dates=True)
        
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the search trends data file")
        return None
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None

def analyze_may_2020(df):
    """
    Analyze search traffic patterns for May 2020.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Filter for May 2020
    may_2020 = df['2020-05']
    
    # Calculate total search traffic
    may_total = may_2020['Search Trends'].sum()
    
    # Calculate median monthly traffic
    monthly = df.resample('M')['Search Trends'].sum()
    median_monthly = monthly.median()
    
    # Calculate percentage difference
    pct_diff = ((may_total - median_monthly) / median_monthly) * 100
    
    results = {
        'may_total': may_total,
        'median_monthly': median_monthly,
        'pct_difference': pct_diff,
        'conclusion': f"May 2020 total search traffic was {pct_diff:.1f}% {'above' if pct_diff > 0 else 'below'} the median"
    }
    
    # Create visualization
    plot = hv.Curve(df['2020-05'], 'Date', 'Search Trends', label='May 2020 Search Traffic')
    plot.opts(title='Search Traffic During May 2020',
            width=800, height=400)
            
    return results, plot

def analyze_hourly_patterns(df):
    """
    Analyze and visualize hourly search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by hour and calculate mean
    hourly_avg = df.groupby(df.index.hour)['Search Trends'].mean()
    
    # Find peak hour
    peak_hour = hourly_avg.idxmax()
    peak_traffic = hourly_avg.max()
    
    results = {
        'peak_hour': peak_hour,
        'peak_traffic': peak_traffic,
        'pattern': f"Search traffic peaks at hour {peak_hour:02d}:00 with average value of {peak_traffic:.1f}"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Hour': hourly_avg.index, 'Traffic': hourly_avg.values})
    plot = hv.Curve(plot_data, 'Hour', 'Traffic', label='Hourly Search Pattern')
    plot.opts(title='Average Search Traffic by Hour',
            width=800, height=400)
            
    return results, plot

def analyze_daily_patterns(df):
    """
    Analyze and visualize daily search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by day of week
    daily_avg = df.groupby(df.index.dayofweek)['Search Trends'].mean()
    
    # Map day numbers to names
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg.index = days
    
    # Find busiest day
    busiest_day = daily_avg.idxmax()
    max_traffic = daily_avg.max()
    
    results = {
        'busiest_day': busiest_day,
        'max_traffic': max_traffic,
        'pattern': f"Search traffic is highest on {busiest_day}s with average value of {max_traffic:.1f}"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Day': daily_avg.index, 'Traffic': daily_avg.values})
    plot = hv.Bars(plot_data, 'Day', 'Traffic', label='Daily Search Pattern')
    plot.opts(title='Average Search Traffic by Day',
            width=800, height=400)
            
    return results, plot

def analyze_weekly_patterns(df):
    """
    Analyze and visualize weekly search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by week of year
    weekly_avg = df.groupby(df.index.isocalendar().week)['Search Trends'].mean()
    
    # Calculate holiday traffic (weeks 40-52)
    holiday_mask = weekly_avg.index.isin(range(40, 53))
    holiday_traffic = weekly_avg[holiday_mask].mean()
    regular_traffic = weekly_avg[~holiday_mask].mean()
    
    # Calculate percentage difference
    pct_diff = ((holiday_traffic - regular_traffic) / regular_traffic) * 100
    
    results = {
        'holiday_traffic': holiday_traffic,
        'regular_traffic': regular_traffic,
        'pct_difference': pct_diff,
        'pattern': f"Holiday period traffic is {abs(pct_diff):.1f}% {'higher' if pct_diff > 0 else 'lower'} than regular weeks"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Week': weekly_avg.index, 'Traffic': weekly_avg.values})
    plot = hv.Curve(plot_data, 'Week', 'Traffic', label='Weekly Pattern')
    plot.opts(title='Average Search Traffic by Week',
            width=800, height=400)
            
    return results, plot

def load_stock_data():
    """
    Load and process the stock price data.
    
    Returns:
        pandas.DataFrame: Processed stock price data
    """
    try:
        print("\nLoading stock price data...")
        print("="*80)
        
        df = pd.read_csv('Resources/mercado_stock_price.csv',
                    index_col='Date',
                    parse_dates=True)
        
        print(f"Loaded {len(df):,} rows of stock price data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the stock price data file")
        return None
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None

def analyze_stock_data(search_df, stock_df):
    """
    Analyze relationship between search trends and stock prices.
    
    Args:
        search_df (pandas.DataFrame): Search trends data
        stock_df (pandas.DataFrame): Stock price data
        
    Returns:
        tuple: Analysis results and plot
    """
    if search_df is None or stock_df is None:
        return None, None
        
    # Merge datasets
    combined = pd.merge(search_df, stock_df,
                    left_index=True,
                    right_index=True,
                    how='inner')
    
    # Calculate lagged search trends and stock metrics
    combined['Lagged_Search'] = combined['Search Trends'].shift(1)
    combined['Stock_Volatility'] = combined['Close'].pct_change().rolling(window=4).std() * np.sqrt(4)
    combined['Stock_Return'] = combined['Close'].pct_change()
    
    # Calculate correlations
    correlations = {
        'search_volatility': combined['Lagged_Search'].corr(combined['Stock_Volatility']),
        'search_return': combined['Lagged_Search'].corr(combined['Stock_Return'])
    }
    
    # Create visualization
    plot1 = hv.Curve(combined['Close'], label='Stock Price')
    plot2 = hv.Curve(combined['Search Trends'], label='Search Trends')
    plot = plot1 * plot2
    plot.opts(title='Stock Price vs Search Trends',
            width=800, height=400)
            
    return correlations, plot

def run_prophet_model(df):
    """
    Create and run Prophet forecasting model.
    
    Args:
        df (pandas.DataFrame): Search trends data
        
    Returns:
        tuple: Forecast results and plots
    """
    if df is None:
        return None, None
        
    # Prepare data for Prophet
    prophet_df = df.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit model
    model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True)
    model.fit(prophet_df)
    
    # Create future dates for prediction
    future = model.make_future_dataframe(periods=24*7, freq='H')
    forecast = model.predict(future)
    
    # Get component plots
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)
    
    # Extract key findings
    results = {
        'peak_hour': forecast.loc[forecast['daily'].idxmax(), 'ds'].hour,
        'peak_day': forecast.loc[forecast['weekly'].idxmax(), 'ds'].day_name(),
        'yearly_low': forecast.loc[forecast['yearly'].idxmin(), 'ds'].strftime('%B %d')
    }
    
    return results, (fig1, fig2)

if __name__ == "__main__":
    # Execute the main analysis
    print("\nExecuting Mercado Analysis...")
    print("=" * 80)
    
    # Load search trends data
    df = load_search_trends()
    
    if df is not None:
        # Perform May 2020 analysis
        results, plot = analyze_may_2020(df)
        
        # Print results
        print("\nMay 2020 Search Traffic Analysis:")
        print("-" * 80)
        print(results['conclusion'])
        hv.save(plot, 'may_2020_traffic.html')
        
        # Perform seasonality analysis
        print("\nAnalyzing Temporal Patterns...")
        print("=" * 80)
        
        # Hourly patterns
        hourly_results, hourly_plot = analyze_hourly_patterns(df)
        if hourly_results:
            print("\nHourly Traffic Pattern:")
            print("-" * 80)
            print(hourly_results['pattern'])
            hv.save(hourly_plot, 'hourly_patterns.html')
        
        # Daily patterns
        daily_results, daily_plot = analyze_daily_patterns(df)
        if daily_results:
            print("\nDaily Traffic Pattern:")
            print("-" * 80)
            print(daily_results['pattern'])
            hv.save(daily_plot, 'daily_patterns.html')
        
        # Weekly patterns
        weekly_results, weekly_plot = analyze_weekly_patterns(df)
        if weekly_results:
            print("\nWeekly Traffic Pattern:")
            print("-" * 80)
            print(weekly_results['pattern'])
            hv.save(weekly_plot, 'weekly_patterns.html')
        
        # Load and analyze stock data
        print("\nAnalyzing Stock Price Patterns...")
        print("=" * 80)
        
        stock_df = load_stock_data()
        if stock_df is not None:
            correlations, plot = analyze_stock_data(df, stock_df)
            if correlations:
                print("\nStock Price Analysis Results:")
                print("-" * 80)
                print(f"Correlation between search trends and stock volatility: {correlations['search_volatility']:.4f}")
                print(f"Correlation between search trends and stock returns: {correlations['search_return']:.4f}")
                
                hv.save(plot, 'stock_trends.html')
        
        # Run Prophet forecast
        print("\nRunning Prophet Time Series Analysis...")
        print("=" * 80)
        
        forecast_results, plots = run_prophet_model(df)
        if forecast_results:
            print("\nForecast Analysis Results:")
            print("-" * 80)
            print(f"Peak Hour of Day: {forecast_results['peak_hour']:02d}:00")
            print(f"Peak Day of Week: {forecast_results['peak_day']}")
            print(f"Yearly Low Point: {forecast_results['yearly_low']}")
            
            # Save Prophet plots
            plots[0].savefig('prophet_forecast.png')
            plots[1].savefig('prophet_components.png')
            
        print("\nAnalysis complete. All visualizations have been saved.")

#!/usr/bin/env python3
"""
MercadoLibre Analysis Script
===========================

This script analyzes Google search trends and stock price data for MercadoLibre.
The analysis includes:
1. Finding unusual patterns in hourly Google search traffic
2. Mining search traffic data for seasonality patterns
3. Analyzing relationship between search trends and stock prices
4. Creating time series forecasts using Prophet

Returns detailed analysis of each component with visualizations and insights.
"""

# Required imports
import pandas as pd
import numpy as np
import holoviews as hv
from prophet import Prophet
from datetime import datetime

# Set up visualization
hv.extension('bokeh')

def load_search_trends():
    """Load and process Google search trends data.
    
    Returns:
        pandas.DataFrame: Search trends data with datetime index
    """
    try:
        print("Loading Google search trends data...")
        print("="*80)
        
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    index_col='Date',
                    parse_dates=True, 
                    date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M'))
        
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find search trends data file")
        return None
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None
    """Load and process Google search trends data.

    Returns:
        pandas.DataFrame: Search trends data with datetime index.
    """
    try:
        print("\nLoading Google search trends data...")
        print("="*80)
        
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    index_col='Date',
                    parse_dates=True,
                    date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M'))
        
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the search trends data file")
        return None
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None

def analyze_may_2020(df):
    """Find unusual patterns in hourly Google search traffic for May 2020.
    
    Analyzes May 2020 search traffic and compares to historical patterns.
    Required by Step 1 of the analysis.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        dict: Analysis results including:
            - Total May 2020 traffic
            - Median monthly traffic
            - Percentage difference
            - Conclusion on May 2020 patterns
    """
    if df is None:
        return None
        
    # Calculate total May 2020 traffic
    may_2020 = df['2020-05']
    may_total = may_2020['Search Trends'].sum()

    # Calculate median monthly traffic for comparison
    monthly_totals = df.resample('M')['Search Trends'].sum()
    median_monthly = monthly_totals.median()

    # Calculate percentage difference 
    pct_diff = ((may_total - median_monthly) / median_monthly) * 100

    # Plot May 2020 traffic
    plot = hv.Curve(may_2020['Search Trends'], 
                'Date', 'Search Traffic',
                label='May 2020 Traffic')

    return {
        'may_total': may_total,
        'median_monthly': median_monthly,
        'pct_difference': pct_diff,
        'plot': plot,
        'conclusion': f"May 2020 search traffic was {pct_diff:.1f}% {'above' if pct_diff > 0 else 'below'} the median"
    }
    """Analyze search traffic for May 2020 compared to overall patterns.

    Args:
        df (pandas.DataFrame): Search trends DataFrame

    Returns:
        dict: Analysis results and metrics
    """
    if df is None:
        return None
        
    # Calculate total May 2020 traffic
    may_2020 = df['2020-05']
    may_total = may_2020['Search Trends'].sum()
    
    # Calculate median monthly traffic
    monthly_totals = df.resample('M')['Search Trends'].sum()
    median_monthly = monthly_totals.median()
    
    # Calculate percentage difference
    pct_diff = ((may_total - median_monthly) / median_monthly) * 100
    
    return {
        'may_total': may_total,
        'median_monthly': median_monthly, 
        'pct_difference': pct_diff,
        'conclusion': f"May 2020 total traffic was {pct_diff:.1f}% {'above' if pct_diff > 0 else 'below'} median"
    }

def analyze_seasonality(df):
    """Mine search traffic data for seasonality patterns.
    
    Analyzes hourly, daily, and weekly patterns in search traffic.
    Required by Step 2 of the analysis.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        dict: Analysis results including:
            - Hourly patterns (peak hours)
            - Daily patterns (busiest days)
            - Weekly patterns (seasonality trends)
            - Visualizations for each pattern
    """
    if df is None:
        return None
        
    # Analyze hourly patterns
    hourly_avg = df.groupby(df.index.hour)['Search Trends'].mean()
    peak_hour = int(hourly_avg.idxmax())
    peak_hourly = float(hourly_avg.max())

    hourly_plot = hv.Curve(hourly_avg, 'Hour', 'Average Traffic',
                        label='Hourly Pattern')
                        
    # Analyze daily patterns
    daily_avg = df.groupby(df.index.dayofweek)['Search Trends'].mean()
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    daily_avg.index = days
    peak_day = daily_avg.idxmax()
    peak_daily = float(daily_avg.max())

    daily_plot = hv.Bars(daily_avg, 'Day', 'Average Traffic',
                    label='Daily Pattern')
                    
    # Analyze weekly patterns
    weekly_avg = df.groupby(df.index.isocalendar().week)['Search Trends'].mean()
    holiday_mask = weekly_avg.index.isin(range(40,53))
    holiday_traffic = weekly_avg[holiday_mask].mean()
    regular_traffic = weekly_avg[~holiday_mask].mean()
    pct_diff = ((holiday_traffic - regular_traffic) / regular_traffic) * 100

    weekly_plot = hv.Curve(weekly_avg, 'Week', 'Average Traffic',
                        label='Weekly Pattern')

    return {
        'hourly': {
            'peak_hour': peak_hour,
            'peak_traffic': peak_hourly,
            'plot': hourly_plot,
            'pattern': f"Search traffic peaks at hour {peak_hour:02d}:00"
        },
        'daily': {
            'peak_day': peak_day,
            'peak_traffic': peak_daily,
            'plot': daily_plot,
            'pattern': f"Search traffic is highest on {peak_day}"
        },
        'weekly': {
            'holiday_traffic': holiday_traffic,
            'regular_traffic': regular_traffic,
            'pct_difference': pct_diff,
            'plot': weekly_plot,
            'pattern': f"Holiday period traffic is {pct_diff:.1f}% {'higher' if pct_diff > 0 else 'lower'} than regular weeks"
        }
    }

def analyze_daily_patterns(df):
    """Analyze daily search traffic patterns.

    Args:
        df (pandas.DataFrame): Search trends DataFrame

    Returns:
        dict: Daily analysis results 
    """
    if df is None:
        return None
        
    # Calculate average traffic by day
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df.groupby(df.index.dayofweek)['Search Trends'].mean()
    daily_avg.index = days
    
    peak_day = daily_avg.idxmax()
    peak_traffic = float(daily_avg.max())
    
    return {
        'peak_day': peak_day,
        'peak_traffic': peak_traffic,
        'pattern': f"Search traffic is highest on {peak_day}s with average value of {peak_traffic:.1f}"
    }

def analyze_weekly_patterns(df):
    """Analyze weekly search traffic patterns.

    Args:
        df (pandas.DataFrame): Search trends DataFrame

    Returns:
        dict: Weekly analysis results
    """
    if df is None:
        return None
        
    # Calculate average traffic by week
    weekly_avg = df.groupby(df.index.isocalendar().week)['Search Trends'].mean()
    
    # Compare holiday period (weeks 40-52) vs regular weeks
    holiday_mask = weekly_avg.index.isin(range(40, 53))
    holiday_traffic = float(weekly_avg[holiday_mask].mean())
    regular_traffic = float(weekly_avg[~holiday_mask].mean())
    
    pct_diff = ((holiday_traffic - regular_traffic) / regular_traffic) * 100
    
    return {
        'holiday_traffic': holiday_traffic,
        'regular_traffic': regular_traffic,
        'pct_difference': pct_diff,
        'pattern': f"Holiday period traffic is {abs(pct_diff):.1f}% {'higher' if pct_diff > 0 else 'lower'} than regular weeks"
    }

def load_stock_data():
    """Load and process stock price data.

    Returns:
        pandas.DataFrame: Stock price data with datetime index
    """
    try:
        print("\nLoading stock price data...")
        print("="*80)
        
        df = pd.read_csv('Resources/mercado_stock_price.csv',
                    index_col='Date',
                    parse_dates=True)
        
        print(f"Loaded {len(df):,} rows of stock price data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the stock price data file")
        return None
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None

def load_stock_data():
    """Load and process stock price data.
    
    Returns:
        pandas.DataFrame: Stock price data with datetime index
    """
    try:
        print("\nLoading stock price data...")
        print("="*80)
        
        df = pd.read_csv('Resources/mercado_stock_price.csv',
                    index_col='Date',
                    parse_dates=True)
                    
        print(f"Loaded {len(df):,} rows of stock price data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find stock price data file")
        return None
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None
        
def analyze_stock_relationship(search_df, stock_df):
    """Analyze relationship between search trends and stock prices.
    
    Required by Step 3 of the analysis.
    
    Args:
        search_df (pandas.DataFrame): Search trends DataFrame
        stock_df (pandas.DataFrame): Stock price DataFrame
        
    Returns:
        dict: Analysis results including:
            - Combined DataFrame with required metrics
            - Correlation analysis
            - Visualizations of relationships
    """
    if search_df is None or stock_df is None:
        return None
        
    # Merge datasets and slice to first half of 2020
    df = pd.merge(search_df, stock_df,
                left_index=True,
                right_index=True,
                how='inner')
                
    h1_2020 = df['2020-01':'2020-06'].copy()

    # Calculate required metrics
    h1_2020['Lagged_Search_Trends'] = h1_2020['Search Trends'].shift(1)
    h1_2020['Stock_Volatility'] = h1_2020['Close'].pct_change().rolling(window=4).std() * np.sqrt(4)
    h1_2020['Hourly_Stock_Return'] = h1_2020['Close'].pct_change()

    # Calculate correlations
    correlations = {
        'search_volatility': h1_2020['Lagged_Search_Trends'].corr(h1_2020['Stock_Volatility']),
        'search_return': h1_2020['Lagged_Search_Trends'].corr(h1_2020['Hourly_Stock_Return'])
    }

    # Create visualization
    plot = hv.Curve(h1_2020['Close'], 'Date', 'Stock Price') * \
        hv.Curve(h1_2020['Search Trends'], 'Date', 'Search Trends')

    return {
        'data': h1_2020,
        'correlations': correlations,
        'plot': plot,
        'conclusion': f"Search trends correlation with stock volatility: {correlations['search_volatility']:.3f}, with returns: {correlations['search_return']:.3f}"
    }
        
    # Merge datasets
    df = pd.merge(search_df, stock_df,
                left_index=True, 
                right_index=True,
                how='inner')
    
    # Calculate required metrics
    df['Lagged_Search'] = df['Search Trends'].shift(1)
    df['Stock_Volatility'] = df['Close'].pct_change().rolling(window=4).std() * np.sqrt(4)
    df['Stock_Return'] = df['Close'].pct_change()
    
    # Calculate correlations
    volatility_corr = df['Lagged_Search'].corr(df['Stock_Volatility'])
    return_corr = df['Lagged_Search'].corr(df['Stock_Return'])
    
    return {
        'volatility_correlation': volatility_corr,
        'return_correlation': return_corr,
        'conclusion': f"Search trends correlation with: Volatility={volatility_corr:.3f}, Returns={return_corr:.3f}"
    }

def run_prophet_forecast(df):
    
    if df is None:
        return None
        
    try:
        # Prepare data for Prophet
        prophet_df = df.reset_index().rename(columns={
            'Date': 'ds',
            'Search Trends': 'y'
        })
        
        # Create and fit model
        model = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True)
        model.fit(prophet_df)
        
        # Create future dates and predict
        future = model.make_future_dataframe(periods=24*7, freq='H')
        forecast = model.predict(future)
        
        # Extract key findings
        results = {
            'peak_hour': forecast.loc[forecast['daily'].idxmax(), 'ds'].hour,
            'peak_day': forecast.loc[forecast['weekly'].idxmax(), 'ds'].day_name(),
            'yearly_low': forecast.loc[forecast['yearly'].idxmin(), 'ds'].strftime('%B %d'),
            'near_term_trend': 'Increasing' if forecast['trend'].diff().tail(24).mean() > 0 else 'Decreasing'
        }
        
        return results
        
    except Exception as e:
        print(f"Error in Prophet modeling: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting MercadoLibre Analysis...")
    
    # Load search trends data
    search_df = load_search_trends()
    if search_df is None:
        print("Error: Could not proceed with analysis due to missing search data")
        exit(1)
        
    # Step 1: Analyze May 2020 patterns
    may_results = analyze_may_2020(search_df)
    if may_results:
        print("\nMay 2020 Analysis Results:")
        print("-" * 40)
        print(may_results['conclusion'])
    
    # Step 2: Analyze seasonality patterns
    hourly_results = analyze_hourly_patterns(search_df)
    if hourly_results:
        print("\nHourly Pattern Analysis:")
        print("-" * 40)
        print(hourly_results['pattern'])
        
    daily_results = analyze_daily_patterns(search_df)
    if daily_results:
        print("\nDaily Pattern Analysis:") 
        print("-" * 40)
        print(daily_results['pattern'])
        
    weekly_results = analyze_weekly_patterns(search_df)
    if weekly_results:
        print("\nWeekly Pattern Analysis:")
        print("-" * 40)
        print(weekly_results['pattern'])
    
    # Step 3: Analyze stock price relationships
    stock_df = load_stock_data()
    if stock_df is not None:
        stock_results = analyze_stock_relationships(search_df, stock_df)
        if stock_results:
            print("\nStock Analysis Results:")
            print("-" * 40)
            print(stock_results['conclusion'])
    
    # Step 4: Run Prophet forecast
    prophet_results = run_prophet_model(search_df)
    if prophet_results:
        print("\nForecast Results:")
        print("-" * 40)
        print(f"Peak hour of day: {prophet_results['peak_hour']:02d}:00")
        print(f"Peak day of week: {prophet_results['peak_day']}")
        print(f"Yearly low point: {prophet_results['yearly_low']}")
        print(f"Near-term trend: {prophet_results['near_term_trend']}")
    """
    Load and process Google search trends data.
    
    Returns:
        pandas.DataFrame: Processed search trends data with datetime index
    """
    try:
        print("\nLoading search trends data...")
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    parse_dates=True,
                    index_col='Date',
                    date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M'))
        
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
        
    except FileNotFoundError:
        print("Error: Search trends data file not found")
        return None
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None

def analyze_may_2020(df):
    """
    Analyze search traffic patterns for May 2020.
    
    Args:
        df: Search trends DataFrame
        
    Returns:
        dict: Analysis results
    """
    if df is None:
        return None
        
    # Filter for May 2020
    may_2020 = df['2020-05']
    
    # Calculate metrics
    may_total = may_2020['Search Trends'].sum()
    monthly_totals = df.resample('M')['Search Trends'].sum()
    median_monthly = monthly_totals.median()
    
    # Calculate percentage difference
    pct_diff = ((may_total - median_monthly) / median_monthly) * 100
    
    return {
        'may_total': may_total,
        'median_monthly': median_monthly,
        'pct_difference': pct_diff,
        'conclusion': f"May 2020 search traffic was {pct_diff:.1f}% {'above' if pct_diff > 0 else 'below'} the median"
    }

def analyze_hourly_patterns(df):
    """
    Analyze hourly search traffic patterns.
    
    Args:
        df: Search trends DataFrame
        
    Returns:
        dict: Analysis results
    """
    if df is None:
        return None
        
    # Calculate hourly averages
    hourly_avg = df.groupby(df.index.hour)['Search Trends'].mean()
    
    # Find peak traffic
    peak_hour = int(hourly_avg.idxmax())
    peak_traffic = float(hourly_avg.max())
    
    return {
        'peak_hour': peak_hour,
        'peak_traffic': peak_traffic,
        'pattern': f"Search traffic peaks at hour {peak_hour:02d}:00 with average traffic of {peak_traffic:.2f}"
    }

def analyze_daily_patterns(df):
    """
    Analyze daily search traffic patterns.
    
    Args:
        df: Search trends DataFrame
        
    Returns:
        dict: Analysis results
    """
    if df is None:
        return None
        
    # Calculate daily averages
    daily_avg = df.groupby(df.index.dayofweek)['Search Trends'].mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg.index = days
    
    # Find peak day
    peak_day = daily_avg.idxmax()
    peak_traffic = float(daily_avg.max())
    
    return {
        'peak_day': peak_day,
        'peak_traffic': peak_traffic,
        'pattern': f"Search traffic is highest on {peak_day}s with average traffic of {peak_traffic:.2f}"
    }

def analyze_weekly_patterns(df):
    """
    Analyze weekly search traffic patterns.
    
    Args:
        df: Search trends DataFrame
        
    Returns:
        dict: Analysis results
    """
    if df is None:
        return None
        
    # Calculate weekly averages
    weekly_avg = df.groupby(df.index.isocalendar().week)['Search Trends'].mean()
    
    # Analyze holiday period (weeks 40-52)
    holiday_mask = weekly_avg.index.isin(range(40, 53))
    holiday_traffic = float(weekly_avg[holiday_mask].mean())
    regular_traffic = float(weekly_avg[~holiday_mask].mean())
    
    # Calculate percentage difference
    pct_diff = ((holiday_traffic - regular_traffic) / regular_traffic) * 100
    
    return {
        'holiday_traffic': holiday_traffic,
        'regular_traffic': regular_traffic,
        'pct_difference': pct_diff,
        'pattern': f"Holiday period traffic is {abs(pct_diff):.1f}% {'higher' if pct_diff > 0 else 'lower'} than regular weeks"
    }

def load_stock_data():
    """
    Load and process stock price data.
    
    Returns:
        pandas.DataFrame: Stock price data with datetime index
    """
    try:
        print("\nLoading stock price data...")
        df = pd.read_csv('Resources/mercado_stock_price.csv',
                    parse_dates=True,
                    index_col='Date')
        
        print(f"Loaded {len(df):,} rows of stock price data")
        return df
        
    except FileNotFoundError:
        print("Error: Stock price data file not found")
        return None
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None

def analyze_stock_trends(search_df, stock_df):
    """
    Analyze relationships between search trends and stock prices.
    
    Args:
        search_df: Search trends DataFrame
        stock_df: Stock price DataFrame
        
    Returns:
        dict: Analysis results
    """
    if search_df is None or stock_df is None:
        return None
        
    # Merge data
    df = pd.merge(search_df, stock_df,
            left_index=True,
            right_index=True,
            how='inner')
    
    # Calculate metrics
    df['Lagged_Search'] = df['Search Trends'].shift(1)
    df['Stock_Volatility'] = df['Close'].pct_change().rolling(window=4).std() * np.sqrt(4)
    df['Stock_Return'] = df['Close'].pct_change()
    
    # Calculate correlations
    volatility_corr = df['Lagged_Search'].corr(df['Stock_Volatility'])
    return_corr = df['Lagged_Search'].corr(df['Stock_Return'])
    
    return {
        'volatility_correlation': volatility_corr,
        'return_correlation': return_corr,
        'conclusion': f"Search trends correlation with: Volatility={volatility_corr:.3f}, Returns={return_corr:.3f}"
    }

def run_prophet_model(df):
    """
    Create and run Prophet forecasting model.
    
    Args:
        df: Search trends DataFrame
        
    Returns:
        dict: Forecast results
    """
    if df is None:
        return None
        
    try:
        # Prepare data
        prophet_df = df.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Create and fit model
        model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True)
        model.fit(prophet_df)
        
        # Create forecast
        future = model.make_future_dataframe(periods=24*7, freq='H')
        forecast = model.predict(future)
        
        # Extract insights
        results = {
            'peak_hour': forecast.loc[forecast['daily'].idxmax(), 'ds'].hour,
            'peak_day': forecast.loc[forecast['weekly'].idxmax(), 'ds'].day_name(),
            'yearly_low': forecast.loc[forecast['yearly'].idxmin(), 'ds'].strftime('%B %d')
        }
        
        return results
        
    except Exception as e:
        print(f"Error in Prophet modeling: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting MercadoLibre Analysis...")
    
    # Load search trends data
    search_df = load_search_trends()
    if search_df is None:
        print("Error: Could not proceed with analysis due to missing search data")
        exit(1)
        
    # Step 1: Analyze May 2020 patterns
    may_results = analyze_may_2020(search_df)
    if may_results:
        print("\nMay 2020 Analysis Results:")
        print("-" * 40)
        print(may_results['conclusion'])
    
    # Step 2: Analyze seasonality patterns
    hourly_results = analyze_hourly_patterns(search_df)
    if hourly_results:
        print("\nHourly Pattern Analysis:")
        print("-" * 40)
        print(hourly_results['pattern'])
        
    daily_results = analyze_daily_patterns(search_df)
    if daily_results:
        print("\nDaily Pattern Analysis:")
        print("-" * 40)
        print(daily_results['pattern'])
        
    weekly_results = analyze_weekly_patterns(search_df)
    if weekly_results:
        print("\nWeekly Pattern Analysis:")
        print("-" * 40)
        print(weekly_results['pattern'])
    
    # Step 3: Analyze stock price relationships
    stock_df = load_stock_data()
    if stock_df is not None:
        stock_results = analyze_stock_trends(search_df, stock_df)
        if stock_results:
            print("\nStock Analysis Results:")
            print("-" * 40)
            print(stock_results['conclusion'])
    
    # Step 4: Run Prophet forecast
    prophet_results = run_prophet_model(search_df)
    if prophet_results:
        print("\nForecast Results:")
        print("-" * 40)
        print(f"Peak hour of day: {prophet_results['peak_hour']:02d}:00")
        print(f"Peak day of week: {prophet_results['peak_day']}")
        print(f"Yearly low point: {prophet_results['yearly_low']}")

#!/usr/bin/env python3
"""
Mercado Analysis Script
======================
This script analyzes Google search trends and stock price data for MercadoLibre.

The analysis includes:
1. Search trend patterns and unusual behavior
2. Seasonality in search traffic
3. Relationship between search trends and stock price
4. Time series forecasting using Prophet

Required packages:
- pandas: data manipulation
- holoviews: interactive visualization
- prophet: time series forecasting
- numpy: numerical operations
"""

import pandas as pd
import holoviews as hv
from prophet import Prophet
import numpy as np
from datetime import datetime

# Set up display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
hv.extension('bokeh')

def load_search_trends():
    """
    Load and process Google search trends data.

    Returns:
        pandas.DataFrame: Processed dataset with datetime index
    """
    try:
        print("\nLoading search trends data...")
        print("="*80)
        
        # Read data with correct date parsing
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    index_col='Date', 
                    parse_dates=True,
                    date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M'))
        
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the search trends data file")
        return None
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None
    try:
        print("\nLoading search trends data...")
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    parse_dates=True,
                    date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M'))
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None
    """
    Load the Google search trends dataset.
    
    Returns:
        pandas.DataFrame: Search trends data with datetime index
    """
    try:
        print("\nLoading search trends data...")
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    index_col='Date',
                    parse_dates=True,
                    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None

def analyze_monthly_traffic(df, target_month='2020-05'):
    """
    Analyze search traffic for a specific month compared to the median.
    
    Args:
        df: DataFrame with search trends
        target_month: Month to analyze in YYYY-MM format
    
    Returns:
        dict: Analysis results
    """
    if df is None:
        return None
        
    # Convert target_month to timestamp for filtering
    start_date = pd.Timestamp(f"{target_month}-01")
    end_date = start_date + pd.offsets.MonthEnd(1)

    # Get data for target month
    monthly_data = df[start_date:end_date]

    # Calculate monthly totals
    monthly_totals = df.resample('M')['Search Trends'].sum()
    median_monthly = monthly_totals.median()
    target_total = monthly_data['Search Trends'].sum()

    # Ensure data exists for target month
    if len(monthly_data) == 0:
        print(f"No data found for {target_month}")
        return None
    
    # Calculate percentage difference
    pct_diff = ((target_total - median_monthly) / median_monthly) * 100
    
    return {
        'target_total': target_total,
        'median_monthly': median_monthly,
        'pct_difference': pct_diff
    }

def analyze_hourly_patterns(df):
    """
    Analyze hourly search traffic patterns.
    
    Args:
        df: DataFrame with search trends
    
    Returns:
        tuple: (results dict, holoviews plot)
    """
    if df is None:
        return None, None
        
    # Calculate hourly averages
    hourly_avg = df.groupby(df.index.hour)['Search Trends'].mean()
    
    # Find peak hour
    peak_hour = hourly_avg.idxmax()
    peak_traffic = hourly_avg.max()
    
    # Create visualization
    plot = hv.Curve(
        pd.DataFrame({'Hour': hourly_avg.index, 'Traffic': hourly_avg.values}),
        'Hour', 'Traffic'
    ).opts(title='Average Search Traffic by Hour')
    
    return {
        'peak_hour': peak_hour,
        'peak_traffic': peak_traffic
    }, plot

def analyze_daily_patterns(df):
    """
    Analyze daily search traffic patterns.
    
    Args:
        df: DataFrame with search trends
    
    Returns:
        tuple: (results dict, holoviews plot)
    """
    if df is None:
        return None, None
        
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df.groupby(df.index.dayofweek)['Search Trends'].mean()
    daily_avg.index = days
    
    peak_day = daily_avg.idxmax()
    peak_traffic = daily_avg.max()
    
    plot = hv.Bars(
        pd.DataFrame({'Day': daily_avg.index, 'Traffic': daily_avg.values}),
        'Day', 'Traffic'
    ).opts(title='Average Search Traffic by Day')
    
    return {
        'peak_day': peak_day,
        'peak_traffic': peak_traffic
    }, plot

def load_stock_data():
    """
    Load stock price data.
    
    Returns:
        pandas.DataFrame: Stock price data with datetime index
    """
    try:
        print("\nLoading stock price data...")
        df = pd.read_csv('Resources/mercado_stock_price.csv',
                    index_col='Date',
                    parse_dates=True)
        print(f"Loaded {len(df):,} rows of stock price data")
        return df
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None

def analyze_stock_trends(search_df, stock_df):
    """
    Analyze relationship between search trends and stock data.
    
    Args:
        search_df: Search trends DataFrame
        stock_df: Stock price DataFrame
    
    Returns:
        tuple: (results dict, holoviews plot)
    """
    if search_df is None or stock_df is None:
        return None, None
        
    # Merge dataframes
    df = pd.merge(search_df, stock_df, 
                left_index=True, 
                right_index=True, 
                how='inner')
    
    # Calculate required metrics
    df['Lagged_Search'] = df['Search Trends'].shift(1)
    df['Stock_Volatility'] = df['Close'].pct_change().rolling(4).std() * np.sqrt(4)
    df['Stock_Return'] = df['Close'].pct_change()
    
    # Calculate correlations
    volatility_corr = df['Lagged_Search'].corr(df['Stock_Volatility'])
    return_corr = df['Lagged_Search'].corr(df['Stock_Return'])
    
    plot = hv.Curve(df['Close'], 'Date', 'Price', label='Stock Price')
    
    return {
        'volatility_correlation': volatility_corr,
        'return_correlation': return_corr
    }, plot

def run_prophet_forecast(df):
    """
    Create and run Prophet forecasting model.
    
    Args:
        df: Search trends DataFrame
    
    Returns:
        dict: Forecast results
    """
    if df is None:
        return None
        
    # Prepare data for Prophet
    prophet_df = df.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit model
    model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True)
    model.fit(prophet_df)
    
    # Create future dates
    future = model.make_future_dataframe(periods=24*7, freq='H')
    forecast = model.predict(future)
    
    return {
        'peak_hour': forecast.loc[forecast['daily'].idxmax(), 'ds'].hour,
        'peak_day': forecast.loc[forecast['weekly'].idxmax(), 'ds'].day_name(),
        'yearly_low': forecast.loc[forecast['yearly'].idxmin(), 'ds'].strftime('%B %d')
    }

if __name__ == "__main__":
    print("Starting Mercado Analysis...\n")
    
    # Load search trends data
    search_df = load_search_trends()
    
    if search_df is not None:
        # Step 1: Analyze May 2020 patterns
        may_results = analyze_monthly_traffic(search_df)
        if may_results:
            print("\nMay 2020 Analysis:")
            print(f"Search traffic vs median: {may_results['pct_difference']:.1f}%")
        
        # Step 2: Analyze temporal patterns
        hourly_results, _ = analyze_hourly_patterns(search_df)
        if hourly_results:
            print("\nHourly Pattern Analysis:")
            print(f"Peak hour: {hourly_results['peak_hour']:02d}:00")
        
        daily_results, _ = analyze_daily_patterns(search_df)
        if daily_results:
            print("\nDaily Pattern Analysis:")
            print(f"Peak day: {daily_results['peak_day']}")
        
        # Step 3: Analyze stock price relationship
        stock_df = load_stock_data()
        if stock_df is not None:
            stock_results, _ = analyze_stock_trends(search_df, stock_df)
            if stock_results:
                print("\nStock Analysis:")
                print(f"Search/Volatility correlation: {stock_results['volatility_correlation']:.3f}")
        
        # Step 4: Prophet forecast
        forecast_results = run_prophet_forecast(search_df)
        if forecast_results:
            print("\nForecast Analysis:")
            print(f"Peak hour of day: {forecast_results['peak_hour']:02d}:00")
            print(f"Peak day of week: {forecast_results['peak_day']}")
            print(f"Yearly low point: {forecast_results['yearly_low']}")

#!/usr/bin/env python3
"""
Mercado Analysis Script
======================
This script performs a comprehensive analysis of Mercado search trends and stock data.

Features:
1. Analysis of unusual patterns in hourly Google search traffic
2. Mining of search traffic data for seasonality patterns
3. Analysis of relationship between search traffic and stock price
4. Time series forecasting using Prophet

Required packages:
- pandas: for data manipulation
- holoviews: for interactive visualization 
- prophet: for time series forecasting
- numpy: for numerical computations

Usage:
$ python analyze_mercado_data.py
"""

import pandas as pd
import holoviews as hv
from prophet import Prophet
import numpy as np
from datetime import datetime
import io

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
hv.extension('bokeh')

def load_search_trends():
    """
    Load and process the Google search trends dataset.
    
    Returns:
        pandas.DataFrame: The processed dataset with datetime index
    """
    try:
        print("\nLoading Google search trends data...")
        print("="*80)
        
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    index_col='Date', 
                    parse_dates=True)
        
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the search trends data file")
        return None
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None

def analyze_may_2020(df):
    """
    Analyze search traffic patterns for May 2020.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Filter for May 2020
    may_2020 = df['2020-05']
    
    # Calculate total search traffic
    may_total = may_2020['Search Trends'].sum()
    
    # Calculate median monthly traffic
    monthly = df.resample('M')['Search Trends'].sum()
    median_monthly = monthly.median()
    
    # Calculate percentage difference
    pct_diff = ((may_total - median_monthly) / median_monthly) * 100
    
    results = {
        'may_total': may_total,
        'median_monthly': median_monthly,
        'pct_difference': pct_diff,
        'conclusion': f"May 2020 total search traffic was {pct_diff:.1f}% {'above' if pct_diff > 0 else 'below'} the median"
    }
    
    # Create visualization
    plot = hv.Curve(df['2020-05'], 'Date', 'Search Trends', label='May 2020 Search Traffic')
    plot.opts(title='Search Traffic During May 2020',
            width=800, height=400)
            
    return results, plot

def analyze_hourly_patterns(df):
    """
    Analyze and visualize hourly search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by hour and calculate mean
    hourly_avg = df.groupby(df.index.hour)['Search Trends'].mean()
    
    # Find peak hour
    peak_hour = hourly_avg.idxmax()
    peak_traffic = hourly_avg.max()
    
    results = {
        'peak_hour': peak_hour,
        'peak_traffic': peak_traffic,
        'pattern': f"Search traffic peaks at hour {peak_hour:02d}:00 with average value of {peak_traffic:.1f}"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Hour': hourly_avg.index, 'Traffic': hourly_avg.values})
    plot = hv.Curve(plot_data, 'Hour', 'Traffic', label='Hourly Search Pattern')
    plot.opts(title='Average Search Traffic by Hour',
            width=800, height=400)
            
    return results, plot

def analyze_daily_patterns(df):
    """
    Analyze and visualize daily search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by day of week
    daily_avg = df.groupby(df.index.dayofweek)['Search Trends'].mean()
    
    # Map day numbers to names
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg.index = days
    
    # Find busiest day
    busiest_day = daily_avg.idxmax()
    max_traffic = daily_avg.max()
    
    results = {
        'busiest_day': busiest_day,
        'max_traffic': max_traffic,
        'pattern': f"Search traffic is highest on {busiest_day}s with average value of {max_traffic:.1f}"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Day': daily_avg.index, 'Traffic': daily_avg.values})
    plot = hv.Bars(plot_data, 'Day', 'Traffic', label='Daily Search Pattern')
    plot.opts(title='Average Search Traffic by Day',
            width=800, height=400)
            
    return results, plot

def analyze_weekly_patterns(df):
    """
    Analyze and visualize weekly search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by week of year
    weekly_avg = df.groupby(df.index.isocalendar().week)['Search Trends'].mean()
    
    # Calculate holiday traffic (weeks 40-52)
    holiday_mask = weekly_avg.index.isin(range(40, 53))
    holiday_traffic = weekly_avg[holiday_mask].mean()
    regular_traffic = weekly_avg[~holiday_mask].mean()
    
    # Calculate percentage difference
    pct_diff = ((holiday_traffic - regular_traffic) / regular_traffic) * 100
    
    results = {
        'holiday_traffic': holiday_traffic,
        'regular_traffic': regular_traffic,
        'pct_difference': pct_diff,
        'pattern': f"Holiday period traffic is {abs(pct_diff):.1f}% {'higher' if pct_diff > 0 else 'lower'} than regular weeks"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Week': weekly_avg.index, 'Traffic': weekly_avg.values})
    plot = hv.Curve(plot_data, 'Week', 'Traffic', label='Weekly Pattern')
    plot.opts(title='Average Search Traffic by Week',
            width=800, height=400)
            
    return results, plot

def load_stock_data():
    """
    Load and process the stock price data.
    
    Returns:
        pandas.DataFrame: Processed stock price data
    """
    try:
        print("\nLoading stock price data...")
        print("="*80)
        
        df = pd.read_csv('Resources/mercado_stock_price.csv',
                    index_col='Date',
                    parse_dates=True)
        
        print(f"Loaded {len(df):,} rows of stock price data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the stock price data file")
        return None
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None

def analyze_stock_data(search_df, stock_df):
    """
    Analyze relationship between search trends and stock prices.
    
    Args:
        search_df (pandas.DataFrame): Search trends data
        stock_df (pandas.DataFrame): Stock price data
        
    Returns:
        tuple: Analysis results and plot
    """
    if search_df is None or stock_df is None:
        return None, None
        
    # Merge datasets
    combined = pd.merge(search_df, stock_df,
                    left_index=True,
                    right_index=True,
                    how='inner')
    
    # Calculate lagged search trends and stock metrics
    combined['Lagged_Search'] = combined['Search Trends'].shift(1)
    combined['Stock_Volatility'] = combined['Close'].pct_change().rolling(window=4).std() * np.sqrt(4)
    combined['Stock_Return'] = combined['Close'].pct_change()
    
    # Calculate correlations
    correlations = {
        'search_volatility': combined['Lagged_Search'].corr(combined['Stock_Volatility']),
        'search_return': combined['Lagged_Search'].corr(combined['Stock_Return'])
    }
    
    # Create visualization
    plot1 = hv.Curve(combined['Close'], label='Stock Price')
    plot2 = hv.Curve(combined['Search Trends'], label='Search Trends')
    plot = plot1 * plot2
    plot.opts(title='Stock Price vs Search Trends',
            width=800, height=400)
            
    return correlations, plot

def run_prophet_model(df):
    """
    Create and run Prophet forecasting model.
    
    Args:
        df (pandas.DataFrame): Search trends data
        
    Returns:
        tuple: Forecast results and plots
    """
    if df is None:
        return None, None
        
    # Prepare data for Prophet
    prophet_df = df.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit model
    model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True)
    model.fit(prophet_df)
    
    # Create future dates for prediction
    future = model.make_future_dataframe(periods=24*7, freq='H')
    forecast = model.predict(future)
    
    # Get component plots
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)
    
    # Extract key findings
    results = {
        'peak_hour': forecast.loc[forecast['daily'].idxmax(), 'ds'].hour,
        'peak_day': forecast.loc[forecast['weekly'].idxmax(), 'ds'].day_name(),
        'yearly_low': forecast.loc[forecast['yearly'].idxmin(), 'ds'].strftime('%B %d')
    }
    
    return results, (fig1, fig2)

def print_analysis_results(results):
    """
    Print formatted analysis results.
    
    Args:
        results (dict): Analysis results to print
    """
    print("\n" + "May 2020 Search Traffic Analysis".center(80))
    print("-" * 80)
    print(f"Total search traffic for May 2020: {results['may_2020_traffic']:,.0f}")
    print(f"Median monthly search traffic: {results['median_monthly_traffic']:,.0f}")
    print(f"Percentage difference: {results['percentage_difference']:,.2f}%")
    print("\nConclusion:")
    if results['percentage_difference'] > 0:
        print("Search traffic increased during May 2020 compared to the median monthly traffic.")
    else:
        print("Search traffic decreased during May 2020 compared to the median monthly traffic.")

def analyze_prophet_results(model, forecast):
    """
    Analyze the results from the Prophet model.
    
    Args:
        model (Prophet): Fitted Prophet model
        forecast (pandas.DataFrame): Forecast results from Prophet
        
    Returns:
        dict: Analysis results including peak times and seasonal patterns
    """
    # Find peak hour
    daily_seasonality = forecast['daily'].max()
    peak_hour = forecast.loc[forecast['daily'] == daily_seasonality, 'ds'].dt.hour.iloc[0]
    
    # Find peak day of week
    weekly_seasonality = forecast['weekly'].max()
    peak_day = forecast.loc[forecast['weekly'] == weekly_seasonality, 'ds'].dt.day_name().iloc[0]
    
    # Find yearly low point
    yearly_min = forecast['yearly'].min()
    low_point_date = forecast.loc[forecast['yearly'] == yearly_min, 'ds'].dt.strftime('%B %d').iloc[0]
    
    results = {
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'yearly_low': low_point_date,
        'near_term_trend': 'Increasing' if forecast['trend'].diff().tail(24).mean() > 0 else 'Decreasing'
    }
    
    return results

if __name__ == "__main__":
    # Execute the main analysis
    print("\nExecuting Mercado Analysis...")
    print("=" * 80)
    
    # Load search trends data
    df = load_search_trends()
    
    if df is not None:
        # Perform May 2020 analysis
        results, plot = analyze_may_2020(df)
        
        # Print results
        print_analysis_results(results)
        
        # Display plot for May 2020 analysis
        hv.save(plot, 'may_2020_traffic.html')
        print("\nVisualization saved as 'may_2020_traffic.html'")
        
        # Load and analyze stock data
        print("\nAnalyzing Stock Price Patterns...")
        print("=" * 80)
        
        stock_df = load_stock_data()
        if stock_df is not None:
            correlations, plot = analyze_stock_data(df, stock_df)
            if correlations is not None:
                print("\nStock Price Analysis Results:")
                print("-" * 80)
                print(f"Correlation between lagged search trends and stock volatility: {correlations['search_volatility']:.4f}")
                print(f"Correlation between lagged search trends and stock returns: {correlations['search_return']:.4f}")
                
                hv.save(plot, 'stock_trends.html')
                print("\nVisualization saved as 'stock_trends.html'")
        
        # Perform seasonality analysis
        print("\nAnalyzing Temporal Patterns...")
        print("=" * 80)
        
        # Hourly patterns
        hourly_results, hourly_plot = analyze_hourly_patterns(df)
        if hourly_results is not None:
            print("\nHourly Traffic Pattern:")
            print("-" * 80)
            print(hourly_results['pattern'])
            hv.save(hourly_plot, 'hourly_patterns.html')
        
        # Daily patterns
        daily_results, daily_plot = analyze_daily_patterns(df)
        if daily_results is not None:
            print("\nDaily Traffic Pattern:")
            print("-" * 80)
            print(daily_results['pattern'])
            hv.save(daily_plot, 'daily_patterns.html')
        
        # Weekly patterns
        weekly_results, weekly_plot = analyze_weekly_patterns(df)
        if weekly_results is not None:
            print("\nWeekly Traffic Pattern:")
            print("-" * 80)
            print(weekly_results['pattern'])
            hv

#!/usr/bin/env python3
"""
Mercado Analysis Script
======================
This script performs a comprehensive analysis of Mercado search trends and stock data.

Features:
1. Analysis of unusual patterns in hourly Google search traffic
2. Mining of search traffic data for seasonality patterns
3. Analysis of relationship between search traffic and stock price
4. Time series forecasting using Prophet

Required packages:
- pandas: for data manipulation
- holoviews: for interactive visualization 
- prophet: for time series forecasting
- numpy: for numerical computations

Usage:
$ python analyze_mercado_data.py
"""

import pandas as pd
import holoviews as hv
from prophet import Prophet
import numpy as np
from datetime import datetime
import io

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
hv.extension('bokeh')

def load_search_trends():
    """
    Load and process the Google search trends dataset.
    
    Returns:
        pandas.DataFrame: The processed dataset with datetime index
    """
    try:
        print("\nLoading Google search trends data...")
        print("="*80)
        
        df = pd.read_csv('Resources/google_hourly_search_trends.csv',
                    index_col='Date', 
                    parse_dates=True)
        
        print(f"Loaded {len(df):,} rows of search trends data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the search trends data file")
        return None
    except Exception as e:
        print(f"Error loading search trends data: {str(e)}")
        return None

def analyze_may_2020(df):
    """
    Analyze search traffic patterns for May 2020.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Filter for May 2020
    may_2020 = df['2020-05']
    
    # Calculate total search traffic
    may_total = may_2020['Search Trends'].sum()
    
    # Calculate median monthly traffic
    monthly = df.resample('M')['Search Trends'].sum()
    median_monthly = monthly.median()
    
    # Calculate percentage difference
    pct_diff = ((may_total - median_monthly) / median_monthly) * 100
    
    results = {
        'may_total': may_total,
        'median_monthly': median_monthly,
        'pct_difference': pct_diff,
        'conclusion': f"May 2020 total search traffic was {pct_diff:.1f}% {'above' if pct_diff > 0 else 'below'} the median"
    }
    
    # Create visualization
    plot = hv.Curve(df['2020-05'], 'Date', 'Search Trends', label='May 2020 Search Traffic')
    plot.opts(title='Search Traffic During May 2020',
            width=800, height=400)
            
    return results, plot

def analyze_hourly_patterns(df):
    """
    Analyze and visualize hourly search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by hour and calculate mean
    hourly_avg = df.groupby(df.index.hour)['Search Trends'].mean()
    
    # Find peak hour
    peak_hour = hourly_avg.idxmax()
    peak_traffic = hourly_avg.max()
    
    results = {
        'peak_hour': peak_hour,
        'peak_traffic': peak_traffic,
        'pattern': f"Search traffic peaks at hour {peak_hour:02d}:00 with average value of {peak_traffic:.1f}"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Hour': hourly_avg.index, 'Traffic': hourly_avg.values})
    plot = hv.Curve(plot_data, 'Hour', 'Traffic', label='Hourly Search Pattern')
    plot.opts(title='Average Search Traffic by Hour',
            width=800, height=400)
            
    return results, plot

def analyze_daily_patterns(df):
    """
    Analyze and visualize daily search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by day of week
    daily_avg = df.groupby(df.index.dayofweek)['Search Trends'].mean()
    
    # Map day numbers to names
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg.index = days
    
    # Find busiest day
    busiest_day = daily_avg.idxmax()
    max_traffic = daily_avg.max()
    
    results = {
        'busiest_day': busiest_day,
        'max_traffic': max_traffic,
        'pattern': f"Search traffic is highest on {busiest_day}s with average value of {max_traffic:.1f}"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Day': daily_avg.index, 'Traffic': daily_avg.values})
    plot = hv.Bars(plot_data, 'Day', 'Traffic', label='Daily Search Pattern')
    plot.opts(title='Average Search Traffic by Day',
            width=800, height=400)
            
    return results, plot

def analyze_weekly_patterns(df):
    """
    Analyze and visualize weekly search traffic patterns.
    
    Args:
        df (pandas.DataFrame): Search trends DataFrame
        
    Returns:
        tuple: Analysis results and plot
    """
    if df is None:
        return None, None
        
    # Group by week of year
    weekly_avg = df.groupby(df.index.isocalendar().week)['Search Trends'].mean()
    
    # Calculate holiday traffic (weeks 40-52)
    holiday_mask = weekly_avg.index.isin(range(40, 53))
    holiday_traffic = weekly_avg[holiday_mask].mean()
    regular_traffic = weekly_avg[~holiday_mask].mean()
    
    # Calculate percentage difference
    pct_diff = ((holiday_traffic - regular_traffic) / regular_traffic) * 100
    
    results = {
        'holiday_traffic': holiday_traffic,
        'regular_traffic': regular_traffic,
        'pct_difference': pct_diff,
        'pattern': f"Holiday period traffic is {abs(pct_diff):.1f}% {'higher' if pct_diff > 0 else 'lower'} than regular weeks"
    }
    
    # Create visualization
    plot_data = pd.DataFrame({'Week': weekly_avg.index, 'Traffic': weekly_avg.values})
    plot = hv.Curve(plot_data, 'Week', 'Traffic', label='Weekly Pattern')
    plot.opts(title='Average Search Traffic by Week',
            width=800, height=400)
            
    return results, plot

def load_stock_data():
    """
    Load and process the stock price data.
    
    Returns:
        pandas.DataFrame: Processed stock price data
    """
    try:
        print("\nLoading stock price data...")
        print("="*80)
        
        df = pd.read_csv('Resources/mercado_stock_price.csv',
                    index_col='Date',
                    parse_dates=True)
        
        print(f"Loaded {len(df):,} rows of stock price data")
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the stock price data file")
        return None
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None

def analyze_stock_data(search_df, stock_df):
    """
    Analyze relationship between search trends and stock prices.
    
    Args:
        search_df (pandas.DataFrame): Search trends data
        stock_df (pandas.DataFrame): Stock price data
        
    Returns:
        tuple: Analysis results and plots
    """
    if search_df is None or stock_df is None:
        return None, None
        
    # Merge datasets
    combined = pd.merge(search_df, stock_df,
                    left_index=True,
                    right_index=True,
                    how='inner')
    
    # Calculate lagged search trends and stock metrics
    combined['Lagged_Search'] = combined['Search Trends'].shift(1)
    combined['Stock_Volatility'] = combined['Close'].pct_change().rolling(window=4).std() * np.sqrt(4)
    combined['Stock_Return'] = combined['Close'].pct_change()
    
    # Calculate correlations
    correlations = {
        'search_volatility': combined['Lagged_Search'].corr(combined['Stock_Volatility']),
        'search_return': combined['Lagged_Search'].corr(combined['Stock_Return'])
    }
    
    # Create visualization
    plot1 = hv.Curve(combined['Close'], label='Stock Price')
    plot2 = hv.Curve(combined['Search Trends'], label='Search Trends')
    plot = plot1 * plot2
    plot.opts(title='Stock Price vs Search Trends',
            width=800, height=400)
            
    return correlations, plot

def run_prophet_model(df):
    """
    Create and run Prophet forecasting model.
    
    Args:
        df (pandas.DataFrame): Search trends data
        
    Returns:
        tuple: Forecast results and plots
    """
    if df is None:
        return None, None
        
    # Prepare data for Prophet
    prophet_df = df.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit model
    model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True)
    model.fit(prophet_df)
    
    # Create future dates for prediction
    future = model.make_future_dataframe(periods=24*7, freq='H')
    forecast = model.predict(future)
    
    # Get component plots
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)
    
    # Extract key findings
    results = {
        'peak_hour': forecast.loc[forecast['daily'].idxmax(), 'ds'].hour,
        'peak_day': forecast.loc[forecast['weekly'].idxmax(), 'ds'].day_name(),
        'yearly_low': forecast.loc[forecast['yearly'].idxmin(), 'ds'].strftime('%B %d')
    }
    
    return results, (fig1, fig2)
def analyze_hourly_patterns(df):
    """
    Analyze and visualize hourly search traffic patterns.
    
    Args:
        df (pandas.DataFrame): The search trends DataFrame
        
    Returns:
        tuple: Analysis results and visualization plot
    """
    # Group by hour and calculate mean search traffic
    hourly_avg = df.groupby(df['Date'].dt.hour)['Search Trends'].mean().reset_index()
    hourly_avg.columns = ['Hour', 'Average Search Traffic']

    # Create visualization
    plot = hv.Curve(hourly_avg, kdims='Hour', vdims='Average Search Traffic',
                label='Hourly Search Trends')
    plot.opts(title='Average Search Traffic by Hour of Day',
            width=800, height=400)
    
    # Find peak hours
    peak_hour = hourly_avg.loc[hourly_avg['Average Search Traffic'].idxmax(), 'Hour']
    peak_traffic = hourly_avg['Average Search Traffic'].max()

    results = {
        'peak_hour': peak_hour,
        'peak_traffic': peak_traffic,
        'hourly_pattern': f"Search traffic peaks during hour {int(peak_hour)} with average value of {peak_traffic:.2f}"
    }
    
    return results, plot

def analyze_daily_patterns(df):
    """
    Analyze and visualize daily search traffic patterns.
    
    Args:
        df (pandas.DataFrame): The search trends DataFrame
        
    Returns:
        tuple: Analysis results and visualization plot
    """
    # Group by day of week and calculate mean search traffic
    daily_avg = df.groupby(df['Date'].dt.dayofweek)['Search Trends'].mean().reset_index()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg.columns = ['DayNum', 'Average Search Traffic']
    daily_avg['Day'] = daily_avg['DayNum'].apply(lambda x: days[x])

    # Create visualization
    plot = hv.Bars(daily_avg, kdims='Day', vdims='Average Search Traffic',
                label='Daily Search Trends')
    plot.opts(title='Average Search Traffic by Day of Week',
            width=800, height=400)
    
    # Find busiest day
    # Find busiest day
    busiest_day_idx = daily_avg['Average Search Traffic'].idxmax()
    busiest_day = daily_avg.loc[busiest_day_idx, 'Day']
    max_traffic = daily_avg['Average Search Traffic'].max()

    results = {
        'busiest_day': busiest_day,
        'max_daily_traffic': max_traffic,
        'daily_pattern': f"Search traffic is highest on {busiest_day} with average value of {max_traffic:.2f}"
    }
    
    return results, plot

def analyze_weekly_patterns(df):
    """
    Analyze and visualize weekly search traffic patterns throughout the year.
    
    Args:
        df (pandas.DataFrame): The search trends DataFrame
        
    Returns:
        tuple: Analysis results and visualization plot
    """
    # Group by week of year and calculate mean search traffic
    weekly_avg = df.groupby(df['Date'].dt.isocalendar().week)['Search Trends'].mean().reset_index()
    weekly_avg.columns = ['Week', 'Average Search Traffic']
    
    # Create visualization
    plot = hv.Curve(weekly_avg, kdims='Week', vdims='Average Search Traffic',
                label='Weekly Search Trends')
    plot.opts(title='Average Search Traffic by Week of Year',
            width=800, height=400)
    
    # Analyze holiday period (weeks 40-52)
    holiday_traffic = float(weekly_avg.loc[weekly_avg['Week'].between(40, 52), 'Average Search Traffic'].mean())
    regular_traffic = float(weekly_avg.loc[weekly_avg['Week'] < 40, 'Average Search Traffic'].mean())
    
    # Calculate percentage difference
    pct_difference = ((holiday_traffic - regular_traffic) / regular_traffic) * 100
    
    # Determine trend direction
    trend_direction = "increase in" if pct_difference > 0 else "decrease in"
    
    results = {
        'holiday_traffic_avg': holiday_traffic,
        'regular_traffic_avg': regular_traffic,
        'holiday_pct_difference': pct_difference,
        'weekly_pattern': f"Holiday period (weeks 40-52) shows {abs(pct_difference):.2f}% {trend_direction} average traffic compared to rest of year (holiday: {holiday_traffic:.2f} vs regular: {regular_traffic:.2f})"
    }
    
    return results, plot

    def run_prophet_model(df, periods=24*7, freq='H'):
        """
        Create and run a Prophet time series model on the search trends data.
        
        Args:
            df (pandas.DataFrame): Search trends DataFrame
            periods (int): Number of periods to forecast (default: 1 week in hours)
            freq (str): Frequency of forecast (default: 'H' for hourly)
            
        Returns:
            tuple: Prophet model, forecast DataFrame, and component plots
        """
        # Prepare data for Prophet
        prophet_df = df.reset_index().rename(columns={
            'Date': 'ds',
            'Search Trends': 'y'
        })
        
        # Create and fit the model
        model = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True)
        model.fit(prophet_df)
        
        # Create future dates for prediction
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast = model.predict(future)
        
        # Create component plots
        fig_comp = model.plot_components(forecast)
        
        # Create forecast plot
        fig_forecast = model.plot(forecast)
        
        return model, forecast, (fig_forecast, fig_comp)

    def load_stock_data():
        """
        Load and process the stock price data.
        
        Returns:
            pandas.DataFrame: Processed stock price data
        """
        try:
            # Read the stock price data
            stock_df = pd.read_csv('Resources/mercado_stock_price.csv', 
                        index_col='Date',
                        parse_dates=True,
                        infer_datetime_format=True)
            
            # Sort index
            stock_df = stock_df.sort_index()
            
            return stock_df
            
        except FileNotFoundError:
            print("Error: Could not find the stock price data file")
            return None
        except Exception as e:
            print(f"Error loading stock data: {str(e)}")
            return None

    def merge_and_analyze_data(search_df, stock_df):
        """  
        Merge stock and search data and perform analysis.
        
        Args:
            search_df (pandas.DataFrame): Search trends data
            stock_df (pandas.DataFrame): Stock price data
            
        Returns:
            tuple: Combined DataFrame and analysis results
        """

    def merge_and_analyze_data(search_df, stock_df):
        """  
        Merge stock and search data and perform analysis.
        
        Args:
            search_df (pandas.DataFrame): Search trends data
            stock_df (pandas.DataFrame): Stock price data
            
        Returns:
            tuple: Combined DataFrame and analysis results
        """
        try:
            # Ensure datetime index for both dataframes
            if not isinstance(search_df.index, pd.DatetimeIndex):
                search_df['Date'] = pd.to_datetime(search_df['Date'])
                search_df.set_index('Date', inplace=True)
            
            # Create lagged search trends
            search_df['Lagged_Search_Trends'] = search_df['Search Trends'].shift(1)
            
            # Merge the dataframes
            combined_df = pd.merge(search_df, stock_df, 
                            how='inner',
                            left_index=True, 
                            right_index=True)
            
            # Calculate stock volatility (4-hour rolling average)  
            combined_df['Stock_Volatility'] = combined_df['Close'].pct_change().rolling(\
                window=4).std() * np.sqrt(4)
            
            # Calculate hourly stock return
            combined_df['Hourly_Stock_Return'] = combined_df['Close'].pct_change()
            
            # Analysis for first half of 2020
            h1_2020 = combined_df.loc['2020-01-01':'2020-06-30']
            
            # Calculate correlations
            correlations = {
                'search_volatility_corr': combined_df['Lagged_Search_Trends'].corr(\
                    combined_df['Stock_Volatility']),
                'search_return_corr': combined_df['Lagged_Search_Trends'].corr(\
                    combined_df['Hourly_Stock_Return'])
            }
            
            return combined_df, h1_2020, correlations
            
        except Exception as e:
            print(f"Error in data merge and analysis: {str(e)}")
            return None, None, None

    def visualize_stock_patterns(combined_df, h1_2020):
        """
        Create visualizations for stock price patterns.
        
        Args:
            combined_df (pandas.DataFrame): Combined stock and search data
            h1_2020 (pandas.DataFrame): First half 2020 data
            
        Returns:
            tuple: Visualization plots
        """
        # Plot full timeline
        full_plot = hv.Curve(combined_df['Close'], label='Stock Price') * \
                hv.Curve(combined_df['Search Trends'], label='Search Trends')
        full_plot.opts(width=800, height=400, title='Stock Price vs Search Trends')
        
        # Plot first half 2020
        h1_plot = hv.Curve(h1_2020['Close'], label='Stock Price') * \
                hv.Curve(h1_2020['Search Trends'], label='Search Trends')
        h1_plot.opts(width=800, height=400, title='First Half 2020: Stock Price vs Search Trends')
        
        return full_plot, h1_plot

    def analyze_may_2020_traffic(df):
        """
        Analyze search traffic data for May 2020 and compare with overall trends.

        Args:
            df (pandas.DataFrame): The search trends DataFrame
            
        Returns:
            dict: Analysis results including total traffic and comparison stats
        """
    # Convert date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
        
    # Extract May 2020 data
    may_2020 = df[df['Date'].dt.strftime('%Y-%m') == '2020-05']
    
    # Calculate total search traffic for May 2020
    may_2020_traffic = may_2020['Search Trends'].sum()
    
    # Calculate median monthly traffic for comparison
    monthly_traffic = df.groupby(df['Date'].dt.strftime('%Y-%m'))['Search Trends'].sum()
    median_monthly_traffic = monthly_traffic.median()
    
    # Calculate the percentage difference
    percentage_diff = ((may_2020_traffic - median_monthly_traffic) / median_monthly_traffic) * 100
    
    results = {
        'may_2020_traffic': may_2020_traffic,
        'median_monthly_traffic': median_monthly_traffic,
        'percentage_difference': percentage_diff
    }
    
    # Create visualization
    plot = hv.Curve(may_2020.set_index('Date')['Search Trends'], 
                'Date', 'Search Trends', 
                label='May 2020 Search Traffic')
    
    return results, plot

def print_analysis_results(results):
    """
    Print formatted analysis results.
    
    Args:
        results (dict): Analysis results to print
    """
    print("\n" + "May 2020 Search Traffic Analysis".center(80))
    print("-" * 80)
    print(f"Total search traffic for May 2020: {results['may_2020_traffic']:,.0f}")
    print(f"Median monthly search traffic: {results['median_monthly_traffic']:,.0f}")
    print(f"Percentage difference: {results['percentage_difference']:,.2f}%")
    print("\nConclusion:")
    if results['percentage_difference'] > 0:
        print("Search traffic increased during May 2020 compared to the median monthly traffic.")
    else:
        print("Search traffic decreased during May 2020 compared to the median monthly traffic.")

def analyze_prophet_results(model, forecast):
    """
    Analyze the results from the Prophet model.
    
    Args:
        model (Prophet): Fitted Prophet model
        forecast (pandas.DataFrame): Forecast results from Prophet
        
    Returns:
        dict: Analysis results including peak times and seasonal patterns
    """
    # Find peak hour
    daily_seasonality = forecast['daily'].max()
    peak_hour = forecast.loc[forecast['daily'] == daily_seasonality, 'ds'].dt.hour.iloc[0]
    
    # Find peak day of week
    weekly_seasonality = forecast['weekly'].max()
    peak_day = forecast.loc[forecast['weekly'] == weekly_seasonality, 'ds'].dt.day_name().iloc[0]
    
    # Find yearly low point
    yearly_min = forecast['yearly'].min()
    low_point_date = forecast.loc[forecast['yearly'] == yearly_min, 'ds'].dt.strftime('%B %d').iloc[0]
    
    results = {
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'yearly_low': low_point_date,
        'near_term_trend': 'Increasing' if forecast['trend'].diff().tail(24).mean() > 0 else 'Decreasing'
    }
    
    return results

if __name__ == "__main__":
    # Execute the main analysis
    print("\nExecuting Mercado Analysis...")
    print("=" * 80)
    
    # Load search trends data
    df = load_and_examine_data()
    
    if df is not None:
        # Perform May 2020 analysis
        results, plot = analyze_may_2020_traffic(df)
        
        # Print results
        print_analysis_results(results)

        # Display plot for May 2020 analysis
        hv.save(plot, 'may_2020_traffic.html')
        print("\nVisualization saved as 'may_2020_traffic.html'")

        # Load and analyze stock data
        print("\nAnalyzing Stock Price Patterns...")
        print("=" * 80)

        stock_df = load_stock_data()
        if stock_df is not None:
            combined_df, h1_2020, correlations = merge_and_analyze_data(df, stock_df)
            if combined_df is not None:
                # Create and save stock visualizations
                full_plot, h1_plot = visualize_stock_patterns(combined_df, h1_2020)
                hv.save(full_plot, 'stock_trends.html')
                hv.save(h1_plot, 'stock_trends_h1_2020.html')
                
                print("\nStock Price Analysis Results:")
                print("-" * 80)
                print(f"Correlation between lagged search trends and stock volatility: {correlations['search_volatility_corr']:.4f}")
                print(f"Correlation between lagged search trends and stock returns: {correlations['search_return_corr']:.4f}")
                
                print("\nVisualizations saved as:")
                print("- stock_trends.html")
                print("- stock_trends_h1_2020.html")
                
                # Perform Prophet Analysis
                print("\nPerforming Prophet Time Series Analysis...")
                print("=" * 80)
                
                model, forecast, plots = run_prophet_model(df)
                if model is not None:
                    print("\nProphet Model Results:")
                    print("-" * 80)
                    forecast_results = analyze_prophet_results(model, forecast)
                    print(f"Peak Hour of Day: {forecast_results['peak_hour']}")
                    print(f"Peak Day of Week: {forecast_results['peak_day']}")
                    print(f"Yearly Low Point: {forecast_results['yearly_low']}")
                    print(f"Near-term Trend: {forecast_results['near_term_trend']}")
        # Perform seasonality analysis
        print("\nAnalyzing Temporal Patterns...")
        print("=" * 80)

        # Hourly patterns
        hourly_results, hourly_plot = analyze_hourly_patterns(df)
        print("\nHourly Traffic Pattern:")
        print("-" * 80)
        print(hourly_results['hourly_pattern'])
        hv.save(hourly_plot, 'hourly_patterns.html')

        # Daily patterns
        daily_results, daily_plot = analyze_daily_patterns(df)
        print("\nDaily Traffic Pattern:")
        print("-" * 80)
        print(daily_results['daily_pattern'])
        hv.save(daily_plot, 'daily_patterns.html')

        # Weekly patterns
        weekly_results, weekly_plot = analyze_weekly_patterns(df)
        print("\nWeekly Traffic Pattern:")
        print("-" * 80)
        print(weekly_results['weekly_pattern'])
        hv.save(weekly_plot, 'weekly_patterns.html')

        print("\nVisualizations saved as:")
        print("- hourly_patterns.html")
        print("- daily_patterns.html")
        print("- weekly_patterns.html")
