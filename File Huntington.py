import streamlit as st
import subprocess
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

st.set_page_config(layout="wide", page_title="Savings Rate Prediction Dashboard")

# Page title and description
st.title("Personal Savings Rate Prediction Dashboard")
st.markdown("""
This dashboard allows you to upload your financial data and predict future personal savings rates
using three different models and their average.
""")

# File upload section
st.subheader("Upload Data")
st.markdown("""
Please upload an Excel file with the following columns:
1. Time Period
2. Personal Savings Rate (%)
3. Unemployment Rate (%)
4. Monthly Disposable Income Total US in Billions ($)
5. Total Consumer Credit Owned and Securitized
6. % Change in S&P 500 Price
7. VIX Close
""")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# Function to safely check stationarity using ADF test
def check_stationarity(series):
    if len(series) < 10:  # Need enough data for the test
        return 1.0  # Return value > 0.05 to indicate non-stationarity
    
    # Make sure the series doesn't have any NaN values
    series = series.dropna()
    
    if len(series) < 10:
        return 1.0
    
    try:
        result = adfuller(series)
        return result[1]  # p-value (if p < 0.05, the series is stationary)
    except:
        return 1.0  # Return a value that indicates non-stationarity if the test fails

# Function to predict with the MLR lagged PSR model
def predict_mlr_lagged_psr(data, forecast_periods=1):
    try:
        # Extract PSR and ensure it's numeric - use column index 1 (second column)
        psr = pd.to_numeric(data.iloc[:, 1], errors='coerce')
        
        # Check if we have enough data
        if len(psr.dropna()) < 13:
            return [float(psr.dropna().mean())] * forecast_periods, None
        
        # Create lagged variables: 1-period lag, 11-period lag, and 12-period lag
        lag_1_psr = psr.shift(1)
        lag_11_psr = psr.shift(11)
        lag_12_psr = psr.shift(12)
        
        # Create DataFrame with lagged variables
        df = pd.DataFrame({
            'Personal_Savings_Rate': psr,
            'Lag_1': lag_1_psr,
            'Lag_11': lag_11_psr,
            'Lag_12': lag_12_psr
        })
        
        # Drop NaN values
        df_clean = df.dropna()
        
        # Check if we have enough data after cleaning
        if len(df_clean) < 5:
            return [float(psr.dropna().mean())] * forecast_periods, None
        
        # Define independent variables
        independent_vars = ['Lag_1', 'Lag_11', 'Lag_12']
        
        # Handle potential zero variance in independent variables
        for var in independent_vars:
            if df_clean[var].std() == 0:
                # Add tiny random noise to avoid zero variance
                df_clean[var] = df_clean[var] + np.random.normal(0, 0.00001, len(df_clean))
        
        # Normalize independent variables
        scaler = StandardScaler()
        df_clean[independent_vars] = scaler.fit_transform(df_clean[independent_vars])
        
        # Define X and y
        X = df_clean[independent_vars]
        X = sm.add_constant(X)  # Add constant
        y = df_clean['Personal_Savings_Rate']
        
        # Fit model
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Prepare for forecasting
        forecasts = []
        
        # Get the last values to start forecasting - ensure it's not empty
        last_psr_values = psr.iloc[-12:].values
        if len(last_psr_values) < 12:
            # Pad with mean if needed
            padding = np.full(12 - len(last_psr_values), psr.mean())
            last_psr_values = np.concatenate([padding, last_psr_values])
        
        # For multiple period forecasting
        for i in range(forecast_periods):
            # Get the lagged values
            lag_1 = last_psr_values[-1]
            lag_11 = last_psr_values[-11] if i == 0 else (last_psr_values[-11+i] if i < 11 else forecasts[i-11])
            lag_12 = last_psr_values[-12] if i == 0 else (last_psr_values[-12+i] if i < 12 else forecasts[i-12])
            
            # Create input data and normalize
            input_data = np.array([[lag_1, lag_11, lag_12]])
            input_data_scaled = scaler.transform(input_data)
            
            # Add constant - manually construct the design matrix with constant
            input_data_with_const = np.column_stack((np.ones(input_data_scaled.shape[0]), input_data_scaled))
            
            # Make prediction
            forecast = results.predict(input_data_with_const)[0]
            forecasts.append(forecast)
            
            # Update the last values for next prediction
            last_psr_values = np.append(last_psr_values[1:], forecast)
        
        return forecasts, results
    except Exception as e:
        st.error(f"Error in MLR Lagged PSR prediction: {str(e)}")
        import traceback
        st.write("MLR Lagged PSR Traceback:", traceback.format_exc())
        # Return mean as fallback
        try:
            mean_psr = float(pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna().mean())
            return [mean_psr] * forecast_periods, None
        except:
            return [5.0] * forecast_periods, None  # Default to 5% if all else fails

# Function to predict with MLR economic indicators model
def predict_mlr_economic(data, forecast_periods=1):
    try:
        # Extract variables and ensure they are numeric
        psr = pd.to_numeric(data.iloc[:, 1], errors='coerce')
        
        # Check if we have enough data
        if len(psr.dropna()) < 5:
            return [float(psr.dropna().mean())] * forecast_periods, None
        
        # Try to extract other columns, with fallbacks
        try:
            unemployment = pd.to_numeric(data.iloc[:, 2], errors='coerce')
        except:
            unemployment = pd.Series([5.0] * len(psr))  # Default value
            
        try:
            disposable_income = pd.to_numeric(data.iloc[:, 3], errors='coerce')
        except:
            disposable_income = pd.Series([1000.0] * len(psr))  # Default value
            
        try:
            consumer_credit = pd.to_numeric(data.iloc[:, 4], errors='coerce')
        except:
            consumer_credit = pd.Series([100.0] * len(psr))  # Default value
            
        try:
            sp500_change = pd.to_numeric(data.iloc[:, 5], errors='coerce')
        except:
            sp500_change = pd.Series([0.0] * len(psr))  # Default value
            
        try:
            vix_close = pd.to_numeric(data.iloc[:, 6], errors='coerce')
        except:
            vix_close = pd.Series([20.0] * len(psr))  # Default value
        
        # Create lagged variables
        lag_unemployment = unemployment.shift(1)
        lag_disposable_income = disposable_income.shift(1)
        lag_consumer_credit = consumer_credit.shift(1)
        lag_sp500_change = sp500_change.shift(1)
        lag_vix_close = vix_close.shift(1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Personal_Savings_Rate': psr,
            'Lag_Unemployment_Rate': lag_unemployment,
            'Lag_Disposable_Income': lag_disposable_income,
            'Lag_Total_Consumer_Credit': lag_consumer_credit,
            'Lag_Change_in_SP500': lag_sp500_change,
            'Lag_VIX_Close': lag_vix_close
        })
        
        # Drop NaN values
        df = df.dropna()
        
        # Check if we have enough data after cleaning
        if len(df) < 5:
            return [float(psr.dropna().mean())] * forecast_periods, None
        
        # Create interaction term
        df['Interaction_SP500_VIX'] = df['Lag_Change_in_SP500'] * df['Lag_VIX_Close']
        
        # Define independent variables
        independent_vars = ['Lag_Unemployment_Rate', 'Lag_Disposable_Income',
                            'Lag_Total_Consumer_Credit', 'Interaction_SP500_VIX']
        
        # Handle potential zero variance in independent variables
        for var in independent_vars:
            if df[var].std() == 0:
                # Add tiny random noise to avoid zero variance
                df[var] = df[var] + np.random.normal(0, 0.00001, len(df))
        
        # Normalize independent variables
        scaler = StandardScaler()
        df[independent_vars] = scaler.fit_transform(df[independent_vars])
        
        # Define X and y
        X = df[independent_vars]
        X = sm.add_constant(X)  # Add constant
        y = df['Personal_Savings_Rate']
        
        # Fit OLS model (changed from GLS for simplicity)
        model = sm.OLS(y, X)
        results = model.fit()
        
        # For forecasting, we need the latest values
        latest_values = {
            'Lag_Unemployment_Rate': unemployment.iloc[-1],
            'Lag_Disposable_Income': disposable_income.iloc[-1],
            'Lag_Total_Consumer_Credit': consumer_credit.iloc[-1],
            'Lag_Change_in_SP500': sp500_change.iloc[-1],
            'Lag_VIX_Close': vix_close.iloc[-1]
        }
        
        # Simple forecasting assumption: use the latest values for all periods
        forecasts = []
        
        for i in range(forecast_periods):
            # Calculate interaction term
            interaction = latest_values['Lag_Change_in_SP500'] * latest_values['Lag_VIX_Close']
            
            # Create input data
            input_data = np.array([[
                latest_values['Lag_Unemployment_Rate'],
                latest_values['Lag_Disposable_Income'],
                latest_values['Lag_Total_Consumer_Credit'],
                interaction
            ]])
            
            # Normalize input data
            input_data_scaled = scaler.transform(input_data)
            
            # Add constant - manually construct the design matrix with constant
            input_data_with_const = np.column_stack((np.ones(input_data_scaled.shape[0]), input_data_scaled))
            
            # Make prediction
            forecast = results.predict(input_data_with_const)[0]
            forecasts.append(forecast)
        
        return forecasts, results
    except Exception as e:
        st.error(f"Error in MLR Economic prediction: {str(e)}")
        import traceback
        st.write("MLR Economic Traceback:", traceback.format_exc())
        # Return mean as fallback
        try:
            mean_psr = float(pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna().mean())
            return [mean_psr] * forecast_periods, None
        except:
            return [5.0] * forecast_periods, None  # Default to 5% if all else fails

# Function to predict with AR optimal lag model
def predict_ar_optimal(data, forecast_periods=1):
    try:
        # Extract and clean the Personal Savings Rate (PSR) series
        psr = pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna()
        
        # Ensure we have enough data points
        if len(psr) < 5:
            return [float(psr.mean())] * forecast_periods, None
        
        # Convert to DataFrame with reset_index to ensure proper indexing
        df = pd.DataFrame({'psr': psr.values})
        
        # Differencing if non-stationary
        is_differenced = False
        stationarity_p_value = check_stationarity(df['psr'])
        
        if stationarity_p_value > 0.05 and len(df) >= 10:  # Ensure enough data for differencing
            df['psr_diff'] = df['psr'].diff().dropna()
            is_differenced = True
            # Make sure we have enough data after differencing
            if len(df['psr_diff'].dropna()) < 5:
                return [float(psr.mean())] * forecast_periods, None
            ar_series = df['psr_diff'].dropna().values
        else:
            ar_series = df['psr'].values
        
        # Try different lag orders and select the best one
        max_lag = min(12, len(ar_series) // 4)  # Ensure we don't exceed reasonable lag length
        max_lag = max(1, max_lag)  # Ensure at least lag 1
        
        best_aic = np.inf
        best_model_aic = None
        
        for k in range(1, max_lag + 1):
            try:
                model = AutoReg(ar_series, lags=k, trend="c").fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model_aic = model
            except:
                continue
        
        # If no model could be fit, use a simple average
        if best_model_aic is None:
            return [float(psr.mean())] * forecast_periods, None
        
        # Make predictions
        try:
            forecasts = best_model_aic.forecast(steps=forecast_periods)
        except:
            # If forecasting fails, use the last value or mean
            return [float(psr.iloc[-1])] * forecast_periods, best_model_aic
        
        # If we differenced the data, we need to revert the differencing
        if is_differenced:
            last_value = psr.iloc[-1]
            reverted_forecasts = [last_value]
            for diff_value in forecasts:
                new_value = reverted_forecasts[-1] + diff_value
                reverted_forecasts.append(new_value)
            forecasts = reverted_forecasts[1:]  # Skip the first element, which is the last observed value
        
        # Ensure we have the right number of forecasts
        if len(forecasts) < forecast_periods:
            # Pad with last value
            last_forecast = forecasts[-1] if len(forecasts) > 0 else float(psr.mean())
            forecasts = list(forecasts) + [last_forecast] * (forecast_periods - len(forecasts))
        elif len(forecasts) > forecast_periods:
            # Truncate
            forecasts = forecasts[:forecast_periods]
        
        return forecasts, best_model_aic
    except Exception as e:
        st.error(f"Error in AR Optimal prediction: {str(e)}")
        import traceback
        st.write("AR Optimal Traceback:", traceback.format_exc())
        # Return mean as fallback
        try:
            mean_psr = float(pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna().mean())
            return [mean_psr] * forecast_periods, None
        except:
            return [5.0] * forecast_periods, None  # Default to 5% if all else fails

# Main application logic
if uploaded_file is not None:
    # Load the data
    try:
        data = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Print column names and dtypes to debug
        st.subheader("Data Information")
        st.write(f"Number of columns: {data.shape[1]}")
        st.write(f"Number of rows: {data.shape[0]}")
        st.write(f"Column names: {list(data.columns)}")
        
        # Display the first few rows of the data
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Ensure we have at least the minimum columns
        if data.shape[1] < 2:
            st.error("Your data needs at least 2 columns: Time Period and Personal Savings Rate.")
            st.stop()
        
        # Apply explicit column access by position rather than names to avoid issues
        st.info("Using the data as-is without renaming columns")
        
        # Verify data types - show a sample of each column
        st.subheader("Data Sample (First 5 rows of each column)")
        cols = st.columns(min(7, data.shape[1]))
        for i in range(min(7, data.shape[1])):
            with cols[i]:
                st.write(f"Column {i}")
                st.write(data.iloc[:5, i])
        
        # Ensure we have numeric data in the Personal Savings Rate column
        psr_data = pd.to_numeric(data.iloc[:, 1], errors='coerce')
        if psr_data.isna().all():
            st.error("Could not convert Personal Savings Rate column to numeric values. Please check your data.")
            st.stop()
        
        # Sidebar for forecast settings
        st.sidebar.header("Forecast Settings")
        forecast_period = st.sidebar.selectbox("Select forecast period", [1, 2, 3, 12], index=0)
        
        # Generate forecasts
        with st.spinner("Generating forecasts..."):
            try:
                # Generate forecasts from each model with error handling
                try:
                    st.write("Attempting MLR Lagged PSR model...")
                    mlr_lagged_forecasts, mlr_lagged_model = predict_mlr_lagged_psr(data, forecast_period)
                except Exception as e:
                    st.error(f"Error in MLR Lagged PSR model: {str(e)}")
                    import traceback
                    st.write("MLR Lagged PSR Traceback:", traceback.format_exc())
                    # Fallback to mean
                    mean_psr = float(psr_data.dropna().mean())
                    mlr_lagged_forecasts = [mean_psr] * forecast_period
                    mlr_lagged_model = None
                
                try:
                    st.write("Attempting MLR Economic model...")
                    mlr_economic_forecasts, mlr_economic_model = predict_mlr_economic(data, forecast_period)
                except Exception as e:
                    st.error(f"Error in MLR Economic model: {str(e)}")
                    import traceback
                    st.write("MLR Economic Traceback:", traceback.format_exc())
                    # Fallback to mean
                    mean_psr = float(psr_data.dropna().mean())
                    mlr_economic_forecasts = [mean_psr] * forecast_period
                    mlr_economic_model = None
                
                try:
                    st.write("Attempting AR Optimal model...")
                    ar_forecasts, ar_model = predict_ar_optimal(data, forecast_period)
                except Exception as e:
                    st.error(f"Error in AR Optimal model: {str(e)}")
                    import traceback
                    st.write("AR Optimal Traceback:", traceback.format_exc())
                    # Fallback to mean
                    mean_psr = float(psr_data.dropna().mean())
                    ar_forecasts = [mean_psr] * forecast_period
                    ar_model = None
                
                # Calculate average forecasts
                avg_forecasts = []
                for i in range(forecast_period):
                    values = [
                        mlr_lagged_forecasts[i] if i < len(mlr_lagged_forecasts) else float(psr_data.dropna().mean()),
                        mlr_economic_forecasts[i] if i < len(mlr_economic_forecasts) else float(psr_data.dropna().mean()),
                        ar_forecasts[i] if i < len(ar_forecasts) else float(psr_data.dropna().mean())
                    ]
                    # Calculate average
                    avg = sum(values) / len(values)
                    avg_forecasts.append(avg)
                
                # Display forecasts in a table
                st.subheader(f"Forecasts for Next {forecast_period} Month(s)")
                
                # Create dataframe for forecasts
                forecast_df = pd.DataFrame({
                    'Month': [f'Month {i+1}' for i in range(forecast_period)],
                    'MLR Lagged PSR': [round(mlr_lagged_forecasts[i], 2) for i in range(forecast_period)],
                    'MLR Economic': [round(mlr_economic_forecasts[i], 2) for i in range(forecast_period)],
                    'AR Optimal': [round(ar_forecasts[i], 2) for i in range(forecast_period)],
                    'Average': [round(avg_forecasts[i], 2) for i in range(forecast_period)]
                })
                
                st.dataframe(forecast_df)
                
                # Historical data and forecasts visualization
                st.subheader("Visualization")
                
                # Get historical data
                try:
                    # Try to convert first column to datetime, but fallback to using it as-is
                    try:
                        historical_dates = pd.to_datetime(data.iloc[:, 0], errors='coerce')
                        if historical_dates.isna().all():
                            historical_dates = pd.Series(range(len(data)))
                            st.warning("Could not parse dates. Using row indices for the time axis.")
                    except:
                        historical_dates = pd.Series(range(len(data)))
                        st.warning("Could not process dates. Using row indices for the time axis.")
                    
                    historical_psr = psr_data
                    
                    # Create future dates for forecasting
                    if isinstance(historical_dates.iloc[0] if not historical_dates.empty else None, pd.Timestamp):
                        # Try to infer frequency from dates
                        try:
                            # Check if the dates appear to be monthly
                            date_diffs = historical_dates.diff().dropna()
                            median_days = date_diffs.median().days if not date_diffs.empty else 30
                            
                            if 28 <= median_days <= 31:
                                freq = 'M'  # Monthly
                            elif 89 <= median_days <= 92:
                                freq = 'Q'  # Quarterly
                            else:
                                freq = None
                                
                            if freq and not historical_dates.empty:
                                future_dates = pd.date_range(start=historical_dates.iloc[-1], 
                                                           periods=forecast_period+1, 
                                                           freq=freq)[1:]
                            else:
                                # Use indices if frequency can't be inferred
                                future_dates = range(len(historical_psr), len(historical_psr) + forecast_period)
                                st.warning("Could not infer date frequency. Using indices for forecast periods.")
                        except Exception as date_err:
                            st.write(f"Date inference error: {str(date_err)}")
                            future_dates = range(len(historical_psr), len(historical_psr) + forecast_period)
                            st.warning("Error inferring date pattern. Using indices for forecast periods.")
                    else:
                        future_dates = range(len(historical_psr), len(historical_psr) + forecast_period)
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Plot historical data
                    ax.plot(range(len(historical_psr)), historical_psr, label='Historical PSR', color='blue')
                    
                    # Plot forecasts
                    ax.plot(range(len(historical_psr), len(historical_psr) + forecast_period), 
                           mlr_lagged_forecasts, label='MLR Lagged PSR Forecast', marker='o', linestyle='--')
                    ax.plot(range(len(historical_psr), len(historical_psr) + forecast_period), 
                           mlr_economic_forecasts, label='MLR Economic Forecast', marker='s', linestyle='--')
                    ax.plot(range(len(historical_psr), len(historical_psr) + forecast_period), 
                           ar_forecasts, label='AR Optimal Forecast', marker='^', linestyle='--')
                    ax.plot(range(len(historical_psr), len(historical_psr) + forecast_period), 
                           avg_forecasts, label='Average Forecast', marker='*', linestyle='-', linewidth=2, color='black')
                    
                    # Customize plot
                    ax.set_title('Personal Savings Rate - Historical Data and Forecasts')
                    ax.set_xlabel('Time Period')
                    ax.set_ylabel('Personal Savings Rate (%)')
                    ax.legend()
                    ax.grid(True)
                    
                    # Show plot
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
                    import traceback
                    st.write("Visualization Traceback:", traceback.format_exc())
                    # Create a simpler fallback visualization
                    try:
                        # Create a simpler plot using indices instead of dates
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Use indices for x-axis
                        hist_indices = range(len(historical_psr))
                        future_indices = range(len(historical_psr), len(historical_psr) + forecast_period)
                        
                        # Plot data
                        ax.plot(hist_indices, historical_psr, label='Historical PSR')
                        ax.plot(future_indices, mlr_lagged_forecasts, label='MLR Lagged PSR', linestyle='--')
                        ax.plot(future_indices, mlr_economic_forecasts, label='MLR Economic', linestyle='--')
                        ax.plot(future_indices, ar_forecasts, label='AR Optimal', linestyle='--')
                        ax.plot(future_indices, avg_forecasts, label='Average', linestyle='-', color='black')
                        
                        ax.set_title('Personal Savings Rate - Historical Data and Forecasts')
                        ax.set_xlabel('Time Period Index')
                        ax.set_ylabel('Personal Savings Rate (%)')
                        ax.legend()
                        ax.grid(True)
                        
                        st.pyplot(fig)
                    except Exception as viz_fallback_err:
                        st.error(f"Could not create visualization. Error: {str(viz_fallback_err)}")
                        st.write("Data may be incompatible.")
                
                # Display model summaries
                st.subheader("Model Summaries")
                
                # Create tabs for model summaries
                tab1, tab2, tab3 = st.tabs(["MLR Lagged PSR", "MLR Economic", "AR Optimal"])
                
                with tab1:
                    st.text("Multiple Linear Regression with Lagged PSR Variables")
                    if mlr_lagged_model is not None:
                        st.text(str(mlr_lagged_model.summary()))
                    else:
                        st.text("Model could not be fitted.")
                
                with tab2:
                    st.text("Multiple Linear Regression with Economic Indicators")
                    if mlr_economic_model is not None:
                        st.text(str(mlr_economic_model.summary()))
                    else:
                        st.text("Model could not be fitted.")
                
                with tab3:
                    st.text("Autoregressive Model with Optimal Lag Selection")
                    if ar_model is not None:
                        st.text(str(ar_model.summary()))
                    else:
                        st.text("Model could not be fitted.")
                
            except Exception as e:
                st.error(f"Detailed error in forecasting: {str(e)}")
                st.write("Exception type:", type(e).__name__)
                import traceback
                st.write("Traceback:", traceback.format_exc())
                st.error("The application encountered an error. Debug information:")
                st.write("Data shape:", data.shape)
                st.write("Data columns:", list(data.columns))
                st.write("Data types:", data.dtypes)
                
                # Inspect first few rows of each column in detail to help debugging
                st.subheader("Detailed Data Inspection")
                for i in range(min(7, data.shape[1])):
                    try:
                        col_name = data.columns[i] if i < len(data.columns) else f"Column {i}"
                        st.write(f"Column {i} ({col_name}) - First 5 values:")
                        for j in range(min(5, data.shape[0])):
                            try:
                                val = data.iloc[j, i]
                                st.write(f"Row {j}: {val} (Type: {type(val).__name__})")
                            except Exception as row_err:
                                st.write(f"Row {j}: Error accessing value - {str(row_err)}")
                    except Exception as col_err:
                        st.write(f"Error inspecting column {i}: {str(col_err)}")
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.write("Data Loading Traceback:", traceback.format_exc())
        st.error("Please ensure your Excel file has the correct structure.")

else:
    # Display sample image or further instructions
    st.info("Upload an Excel file to see predictions.")
    
    # Explain each model
    st.subheader("About the Models")
    
    st.markdown("""
    This dashboard uses three different models to predict future personal savings rates:
    
    1. **MLR Lagged PSR Model**: A multiple linear regression model that uses lagged values of the Personal Savings Rate itself (1, 11, and 12 months ago).
    
    2. **MLR Economic Model**: A multiple linear regression model that incorporates various economic indicators including unemployment rate, disposable income, consumer credit, S&P 500 changes, and VIX.
    
    3. **AR Optimal Model**: An autoregressive model that automatically identifies the optimal lag structure based on AIC and statistical significance.
    
    The dashboard also provides an average forecast combining all three models, which may provide more balanced predictions.
    """)
    
    # Example data format
    st.subheader("Data Format")
    
    # Create a sample dataframe
    sample_df = pd.DataFrame({
        "Time Period": ["2010-01-01", "2010-02-01", "2010-03-01", "2010-04-01", "2010-05-01"],
        "Personal Savings Rate (%)": [5.2, 5.5, 6.0, 6.2, 6.4],
        "Unemployment Rate (%)": [10.0, 9.8, 9.9, 9.7, 9.6],
        "Monthly Disposable Income Total US in Billions ($)": [1100, 1105, 1110, 1112, 1115],
        "Total Consumer Credit Owned and Securitized": [2400, 2410,]
    })
    
