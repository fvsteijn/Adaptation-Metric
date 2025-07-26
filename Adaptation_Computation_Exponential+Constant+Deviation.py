# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# import scipy.stats as stats

# # Load the updated DataFrame
# df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_2.pkl')

# # Extract R2 values and convert to numeric, forcing errors to NaN
# df['R2'] = pd.to_numeric(df['R2'], errors='coerce')

# # Remove rows with NaN or infinite R2 values
# df = df[np.isfinite(df['R2'])]

# # Plot the histogram
# plt.figure(figsize=(10, 6))
# plt.hist(df['R2'], bins=100, color='blue', edgecolor='black', alpha=0.7)
# plt.title('Histogram of $r^2$ values')
# plt.xlabel('$r^2$ values')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# # Calculate the average and standard deviation of the R2 values
# r2_mean = np.mean(df['R2'])
# r2_std = np.std(df['R2'])

# print(f"Average R2 value: {r2_mean:.4f}")
# print(f"Standard deviation of R2 values: {r2_std:.4f}")
# print(f"Median R2 value: {np.median(df['R2']):.4f}")
# print(f"Percentage of values higher than 0.9: {np.sum(df['R2'] > 0.9) / len(df['R2']) * 100:.2f}%")

# # Now make a list where we add the number of fitted waves for each country
# n_waves = []
# for index, row in df.iterrows():
#     fitted_params = row.get('Fitted_Parameters', [])
#     n_waves.append(len(fitted_params) // 4)
    
# # Add the number of waves to the DataFrame
# df['Number_of_Waves'] = n_waves

# # Now, count the number of countries with less than 3 waves
# n_3_waves = len(df[df['Number_of_Waves'] < 3])


import pandas as pd
import numpy as np

# Load the DataFrame
df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_2.pkl')

# Convert R2 to numeric and remove invalid entries
df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
df = df[np.isfinite(df['R2'])]

# Count the number of fitted waves for each entry
n_waves = []
for index, row in df.iterrows():
    fitted_params = row.get('Fitted_Parameters', [])
    n_waves.append(len(fitted_params) // 4)

# Add the number of waves to the DataFrame
df['Number_of_Waves'] = n_waves

# Filter entries with 3 or more waves
df_filtered = df[df['Number_of_Waves'] >= 3]

# Save the ISO codes (index) to a CSV file
df_filtered.index.to_series().to_csv('iso_codes_3_or_more_waves.csv', index=False)

print("ISO codes of countries with 3 or more waves have been saved to 'iso_codes_3_or_more_waves.csv'.")









# # Now make a scatterplot of the number of waves versus the R2 values
# plt.figure(figsize=(10, 6))
# plt.scatter(df['Number_of_Waves'], df['R2'], color='blue', alpha=0.7)
# plt.title('Number of Waves vs R2 values')
# plt.xlabel('Number of Waves')
# plt.ylabel('R2 values')
# plt.grid(True)
# plt.show()

# # Now make a scatterplot of the number of waves versus the R2 values, with a regression line
# plt.figure(figsize=(10, 6))
# plt.scatter(df['Number_of_Waves'], df['R2'], color='blue', alpha=0.7)
# plt.title('Number of Waves vs R2 values')
# plt.xlabel('Number of Waves')
# plt.ylabel('R2 values')
# plt.grid(True)

# # Fit a linear regression model
# model = np.polyfit(df['Number_of_Waves'], df['R2'], 1)
# x_range = np.linspace(min(df['Number_of_Waves']), max(df['Number_of_Waves']), 100)
# y_range = np.polyval(model, x_range)
# plt.plot(x_range, y_range, color='red', linestyle='--', label=f'Linear fit: y = {model[0]:.4f}x + {model[1]:.4f}')

# plt.legend()
# plt.show()

# # Now compute the average, sd, median and percentage of R2 values, only on the subset of countries with more than 5 waves
# df_subset = df[df['Number_of_Waves'] > 5]

# # Calculate the average and standard deviation of the R2 values
# r2_mean_subset = np.mean(df_subset['R2'])
# r2_std_subset = np.std(df_subset['R2'])
# r2_median_subset = np.median(df_subset['R2'])
# r2_percentage_subset = np.sum(df_subset['R2'] > 0.9) / len(df_subset['R2']) * 100

# print(f"Average R2 value for countries with more than 5 waves: {r2_mean_subset:.4f}")
# print(f"Standard deviation of R2 values for countries with more than 5 waves: {r2_std_subset:.4f}")
# print(f"Median R2 value for countries with more than 5 waves: {r2_median_subset:.4f}")
# print(f"Percentage of values higher than 0.9 for countries with more than 5 waves: {r2_percentage_subset:.2f}%")





# Define a fitting function with a constant term
def exponential_fit(x, a, b, c):
    return a * np.exp(-b * x) + c

# Calculate deviation as mean absolute error
def calculate_r2(y_values, fitted_curve):
    return 1 - np.sum((y_values - fitted_curve) ** 2) / np.sum((y_values - np.mean(y_values)) ** 2)

# Collect DataFrames for concatenation
coeffs_dataframes = []

# List of metric columns to process
metric_columns = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']
metrics_labels = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']

for index, row in df_filtered.iterrows():
    country = index  # Assuming country codes are the index
    fitted_params = row.get('Fitted_Parameters', [])
    
    wave_count = len(fitted_params) // 4
    peak_times = [fitted_params[4 * j] for j in range(wave_count)]
    
    # Process each metric
    for metric, label in zip(metric_columns, metrics_labels):
        y_values = row.get(metric, None)
        
        if y_values is None or not isinstance(y_values, list) or not y_values:
            print(f"Skipping {country} for metric {label} due to missing data.")
            continue
        
        valid_y_values = np.array([y for y in y_values if y is not None and not np.isnan(y)])
        valid_peak_times = np.array(peak_times[:len(valid_y_values)])
        
        first_peak_value = valid_y_values[0]
        
        
        
        
        # Proceed only if there are enough data points
        if len(valid_peak_times) >= 2 and len(valid_y_values) >= 2:  # At least 2 points for 3 parameters
            # Shift peak times
            first_peak_time = valid_peak_times[0]
            valid_peak_times_shifted = valid_peak_times - first_peak_time
            
            # Fit to exponential with a constant term
            try:
                initial_guess = [0, 0, 0]  # Initial guesses for a, b, c
                bounds = ([-10*first_peak_value, -2, -10*first_peak_value], 
                          [10*first_peak_value, 2, 10*first_peak_value])  # No bounds on a, b, c
                params, _ = curve_fit(
                    exponential_fit, valid_peak_times_shifted, valid_y_values,
                    p0=initial_guess, maxfev=50000, 
                    bounds=bounds,
                    method='trf')
                a, b, c = params
                
                # Calculate fitted curve values
                fitted_curve = exponential_fit(valid_peak_times_shifted, a, b, c)
                
                # Calculate deviation
                deviation_metric = calculate_r2(valid_y_values, fitted_curve)
                
                # Calculate y_400 and y_800
                # y_400 = (exponential_fit(400 - first_peak_time, a, b, c) / exponential_fit(0 - first_peak_time, a, b, c)) * 100
                # y_800 = (exponential_fit(800 - first_peak_time, a, b, c) / exponential_fit(0 - first_peak_time, a, b, c)) * 100
                
                # y_400 = ((exponential_fit(200, a, b, c) - exponential_fit(0, a, b, c)) / exponential_fit(0, a, b, c)) * 100
                # y_800 = (exponential_fit(800, a, b, c) - exponential_fit(0, a, b, c) / exponential_fit(0, a, b, c)) * 100
                # y_800 = ((exponential_fit(400, a, b, c) - exponential_fit(200, a, b, c)) / exponential_fit(0, a, b, c)) * 100
                
            except RuntimeError:
                print(f"Exponential fit failed for {country} in metric {label}.")    
                a = b = c = deviation_metric = None         
                # a = b = c = deviation_metric = y_400 = y_800 = None          
                             
            
            
            
            # Store coefficients and deviation metric
            coeffs_df = pd.DataFrame({
                'Country': [country],
                'Metric': [label],
                'A': [a],
                'B': [b],
                'C': [c],
                'Deviation': [deviation_metric],
                # 'y_400': [y_400],
                # 'y_800': [y_800]
            })
            coeffs_dataframes.append(coeffs_df)

# Concatenate all DataFrames into one big table
final_df = pd.concat(coeffs_dataframes)

# Save the final DataFrame to a new pickle file
# final_df.to_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')
print("Processing complete. The final DataFrame has been saved to 'Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl'.")






# Load the pickle file
df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# Exclude countries with fewer than 3 waves
df_filtered = df[~df['Country'].isin(iso_codes_less_than_3_waves)]

# Plot a histogram of the 'Deviation' (R²) values for the remaining countries
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['Deviation'].dropna(), bins=100, edgecolor='black')
plt.title('Histogram of R² Values from Exponential Fits (≥ 3 Waves)')
plt.xlabel('R² Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Count the number of non-null R² values in the filtered DataFrame
r2_count_filtered = df_filtered['Deviation'].dropna().shape[0]
print(f"The histogram contains {r2_count_filtered} R² values from countries with ≥ 3 waves.")










# Load the pickle file
df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# Display the first few rows of the DataFrame
df.head()


# Plot a histogram of the 'Deviation' (R²) values
plt.figure(figsize=(10, 6))
plt.hist(df['Deviation'].dropna(), bins=100, edgecolor='black')
plt.title('Histogram of R² Values from Exponential Fits')
plt.xlabel('R² Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Count the number of non-null R² values
r2_count = df['Deviation'].dropna().shape[0]

print(f"The histogram will contain {r2_count} R² values.")







# Concatenate all coefficient DataFrames if available
if coeffs_dataframes:
    exponential_coefficients_df = pd.concat(coeffs_dataframes, ignore_index=True)
    
    # Reset index for merging
    df_reset = df.reset_index()
    df_reset.rename(columns={'index': 'Country'}, inplace=True)
    
    # Add 'Metric' column to df_reset with NaN values
    df_reset['Metric'] = np.nan
    df_reset['Country'] = df_reset['Country'].astype(str)
    exponential_coefficients_df['Country'] = exponential_coefficients_df['Country'].astype(str)
    df_reset['Metric'] = df_reset['Metric'].astype(str)
    exponential_coefficients_df['Metric'] = exponential_coefficients_df['Metric'].astype(str)
    
    # Merge on 'Country' and 'Metric'
    merged_df = df_reset.merge(exponential_coefficients_df, on=['Country', 'Metric'], how='left')
    
    # Optionally set the index back to 'Country'
    merged_df.set_index('Country', inplace=True)
    
    # Save the merged DataFrame with exponential fit coefficients
    # merged_df.to_pickle('Final_Analysis_with_Exponential_Fits_and_Metrics_no_bounds.pkl')
else:
    print("No valid coefficients to concatenate.")
    
def plot_covid_analysis(df, exponential_coefficients_df, country_code, metrics=['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']):
    # Ensure the country_code exists
    if country_code not in df.index:
        print(f"Country code '{country_code}' not found in the DataFrame.")
        return
    
    # Retrieve country-specific data
    country_data = df.loc[country_code]
    fitted_params = country_data['Fitted_Parameters']
    covid_deaths = country_data.get('COVID_Deaths', [])
    
    # Plot COVID deaths over time
    plt.figure(figsize=(12, 6))
    plt.plot(covid_deaths, label='COVID Deaths')
    plt.title(f'COVID Deaths over Time for {country_code}')
    plt.xlabel('Time')
    plt.ylabel('Deaths')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot fitted skewnormal curves
    n_waves = len(fitted_params) // 4
    plt.figure(figsize=(12, 6))
    for j in range(n_waves):
        mu, sigma, skew, scale = fitted_params[4 * j: 4 * j + 4]
        x_range = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        wave = scale * stats.skewnorm.pdf(x_range, skew, loc=mu, scale=sigma)
        plt.plot(x_range, wave, label=f'Wave {j + 1}')
    plt.title(f'Fitted Skewnormal Curves for {country_code}')
    plt.xlabel('Time')
    plt.ylabel('COVID deaths per million')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot metrics with adaptation dimensions using shifted time
    for metric in metrics:
        metric_data = exponential_coefficients_df[(exponential_coefficients_df['Country'] == country_code) &
                                                  (exponential_coefficients_df['Metric'] == metric)]
        if metric_data.empty:
            print(f"No data for metric '{metric}' for country '{country_code}'.")
            continue
        
        a = metric_data['A'].values[0]
        b = metric_data['B'].values[0]
        c = metric_data['C'].values[0]
        deviation = metric_data['Deviation'].values[0]
        # y_400 = metric_data['y_400'].values[0]
        # y_800 = metric_data['y_800'].values[0]
        y_values = country_data.get(metric, [])
        peak_times = [fitted_params[4 * j] for j in range(len(y_values))]
        
        if not y_values or len(y_values) < 2:
            print(f"Insufficient data for metric '{metric}' for country '{country_code}'.")
            a = b = c = deviation = None
            # y_400 = y_800 = None
        
        # Shift peak times for plotting
        first_peak_time = peak_times[0]
        shifted_peak_times = np.array(peak_times) - first_peak_time
        x_range = np.linspace(min(shifted_peak_times), max(shifted_peak_times), 100)
        if metric == 'Time_to_Next_Wave':
            x_range = np.linspace(min(shifted_peak_times), max(shifted_peak_times[:-1]), 100)
        
        # Create exponential fit over shifted time
        fit_line = a * np.exp(-b * x_range) + c
        plt.figure(figsize=(10, 6))
        plt.scatter(shifted_peak_times, y_values, label='Data Points', color='blue', s=50)
        plt.plot(x_range, fit_line, label='Exponential Fit', color='orange', linestyle='--', linewidth=2)
        
        # Annotate figure with adaptation dimensions
        plt.annotate(f'A: {a:.4f}', xy=(0.05, 0.90), xycoords='axes fraction')
        plt.annotate(f'B: {b:.4f}', xy=(0.05, 0.85), xycoords='axes fraction')
        plt.annotate(f'C: {c:.4f}', xy=(0.05, 0.80), xycoords='axes fraction')
        plt.annotate(f'Deviation: {deviation:.4f}', xy=(0.05, 0.75), xycoords='axes fraction')
        # plt.annotate(f'y_400: {y_400:.2f}%', xy=(0.05, 0.70), xycoords='axes fraction')
        # plt.annotate(f'y_800: {y_800:.2f}%', xy=(0.05, 0.65), xycoords='axes fraction')
        plt.title(f'{metric} for {country_code}')
        plt.xlabel('Shifted Time')
        plt.ylabel(metric)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

# Example usage
plot_covid_analysis(df, exponential_coefficients_df, 'CHL')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Define the metrics
metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']

def discard_percentiles(nparray, lower = 0, upper = 1):
    
    # check if the input is a numpy array
    if not isinstance(nparray, np.ndarray):
        raise ValueError('Input is not a numpy array')
    
    
    lower_bound = np.nanquantile(nparray, lower, axis=0)
    upper_bound = np.nanquantile(nparray, upper, axis=0)
    
    discarded_idx = np.where((nparray < lower_bound) | (nparray > upper_bound))
    
    return discarded_idx

# Plot scatter plots for each metric separately
for metric in metrics:
    metric_data = final_df[final_df['Metric'] == metric]
    a_values = metric_data['A'].values
    b_values = metric_data['B'].values
    c_values = metric_data['C'].values
    deviation_values = metric_data['Deviation'].values
    
    # check if lengtsh are all the same
    if not len(a_values) == len(b_values) == len(c_values) == len(deviation_values):
        raise ValueError('Lengths of the arrays are not the same')
    
    #remove nan values, if they exist. Remove from both a_plus_c_values and b_values
    nan_indices = np.isnan(a_values) | np.isnan(b_values) | np.isnan(c_values) | np.isnan(deviation_values)
    print(f'Number of discarded values for {metric}: {np.sum(nan_indices)}')
    a_values = a_values[~nan_indices]
    b_values = b_values[~nan_indices]
    c_values = c_values[~nan_indices]
    deviation_values = deviation_values[~nan_indices]
    
    if not len(a_values) == len(b_values) == len(c_values) == len(deviation_values):
        raise ValueError('Lengths of the arrays are not the same')
     
    # Remove data where the deviation is higher than 0.6 and smaller than 0.95
    valid_indices = (deviation_values > 0.3) & (deviation_values < 0.99)
    print(f'Number of discarded low-deviation values for {metric}: {np.sum(~valid_indices)}')

    a_values = a_values[valid_indices]
    b_values = b_values[valid_indices]
    c_values = c_values[valid_indices]
    deviation_values = deviation_values[valid_indices]
    
    if not len(a_values) == len(b_values) == len(c_values) == len(deviation_values):
        raise ValueError('Lengths of the arrays are not the same')
    else: 
        # print the number of values that are left
        print(f'Number of values left for {metric}: {len(a_values)}')
    
    a_plus_c_values = a_values + c_values
    df_dx_0 = - a_values * b_values
    ddf_dx_0 = a_values * b_values * b_values
    
    df_dx_0_normalised = df_dx_0 / a_plus_c_values
    ddf_dx_0_normalised = ddf_dx_0 / a_plus_c_values
    
    
    # # Fit a linear regression model on A + C and B
    # model = LinearRegression()
    # model.fit(a_plus_c_values.reshape(-1, 1), b_values)
    
    # alpha = model.coef_[0]
    # beta = model.intercept_
    # error = model.score(a_plus_c_values.reshape(-1, 1), b_values)
    
    # discarded_idx = discard_percentiles(df_dx_0)
    # b_values_discarded = np.delete(b_values, discarded_idx)
    # a_plus_c_values_discarded = np.delete(a_plus_c_values, discarded_idx)
    # df_dx_0_discarded = np.delete(df_dx_0, discarded_idx)
    # ddf_dx_0_discarded = np.delete(ddf_dx_0, discarded_idx)
    
    # print(f'Number of discarded values for {metric}: {len(b_values) - len(b_values_discarded)}')
        
    # plot a versus b
    plt.figure(figsize=(10, 6))
    plt.scatter(a_values, b_values, label=metric)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.title(f'Scatter plot of B vs A for {metric}')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    
        
    # plt.figure(figsize=(10, 6))
    # print(len(a_plus_c_values), len(b_values))
    # plt.scatter(a_plus_c_values, b_values, label=metric)
    # plt.xlabel('A + C')
    # plt.ylabel('B')
    # plt.xscale('symlog')
    # plt.yscale('symlog')
    # plt.title(f'Scatter plot of B vs A + C for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    # # plot the deviation metric versus A + C
    # plt.figure(figsize=(10, 6))
    # print(len(a_plus_c_values), len(deviation_metric))
    # plt.scatter(a_plus_c_values, deviation_values, label=metric)
    # # plt.xscale('symlog')
    # # plt.yscale('symlog')
    # plt.xlabel('A + C')
    # plt.ylabel('Deviation')
    # plt.title(f'Scatter plot of Deviation vs A + C for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # now deviation versus b
    # plt.figure(figsize=(10, 6))
    # plt.scatter(b_values, deviation_values, label=metric)
    # plt.xscale('symlog')
    # # plt.yscale('symlog')
    # plt.xlabel('B')
    # plt.ylabel('Deviation')
    # plt.title(f'Scatter plot of Deviation vs B for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # now deviation versus df_dx_0 / a_plus_c_values
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df_dx_0_normalised, deviation_values, label=metric)
    # plt.xscale('symlog')
    # # plt.yscale('symlog')
    # plt.xlabel(r'$\frac{dF}{dx}(0) / (A + C)$')
    # plt.ylabel('Deviation')
    # plt.title(f'Scatter plot of Deviation vs dF/dx(0) / (A + C) for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # now deviation versus ddf_dx_0 / a_plus_c_values
    # plt.figure(figsize=(10, 6))
    # plt.scatter(ddf_dx_0_normalised, deviation_values, label=metric)
    # plt.xscale('symlog')
    # # plt.yscale('symlog')
    # plt.xlabel(r'$\frac{d^2F}{dx^2}(0) / (A + C)$')
    # plt.ylabel('Deviation')
    # plt.title(f'Scatter plot of Deviation vs d^2F/dx^2(0) / (A + C) for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    
    
    
    # # # linear regression model for neg_ab_values and a_plus_c_values
    # # model = LinearRegression()
    # # model.fit(a_plus_c_values.reshape(-1, 1), df_dx_0_discarded)
    
    # # alpha = model.coef_[0]
    # # beta = model.intercept_
    # # error = model.score(a_plus_c_values.reshape(-1, 1), df_dx_0_discarded)
    
    
    
    # # plot the neg_ab and a_plus_c_values
    # plt.figure(figsize=(10, 6))
    # plt.scatter(a_plus_c_values, df_dx_0, label=metric)
    # # plt.plot(a_plus_c_values, alpha * a_plus_c_values + beta, color='red', label=f'Linear fit: y = {alpha:.3f}x + {beta:.3f}, R^2 = {error:.3f}')
    # plt.xscale('symlog')
    # plt.yscale('symlog')
    # plt.xlabel('A + C')
    # plt.ylabel(r'$\frac{dF}{dx}(0)$')
    # plt.title(f'Scatter plot of first derivative at time 0 (-A*B) vs A + C for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
        
    
    # # plot first and second derivative values. Dont plot regression line
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df_dx_0, ddf_dx_0, label=metric, alpha=0.5)
    # plt.xlabel(r'$\frac{dF}{dx}(0)$')
    # plt.ylabel(r'$\frac{d^2F}{dx^2}(0)$')
    # plt.xscale('symlog')
    # plt.yscale('symlog')    
    # plt.title(f'Scatter plot of first derivative vs second derivative at time 0 for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    # # plot the first derivative divided by A + C, versus A + C.
    # plt.figure(figsize=(10, 6))
    # plt.scatter(a_plus_c_values, df_dx_0_normalised, label=metric)
    # plt.xscale('symlog')
    # plt.yscale('symlog')
    # plt.xlabel('A + C')
    # plt.ylabel(r'$\frac{dF}{dx}(0) / (A + C)$')
    # plt.title(f'Scatter plot of first derivative at time 0 divided by A + C vs A + C for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # Now make the previous plot with plotly, such that we can click on the points and see the country names
    # import plotly.express as px
    # import plotly.graph_objects as go
    
    # fig = px.scatter(x=a_plus_c_values, y=df_dx_0_normalised, labels={'x': 'A + C', 'y': 'dF/dx(0) / (A + C)'}, title=f'Scatter plot of first derivative at time 0 divided by A + C vs A + C for {metric}')
    # fig.update_traces(marker=dict(size=12,
    #                                 line=dict(width=2,
    #                                             color='DarkSlateGrey')),
    #                 selector=dict(mode='markers'),
    #                 text=metric_data['Country'])
    
    # fig.show()
    
    # # plot the second derivative divided by A + C, versus A + C.
    # plt.figure(figsize=(10, 6))
    # plt.scatter(a_plus_c_values, ddf_dx_0_normalised, label=metric)
    # plt.xscale('symlog')
    # plt.yscale('symlog')
    # plt.xlabel('A + C')
    # plt.ylabel(r'$\frac{d^2F}{dx^2}(0) / (A + C)$')
    # plt.title(f'Scatter plot of second derivative at time 0 divided by A + C vs A + C for {metric}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    
    









# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the final DataFrame with y_400 and y_800
# final_df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# # List of metrics to plot
# metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']

# # Function to discard the 3rd and 97th percentile of a metric
# def discard_percentiles(df, column):
#     lower_bound = df[column].quantile(0.07)
#     upper_bound = df[column].quantile(0.93)
#     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# # Plot 2D plots for each metric
# for metric in metrics:
#     metric_data = final_df[final_df['Metric'] == metric]
    
#     # Discard the 3rd and 97th percentile of y_400 and y_800
#     metric_data = discard_percentiles(metric_data, 'y_400')
#     metric_data = discard_percentiles(metric_data, 'y_800')
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(metric_data['y_400'], metric_data['y_800'], label=metric, color='blue')
    
#     # for i, country in enumerate(metric_data['Country']):
#     #     plt.annotate(country, (metric_data['y_400'].iloc[i], metric_data['y_800'].iloc[i]))
    
#     plt.title(f'y_400 vs y_800 for {metric}')
#     plt.xlabel('y_400 (%)')
#     plt.ylabel('y_800 (%)')
#     plt.grid(True)
#     plt.show()



# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt

# # Function to plot metrics on a world map
# def plot_metrics_on_world_map(final_df, metrics):
#     # Load a shapefile of the world map
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#     for metric in metrics:
#         # Filter for the current metric
#         metric_df = final_df[final_df['Metric'] == metric].set_index('Country')

#         # Merge with the world map DataFrame
#         plot_df = world.set_index('iso_a3').join(metric_df)

#         # Create plots for y_400 and y_800
#         fig, ax = plt.subplots(1, 2, figsize=(20, 10))

#         # Plot y_400
#         plot_df.plot(
#             column='y_400',
#             cmap='viridis',
#             legend=True,
#             ax=ax[0],
#             missing_kwds={"color": "lightgrey"},
#             scheme='Quantiles',
#             classification_kwds={'k': 6},
#             # vmin=plot_df['y_400'].quantile(0.01),  # Trim outliers
#             # vmax=plot_df['y_400'].quantile(0.99)
#         )
#         ax[0].set_title(f'{metric}: y_400 on World Map')

#         # Plot y_800
#         plot_df.plot(
#             column='y_800',
#             cmap='viridis',
#             legend=True,
#             ax=ax[1],
#             missing_kwds={"color": "lightgrey"},
#             scheme='Quantiles',
#             classification_kwds={'k': 6},
#             # vmin=plot_df['y_800'].quantile(0.01),  # Trim outliers
#             # vmax=plot_df['y_800'].quantile(0.99)
#         )
#         ax[1].set_title(f'{metric}: y_800 on World Map')

#         # Display the plots
#         plt.tight_layout()
#         plt.show()

# # Load the final DataFrame with y_400 and y_800
# final_df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# # List of metrics to plot
# metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']

# # Plot the results on a world map
# plot_metrics_on_world_map(final_df, metrics)





# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt
# import mapclassify

# # Function to plot metrics on a world map
# def plot_metrics_on_world_map(final_df, metrics):
#     # Load a shapefile of the world map
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#     # Merge y_400 and y_800 for joint classification
#     merged_values = pd.concat([final_df['y_400'], final_df['y_800']])
#     merged_values_clean = merged_values.dropna()

#     # Apply Jenks Natural Breaks classification on merged data
#     jenks = mapclassify.NaturalBreaks(merged_values_clean, k=6)  # Customize number of classes as needed
#     print("Jenks Breaks:", jenks.bins)  # Display break boundaries

#     for metric in metrics:
#         # Filter for the current metric
#         metric_df = final_df[final_df['Metric'] == metric].set_index('Country')

#         # Merge with the world map DataFrame
#         plot_df = world.set_index('iso_a3').join(metric_df)

#         # Create plots for y_400 and y_800
#         fig, ax = plt.subplots(1, 2, figsize=(20, 10))

#         # Plot y_400 using Jenks classification
#         plot_df.plot(
#             column='y_400',
#             cmap='viridis',
#             legend=True,
#             ax=ax[0],
#             scheme='NaturalBreaks',
#             classification_kwds={'bins': jenks.bins},
#             legend_kwds={
#                 'loc': 'lower left',
#                 'fmt': '{:.2f}',  # Format to show two decimal places
#                 'title': 'y_400 (Jenks)'
#             },
#             missing_kwds={"color": "lightgrey"}
#         )
#         ax[0].set_title(f'{metric}: y_400 on World Map')

#         # Plot y_800 using Jenks classification
#         plot_df.plot(
#             column='y_800',
#             cmap='viridis',
#             legend=True,
#             ax=ax[1],
#             scheme='NaturalBreaks',
#             classification_kwds={'bins': jenks.bins},
#             legend_kwds={
#                 'loc': 'lower left',
#                 'fmt': '{:.2f}',  # Format to show two decimal places
#                 'title': 'y_800 (Jenks)'
#             },
#             missing_kwds={"color": "lightgrey"}
#         )
#         ax[1].set_title(f'{metric}: y_800 on World Map')

#         # Display the plots
#         plt.tight_layout()
#         plt.show()

# # Load the final DataFrame with y_400 and y_800
# final_df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# # List of metrics to plot
# metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']

# # Plot the results on a world map
# plot_metrics_on_world_map(final_df, metrics)







# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt

# # Function to plot metrics on a world map
# def plot_metrics_on_world_map(final_df, metrics):
#     # Load a shapefile of the world map
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#     for metric in metrics:
#         # Filter for the current metric
#         metric_df = final_df[final_df['Metric'] == metric].set_index('Country')

#         # Merge with the world map DataFrame
#         plot_df = world.set_index('iso_a3').join(metric_df)

#         # Create plots for y_400 and y_800
#         fig, ax = plt.subplots(1, 2, figsize=(20, 10))

#         # Plot y_400
#         plot_df.plot(
#             column='y_400',
#             cmap='coolwarm',
#             legend=True,
#             ax=ax[0],
#             missing_kwds={"color": "lightgrey"},
#             vmin=plot_df['y_400'].quantile(0.03),  # Trim outliers
#             vmax=plot_df['y_400'].quantile(0.97)
#         )
#         ax[0].set_title(f'{metric}: y_400 on World Map')

#         # Plot y_800
#         plot_df.plot(
#             column='y_800',
#             cmap='viridis',
#             legend=True,
#             ax=ax[1],
#             missing_kwds={"color": "lightgrey"},
#             vmin=plot_df['y_800'].quantile(0.03),  # Trim outliers
#             vmax=plot_df['y_800'].quantile(0.97)
#         )
#         ax[1].set_title(f'{metric}: y_800 on World Map')

#         # Display the plots
#         plt.tight_layout()
#         plt.show()

# # Load the final DataFrame with y_400 and y_800
# final_df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# # List of metrics to plot
# metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']

# # Plot the results on a world map
# plot_metrics_on_world_map(final_df, metrics)



# import geopandas as gpd
# import matplotlib.pyplot as plt
# import pandas as pd
# import mapclassify

# # Function to plot metrics on a world map
# def plot_metrics_on_world_map(final_df, metrics):
#     # Load a shapefile of the world map
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
#     for metric in metrics:
#         # Filter for the current metric
#         metric_df = final_df[final_df['Metric'] == metric].set_index('Country')
#         # Merge with the world map DataFrame
#         plot_df = world.set_index('iso_a3').join(metric_df)
        
#         # Create plots for y_400 and y_800
#         fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        
#         # Plot y_400 with Jenks binning
#         plot_df.plot(
#             column='y_400',
#             cmap='coolwarm',
#             legend=True,
#             ax=ax[0],
#             scheme='Quantiles',
#             classification_kwds={'k': 10},
#             legend_kwds={
#                 'loc': 'lower left',
#                 'fmt': '{:.2f}',  # Format to show two decimal places
#                 'title': 'y_400 (Jenks)'
#             },
#             missing_kwds={"color": "lightgrey"}
#         )
#         ax[0].set_title(f'{metric}: y_400 on World Map')

#         # Plot y_800 with Jenks binning
#         plot_df.plot(
#             column='y_800',
#             cmap='viridis',
#             legend=True,
#             ax=ax[1],
#             scheme='NaturalBreaks',
#             classification_kwds={'k': 10},
#             legend_kwds={
#                 'loc': 'lower left',
#                 'fmt': '{:.2f}',  # Format to show two decimal places
#                 'title': 'y_800 (Jenks)'
#             },
#             missing_kwds={"color": "lightgrey"}
#         )
#         ax[1].set_title(f'{metric}: y_800 on World Map')
        
#         # Display the plots
#         plt.tight_layout()
#         plt.show()

# # Load the final DataFrame with y_400 and y_800
# final_df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# # List of metrics to plot
# metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']

# # Plot the results on a world map
# plot_metrics_on_world_map(final_df, metrics)


# # Concatenate all coefficient DataFrames if available
# if coeffs_dataframes:
#     exponential_coefficients_df = pd.concat(coeffs_dataframes, ignore_index=True)
    
#     # Reset index for merging
#     df_reset = df.reset_index()
#     df_reset.rename(columns={'index': 'Country'}, inplace=True)
    
#     # Add 'Metric' column to df_reset with NaN values
#     df_reset['Metric'] = np.nan
#     df_reset['Country'] = df_reset['Country'].astype(str)
#     exponential_coefficients_df['Country'] = exponential_coefficients_df['Country'].astype(str)
#     df_reset['Metric'] = df_reset['Metric'].astype(str)
#     exponential_coefficients_df['Metric'] = exponential_coefficients_df['Metric'].astype(str)
    
#     # Merge on 'Country' and 'Metric'
#     merged_df = df_reset.merge(exponential_coefficients_df, on=['Country', 'Metric'], how='left')
    
#     # Optionally set the index back to 'Country'
#     merged_df.set_index('Country', inplace=True)
    
#     # Save the merged DataFrame with exponential fit coefficients
#     merged_df.to_pickle('Final_Analysis_with_Exponential_Fits_and_Metrics_no_bounds.pkl')
# else:
#     print("No valid coefficients to concatenate.")
    
# def plot_covid_analysis(df, exponential_coefficients_df, country_code, metrics=['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']):
#     # Ensure the country_code exists
#     if country_code not in df.index:
#         print(f"Country code '{country_code}' not found in the DataFrame.")
#         return
    
#     # Retrieve country-specific data
#     country_data = df.loc[country_code]
#     fitted_params = country_data['Fitted_Parameters']
#     covid_deaths = country_data.get('COVID_Deaths', [])
    
#     # Plot COVID deaths over time
#     plt.figure(figsize=(12, 6))
#     plt.plot(covid_deaths, label='COVID Deaths')
#     plt.title(f'COVID Deaths over Time for {country_code}')
#     plt.xlabel('Time')
#     plt.ylabel('Deaths')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
#     # Plot fitted skewnormal curves
#     n_waves = len(fitted_params) // 4
#     plt.figure(figsize=(12, 6))
#     for j in range(n_waves):
#         mu, sigma, skew, scale = fitted_params[4 * j: 4 * j + 4]
#         x_range = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
#         wave = scale * stats.skewnorm.pdf(x_range, skew, loc=mu, scale=sigma)
#         plt.plot(x_range, wave, label=f'Wave {j + 1}')
#     plt.title(f'Fitted Skewnormal Curves for {country_code}')
#     plt.xlabel('Time')
#     plt.ylabel('COVID deaths per million')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
#     # Plot metrics with adaptation dimensions using shifted time
#     for metric in metrics:
#         metric_data = exponential_coefficients_df[(exponential_coefficients_df['Country'] == country_code) &
#                                                   (exponential_coefficients_df['Metric'] == metric)]
#         if metric_data.empty:
#             print(f"No data for metric '{metric}' for country '{country_code}'.")
#             continue
        
#         a = metric_data['A'].values[0]
#         b = metric_data['B'].values[0]
#         c = metric_data['C'].values[0]
#         deviation = metric_data['Deviation'].values[0]
#         y_values = country_data.get(metric, [])
#         peak_times = [fitted_params[4 * j] for j in range(len(y_values))]
        
#         if not y_values or len(y_values) < 2:
#             print(f"Insufficient data for metric '{metric}' for country '{country_code}'.")
#             continue
        
#         # Shift peak times for plotting
#         first_peak_time = peak_times[0]
#         shifted_peak_times = np.array(peak_times) - first_peak_time
#         x_range = np.linspace(min(shifted_peak_times), max(shifted_peak_times), 100)
#         if metric == 'Time_to_Next_Wave':
#             x_range = np.linspace(min(shifted_peak_times), max(shifted_peak_times[:-1]), 100)
        
#         # Create exponential fit over shifted time
#         fit_line = a * np.exp(-b * x_range) + c
#         plt.figure(figsize=(10, 6))
#         plt.scatter(shifted_peak_times, y_values, label='Data Points', color='blue', 
#                     s = 50)
#         plt.plot(x_range, fit_line, label='Exponential Fit', color='orange', linestyle='--', linewidth=2)
        
#         # Annotate figure with adaptation dimensions
#         plt.annotate(f'A: {a:.4f}', xy=(0.05, 0.90), xycoords='axes fraction')
#         plt.annotate(f'B: {b:.4f}', xy=(0.05, 0.85), xycoords='axes fraction')
#         plt.annotate(f'C: {c:.4f}', xy=(0.05, 0.80), xycoords='axes fraction')
#         plt.annotate(f'Deviation: {deviation:.4f}', xy=(0.05, 0.75), xycoords='axes fraction')
#         plt.title(f'{metric} for {country_code}')
#         plt.xlabel('Shifted Time')
#         plt.ylabel(metric)
#         plt.legend(loc='upper right')
#         plt.grid(True)
#         plt.show()

# # Example usage
# plot_covid_analysis(df, exponential_coefficients_df, 'BEL')







            
            
            # try:
            #     initial_guess = [0, 0, 0]  # Initial guesses for a, b, c


            #     # Strategy 1: a >= 0
            #     bounds_positive_a = (
            #         [0, -2, -5 * first_peak_value],  # a >= 0
            #         [5 * first_peak_value, 2, 5 * first_peak_value]
            #     )
            #     try:
            #         params_positive, _ = curve_fit(
            #             exponential_fit, valid_peak_times_shifted, valid_y_values,
            #             p0=initial_guess, maxfev=1000,
            #             bounds=bounds_positive_a,
            #             method='trf'
            #         )
            #         a_pos, b_pos, c_pos = params_positive
            #         fitted_curve_pos = exponential_fit(valid_peak_times_shifted, a_pos, b_pos, c_pos)
            #         deviation_metric_pos = calculate_deviation(valid_y_values, fitted_curve_pos)
            #     except RuntimeError:
            #         a_pos = b_pos = c_pos = deviation_metric_pos = float('inf')


            #     # Strategy 2: a <= 0
            #     bounds_negative_a = (
            #         [-5 * first_peak_value, -2, -5 * first_peak_value],  # a <= 0
            #         [0, 2, 5 * first_peak_value]
            #     )
            #     try:
            #         params_negative, _ = curve_fit(
            #             exponential_fit, valid_peak_times_shifted, valid_y_values,
            #             p0=initial_guess, maxfev=1000,
            #             bounds=bounds_negative_a,
            #             method='trf'
            #         )
            #         a_neg, b_neg, c_neg = params_negative
            #         fitted_curve_neg = exponential_fit(valid_peak_times_shifted, a_neg, b_neg, c_neg)
            #         deviation_metric_neg = calculate_deviation(valid_y_values, fitted_curve_neg)
            #     except RuntimeError:
            #         a_neg = b_neg = c_neg = deviation_metric_neg = float('inf')


            #     # Compare deviation metrics and choose the best fit
            #     if deviation_metric_pos < deviation_metric_neg:
            #         a, b, c = a_pos, b_pos, c_pos
            #         deviation_metric = deviation_metric_pos
            #     else:
            #         a, b, c = a_neg, b_neg, c_neg
            #         deviation_metric = deviation_metric_neg


            #     # Calculate y_400 and y_800
            #     y_400 = ((exponential_fit(200, a, b, c) - exponential_fit(0, a, b, c)) 
            #             / exponential_fit(0, a, b, c)) * 100
            #     y_800 = ((exponential_fit(400, a, b, c) - exponential_fit(200, a, b, c)) 
            #             / exponential_fit(0, a, b, c)) * 100


            # except RuntimeError:
            #     print(f"Exponential fit failed for {country} in metric {label}.")
            #     a = b = c = deviation_metric = y_400 = y_800 = None
                
            
            
            