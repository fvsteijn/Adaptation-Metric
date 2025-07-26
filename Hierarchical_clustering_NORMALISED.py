import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import shapiro, kruskal, mannwhitneyu
import ptitprince as pt



# Load the pickle file
with open(r'c:\\Users\\fpvansteijn\\OneDrive - Delft University of Technology\\Documents\\Scripts\\TBCOV data set\\Cluster_Data\\Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl', 'rb') as f:
    df_fitted = pickle.load(f)

# Rename entries in the 'Metric' column
df_fitted['Metric'] = df_fitted['Metric'].replace('Recovery_Speed', 'Disruption_Duration')

# Display the updated DataFrame
print(df_fitted.head())

# Load the filtered ISO codes from the CSV file
iso_codes_filtered = np.array(pd.read_csv('iso_codes_3_or_more_waves.csv', header=None)[0].tolist())[1:]


# Load the cluster labels
df_deaths = pd.read_csv(r'c:\\Users\\fpvansteijn\\OneDrive - Delft University of Technology\\Documents\\Scripts\\TBCOV data set\\Cluster_Data\\data_new_deaths_smoothed_per_million.csv')
iso_code_deaths = df_deaths['iso_code'].tolist()
df_deaths = df_deaths.drop('iso_code', axis=1)

# Perform hierarchical clustering
array_deaths = df_deaths.iloc[:, 1:].to_numpy()
array_deaths = np.delete(array_deaths, [139, 191], axis=0)
iso_code_deaths = [iso_code for idx, iso_code in enumerate(iso_code_deaths) if idx != 139 and idx != 191]

for i in range(array_deaths.shape[0]):
    array_deaths[i] = (array_deaths[i] - np.min(array_deaths[i])) / (np.max(array_deaths[i]) - np.min(array_deaths[i]))



dist = pdist(array_deaths, metric='euclidean')
linkage_matrix = linkage(dist, method='ward')


# Elbow plot to determine the optimal number of clusters
last = linkage_matrix[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.figure(figsize=(5, 3))
plt.plot(idxs, last_rev)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.xticks(range(1,11))
plt.ylabel('Linkage Distance')
plt.show()






N_Clusters = 5
labels = fcluster(linkage_matrix, N_Clusters, criterion='maxclust')

# Add cluster labels to the fitted DataFrame
df_fitted['Cluster'] = df_fitted['Country'].map(dict(zip(iso_code_deaths, labels)))


iso_codes_filtered = np.array(pd.read_csv('iso_codes_3_or_more_waves.csv', header=None)[0].tolist())[1:]

# Filter the main DataFrame to include only countries in the filtered ISO code list
df_filtered = df_fitted[df_fitted['Country'].isin(iso_codes_filtered)]
df_fitted = df_filtered





import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Define the number of clusters and color scheme
color_scheme = sns.color_palette("tab10", N_Clusters)

# Define countries to highlight
highlight_countries = ['BGR', 'EST']

# Create date range
date = pd.date_range(start='2020-01-08', periods=array_deaths.shape[1], freq='D')
tick_dates = pd.date_range(start='2020-01-01', end=date[-1], freq='3MS')

# Define countries to highlight and their unique line styles
highlight_styles = {
    'BGR': {'linestyle': 'dotted', 'label': 'Bulgaria'},
    'EST': {'linestyle': 'solid', 'label': 'Estonia'},
    'BRA': {'linestyle': 'dashed', 'label': 'Brazil'},
    'ZAF': {'linestyle': 'dashdot', 'label': 'South Africa'},
    'PRT': {'linestyle': (0, (3, 1, 1, 1)), 'label': 'Portugal'},
    'ECU': {'linestyle': (0, (5, 5)), 'label': 'Ecuador'},
    'ISL': {'linestyle': (0, (1, 1)), 'label': 'Iceland'},
    'MEX': {'linestyle': (0, (3, 5, 1, 5)), 'label': 'Mexico'},
    'IRN': {'linestyle': (0, (2, 2, 10, 2)), 'label': 'Iran'},
    'IDN': {'linestyle': (0, (4, 2, 1, 2)), 'label': 'Indonesia'}
}

for i in range(N_Clusters):
    cluster_index = i + 1
    idx_cluster = np.where(labels == cluster_index)[0]
    array_deaths_cluster = array_deaths[idx_cluster]
    iso_codes_cluster = [iso_code_deaths[idx] for idx in idx_cluster]

    plt.figure(figsize=(15, 9))

    for j, series in enumerate(array_deaths_cluster):
        iso_code = iso_codes_cluster[j]
        if iso_code in highlight_styles:
            style = highlight_styles[iso_code]
            plt.plot(series, color='black', linestyle=style['linestyle'], 
                     linewidth=1.5, label=style['label'])
        else:
            plt.plot(series, color=color_scheme[i], alpha=0.15)

    plt.plot(np.mean(array_deaths_cluster, axis=0), color=color_scheme[i], 
             alpha=1, linewidth=3.5)

    plt.title(f'Cluster {cluster_index}', fontsize=35)
    plt.ylabel('Fatalities per million (normalized)', fontsize=31)
    plt.yticks(fontsize = 25)
    plt.xticks(
        ticks=[(d - date[0]).days for d in tick_dates if d >= date[0]],
        labels=[d.strftime('%b %Y') for d in tick_dates if d >= date[0]],
        rotation=45, fontsize=22
    )
    plt.tight_layout()
    plt.legend(fontsize = 30)
    plt.show()



# MAKING SUPPLEMENTARY TABLES FOR GOODNESS OF EXPONENTIAL FIT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create summary tables for each metric
metrics = df_filtered['Metric'].unique()
summary_tables = {}

for metric in metrics:
    df_metric = df_filtered[df_filtered['Metric'] == metric]
    
    # Summary per cluster
    summary = df_metric.groupby('Cluster')['Deviation'].agg(
        Median_R2='median',
        Percent_Less_Than_0_5=lambda x: (x < 0.5).mean() * 100,
        Percent_Greater_Than_0_9=lambda x: (x > 0.9).mean() * 100
    ).reset_index().round(2)
    
    # Summary for all countries
    all_countries = pd.DataFrame({
        'Cluster': ['All Countries'],
        'Median_R2': [df_metric['Deviation'].median()],
        'Percent_Less_Than_0_5': [(df_metric['Deviation'] < 0.5).mean() * 100],
        'Percent_Greater_Than_0_9': [(df_metric['Deviation'] > 0.9).mean() * 100]
    }).round(2)

    # Append the all countries row
    summary = pd.concat([summary, all_countries], ignore_index=True)
    summary_tables[metric] = summary

# Display the summary tables
for metric, table in summary_tables.items():
    print(f"\nSummary for Metric: {metric}")
    print(table)









# print a histogram of cluster allocations
plt.hist(labels, bins = N_Clusters)
plt.title('Cluster allocations')
plt.xlabel('Cluster')
plt.ylabel('Number of countries')
plt.show()


# Plot the dendrogram with appropriate colours
from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(20, 10))
dendrogram(linkage_matrix, labels=iso_code_deaths, color_threshold=linkage_matrix[-N_Clusters, 2])
plt.title('Dendrogram of countries based on COVID deaths')
plt.show()



# Pivot the DataFrame to have metrics as columns
df_pivot = df_fitted.pivot(index='Country', columns='Metric', values=['A', 'B', 'C', 'Deviation']).reset_index()
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
df_pivot = df_pivot.rename(columns={'Country_': 'Country', 'Cluster_': 'Cluster'})

# Merge the pivoted DataFrame with the cluster labels
df_pivot = df_pivot.merge(df_fitted[['Country', 'Cluster']], on='Country')

# Define the metrics lists
metrics_A = [col for col in df_pivot.columns if col.startswith('A_')]
metrics_B = [col for col in df_pivot.columns if col.startswith('B_')]
metrics_C = [col for col in df_pivot.columns if col.startswith('C_')]
metrics_Deviation = [col for col in df_pivot.columns if col.startswith('Deviation_')]

resilience_metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Disruption_Duration', 'Time_to_Next_Wave']

# Devise new metrics based on A, B, C, and Deviation
new_metrics = []
i = 0
for metric_A, metric_C in zip(metrics_A, metrics_C):
    resilience_metric_name = resilience_metrics[i]
    new_metric_sum = f"(A_plus_C)_{resilience_metric_name}"
    new_metric_ratio = f"(C_div_A_plus_C)_{resilience_metric_name}"
    df_pivot[new_metric_sum] = df_pivot[metric_A] + df_pivot[metric_C]
    df_pivot[new_metric_ratio] = df_pivot[metric_C] / (df_pivot[metric_A] + df_pivot[metric_C])
    new_metrics.extend([new_metric_sum, new_metric_ratio])
    i += 1

# Reorder columns by resilience metric
ordered_columns = ['Country', 'Cluster']

for metric in resilience_metrics:
    ordered_columns.extend([col for col in df_pivot.columns if metric in col and not col.startswith('(A_plus_C)') and not col.startswith('(C_div_A_plus_C)')])
    ordered_columns.extend([col for col in df_pivot.columns if metric in col and (col.startswith('(A_plus_C)') or col.startswith('(C_div_A_plus_C)'))])

df_pivot = df_pivot[ordered_columns]

# Remove duplicate rows for each country
df_pivot = df_pivot.drop_duplicates(subset=['Country'])

# Resilience metrics list for plotting
resilience_metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Disruption_Duration', 'Time_to_Next_Wave']

# COLORSSSSS
color_scheme =  sns.color_palette("tab10", N_Clusters)
color_scheme_RGBA = color_scheme

for resilience_metric in resilience_metrics:
    print(f"Processing resilience metric: {resilience_metric}")
    
    # Filter metrics for the current resilience metric
    current_metrics_A = [col for col in metrics_A if resilience_metric in col]
    current_metrics_B = [col for col in metrics_B if resilience_metric in col]
    current_metrics_C = [col for col in metrics_C if resilience_metric in col]
    current_metrics_Deviation = [col for col in metrics_Deviation if resilience_metric in col]
    current_new_metrics = [col for col in new_metrics if resilience_metric in col]
    print(current_new_metrics)

    # Combine all metrics for the current resilience metric
    current_metrics = current_metrics_A + current_metrics_B + current_metrics_C + current_metrics_Deviation + current_new_metrics
    print(np.array(current_metrics).T)
    
    print(df_pivot.head())
    
    # now, add the new metrics to the dataframe
    df_pivot['A_plus_C'] = df_pivot[current_metrics_A].sum(axis=1)
    df_pivot['C_div_A_plus_C'] = df_pivot[current_metrics_C].sum(axis=1) / df_pivot['A_plus_C']
    
    for metric in current_metrics:   
        print(f"Processing metric: {metric}")
        
        # Test for normality using Shapiro-Wilk test
        normality_results = {}
        for cluster in df_pivot['Cluster'].unique():
            cluster_data = df_pivot[df_pivot['Cluster'] == cluster][metric].dropna()  # Drop NaN values
            stat, p_value = shapiro(cluster_data)
            normality_results[cluster] = p_value
        
        # Print normality test results
        print(f"Normality test results for {metric}:")
        for cluster, p_value in normality_results.items():
            print(f"Cluster {cluster}: p-value={p_value:.3f}")
        
        # Perform Kruskal-Wallis test on all clusters at the same time
        cluster_data_list = [df_pivot[df_pivot['Cluster'] == cluster][metric].dropna() for cluster in df_pivot['Cluster'].unique()]
        stat, p_value = kruskal(*cluster_data_list)
        print(f"Kruskal-Wallis test result for {metric}: p-value={p_value:.3f}")
                
        # Perform Mann-Whitney U test on all pairs of clusters and collect significant ones
        clusters = sorted(df_pivot['Cluster'].unique())
        significant_pairs = []

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster_i_data = df_pivot[df_pivot['Cluster'] == clusters[i]][metric].dropna()
                cluster_j_data = df_pivot[df_pivot['Cluster'] == clusters[j]][metric].dropna()
                stat, p_value = mannwhitneyu(cluster_i_data, cluster_j_data)
                if p_value < 0.05:
                    significant_pairs.append((clusters[i], clusters[j]))

        # Format the annotation text
        from collections import defaultdict
        summary = defaultdict(list)
        for a, b in significant_pairs:
            summary[a].append(str(b))

        annotation_lines = []
        for a in sorted(summary):
            annotation_lines.append(f"Cluster {a} vs {','.join(summary[a])}")
        annotation_text = "p-value < 0.05 for " + "; ".join(annotation_lines)

        # Plot
        plt.figure(figsize=(10, 6))
        
        sns.boxplot(x='Cluster', y=metric, data=df_pivot, palette=color_scheme)
        
        
# # AAA Define countries to highlight and their unique markers AAA
#         highlight_countries = {
#             'IDN': {'marker': 'H', 'label': 'Indonesia'},
#             'PRT': {'marker': 'D', 'label': 'Portugal'},
#             'ECU': {'marker': 'P', 'label': 'Ecuador'},
#             'ISL': {'marker': '*', 'label': 'Iceland'},
#             'BRA': {'marker': '^', 'label': 'Brazil'},
#             'IRN': {'marker': 'h', 'label': 'Iran'},
#             'ZAF': {'marker': 'v', 'label': 'South Africa'},       
#             'MEX': {'marker': 'X', 'label': 'Mexico'},
#             'BGR': {'marker': 'o', 'label': 'Bulgaria'},
#             'EST': {'marker': 's', 'label': 'Estonia'},
#         }

#         # Overlay highlighted countries
#         for iso_code, style in highlight_countries.items():
#             if iso_code in df_pivot['Country'].values:
#                 row = df_pivot[df_pivot['Country'] == iso_code]
#                 cluster = row['Cluster'].values[0]
#                 y_val = row[metric].values[0]
#                 plt.scatter(x=cluster - 1, y=y_val, color='black', s=100, marker=style['marker'], label=style['label'])

#         # Avoid duplicate legend entries
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         plt.legend(by_label.values(), by_label.keys(), loc='best')
# # AAA end of added countries points AAA
        
        
        # Set x-axis ticks to just show cluster numbers
        plt.xticks(ticks=range(len(clusters)), 
                   labels=[str(c) for c in sorted(clusters)], fontsize = 17)

        # Add annotation text to the plot
        plt.gcf().text(0.5, 0.9, annotation_text, ha='center', 
                       fontsize=17, wrap=True)

        
        # if the metric is B, set the y-axis to log scale
        if metric.startswith('B_'):
            plt.yscale('symlog', linthresh=1e-3)
            y_ticks = [-1e-3, -1e-2, -1e-1, -1, 0, 1e-3, 1e-2, 1e-1, 1]
            plt.yticks(ticks=y_ticks, fontsize = 14) 
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # plt.title(f'Boxplot of {metric} by Cluster')
        plt.yticks(fontsize = 14)
        plt.xlabel('Cluster', fontsize = 15)
        plt.ylabel(metric, fontsize = 17)
                 
        plt.show()


### RUN TO THIS LINE ###







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the main DataFrame
df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# Load the filtered ISO codes from the previously saved CSV file
# Load the filtered ISO codes from the CSV file
iso_codes_filtered = np.array(pd.read_csv('iso_codes_3_or_more_waves.csv', header=None)[0].tolist())[1:]

# Filter the main DataFrame to include only countries in the filtered ISO code list
df_filtered = df[df.index.isin(iso_codes_filtered)]

# Define resilience metrics and their labels
resilience_metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']
metric_labels = ['MPL', 'TPL', 'DD', 'TTND']

# Set plot style
sns.set(style="whitegrid")

# Create histograms for each resilience metric using the filtered DataFrame
for metric, label in zip(resilience_metrics, metric_labels):
    metric_df = df_filtered[df_filtered['Metric'] == metric]
    deviations = metric_df['Deviation'].dropna()
    count = deviations.shape[0]

    plt.figure(figsize=(8, 5))
    sns.histplot(deviations, bins=20, kde=False, color='skyblue')
    plt.title(f'Histogram of R² Values for {label} ({count} values)', fontsize=14)
    plt.xlabel('R² Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()













import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DataFrame from the pickle file
df = pd.read_pickle('Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl')

# Define resilience metrics
resilience_metrics = ['Max_Performance_Loss', 'Total_Performance_Loss', 'Recovery_Speed', 'Time_to_Next_Wave']
metric_labels = ['MPL', 'TPL', 'DD', 'TTND']

# Set plot style
sns.set(style="whitegrid")

# Create histograms for each resilience metric
for metric, label in zip(resilience_metrics, metric_labels):
    # Filter the DataFrame for the current metric
    metric_df = df[df['Metric'] == metric]
    
    # Extract deviation values and drop NaNs
    deviations = metric_df['Deviation'].dropna()
    count = deviations.shape[0]
    
    # Plot histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(deviations, bins=20, kde=False, color='skyblue')
    plt.title(f'Histogram of R² Values for {label} ({count} values)', fontsize=14)
    plt.xlabel('R² Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()
















# # plot, per resilience metric, the B versus the A+C
# for resilience_metric in resilience_metrics:
#     print(f"Processing resilience metric: {resilience_metric}")
    
#     # Filter metrics for the current resilience metric
#     current_metrics_A = [col for col in metrics_A if resilience_metric in col]
#     current_metrics_B = [col for col in metrics_B if resilience_metric in col]
#     current_metrics_C = [col for col in metrics_C if resilience_metric in col]
#     current_metrics_Deviation = [col for col in metrics_Deviation if resilience_metric in col]
#     current_new_metrics = [col for col in new_metrics if resilience_metric in col]
    
#     # Combine all metrics for the current resilience metric
#     current_metrics = current_metrics_A + current_metrics_B + current_metrics_C + current_metrics_Deviation + current_new_metrics
#     print(np.array(current_metrics).T)
    
#     a_list = df_pivot[current_metrics_A]
#     b_list = df_pivot[current_metrics_B]
#     c_list = df_pivot[current_metrics_C]
    
#     for i in range(len(df_pivot)):
#         a = a_list.iloc[i].values
#         b = b_list.iloc[i].values
#         c = c_list.iloc[i].values
        
#         x = np.linspace(0, 1200, 1200)
#         y = a + c
        
#         # color according to the cluster color scheme
#         plt.scatter(y, b, color = color_scheme[df_pivot['Cluster'].iloc[i]-1], alpha = 0.7)
    
#     plt.title(f'B versus A+C for {resilience_metric}')
#     plt.xlabel('A+C')
#     plt.ylabel('B')
#     plt.yscale('symlog', linthresh=1e-4)
#     plt.show()



# # Plot fitted exponential curves divided by (A+C) for each resilience metric
# # so that the curves are normalised by the maximum value of the curve. The curve function is f(x) = A * exp(-B * x) + C
# def norm_exp_function(x, A, B, C):
#     return (A * np.exp(-B * x) + C) / (A + C)

# for resilience_metric in resilience_metrics:
#     print(f"Processing resilience metric: {resilience_metric}")
    
#     # Filter metrics for the current resilience metric
#     current_metrics_A = [col for col in metrics_A if resilience_metric in col]
#     current_metrics_B = [col for col in metrics_B if resilience_metric in col]
#     current_metrics_C = [col for col in metrics_C if resilience_metric in col]
#     current_metrics_Deviation = [col for col in metrics_Deviation if resilience_metric in col]
#     current_new_metrics = [col for col in new_metrics if resilience_metric in col]
    
#     # Combine all metrics for the current resilience metric
#     current_metrics = current_metrics_A + current_metrics_B + current_metrics_C + current_metrics_Deviation + current_new_metrics
#     print(np.array(current_metrics).T)
    
#     a_list = df_pivot[current_metrics_A]
#     b_list = df_pivot[current_metrics_B]
#     c_list = df_pivot[current_metrics_C]
    
#     for i in range(len(df_pivot)):
#         a = a_list.iloc[i].values
#         b = b_list.iloc[i].values
#         c = c_list.iloc[i].values
        
#         x = np.linspace(0, 1200, 1200)
#         y = norm_exp_function(x, a, b, c)
        
#         # color according to the cluster color scheme
#         plt.plot(x, y, color = color_scheme[df_pivot['Cluster'].iloc[i]-1], alpha = 0.7, 
#                  linewidth = 2.5)
    
#     plt.title(f'Fitted exponential curves normalised by (A+C) for {resilience_metric}')
#     plt.xlabel('Time (days)')
#     plt.ylabel('f(x)')
#     # ymax and ymin are 100 and 0
#     plt.ylim(0, 2)
#     # plt.yscale('symlog', linthresh=1e-3)
    
#     plt.show()
    

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Load the world map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Add the iso codes to df_deaths
iso_code_deaths = pd.DataFrame(iso_code_deaths, columns=['iso_code'])
# Add the clusters to the iso codes
iso_code_deaths['Cluster'] = labels

# Merge the iso codes with the world map data
world = world.merge(iso_code_deaths, left_on='iso_a3', right_on='iso_code', how='left')
# world = world.to_crs('+proj=eqearth')


# Filter out Antarctica and countries with no population data
world = world[(world.pop_est > 0) & (world.name != "Antarctica")]

# Define the color scheme for clusters
cmap = ListedColormap(color_scheme.as_hex())

# Plot the world map with cluster colors
fig, ax = plt.subplots(1, 1, figsize=(30, 30))
world.boundary.plot(ax=ax, linewidth=.3, color='black')  # Change border color to black
world.plot(column='Cluster', ax=ax, legend=True, categorical=True,
           cmap=cmap, edgecolor='black', linewidth=0.2)
plt.title('Clusters of countries based on COVID deaths')
plt.show()





# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import ListedColormap
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # Load the world map data
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# # Add the iso codes to df_deaths
# iso_code_deaths = pd.DataFrame(iso_code_deaths, columns=['iso_code'])
# # Add the clusters to the iso codes
# iso_code_deaths['Cluster'] = labels

# # Merge the iso codes with the world map data
# world = world.merge(iso_code_deaths, left_on='iso_a3', right_on='iso_code', how='left')
# world = world.to_crs('+proj=eqearth')

# # Filter out Antarctica and countries with no population data
# world = world[(world.pop_est > 0) & (world.name != "Antarctica")]

# # Define the color scheme for clusters
# cmap = ListedColormap(color_scheme.as_hex())

# # Plot the world map with cluster colors
# fig, ax = plt.subplots(1, 1, figsize=(30, 30), subplot_kw={'projection': ccrs.EqualEarth()})
# ax.set_global()

# # Add map boundaries and latitude/longitude lines
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# # Plot the world map with cluster colors
# world.boundary.plot(ax=ax, linewidth=0.3, color='black', transform=ccrs.PlateCarree())  # Change border color to black
# world.plot(column='Cluster', ax=ax, legend=True, categorical=True,
#            cmap=cmap, edgecolor='black', linewidth=0.2, transform=ccrs.PlateCarree())
# plt.title('Clusters of countries based on COVID deaths')
# plt.show()






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import csv
import seaborn as sns


def read_data(folder):
    """
    This function reads the data from the folder containing the data sets.
    
    Folder: string containing the folder name
    """
    
    # Read all the .csv files in the folder
    files = glob.glob('*.csv')
    csv_list = []
    iso_code_list = []
    for file in files:
        df = pd.read_csv(file)
        # print(df.head())
        # remove the iso_code column, and make it a seperate list
        iso_code = df['iso_code'].tolist()
        df = df.drop('iso_code', axis = 1)
        csv_list.append(df)
        iso_code_list.append(iso_code)
    return csv_list, iso_code_list

# df_list, iso_codes_list = read_data('Cluster_Data')
# df_deaths = df_list[0]
# iso_codes_deaths = iso_codes_list[0]

# r'C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data'

# 'C:/Users/fpvansteijn/OneDrive - Delft University of Technology/Documents/Scripts/TBCOV data set/Cluster_Data/data_new_deaths_smoothed_per_million.csv'

# try:
#     with open(r'C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data', mode='r', newline='') as file:
#            reader = csv.reader(file)
#            for row in reader:
#                print(row)
# except FileNotFoundError:
#     print("The file does not exist")
# except Exception as e:
#        print("An error occurred:", e)



# try:
#     os.chdir(os.path.join(os.getcwd(), '/fpvansteijn/OneDrive - Delft University of Technology/Documents/Scripts/TBCOV data set/Cluster_Data')) # '.' if the path is to current folder
#     print(os.getcwd())
# except:
#     pass



# Open the data set containing the daily deaths
# which is at C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data
#os.chdir("C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data")
#os.chdir("C:/Users/fpvansteijn/OneDrive - Delft University of Technology/Documents/Scripts/TBCOV data set/Cluster_Data/data_new_deaths_smoothed_per_million.csv")
# df_deaths = pd.read_csv('C:/Users/fpvansteijn/OneDrive - Delft University of Technology/Documents/Scripts/TBCOV data set/Cluster_Data/data_new_deaths_smoothed_per_million.csv')

df_deaths = pd.read_csv(r'c:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\data_new_deaths_smoothed_per_million.csv')
iso_code_deaths = df_deaths['iso_code'].tolist()
df_deaths = df_deaths.drop('iso_code', axis = 1)

# Do hierarchical clustering on the country daily deaths data
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

array_deaths = df_deaths.iloc[:,1:].to_numpy()
#delete the rows with Nan values at 139 and 191
array_deaths = np.delete(array_deaths, [139, 191], axis = 0)
array_deaths_no_norm = array_deaths.copy()
iso_code_deaths = [iso_code for idx, iso_code in enumerate(iso_code_deaths) if idx != 139 and idx != 191]


# is there any element that is equal to Nan in the data? print the index of the element, and the column and row in which it is located
print(np.isnan(array_deaths).any())
print(np.argwhere(np.isnan(array_deaths)))



# # # Now, we Normalise the data by taking the highest value for each country to be equal to 1, the lowest value to be equal to 0, and the rest of the values to be in between 0 and 1
for i in range(array_deaths.shape[0]):
    # if i == 139 or i == 191:
    #     continue
    array_deaths[i] = (array_deaths[i] - np.min(array_deaths[i])) / (np.max(array_deaths[i]) - np.min(array_deaths[i]))
    
print(np.isnan(array_deaths).any())
print(np.argwhere(np.isnan(array_deaths[:,0])))


# remove features with the same value for all countries
# array_deaths = array_deaths[:, np.std(array_deaths, axis = 0) != 0]

# Calculate the distance matrix
dist = pdist(array_deaths, metric = 'euclidean')
# dist = pdist(array_deaths, metric = 'correlation')


# Perform hierarchical clustering
linkage_matrix = linkage(dist, method = 'ward')
# linkage_matrix = linkage(dist, method = 'average')





# # save the linkage matrix to a .csv file
# np.savetxt(r'C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\linkage_matrix_deaths.csv', linkage_matrix, delimiter = ',')
# # save the iso codes to a .csv file
# np.savetxt(r'C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\iso_codes_deaths.csv', iso_code_deaths, delimiter = ',', fmt = '%s')


# Plot the dendrogram
plt.figure(figsize = (50, 40))
dendrogram(linkage_matrix, labels = iso_code_deaths)
plt.xticks(rotation = 70, fontsize = 12)
plt.show()
# plt.savefig('dendrogram_deaths.png')

# Plot the members of the clusters in a scatterplot, based on the hierarchical clustering
from scipy.cluster.hierarchy import fcluster
N_Clusters = 5
labels = fcluster(linkage_matrix, N_Clusters, criterion = 'maxclust')
# labels = fcluster(linkage_matrix, 20, criterion = 'distance')
# labels = fcluster(linkage_matrix, N_Clusters, criterion = 'maxclust_monocrit')

# ADD labels to the dataframe

# df_deaths['Cluster'] = labels

# # NOW save the dataframe to a .csv file
# df_deaths.to_csv(r'C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\data_new_deaths_smoothed_per_million_clustered_WITH_4_CLUSTERS.csv', index = False)




# make a color scheme, dependent on N_Clusters, such that in every figure the same cluster has the same color
color_scheme = plt.cm.jet(np.linspace(0, 1, N_Clusters))
color_scheme_RGBA = color_scheme[:, :3]


# import matplotlib
# # import a colorscheme from colorbrewer: https://colorbrewer2.org/?type=diverging&scheme=RdYlBu&n=5
# color_scheme = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027']
# color_scheme_RGBA = [matplotlib.colors.to_rgba(color) for color in color_scheme]


# PRINTING THE CLUSTERS TIME SERIES 
# based on these labels, make a plot for every cluster, where the mean of the cluster is highlighted
# Day 1 is 8 january 2020, show dates on the x-axis, and deaths per million on the y-axis. Fir example, just the start of every 3rd month, starting on 1 feb 2020 


import matplotlib.dates as mdates
import locale

# Ensure English locale for month names
try:
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
except locale.Error:
    pass  # If the locale is not available, skip setting it


# Date range starting from Jan 8, 2020
date = pd.date_range(start='2020-01-08', periods=array_deaths.shape[1], freq='D')

# Define tick positions starting from Jan 1, 2020 every 3 months
tick_start = pd.Timestamp('2020-01-01')
tick_end = date[-1]
tick_dates = pd.date_range(start=tick_start, end=tick_end, freq='3MS')

highlight_countries = ['BGR', 'EST']



for i in range(N_Clusters):
    cluster_index = i + 1
    idx_cluster = np.where(labels == cluster_index)[0]
    
    # if the country is bulgaria 
    
    array_deaths_cluster = array_deaths[idx_cluster]

    plt.figure(figsize=(10, 6))
    plt.plot(array_deaths_cluster.T, color=color_scheme[i], alpha=0.15)
    plt.plot(np.mean(array_deaths_cluster, axis=0).T, color=color_scheme[i], alpha=1, linewidth=3.5)

    plt.title(f'Cluster {cluster_index}', fontsize=20)
    # plt.xlabel('Date', fontsize=14)
    plt.ylabel('Fatalities per million (normalised)', fontsize=16)

    plt.xticks(ticks=[(d - date[0]).days for d in tick_dates if d >= date[0]],
               labels=[d.strftime('%b %Y') for d in tick_dates if d >= date[0]],
               rotation=45, fontsize=11)

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Assuming array_deaths, labels, iso_code_deaths, and date are already defined
# Define the ISO codes to highlight
highlight_countries = ['BGR', 'EST']

# Number of clusters
N_Clusters = len(set(labels))

# Define color scheme
color_scheme = sns.color_palette("tab10", N_Clusters)

# Define tick positions starting from Jan 1, 2020 every 3 months
tick_start = pd.Timestamp('2020-01-01')
tick_end = date[-1]
tick_dates = pd.date_range(start=tick_start, end=tick_end, freq='3MS')

# Plot each cluster
for i in range(N_Clusters):
    cluster_index = i + 1
    idx_cluster = np.where(labels == cluster_index)[0]
    array_deaths_cluster = array_deaths[idx_cluster]
    iso_cluster = [iso_code_deaths[idx] for idx in idx_cluster]

    plt.figure(figsize=(10, 6))
    plt.plot(array_deaths_cluster.T, color=color_scheme[i], alpha=0.15)
    plt.plot(np.mean(array_deaths_cluster, axis=0).T, color=color_scheme[i], alpha=1, linewidth=3.5)

    # Highlight Bulgaria and Estonia
    for country_code in highlight_countries:
        if country_code in iso_cluster:
            idx = iso_cluster.index(country_code)
            plt.plot(array_deaths_cluster[idx], color='black', linewidth=1, label=country_code)

    plt.title(f'Cluster {cluster_index}', fontsize=20)
    plt.ylabel('Fatalities per million (normalised)', fontsize=16)
    plt.xticks(ticks=[(d - date[0]).days for d in tick_dates if d >= date[0]],
               labels=[d.strftime('%b %Y') for d in tick_dates if d >= date[0]],
               rotation=45, fontsize=11)
    plt.tight_layout()
    plt.legend()
    plt.show()

















date = pd.date_range(start='2020-01-08', periods=array_deaths.shape[1], freq='D')

for i in range(N_Clusters):
    i = i + 1
    
    # idx_cluster = np.where(labels == i)[0]
    # array_deaths_cluster_no_norm = array_deaths_no_norm[idx_cluster]
    # plt.plot(array_deaths_cluster_no_norm.T, color = color_scheme[i-1], alpha = 0.1)
    # plt.plot(np.mean(array_deaths_cluster_no_norm, axis = 0).T, color = color_scheme[i-1], alpha = 1, linewidth = 3)
    
    idx_cluster = np.where(labels == i)[0]
    array_deaths_cluster = array_deaths[idx_cluster]
    plt.plot(array_deaths_cluster.T, color = color_scheme[i-1], alpha = 0.15)
    plt.plot(np.mean(array_deaths_cluster, axis = 0).T, color = color_scheme[i-1], alpha = 1, linewidth = 3.5)
    
    plt.title('Cluster ' + str(i))
    plt.xlabel('Date')
    plt.xticks(ticks=np.arange(0, len(date), 60), labels=date[::60].strftime('%Y-%m-%d'), rotation=45)
    
    plt.ylabel('Deaths per million (normalised)')
    # rotate the x-axis labels
    plt.xticks(rotation = 45)
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.show()
    

# open Final_Analysis_with_Exponential_Fits_and_Metrics_no_bounds.pkl, containing the exponential fits and the metrics
import pickle
with open(r'c:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl', 'rb') as f:
    df_fitted = pickle.load(f)
    
print(df_fitted.head(10)) 
    


import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import shapiro, kruskal, mannwhitneyu

# Load the data
with open(r'c:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\Updated_Optimal_Fitted_Wave_Metrics_with_y400_y800.pkl', 'rb') as f:
    df_fitted = pickle.load(f)

# Load the cluster labels
df_deaths = pd.read_csv(r'c:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\data_new_deaths_smoothed_per_million.csv')
iso_code_deaths = df_deaths['iso_code'].tolist()
df_deaths = df_deaths.drop('iso_code', axis=1)

# Perform hierarchical clustering
array_deaths = df_deaths.iloc[:, 1:].to_numpy()
array_deaths = np.delete(array_deaths, [139, 191], axis=0)
iso_code_deaths = [iso_code for idx, iso_code in enumerate(iso_code_deaths) if idx != 139 and idx != 191]

for i in range(array_deaths.shape[0]):
    array_deaths[i] = (array_deaths[i] - np.min(array_deaths[i])) / (np.max(array_deaths[i]) - np.min(array_deaths[i]))

dist = pdist(array_deaths, metric='euclidean')
linkage_matrix = linkage(dist, method='ward')
N_Clusters = 5
labels = fcluster(linkage_matrix, N_Clusters, criterion='maxclust')

# Add cluster labels to the fitted DataFrame
df_fitted['Cluster'] = df_fitted['Country'].map(dict(zip(iso_code_deaths, labels)))

# Pivot the DataFrame to have metrics as columns
df_pivot = df_fitted.pivot(index='Country', columns='Metric', values=['A', 'B', 'C', 'Deviation']).reset_index()
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
df_pivot = df_pivot.rename(columns={'Country_': 'Country', 'Cluster_': 'Cluster'})

# Merge the pivoted DataFrame with the cluster labels
df_pivot = df_pivot.merge(df_fitted[['Country', 'Cluster']], on='Country')

# Define the metrics lists
metrics_A = [col for col in df_pivot.columns if col.startswith('A_')]
metrics_B = [col for col in df_pivot.columns if col.startswith('B_')]
metrics_C = [col for col in df_pivot.columns if col.startswith('C_')]
metrics_Deviation = [col for col in df_pivot.columns if col.startswith('Deviation_')]

# Devise new metrics based on A, B, C, and Deviation
new_metrics = []
for metric_A, metric_C in zip(metrics_A, metrics_C):
    resilience_metric_name = metric_A.split('_')[1]
    new_metric_sum = f"(A_plus_C)_{resilience_metric_name}"
    new_metric_ratio = f"(C_div_A_plus_C)_{resilience_metric_name}"
    df_pivot[new_metric_sum] = df_pivot[metric_A] + df_pivot[metric_C]
    df_pivot[new_metric_ratio] = df_pivot[metric_C] / (df_pivot[metric_A] + df_pivot[metric_C])
    new_metrics.extend([new_metric_sum, new_metric_ratio])


for metric in metrics_A + metrics_B + metrics_C + metrics_Deviation + new_metrics:   
    
    # Test for normality using Shapiro-Wilk test
    normality_results = {}
    for cluster in df_pivot['Cluster'].unique():
        cluster_data = df_pivot[df_pivot['Cluster'] == cluster][metric].dropna()  # Drop NaN values
        stat, p_value = shapiro(cluster_data)
        normality_results[cluster] = p_value
    
    # Print normality test results
    print(f"Normality test results for {metric}:")
    for cluster, p_value in normality_results.items():
        print(f"Cluster {cluster}: p-value={p_value:.3f}")
    
    # Perform Kruskal-Wallis test on all clusters at the same time
    cluster_data_list = [df_pivot[df_pivot['Cluster'] == cluster][metric].dropna() for cluster in df_pivot['Cluster'].unique()]
    stat, p_value = kruskal(*cluster_data_list)
    print(f"Kruskal-Wallis test result for {metric}: p-value={p_value:.3f}")
    
    # Perform Mann-Whitney U test on all pairs of clusters and print p-values
    clusters = df_pivot['Cluster'].unique()
    print(f"Mann-Whitney U test results for {metric}:")
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            cluster_i_data = df_pivot[df_pivot['Cluster'] == clusters[i]][metric].dropna()  # Drop NaN values
            cluster_j_data = df_pivot[df_pivot['Cluster'] == clusters[j]][metric].dropna()  # Drop NaN values
            stat, p_value = mannwhitneyu(cluster_i_data, cluster_j_data)
            print(f"p({clusters[i]}-{clusters[j]})={p_value:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=metric, data=df_pivot)
    plt.title(f'Boxplot of {metric} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(metric)
    plt.show()







    
# plot the countries on the world map, with the colors of the clusters. Use the iso_codes_deaths list
import geopandas as gpd 
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# add the iso codes to df_deaths
iso_code_deaths = pd.DataFrame(iso_code_deaths, columns = ['iso_code'])
# add the clusters to the iso codes
iso_code_deaths['Cluster'] = labels

# # also add the data from Learning_data.csv to the iso_code_deaths dataframe
# df_learning = pd.read_csv(r'c:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\Learning_data.csv')
# iso_code_deaths['Unlearning_Occasions'] = df_learning['Unlearning_Occasions_Height']
# iso_code_deaths['Consecutive_Learning'] = df_learning['Consecutive_Learning_Height']
# iso_code_deaths['Additive_Method'] = df_learning['Additive_Height']
# iso_code_deaths['Multiplicative_Method'] = df_learning['Multiplicative_Height']

# # make a histogram for each metric in the iso_code_deaths dataframe
# for column in iso_code_deaths.columns[1:]:
#     plt.hist(iso_code_deaths[column], bins = 20)
#     plt.title(column)
#     plt.show()






# make sure that the iso codes are in the same order as the countries in the world map
world = world.merge(iso_code_deaths, left_on='iso_a3', right_on='iso_code', how='left')

# plot the world map
world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
# make the world map with colours from the colour_scheme
world.plot(column = 'Cluster', legend = False, figsize = (20, 20), edgecolor = 'black', linewidth = 0.5, 
        #    cmap = matplotlib.colors.ListedColormap(color_scheme_RGBA[:N_Clusters]),
           cmap = plt.cm.jet, 
           alpha = 1)
# Add a legend where the clusters are explained, but not in a bar, but in a list

plt.title('Clusters of countries based on COVID deaths')
plt.show()


# plt.savefig('world_map_deaths_NORMALISED_' + str(N_Clusters) + '_clusters.png')
# np.savetxt(r'C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\linkage_matrix_deaths_NORMALISED_' + str(N_Clusters) + '_clusters.csv', linkage_matrix, delimiter = ',')
# # now save the iso codes and clusters to a .csv file
# iso_code_deaths.to_csv(r'C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\iso_codes_deaths_NORMALISED_' + str(N_Clusters) + '_clusters.csv', index = False)


# # plot the Learning data on the world map, with a good color scheme for continuous data (not cm.jet)
# # choose a different projection type, because the world map is not very clear
# world.plot(column = 'Unlearning_Occasions', legend = False, figsize = (20, 20), edgecolor = 'black', linewidth = 0.5,
#               cmap = 'plasma_r', alpha = 1)

# plt.title('Unlearning occasions of countries')
# plt.show()

# # plot the Learning data on the world map, with a good color scheme for continuous data (not cm.jet)
# world.plot(column = 'Consecutive_Learning', legend = False, figsize = (20, 20), edgecolor = 'black', linewidth = 0.5,
#                 cmap = 'plasma', alpha = 1)
# plt.title('Consecutive learning of countries')
# plt.show()

# world.plot(column = 'Additive_Method', legend = False, figsize = (20, 20), edgecolor = 'black', linewidth = 0.5,
#                 cmap = 'plasma', alpha = 1) 
# plt.title('Additive method of countries')
# plt.show()

# # world.plot(column = 'Multiplicative_Method', legend = False, figsize = (20, 20), edgecolor = 'black', linewidth = 0.5,
# #                 cmap = 'plasma', alpha = 1) 
# # plt.title('Multiplicative method of countries')
# # plt.show()






    
# # open the Lognormal_Fitted_Curves_Deaths_N11.pkl file
# import pickle
# with open(r'C:\Users\fpvansteijn\OneDrive - Delft University of Technology\Documents\Scripts\TBCOV data set\Cluster_Data\Lognormal_Fitted_Curves_Deaths_N11.pkl', 'rb') as f:
#     df_fitted = pickle.load(f)

# df_R2 = df_fitted['R2']
# df_moments = df_fitted['Moments']

# # for each country, get the suboptimal fits
# N_suboptimal_list = []
# for country_idx in range(len(df_R2)):
#     R2_list = df_R2.iloc[country_idx]
    
#     if R2_list == 'Not enough data':
#         N_suboptimal_list.append('Not enough data')
#         continue
    
#     max_R2_idx = np.argmax(R2_list)
#     # Now get the indices of the suboptimal fits
#     for R2_idx in range(len(R2_list)):
#         if R2_list[max_R2_idx] - R2_list[R2_idx] < 0.01:
#             N_suboptimal_list.append(R2_idx+2)
#             break


# # first, from the N_suboptimal_list, get the indices of the suboptimal fits, by substracting every element from 2
# # make an exception for the 'Not enough data' entries, then make the entry 'Not enough data'
# idx_suboptimal_list = []
# for N_suboptimal in N_suboptimal_list:
#     if N_suboptimal == 'Not enough data':
#         idx_suboptimal_list.append('Not enough data')
#     else:
#         idx_suboptimal_list.append(N_suboptimal - 2)


# # get the moments of the suboptimal fits
# moments_suboptimal_list = []
# for country_idx in range(len(df_moments)):
#     moments_list = df_moments.iloc[country_idx]
    
#     if moments_list == 'Not enough data':
#         moments_suboptimal_list.append('Not enough data')
#         continue
    
#     suboptimal_idx = idx_suboptimal_list[country_idx]
    
#     moments_suboptimal_list.append(moments_list[suboptimal_idx])


# # for every cluster, make a boxplot of the amount of N suboptimal fits, and put it all in one plot
# plt.figure(figsize = (10, 10))

# N_for_all_clusters = []
# for i in range(N_Clusters):
#     i = i + 1
#     idx_cluster = np.where(labels == i)[0]
#     N_suboptimal_cluster = [N_suboptimal_list[idx] for idx in idx_cluster]
#     # remove the 'Not enough data' entries
#     N_suboptimal_cluster = [N_suboptimal for N_suboptimal in N_suboptimal_cluster if N_suboptimal != 'Not enough data']
#     print(N_suboptimal_cluster)

#     N_for_all_clusters.append(N_suboptimal_cluster)

# sns.boxplot(data = N_for_all_clusters, palette = color_scheme_RGBA)
# plt.title('Amount of COVID deaths curves according to the suboptimal rule')

# # add, for every cluster, the amount of data points in the cluster
# N_data_points = [len(np.where(labels == i)[0]) for i in range(1, N_Clusters+1)]
# for i in range(N_Clusters):
#     plt.text(i, 10, 'N = ' + str(N_data_points[i]), horizontalalignment = 'center', size = 'medium', color = 'black', weight = 'semibold')
       
# plt.xticks(ticks = range(N_Clusters), labels = range(1, N_Clusters+1))
# plt.yticks(ticks = range(0, 11))
# plt.xlabel('Cluster')
# plt.ylabel('Amount of curves')
# plt.show()


# # DO statistical analysis on whether the amount of suboptimal fits is significantly different between the clusters
# from scipy.stats import kruskal
# H, p = kruskal(*N_for_all_clusters)
# print('H = ', H)
# print('p = ', p)

# # and find pairwise differences in significance between the clusters
# from scipy.stats import mannwhitneyu
# for i in range(N_Clusters):
#     for j in range(i+1, N_Clusters):
#         U, p = mannwhitneyu(N_for_all_clusters[i], N_for_all_clusters[j])
#         print('Cluster ', i+1, ' and ', j+1, ': U = ', U, ' p = ', np.round(p, 3))

# # Now, instead of the amount of suboptimal fits, do the same for the moments of the suboptimal fits
# # start with the average of the means
# mean_moments_suboptimal_list = []
# for moments in moments_suboptimal_list:
#     if moments == 'Not enough data':
#         mean_moments_suboptimal_list.append('Not enough data')
#     else:
#         mean_moments_suboptimal_list.append(np.mean(moments))
        
# # for every cluster, make a boxplot of the average of the means of the moments of the suboptimal fits, and put it all in one plot
# plt.figure(figsize = (10, 10))
# plt.title('Average of the means of the moments of the suboptimal fits')
# plt.ylabel('Average of the means of the moments')
# N_for_all_clusters = []
# for i in range(N_Clusters):
#     i = i + 1
#     idx_cluster = np.where(labels == i)[0]
#     mean_moments_cluster = [mean_moments_suboptimal_list[idx] for idx in idx_cluster]
#     # remove the 'Not enough data' entries
#     mean_moments_cluster = [mean_moments for mean_moments in mean_moments_cluster if mean_moments != 'Not enough data']
#     print(mean_moments_cluster)

#     N_for_all_clusters.append(mean_moments_cluster)

# sns.boxplot(data = N_for_all_clusters, palette = color_scheme_RGBA)
# plt.title('Average of the means of the moments of the suboptimal fits')
# plt.xticks(ticks = range(N_Clusters), labels = range(1, N_Clusters+1))
# plt.xlabel('Cluster')
# plt.ylabel('Average of the means of the moments')
# plt.show()
















# # print countries on the world map
# import geopandas as gpd
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# # make sure that the iso codes are in the same order as the countries in the world map
# world = world.merge(df_deaths, left_on='iso_a3', right_on='iso_code', how='left')

# world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
# # world = world.set_index('iso_a3')

# exclude = ['AND', 'ATG', 'ABW', 'BHR', 'BRB', 'BMU', 'CPV', 'CYM', 'COM', 'CUW', 'DMA', 'FRO', 'GUF', 'PYF', 'GRD', 'GLP', 'GUM', 'GGY', 'IMN', 'JEY', 'KIR', 'MDV', 'MLT', 'MTQ', 'MUS', 'MYT', 'FSM', 'REU', 'LCA', 'VCT', 'WSM', 'STP', 'SYC', 'SGP', 'TON', 'VIR']
# iso_code_deaths = [x for x in iso_code_deaths if x not in exclude]
# world = world.loc[iso_code_deaths]
# world.plot()
# plt.show()
# # # plt.savefig('world_map.png')
