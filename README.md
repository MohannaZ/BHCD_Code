##Finding Overlapping Catchments
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import fnmatch
import pandas as pd
import numpy as np
import itertools
import sys

data_dir = r"data_dir"
result_dir = r"result_dir"
data_dir = str(sys.argv[1])
result_dir = str(sys.argv[2])
result_filename = str(sys.argv[3])

c_shp=[] 
list_files = os.listdir(data_dir) 
for each_file in list_files :  
  if fnmatch.fnmatch(each_file, "*.shp"): 
        print(each_file)
        c_shp.append(each_file) 

c_combos = list(itertools.combinations(c_shp, 2))
no_overlap_result=[]
overlap_result=[]
for c in c_combos: 
   
    file_name_0 = c[0] 
    file_name_1 = c[1] 
    data_0 = gpd.read_file(os.path.join(data_dir, file_name_0)) 
    
    data_1 = gpd.read_file(os.path.join(data_dir, file_name_1))
    data_0 = data_0.to_crs(crs=3857)
    data_1 = data_1.to_crs(crs=3857)
    result = gpd.overlay(data_0, data_1, how='intersection', keep_geom_type=False) 
    result['area'] = result['geometry'].area
    if result.shape[0]> 0:
           overlap_result.append([c[0],c[1],result.area.values[0]]) 
    else:
           no_overlap_result.append([c[0],c[1]]) 
    
df= pd.DataFrame(overlap_result, columns = ['catchment1', 'catchment2', 'area'])
print(df)  
results_full_filename = os.path.join(result_dir, result_filename) 
df.to_csv(results_full_filename, sep=';', index=False)
####END HERE####

#%%
##Boxplot for all variables (Here we considered T as an example)
import matplotlib.pyplot as plt
import pandas as pd
import os

data1 = pd.read_csv("data_dir_Obs")
data2 = pd.read_csv("data_dir_Mixed")
data3 = pd.read_csv("data_dir_GLDAS")
data4 = pd.read_csv("data_dir_ERA5")


datasets = [data1["Mean T (C)"], data2["Mean T (C)"], data3["Mean T (C)"], data4["Mean T (C)"]]
box_colors = ["blue", "brown", "green", "purple"]
median_color = "black"
# Create the boxplot
plt.figure(figsize=(8, 6))
boxplot = plt.boxplot(datasets, patch_artist=True, showfliers=False)  

for patch, color in zip(boxplot['boxes'], box_colors):
    patch.set_facecolor(color)
for median in boxplot['medians']:
    median.set(color=median_color)

plt.ylabel("Mean Temperature (C)", fontsize=20)
labels = ["Obs", "Mixed", "GLDAS", "ERA5"]
plt.xticks(range(1, len(datasets) + 1), labels, fontsize=20)
plt.yticks(fontsize=20)

desktop_path = os.path.expanduser("data_dir")  
file_path = os.path.join(desktop_path, "Global-T.tiff")  
plt.savefig(file_path, dpi=300)
plt.show()
####END HERE####
#%%

##Budyko Plot (Here we consider Obs dataset as an example)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
data =pd.read_csv("data_dir")
PETPratio = data['PET/P']
ETPratio = data['ET/P']
# Plot the data points
plt.scatter(PET/P, ET/P, color='purple', s=8, label='Data Points')
# Define a range of values for potential evapotranspiration (PET) fraction (ET/P)
budyko_curve_x = np.arange(0, 8, 0.05)
energy_limit_x = np.arange(0, 1.0001, 0.05)
x = np.arange(0, 1.0001, 0.05)
budyko_curve_y = np.power((budyko_curve_x*np.tanh(1/budyko_curve_x)*(1-np.exp(-budyko_curve_x))),0.5)
water_limit_y = 1+budyko_curve_x*0
energy_limit_y = energy_limit_x
y = 1 + x*0
plt.plot(budyko_curve_x,budyko_curve_y,  linestyle='--')
plt.plot(energy_limit_y,energy_limit_x, c='black')
plt.plot(budyko_curve_x,water_limit_y,c='black')
plt.plot(y,x,linestyle='-', c='red',label='_nolegend_')
plt.ylabel("ET/P")
plt.xlabel("PET/P")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
desktop_path = os.path.expanduser("data_dir")  
file_path = os.path.join(desktop_path, "Global_budyko_Obs.tiff")  
plt.savefig(file_path, dpi=300)
plt.show()
####END HERE####

#%%
##Latitudinal line plot
import pandas as pd
import matplotlib.pyplot as plt
file_paths = [
    r"data_dir_Obs",
    r" data_dir_Mixed ",
    r" data_dir_GLDAS ",
    r" data_dir_ERA5 "
]
dataset_labels = ["Obs", "Mixed", "GLDAS", "ERA5"]
colors = ["royalblue", "maroon", "olive", "purple"]  
plt.figure(figsize=(12, 6))
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
        catchments = df['Catchment']  
    precipitation = df['Mean P (mm/year)']  
        plt.plot(catchments, precipitation, label=dataset_labels[i], color=colors[i], linewidth=2)
plt.xlabel("Catchments_Latitudinal Order", fontsize=14)
plt.ylabel("Mean Precipitation (mm/year)", fontsize=14)
plt.title("Latitudinal Distribution of Precipitation", fontsize=16)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16, loc="upper right")  
plt.grid(False)
desktop_path = r"desktop_path"
file_path = desktop_path + "\\Precipitation_Latitudinal_Distribution.tiff"
plt.savefig(file_path, dpi=300)
plt.show()
####END HERE####

#%%
##area-weighted regional change trend all in one plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
datasets = {
    "Obs": r"data_dir_Obs",
    "Mixed": r"data_dir_Mixed ",
    "GLDAS": r"data_dir_GLDAS ",
    "ERA5": r"data_dir_ERA5 "
}
plot_colors = ['royalblue', 'maroon', 'olive', 'purple']
shade_colors = ['lightblue', 'pink', 'lightgreen', 'darkorchid']
plt.figure(figsize=(8, 6), facecolor='white')
legend_labels = []  
for i, (name, file_path) in enumerate(datasets.items()):  
    df = pd.read_csv(file_path)
    sns.lineplot(x='Year', y='All', data=df, errorbar='sd', color=plot_colors[i], marker='o', linewidth=2.5)
    plt.fill_between(df['Year'], df['All'] - df['All'].std(), df['All'] + df['All'].std(), color=shade_colors[i], alpha=0.2)
    X = df['Year'].values.reshape(-1, 1)
    y = df['All'].values
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    r_squared = reg.score(X, y)
    x_range = [min(df['Year']), max(df['Year'])]
    plt.plot(x_range, reg.predict([[x] for x in x_range]), color=plot_colors[i], linestyle='--', linewidth=2)
    legend_labels.append(f"{name}: Slope={slope:.2f}, R²={r_squared:.2f}")
plt.legend(legend_labels, loc='upper left', fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-300, 300)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Delta Storage (mm/year)", fontsize=20)
plt.grid(False)
desktop_path = os.path.expanduser(r"data_dir")
file_path = os.path.join(desktop_path, "DS_Combined.tiff")
plt.savefig(file_path, dpi=300)
plt.show()
####END HERE####

#%%
##1st percentile value over datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
dataset_paths = {
    "Obs": r"data_dir_Obs",
    "Mixed": r"data_dir_Mixed",
    "GLDAS": r"data_dir_GLDAS",
    "ERA5": r"data_dir_ERA5"
}
season_mapping = {
    'Jan': 'Winter', 'Feb': 'Winter', 'Mar': 'Spring', 'Apr': 'Spring', 'May': 'Spring',
    'Jun': 'Summer', 'Jul': 'Summer', 'Aug': 'Summer', 'Sep': 'Autumn', 'Oct': 'Autumn', 'Nov': 'Autumn',
    'Dec': 'Winter'
}
def process_dataset(file_path):
    df = pd.read_csv(file_path, header=None)
    years = df.iloc[0, :-1].values
    months = df.iloc[1, :-1].values
    runoff_data = df.iloc[2:, :-1].values
    catchment_ids = df.iloc[2:, -1].values
    years_repeated = np.tile(years, len(catchment_ids))
    months_repeated = np.tile(months, len(catchment_ids))
    catchments_repeated = np.repeat(catchment_ids, len(years))
    runoff_flattened = runoff_data.flatten()
    reshaped_data = pd.DataFrame({
        'Year': years_repeated,
        'Month': months_repeated,
        'Catchment': catchments_repeated,
        'Runoff': runoff_flattened
    })
    reshaped_data['Year'] = reshaped_data['Year'].astype(int)
    reshaped_data['Runoff'] = pd.to_numeric(reshaped_data['Runoff'], errors='coerce')
    reshaped_data['Season'] = reshaped_data['Month'].map(season_mapping)
    window1 = reshaped_data[(reshaped_data['Year'] >= 1980) & (reshaped_data['Year'] <= 1994)]
    window2 = reshaped_data[(reshaped_data['Year'] >= 1995) & (reshaped_data['Year'] <= 2010)]
    monthly_percentiles_window1 = window1.groupby('Month')['Runoff'].quantile(0.01).reset_index()
    monthly_percentiles_window1.columns = ['Month', '1st_percentile']
    monthly_percentiles_window1['Season'] = monthly_percentiles_window1['Month'].map(season_mapping)
    seasonal_percentiles_window1 = monthly_percentiles_window1.groupby('Season')['1st_percentile'].mean()
    monthly_percentiles_window2 = window2.groupby('Month')['Runoff'].quantile(0.01).reset_index()
    monthly_percentiles_window2.columns = ['Month', '1st_percentile']
    monthly_percentiles_window2['Season'] = monthly_percentiles_window2['Month'].map(season_mapping)
    seasonal_percentiles_window2 = monthly_percentiles_window2.groupby('Season')['1st_percentile'].mean()
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_percentiles_window1 = seasonal_percentiles_window1.reindex(season_order)
    seasonal_percentiles_window2 = seasonal_percentiles_window2.reindex(season_order)
    return seasonal_percentiles_window1, seasonal_percentiles_window2
data_results = {}
for dataset_name, file_path in dataset_paths.items():
    seasonal_p1_window1, seasonal_p1_window2 = process_dataset(file_path)
    data_results[dataset_name] = {
        "1980-1994": seasonal_p1_window1,
        "1995-2010": seasonal_p1_window2
    }
df_plot = pd.DataFrame({
    (dataset, period): data_results[dataset][period]
    for dataset in dataset_paths.keys()
    for period in ["1980-1994", "1995-2010"]
})
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.18
season_spacing = 0.5  
x = np.arange(len(df_plot.index)) * (3 + season_spacing)
colors = {
    "Obs": ["darkblue", "dodgerblue"],
    "Mixed": ["darkred", "orangered"],
    "GLDAS": ["darkgreen", "lightgreen"],
    "ERA5": ["purple", "violet"]
}
offset = -1.5 * bar_width
for dataset in dataset_paths.keys():
    bars1 = ax.bar(x + offset, df_plot[(dataset, "1980-1994")], bar_width, label=f"{dataset} (1980-1994)", color=colors[dataset][0])
    bars2 = ax.bar(x + offset + bar_width, df_plot[(dataset, "1995-2010")], bar_width, label=f"{dataset} (1995-2010)", color=colors[dataset][1])
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold', rotation=90)
    offset += 2 * bar_width
ax.set_xticks(x)
ax.set_xticklabels(df_plot.index, fontsize=14)
ax.set_ylabel("1st Percentile Runoff (mm/month)", fontsize=20)
ax.set_xlabel("Season", fontsize=14)
ax.set_title("Seasonal 1st Percentile Comparison Across Datasets (1980-1994 vs 1995-2010)", fontsize=16)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.set_ylim(0, 50)
ax.legend(fontsize=14, loc='upper left')
ax.grid(False)
desktop_path = os.path.expanduser(r"data_dir")
file_path = os.path.join(desktop_path, "R_1st_Percentile.tiff")
plt.savefig(file_path, dpi=300)
plt.show()
####END HERE####

#%%
##99th percentile value over datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
dataset_paths = {
   "Obs": r"data_dir_Obs",
   "Mixed": r" data_dir_Mixed ",
   "GLDAS": r" data_dir_GLDAS ",
   "ERA5": r" data_dir_ERA5 " 
}
season_mapping = {
    'Jan': 'Winter', 'Feb': 'Winter', 'Mar': 'Spring', 'Apr': 'Spring', 'May': 'Spring',
    'Jun': 'Summer', 'Jul': 'Summer', 'Aug': 'Summer', 'Sep': 'Autumn', 'Oct': 'Autumn', 'Nov': 'Autumn',
    'Dec': 'Winter'
}
def process_dataset(file_path):
    df = pd.read_csv(file_path, header=None)
    years = df.iloc[0, :-1].values
    months = df.iloc[1, :-1].values
    runoff_data = df.iloc[2:, :-1].values
    catchment_ids = df.iloc[2:, -1].values
    years_repeated = np.tile(years, len(catchment_ids))
    months_repeated = np.tile(months, len(catchment_ids))
    catchments_repeated = np.repeat(catchment_ids, len(years))
    runoff_flattened = runoff_data.flatten()  
    reshaped_data = pd.DataFrame({
        'Year': years_repeated,
        'Month': months_repeated,
        'Catchment': catchments_repeated,
        'Runoff': runoff_flattened
    })
    reshaped_data['Year'] = reshaped_data['Year'].astype(int)
    reshaped_data['Runoff'] = pd.to_numeric(reshaped_data['Runoff'], errors='coerce')
    reshaped_data['Season'] = reshaped_data['Month'].map(season_mapping)
    window1 = reshaped_data[(reshaped_data['Year'] >= 1980) & (reshaped_data['Year'] <= 1994)]
    window2 = reshaped_data[(reshaped_data['Year'] >= 1995) & (reshaped_data['Year'] <= 2010)]
    monthly_percentiles_window1 = window1.groupby('Month')['Runoff'].quantile(0.99).reset_index(name="99th_Percentile")
    monthly_percentiles_window2 = window2.groupby('Month')['Runoff'].quantile(0.99).reset_index(name="99th_Percentile")
    monthly_percentiles_window1['Season'] = monthly_percentiles_window1['Month'].map(season_mapping)
    monthly_percentiles_window2['Season'] = monthly_percentiles_window2['Month'].map(season_mapping)
    seasonal_percentiles_window1 = monthly_percentiles_window1.groupby('Season')["99th_Percentile"].mean()
    seasonal_percentiles_window2 = monthly_percentiles_window2.groupby('Season')["99th_Percentile"].mean()
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_percentiles_window1 = seasonal_percentiles_window1.reindex(season_order)
    seasonal_percentiles_window2 = seasonal_percentiles_window2.reindex(season_order)
    return seasonal_percentiles_window1, seasonal_percentiles_window2
data_results = {}
for dataset_name, file_path in dataset_paths.items():
    seasonal_p99_window1, seasonal_p99_window2 = process_dataset(file_path)
    data_results[dataset_name] = {
        "1980-1994": seasonal_p99_window1,
        "1995-2010": seasonal_p99_window2
    }
df_plot_99th = pd.DataFrame({
    (dataset, period): data_results[dataset][period]
    for dataset in dataset_paths.keys()
    for period in ["1980-1994", "1995-2010"]
})
fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.18
season_spacing = 0.4  
x = np.arange(len(df_plot_99th.index)) * (2 + season_spacing)  
colors = {
    "Obs": ["darkblue", "dodgerblue"],
    "Mixed": ["darkred", "orangered"],
    "GLDAS": ["darkgreen", "lightgreen"],
    "ERA5": ["purple", "violet"]  
}
offset = -1.5 * bar_width - bar_width / 2  
for dataset in dataset_paths.keys():
    bars1 = ax.bar(x + offset, df_plot_99th[(dataset, "1980-1994")], bar_width, label=f"{dataset} (1980-1994)", color=colors[dataset][0])
    bars2 = ax.bar(x + offset + bar_width, df_plot_99th[(dataset, "1995-2010")], bar_width, label=f"{dataset} (1995-2010)", color=colors[dataset][1])
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold', rotation=90)
    offset += 2 * bar_width  
ax.set_xticks(x)
ax.set_xticklabels(df_plot_99th.index, fontsize=14)
ax.set_ylabel("99th Percentile Runoff (mm/month)", fontsize=20)  
ax.set_xlabel("Season", fontsize=14)
ax.set_title("Seasonal 99th Percentile Comparison Across Datasets (1980-1994 vs 1995-2010)", fontsize=16)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.set_ylim(0, 400)  
ax.legend(fontsize=14, loc='upper left')
ax.grid(False)
desktop_path = os.path.expanduser(r"data_dir")
file_path = os.path.join(desktop_path, "R_99th_Percentile.tiff")
plt.savefig(file_path, dpi=300)
plt.show()
####END HERE####

