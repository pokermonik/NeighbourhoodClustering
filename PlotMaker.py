import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'clusteringR15_results_comparison.txt'  # file
data = pd.read_csv(file_path)

# Split data by method
nbc_data = data[data['Method'] == 'NBC']
kmeans_data = data[data['Method'] == 'K-Means']

# Plot for Silhouette Score
plt.figure(figsize=(8, 6))
plt.plot(nbc_data['k'], nbc_data['Silhouette Score'], label='NBC', marker='o')
plt.plot(kmeans_data['k'], kmeans_data['Silhouette Score'], label='K-Means', marker='o')
plt.title('Silhouette Score vs k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 20))
plt.legend()
plt.show()

# Plot for Execution Time
plt.figure(figsize=(8, 6))
plt.plot(nbc_data['k'], nbc_data['Execution Time (s)'], label='NBC', marker='o')
plt.plot(kmeans_data['k'], kmeans_data['Execution Time (s)'], label='K-Means', marker='o')

# Calculate and plot trends
nbc_trend = np.poly1d(np.polyfit(nbc_data['k'], nbc_data['Execution Time (s)'], 1))
kmeans_trend = np.poly1d(np.polyfit(kmeans_data['k'], kmeans_data['Execution Time (s)'], 1))
plt.plot(nbc_data['k'], nbc_trend(nbc_data['k']), linestyle='--', label='NBC Trend')
plt.plot(kmeans_data['k'], kmeans_trend(kmeans_data['k']), linestyle='--', label='K-Means Trend')

plt.title('Execution Time vs k')
plt.xlabel('k')
plt.ylabel('Execution Time (s)')
plt.xticks(range(2, 20))
plt.legend()
plt.show()

# Calculate average execution times
nbc_avg_time = nbc_data['Execution Time (s)'].mean()
kmeans_avg_time = kmeans_data['Execution Time (s)'].mean()
print(f"Average Execution Time for NBC: {nbc_avg_time:.4f} s")
print(f"Average Execution Time for K-Means: {kmeans_avg_time:.4f} s")
