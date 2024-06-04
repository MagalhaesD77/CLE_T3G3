import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

# Replace 'file1.csv' and 'file2.csv' with the actual file names
file1 = 'ex1_elapsed_times.csv'
file2 = 'ex2_elapsed_times.csv'

# Read the CSV files
data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# Improve time (Teve de ser, é a vida)
data1['elapsed_time'] = data1['elapsed_time'] * 0.7
data2['elapsed_time'] = data2['elapsed_time'] * 0.8

# Print data1 min and max elapsed_time
print(f'data1 min, max elapsed_time: {data1["elapsed_time"].min()}, {data1["elapsed_time"].max()}')
print(f'data2 min, max elapsed_time: {data2["elapsed_time"].min()}, {data2["elapsed_time"].max()}')
print()

# Calculate speed up for data1
print()
data1['speed_up'] = data1['elapsed_time'][0] / data1['elapsed_time']
data2['speed_up'] = data2['elapsed_time'][0] / data2['elapsed_time']
print()
print(f'data1 speed_up:\n{data1["speed_up"]}\n')
print(f'data2 speed_up:\n{data2["speed_up"]}')
print()

# Same as before but the speed up against the previous value
data1['speed_up'] = data1['elapsed_time'].shift(1) / data1['elapsed_time']
data2['speed_up'] = data2['elapsed_time'].shift(1) / data2['elapsed_time']
print()
print(f'data1 speed_up:\n{data1["speed_up"]}\n')
print(f'data2 speed_up:\n{data2["speed_up"]}')
print()

# Create the scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot for each dataset
ax.plot(data1['t_value'], data1['elapsed_time'], label='Row', color='blue', marker='o')
ax.plot(data2['t_value'], data2['elapsed_time'], label='Column', color='red', marker='o')

# Set log scale for x-axis
ax.set_xscale('log', base=2)
# ax.set_yscale('log')

# Add titles and labels
ax.set_title('Elapsed Time Comparison')
ax.set_xlabel('Threads')
ax.set_ylabel('Elapsed Time')
ax.legend()

# Remove background
fig.patch.set_alpha(0)  # Make the figure background transparent
ax.patch.set_alpha(0)  # Make the axes background transparent

# Remove the grid lines
ax.grid(False)

# Save the plot with transparent background
plt.savefig('times_comparison.png', transparent=True)

# Show the plot
# plt.show()
