import pandas as pd
import matplotlib.pyplot as plt

# Replace 'file1.csv' and 'file2.csv' with the actual file names
file1 = 'ex1_elapsed_times.csv'
file2 = 'ex1_elapsed_times.csv'

# Read the CSV files
data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data1['t_value'], data1['elapsed_time'], label='File 1', color='blue', marker='o')
plt.scatter(data2['t_value'], data2['elapsed_time'], label='File 2', color='red', marker='x')

# Add titles and labels
plt.title('Scatter Plot of Elapsed Time vs. t_value')
plt.xlabel('t_value')
plt.ylabel('elapsed_time')
plt.legend()

# Show the plot
plt.show()
