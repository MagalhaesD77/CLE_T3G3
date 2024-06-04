import subprocess
import re
import csv

data = "dataSet2/datSeq1M.bin"
exercise = "prog1/ex1"

# Function to run the command and extract elapsed time
def get_elapsed_time(t_value, data):
    command = f'./{exercise} -f {data} -t {t_value}'
    while True:
        try:
            result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = result.stdout.decode('utf-8')
            # Check for CUDA error
            if "all CUDA-capable devices are busy or unavailable" in output:
                print("CUDA-capable devices are busy or unavailable. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            # Extract elapsed time using regex
            elapsed_time = re.search(r'Elapsed time = ([0-9.]+) s', output).group(1)
            print(f"t = {t_value}, Elapsed time = {elapsed_time} s")
            return float(elapsed_time)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

# List to store t values and corresponding elapsed times
t_values = [2**i for i in range(11)]  # 2^0 to 2^10
results = []

# Run the command for each t value and collect elapsed times
for t in t_values:
    elapsed_time = get_elapsed_time(t, data)
    if elapsed_time is not None:
        results.append((t, elapsed_time))

# Write results to a CSV file
csv_filename = f'{exercise.split("/")[-1]}_elapsed_times.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['t_value', 'elapsed_time'])
    writer.writerows(results)

print(f'Results have been written to {csv_filename}')
