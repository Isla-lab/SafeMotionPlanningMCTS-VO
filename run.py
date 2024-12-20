import multiprocessing

with open('experiment_list.txt') as f:
    processes = f.readlines()

# Define the maximum number of concurrent processes
max_processes = 10


# Define a function that runs a process using subprocess
def run_process(process):
    import subprocess
    print(process)
    subprocess.run(process, shell=True)


# Create a pool of 4 worker processes
pool = multiprocessing.Pool(max_processes)

# Create an empty list to store the AsyncResult objects
results = []

# Loop through the processes list
for process in processes:
    # Submit a task to the pool and append the AsyncResult object to the list
    results.append(pool.apply_async(run_process, (process,)))

# Loop through the results list
for result in results:
    # Wait for the task to finish and get the result
    result.wait()
    result.get()

# Close and join the pool
pool.close()
pool.join()

# Print a message when done
print("All processes completed.")
