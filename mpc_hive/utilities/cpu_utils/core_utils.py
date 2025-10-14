import os
import re
import psutil
import tracemalloc
import sys

def get_isolated_cores():
    # Read the kernel boot parameters
    with open('/proc/cmdline', 'r') as file:
        cmdline = file.read()

    # Find the isolcpus parameter
    match = re.search(r'isolcpus=([0-9,-]+)', cmdline)
    if not match:
        return "No isolated cores found.", []

    # Extract the core indices
    isolated_cores_str = match.group(1).split(',')
    isolated_cores = []

    for core_str in isolated_cores_str:
        if '-' in core_str:
            start, end = map(int, core_str.split('-'))
            isolated_cores.extend(range(start, end + 1))
        else:
            isolated_cores.append(int(core_str))

    return len(isolated_cores), isolated_cores

# Get and print the number of isolated cores and their IDs
num_isolated, isolated_core_ids = get_isolated_cores()
print(f"Number of isolated cores: {num_isolated}")
print(f"Isolated core IDs: {isolated_core_ids}")

def get_memory_usage(db_print:bool=True):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024**3)  # Convert bytes to gigabytes
    if db_print:
        print(f"[PID {os.getpid()}] Memory usage now: {memory_usage_gb} GB")
    return memory_usage_gb

def get_system_memory(label="", prev=0.0, db_print:bool=True):
    """Get system memory usage."""
    mem = psutil.virtual_memory()
    used=mem.used / (1024 ** 3)
    avail=mem.available / (1024 ** 3)
    if db_print:
        print(f"{label} System Memory: Used = {used} GB, Available = {avail} GB,differece {used-prev}")
    return (used, avail)

def get_lib_usage_by_name():
    """
    Returns a dictionary with the memory usage of each imported library in GB.
    Keys are module names and values are the memory usage in GB.
    """
    # Start tracking memory allocations
    

    # Get the current process memory info
    process = psutil.Process(os.getpid())
    total_memory = process.memory_info().rss  # Total RSS memory in bytes

    # Take a snapshot of memory usage
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('filename')

    # Initialize dictionary to store memory usage by module
    module_memory = {}

    for stat in stats:
        # Extract the file/module name
        module_name = stat.traceback[0].filename

        # Ignore system or unknown files
        if module_name and "site-packages" in module_name:
            # Extract just the package name (last directory before file name)
            module_name = module_name.split(os.sep)[-2]

        elif module_name and os.path.basename(module_name) in sys.modules:
            module_name = os.path.basename(module_name)
        else:
            module_name = "unknown"

        # Accumulate memory usage
        if module_name in module_memory:
            module_memory[module_name] += stat.size
        else:
            module_memory[module_name] = stat.size

    # Convert memory usage to GB and return as a dictionary
    module_memory_gb = {module: size / (1024 ** 3) for module, size in module_memory.items()}
    
    tracemalloc.stop()
    return module_memory_gb


# Example Usage in Child Process
if __name__ == "__main__":
    import time
    tracemalloc.start()
    # Simulate importing libraries
    import numpy as np
    time.sleep(1)

    # Call the function
    library_usage = get_lib_usage_by_name()
    print("Library Memory Usage (in GB):")
    for lib, usage in library_usage.items():
        print(f"{lib}: {usage:.6f} GB")