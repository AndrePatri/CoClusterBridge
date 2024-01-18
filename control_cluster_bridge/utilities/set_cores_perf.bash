#!/bin/bash

# Script Name: set_isolated_cores_mode.bash

# Function to check if a number is valid
is_number() {
    [[ $1 =~ ^[0-9]+$ ]]
}

# Function to print the current performance mode of a core
print_core_mode() {
    core=$1
    governor_path="/sys/devices/system/cpu/cpu$core/cpufreq/scaling_governor"
    if [ -f "$governor_path" ]; then
        current_mode=$(cat "$governor_path")
        echo "Current mode for core $core: $current_mode"
    else
        echo "Core $core not found or not available for modification."
    fi
}

# Function to set the CPU governor for a core
set_core_mode() {
    core=$1
    mode=$2
    governor_path="/sys/devices/system/cpu/cpu$core/cpufreq/scaling_governor"
    if [ -f "$governor_path" ]; then
        echo "Setting core $core to $mode mode."
        echo $mode | sudo tee $governor_path
    else
        echo "Core $core not found or not available for modification."
    fi
}

# Function to boost the maximum frequency for a core
boost_core() {
    core=$1
    echo "Boosting core $core to maximum performance."
    sudo cpufreq-set -c $core -g performance
}

# Read the list of isolated cores (default to all cores if not provided)
read -p "Enter the list of isolated cores (separated by space, press Enter for all cores): " -a cores_array

# If no cores are provided, get all available cores
if [ ${#cores_array[@]} -eq 0 ]; then
    total_cores=$(nproc)
    cores_array=($(seq 0 $((total_cores-1))))
fi

# Print the current status of the cores before setting the mode
for core in "${cores_array[@]}"; do
    if is_number "$core"; then
        print_core_mode "$core"
    else
        echo "Invalid core index: $core. Core indices should be numeric."
    fi
done

# Read the mode to set (performance or powersave), default to performance
read -p "Enter the mode (performance/powersave), default is performance: " mode
mode=${mode:-performance}

# Validate the mode
if [[ "$mode" != "performance" ]] && [[ "$mode" != "powersave" ]]; then
    echo "Invalid mode: $mode. Please enter 'performance' or 'powersave'."
    exit 1
fi

# Set the specified cores to the chosen mode and boost their maximum frequency
for core in "${cores_array[@]}"; do
    if is_number "$core"; then
        set_core_mode "$core" "$mode"
        boost_core "$core"
    else
        echo "Invalid core index: $core. Core indices should be numeric."
    fi
done

echo "Mode set and cores boosted for specified cores."



