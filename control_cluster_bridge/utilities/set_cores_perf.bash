#!/bin/bash

# Script Name: set_isolated_cores_mode.bash

# Function to check if a number is valid
is_number() {
    [[ $1 =~ ^[0-9]+$ ]]
}

# Read the list of isolated cores
read -p "Enter the list of isolated cores (separated by space): " -a cores_array

# Read the mode to set (performance or powersave), default to performance
read -p "Enter the mode (performance/powersave), default is performance: " mode
mode=${mode:-performance}

# Validate the mode
if [[ "$mode" != "performance" ]] && [[ "$mode" != "powersave" ]]; then
    echo "Invalid mode: $mode. Please enter 'performance' or 'powersave'."
    exit 1
fi

# Set the specified cores to the chosen mode
for core in "${cores_array[@]}"; do
    if ! is_number "$core"; then
        echo "Invalid core index: $core. Core indices should be numeric."
        continue
    fi

    # Construct the path to the governor setting for the core
    governor_path="/sys/devices/system/cpu/cpu$core/cpufreq/scaling_governor"

    # Check if the path exists
    if [ -f "$governor_path" ]; then
        echo "Setting core $core to $mode mode."
        echo $mode | sudo tee $governor_path
    else
        echo "Core $core not found or not available for modification."
    fi
done

echo "Mode set for specified cores."
