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

# Function to set the CPU governor and frequency for a core
set_core_mode() {
    core=$1
    mode=$2
    governor_path="/sys/devices/system/cpu/cpu$core/cpufreq/scaling_governor"
    max_freq_path="/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_max_freq"
    base_freq_path="/sys/devices/system/cpu/cpu$core/cpufreq/base_frequency"

    if [ -f "$governor_path" ] && [ -f "$max_freq_path" ]; then
        echo "Setting core $core to $mode mode."

        # Print the actual maximum frequency before setting
        actual_max_freq=$(cat "$max_freq_path")
        echo "Actual maximum frequency for core $core: $actual_max_freq kHz"

        echo $mode | sudo tee "$governor_path"

        if [ "$mode" == "performance" ]; then
            sudo cpufreq-set -c $core -g performance
            echo "Setting maximum frequency for core $core to: $actual_max_freq kHz"
        elif [ "$mode" == "powersave" ] && [ -f "$base_freq_path" ]; then
            base_freq=$(cat "$base_freq_path")
            sudo cpufreq-set -c $core -u $base_freq
            echo "Setting frequency for core $core to: $base_freq kHz"
        fi

        # Verify the updated frequency
        updated_freq=$(cat "$max_freq_path")
        echo "Updated frequency for core $core: $updated_freq kHz"
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

# Read the core range (start_idx end_idx, press Enter for all cores)
read -p "Enter the core range (start_idx end_idx, press Enter for all cores): " -a core_range

# If no cores are provided, get all available cores
if [ ${#core_range[@]} -eq 0 ]; then
    total_cores=$(nproc)
    core_range=($(seq 0 $((total_cores-1))))
fi

# Parse the start and end indexes from the range
start_idx=${core_range[0]}
end_idx=${core_range[1]:-$start_idx}

# Print the current status of the cores before setting the mode
for ((core=start_idx; core<=end_idx; core++)); do
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

# Read the boost argument
read -p "To boost the cores, enter 'bOOst' (without quotes), otherwise press Enter: " boost_input

# Set the specified cores to the chosen mode and boost their maximum frequency
for ((core=start_idx; core<=end_idx; core++)); do
    if is_number "$core"; then
        set_core_mode "$core" "$mode"

        # Boost only if the correct boost argument is provided
        if [ "$mode" == "performance" ] && [ "$boost_input" == "bOOst" ]; then

            echo "Danger: boosting frequency for core $core"
            boost_core "$core"
        fi
    else
        echo "Invalid core index: $core. Core indices should be numeric."
    fi
done

echo "Mode set and cores boosted for specified cores. You may need to reboot your system for changes to take effect."
