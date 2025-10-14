#!/bin/bash

# Script Name: isolate_core.bash

# Generate a unique identifier based on the current date and time
backup_suffix=$(date +"%Y%m%d-%H%M%S")

# Get the total number of CPU cores
total_cores=$(nproc)
echo "Total available cores: $total_cores"

# Ask the user for the number of cores they want to isolate
while true; do
    read -p "Enter the number of cores to isolate: " num_to_isolate
    if [[ "$num_to_isolate" =~ ^[0-9]+$ ]]; then
        break
    else
        echo "Please enter a valid numeric value."
    fi
done

# Calculate the maximum allowed cores to isolate
max_cores_to_isolate=$((total_cores - 4))

# Validate the input
if [ "$num_to_isolate" -le 0 ] || [ "$num_to_isolate" -gt "$max_cores_to_isolate" ]; then
    echo "Invalid number of cores to isolate. Please choose a number between 1 and $max_cores_to_isolate."
    exit 1
fi

# Calculate the starting core index to isolate
start_core=$((total_cores - num_to_isolate))

# Isolate the cores
isolated_cores=()
for (( i=start_core; i<total_cores; i++ )); do
    isolated_cores+=($i)
done

# Convert the array of isolated cores into a comma-separated string
isolated_cores_string=$(IFS=,; echo "${isolated_cores[*]}")

# Print the isolated core indices
echo "Isolated cores: $isolated_cores_string"

# Backup and Update GRUB configuration (requires superuser privileges)
echo "Backing up and updating GRUB configuration to isolate cores. You might need to enter your password."

sudo cp /etc/default/grub /etc/default/grub.$backup_suffix.bak
if sudo grep -q "GRUB_CMDLINE_LINUX_DEFAULT=" /etc/default/grub; then
    sudo sed -i "/^GRUB_CMDLINE_LINUX_DEFAULT=/ s/\"$/ isolcpus=$isolated_cores_string\"/" /etc/default/grub
    if sudo update-grub; then
        echo "GRUB configuration updated successfully."
    else
        echo "Error updating GRUB. Please check the configuration."
        exit 1
    fi
else
    echo "GRUB_CMDLINE_LINUX_DEFAULT not found in /etc/default/grub. Please check the configuration."
    exit 1
fi

# Suggest rebooting the system
echo "Please reboot your system for the changes to take effect."


