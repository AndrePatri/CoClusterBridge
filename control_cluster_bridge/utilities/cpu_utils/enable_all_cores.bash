#!/bin/bash

# Script Name: enable_all_cores.bash

# Backup and Update GRUB configuration (requires superuser privileges)
echo "Backing up and updating GRUB configuration to re-enable all cores. You might need to enter your password."

# Generate a unique identifier based on the current date and time
backup_suffix=$(date +"%Y%m%d-%H%M%S")

# Backup the current GRUB configuration with the unique identifier
sudo cp /etc/default/grub /etc/default/grub.$backup_suffix.bak

# Clear GRUB_CMDLINE_LINUX_DEFAULT
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=".*"/GRUB_CMDLINE_LINUX_DEFAULT=""/' /etc/default/grub

# Update GRUB
if sudo update-grub; then
    echo "GRUB configuration updated successfully. All cores re-enabled."
else
    echo "Error updating GRUB. Please check the configuration."
    exit 1
fi

# Suggest rebooting the system
echo "Please reboot your system for the changes to take effect."
