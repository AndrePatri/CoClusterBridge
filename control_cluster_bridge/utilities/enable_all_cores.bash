#!/bin/bash

# Script Name: enable_all_cores.bash

# Backup and Update GRUB configuration (requires superuser privileges)
echo "Backing up and updating GRUB configuration to re-enable all cores. You might need to enter your password."

# Backup the current GRUB configuration
sudo cp /etc/default/grub /etc/default/grub.bak

# Check if isolcpus is set and remove it
if sudo grep -q "isolcpus=" /etc/default/grub; then
    sudo sed -i '/isolcpus=/ s/isolcpus=[^ ]* //' /etc/default/grub
    if sudo update-grub; then
        echo "GRUB configuration updated successfully. All cores re-enabled."
    else
        echo "Error updating GRUB. Please check the configuration."
        exit 1
    fi
else
    echo "No isolated cores found in GRUB configuration. No changes made."
fi

# Suggest rebooting the system
echo "Please reboot your system for the changes to take effect."
