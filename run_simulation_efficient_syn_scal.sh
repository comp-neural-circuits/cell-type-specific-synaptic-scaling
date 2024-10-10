#!/bin/bash

# Read the parameter values from the txt file
parameter_file="param3.txt"

# Define the function to execute the Julia script
run_simulation() {
    parameter=$1
    nice python3 /home/ge74coy/mnt/naspersonal/Code/synaptic_scaling/parameter_exploration.py "$parameter"
}

# Set the time to wait (in seconds) after each process
wait_time=1
wait_time_full_ram=3
wait_time_full_cpu=3

# Set the RAM threshold (in megabytes) and CPU threshold (percentage) that will remain free
ram_threshold=140000
cpu_threshold=99

# Function to check available RAM in megabytes
get_available_ram() {
    available_ram=$(free -m | awk 'NR==2 {print $7}')
    echo "$available_ram"
}

# Function to check CPU usage percentage
get_cpu_usage() {
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d. -f1)
    echo "$cpu_usage"
}

# Read the parameter values line by line from the file
while IFS= read -r parameter; do
    # Skip empty lines or lines starting with a comment character
    if [[ -z "$parameter" || "$parameter" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Run the simulation
    run_simulation "$parameter" &

    # Sleep for the specified wait time after running each process
    sleep "$wait_time"

    # Check available RAM
    available_ram=$(get_available_ram)

    # Check if available RAM is below the threshold
    while ((available_ram < ram_threshold)); do
        echo "Available RAM ($available_ram MB) below threshold. Waiting for processes to finish."
        sleep "$wait_time_full_ram"
        available_ram=$(get_available_ram)
    done

    # Check CPU usage
    cpu_usage=$(get_cpu_usage)

    # Check if CPU usage is above the threshold
    while ((cpu_usage >= cpu_threshold)); do
        echo "CPU Usage ($cpu_usage%) above threshold. Waiting for processes to finish."
        sleep "$wait_time_full_cpu"
        cpu_usage=$(get_cpu_usage)
    done

done < "$parameter_file"

# Wait for all remaining background processes to finish
wait
