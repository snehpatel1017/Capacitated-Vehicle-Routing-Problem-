#!/bin/bash

# This script automates the process of running the HGS solver on all CVRP instances.
# It should be executed from the root directory that contains 'build' and 'Instances'.

# --- Configuration ---
# Directory containing the instance files (.vrp)
INSTANCE_DIR="Instances/CVRP"

# Directory where output files (.sol and .txt) will be stored
ANALYSIS_DIR="analysis"

# Path to the HGS executable
EXECUTABLE="build/hgs"

# --- Script Logic ---

# Exit immediately if a command exits with a non-zero status.
set -e

# Create the analysis directory if it doesn't exist.
# The -p flag ensures no error is thrown if the directory already exists.
echo "Creating analysis directory at '$ANALYSIS_DIR'..."
mkdir -p "$ANALYSIS_DIR"

# Check if the executable exists before starting the loop
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at '$EXECUTABLE'"
    echo "Please make sure you have built the project correctly."
    exit 1
fi

# Find and loop through all .vrp files in the instance directory.
for instance_path in "$INSTANCE_DIR"/*.vrp
do
    # Get the base name of the file (e.g., "CMT1" from "Instances/CVRP/CMT1.vrp").
    base_name=$(basename "$instance_path" .vrp)

    echo "----------------------------------------------------"
    echo "Processing instance: $base_name"

    # Define the full path for the output solution file.
    solution_file="$ANALYSIS_DIR/${base_name}.sol"

    # Define the full path for the command's output log.
    output_log="$ANALYSIS_DIR/${base_name}_output.txt"

    # Construct and run the command.
    # The standard output is redirected (>) to the log file.
    echo "Command: $EXECUTABLE $instance_path $solution_file -seed 1 -log 0"
    "$EXECUTABLE" "$instance_path" "$solution_file" -seed 1 -log 0 > "$output_log"

    echo "Successfully generated:"
    echo "  - Solution: $solution_file"
    echo "  - Log:      $output_log"
done

echo "----------------------------------------------------"
echo "✅ All instances have been processed successfully."
echo "All output files are located in the '$ANALYSIS_DIR' directory."