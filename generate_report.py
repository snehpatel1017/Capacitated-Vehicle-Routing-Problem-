import os
import re
import csv

def parse_time_file(filepath):
    """
    Parses the _output.txt file to extract timing information.
    Returns a dictionary of the timing values.
    """
    timings = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    try:
                        # Extract only the floating-point number
                        numeric_value = float(re.findall(r"[-+]?\d*\.\d+|\d+", value)[0])
                        timings[key] = numeric_value
                    except (ValueError, IndexError):
                        # Handle cases where conversion to float fails or no number is found
                        pass
    except FileNotFoundError:
        print(f"Warning: Time file not found at {filepath}")
    return timings

def parse_solution_file(filepath):
    """
    Parses the _solution.sol file to extract the final cost and the
    total number of unique customers.
    Returns a tuple (cost, num_customers).
    """
    cost = None
    customers = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Cost" in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            cost = int(parts[-1])
                        except ValueError:
                            pass
                elif "Route" in line:
                    # Find all numbers in the route line and add them to the set
                    route_customers = re.findall(r'\d+', line.split(':')[1])
                    for cust in route_customers:
                        customers.add(int(cust))
    except FileNotFoundError:
        print(f"Warning: Solution file not found at {filepath}")
        
    num_customers = len(customers) if customers else 0
    return cost, num_customers

def main():
    """
    Main function to find file pairs, parse them, and write to a CSV.
    """
    # Use relative paths based on the script's location
    # Assumes the script is run from a directory that is a sibling to 'analysis'
    analysis_dir = 'analysis' 
    output_csv_path = os.path.join(analysis_dir, 'performance_summary.csv')
    
    if not os.path.isdir(analysis_dir):
        print(f"Error: The directory '{analysis_dir}' was not found. Please run this script from the correct parent directory.")
        return

    # Dictionary to hold the data for each instance
    instance_data = {}

    # Find all files and group them by instance name
    for filename in os.listdir(analysis_dir):
        if filename.endswith("_output.txt") or filename.endswith("_solution.sol"):
            instance_name = filename.split('_')[0]
            if instance_name not in instance_data:
                instance_data[instance_name] = {}
            if filename.endswith("_output.txt"):
                instance_data[instance_name]['txt_file'] = os.path.join(analysis_dir, filename)
            elif filename.endswith("_solution.sol"):
                 instance_data[instance_name]['sol_file'] = os.path.join(analysis_dir, filename)
    
    # Prepare data for CSV writing
    csv_data = []
    
    for name, files in sorted(instance_data.items()):
        if 'txt_file' in files and 'sol_file' in files:
            print(f"Processing instance: {name}")
            
            timings = parse_time_file(files['txt_file'])
            cost, num_customers = parse_solution_file(files['sol_file'])
            
            # Prepare a row for the CSV in the desired order
            row = {
                'Instance name': name,
                'number of customers': num_customers,
                'cost': cost,
                'generate population time': timings.get('GeneratePopulation Completed'),
                'crossover time': timings.get('crossoverOx time'),
                'localsearch time': timings.get('localSearch time'),
                'add Individual  to population time': timings.get('addIndividual time')
            }
            csv_data.append(row)
        else:
            print(f"Warning: Missing a file for instance {name}. Skipping.")

    # Write the collected data to a CSV file
    if not csv_data:
        print("No data was processed. The CSV file will not be created.")
        return
        
    headers = [
        'Instance name', 'number of customers', 'cost', 'generate population time', 
        'crossover time', 'localsearch time', 'add Individual  to population time'
    ]
    
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\nSuccessfully generated CSV file at: {output_csv_path}")
    except IOError:
        print(f"Error: Could not write to the file at {output_csv_path}. Please check permissions.")

if __name__ == "__main__":
    main()
