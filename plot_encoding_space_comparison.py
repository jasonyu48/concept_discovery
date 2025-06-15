import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_label_from_experiment_details(experiment_details):
    """
    Convert experiment details to proper labels based on the mapping rules
    """
    if "Default" in experiment_details:
        return "tdmpc2"
    elif "GradFromPolicy" in experiment_details:
        return "gradient from policy"
    elif "GradFromQ" in experiment_details and "GradFromQR" not in experiment_details:
        return "gradient from Q"
    elif "GradFromR" in experiment_details and "GradFromQR" not in experiment_details:
        return "gradient from reward"
    elif "GradFromQR" in experiment_details:
        return "gradient from Q and reward"
    else:
        return experiment_details

def find_encoding_files(logs_dir):
    """
    Find all simple_monitoring_data.pkl files and organize by environment
    """
    encoding_files = {}
    
    # Define the RL environments to process
    rl_envs = ["fish-swim", "hopper-hop", "walker-walk"]
    
    for rl_env in rl_envs:
        encoding_files[rl_env] = []
        env_path = os.path.join(logs_dir, rl_env)
        
        if not os.path.exists(env_path):
            print(f"Warning: Environment path {env_path} does not exist")
            continue
        
        # Look for seed folders (like 2016)
        for seed_folder in os.listdir(env_path):
            seed_path = os.path.join(env_path, seed_folder)
            if not os.path.isdir(seed_path):
                continue
                
            # Look for experiment folders
            for experiment_folder in os.listdir(seed_path):
                experiment_path = os.path.join(seed_path, experiment_folder)
                if not os.path.isdir(experiment_path):
                    continue
                    
                encoding_monitor_path = os.path.join(experiment_path, "encoding_monitor", "simple_monitoring_data.pkl")
                if os.path.exists(encoding_monitor_path):
                    encoding_files[rl_env].append({
                        'path': encoding_monitor_path,
                        'experiment_details': experiment_folder,
                        'seed': seed_folder
                    })
                    
    return encoding_files

def plot_environment_encoding_space(env_name, encoding_files_info, output_dir="./"):
    """
    Plot encoding space size curves for a single environment
    """
    # Define consistent colors for each experiment type
    color_map = {
        "tdmpc2": "green",
        "gradient from policy": "red", 
        "gradient from Q": "orange",
        "gradient from reward": "purple",
        "gradient from Q and reward": "blue"
    }
    
    plt.figure(figsize=(12, 8))
    
    for file_info in encoding_files_info:
        try:
            # Load the pickle file
            with open(file_info['path'], 'rb') as f:
                data = pickle.load(f)
            
            # Check if the required keys exist
            if 'steps' not in data or 'encoding_space_size' not in data:
                print(f"Warning: Required keys not found in {file_info['path']}")
                continue
            
            # Get the proper label
            label = get_label_from_experiment_details(file_info['experiment_details'])
            
            # Get the color for this experiment type
            color = color_map.get(label, "black")  # Default black for unknown types
            
            # Plot the encoding space size curve
            plt.plot(data['steps'], data['encoding_space_size'], 
                    label=f"{label} (seed={file_info['seed']})", 
                    linewidth=2, 
                    color=color)
            
        except Exception as e:
            print(f"Error processing {file_info['path']}: {e}")
            continue
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Encoding Space Size (log scale)', fontsize=12)
    plt.yscale('log')
    plt.title(f'Encoding Space Size Comparison - {env_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"{env_name}_encoding_space_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved encoding space plot for {env_name} to {output_path}")
    plt.close()

def main():
    # Set the logs directory path
    logs_dir = "./tdmpc2/logs"
    
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory {logs_dir} does not exist")
        return
    
    # Find all encoding monitoring files
    print("Searching for encoding monitoring files...")
    encoding_files = find_encoding_files(logs_dir)
    
    # Display found files
    for env, files in encoding_files.items():
        print(f"\n{env}: Found {len(files)} encoding monitoring files")
        for file_info in files:
            print(f"  - {file_info['experiment_details']} (seed: {file_info['seed']})")
    
    # Plot for each environment
    print("\nGenerating encoding space plots...")
    for env_name, files_info in encoding_files.items():
        if files_info:  # Only plot if there are files
            plot_environment_encoding_space(env_name, files_info)
        else:
            print(f"No encoding monitoring files found for {env_name}")
    
    print("\nEncoding space plotting completed!")

if __name__ == "__main__":
    main() 