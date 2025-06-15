import os
import pandas as pd
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

def find_eval_files(logs_dir):
    """
    Find all eval.csv files and organize by environment
    """
    eval_files = {}
    
    # Define the RL environments to process
    rl_envs = ["fish-swim", "hopper-hop", "walker-walk"]
    
    for rl_env in rl_envs:
        eval_files[rl_env] = []
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
                    
                eval_file_path = os.path.join(experiment_path, "eval.csv")
                if os.path.exists(eval_file_path):
                    eval_files[rl_env].append({
                        'path': eval_file_path,
                        'experiment_details': experiment_folder,
                        'seed': seed_folder
                    })
                    
    return eval_files

def plot_environment_rewards(env_name, eval_files_info, output_dir="./"):
    """
    Plot reward curves for a single environment
    """
    plt.figure(figsize=(12, 8))
    
    for file_info in eval_files_info:
        try:
            # Read the eval.csv file
            df = pd.read_csv(file_info['path'])
            
            # Check if the required columns exist
            if 'step' not in df.columns or 'episode_reward' not in df.columns:
                print(f"Warning: Required columns not found in {file_info['path']}")
                continue
            
            # Get the proper label
            label = get_label_from_experiment_details(file_info['experiment_details'])
            
            # Plot the reward curve
            plt.plot(df['step'], df['episode_reward'], label=f"{label} (seed={file_info['seed']})", linewidth=2)
            
        except Exception as e:
            print(f"Error processing {file_info['path']}: {e}")
            continue
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title(f'Reward Comparison - {env_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"{env_name}_reward_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot for {env_name} to {output_path}")
    plt.close()

def main():
    # Set the logs directory path
    logs_dir = "./tdmpc2/logs"
    
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory {logs_dir} does not exist")
        return
    
    # Find all eval.csv files
    print("Searching for eval.csv files...")
    eval_files = find_eval_files(logs_dir)
    
    # Display found files
    for env, files in eval_files.items():
        print(f"\n{env}: Found {len(files)} eval.csv files")
        for file_info in files:
            print(f"  - {file_info['experiment_details']} (seed: {file_info['seed']})")
    
    # Plot for each environment
    print("\nGenerating plots...")
    for env_name, files_info in eval_files.items():
        if files_info:  # Only plot if there are files
            plot_environment_rewards(env_name, files_info)
        else:
            print(f"No eval.csv files found for {env_name}")
    
    print("\nPlotting completed!")

if __name__ == "__main__":
    main() 