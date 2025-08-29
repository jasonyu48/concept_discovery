#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
import numpy as np

def extract_encoding_space_data(log_file_path):
    """
    Extract encoding space size and reward data from a log file.
    Returns three lists: training_steps, encoding_space_sizes, and rewards
    """
    training_steps = []
    encoding_space_sizes = []
    rewards = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Look for lines that contain "Encoding space size:"
    for i, line in enumerate(lines):
        if "Encoding space size:" in line:
            # Extract the encoding space size value
            match = re.search(r"Encoding space size:\s+([\d\.e\-\+]+)", line)
            if match:
                encoding_size = float(match.group(1))
                
                # Look backwards to find "Exist condition holds:" and then the reward line
                exist_line_idx = None
                for j in range(i-1, max(i-10, 0), -1):
                    if "Exist condition holds:" in lines[j]:
                        exist_line_idx = j
                        break
                
                if exist_line_idx is not None:
                    # The reward line should be 2 lines before "Exist condition holds:"
                    reward_line_idx = exist_line_idx - 2
                    if reward_line_idx >= 0:
                        reward_line = lines[reward_line_idx]
                        
                        # Extract training step and reward from the same line
                        step_match = re.search(r"I:\s+([\d,]+)", reward_line)
                        reward_match = re.search(r"R:\s+([\d\.]+)", reward_line)
                        
                        if step_match and reward_match:
                            step = int(step_match.group(1).replace(',', ''))
                            reward = float(reward_match.group(1))
                            
                            training_steps.append(step)
                            encoding_space_sizes.append(encoding_size)
                            rewards.append(reward)
    
    return training_steps, encoding_space_sizes, rewards

def plot_encoding_space_comparison():
    """
    Plot the encoding space size and reward comparison between the two experiments.
    """
    # File paths
    qr_log_path = "/home/tdmpc2/tdmpc2/logs/cheetah-run/1/grad_from_Q_R_011_full_rank_rgb_no_linear_no_simnorm_size_check/log_QR.txt"
    policy_log_path = "/home/tdmpc2/tdmpc2/logs/cheetah-run/1/grad_from_policy_011_full_rank_rgb_no_linear_no_simnorm_size_check/log_c.txt"
    
    # Extract data from both files
    qr_steps, qr_sizes, qr_rewards = extract_encoding_space_data(qr_log_path)
    policy_steps, policy_sizes, policy_rewards = extract_encoding_space_data(policy_log_path)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot encoding space sizes
    ax1.plot(qr_steps, qr_sizes, 'b-o', label='gradient from Q and R', linewidth=2, markersize=6)
    ax1.plot(policy_steps, policy_sizes, 'r-s', label='gradient from policy', linewidth=2, markersize=6)
    
    ax1.set_ylabel('Encoding Space Size', fontsize=14)
    ax1.set_title('Encoding Space Size vs Training Steps', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot rewards
    # ax2.plot(qr_steps, qr_rewards, 'b-o', label='gradient from Q and R', linewidth=2, markersize=6)
    ax2.plot(policy_steps, policy_rewards, 'r-s', label='gradient from policy', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Training Steps', fontsize=14)
    ax2.set_ylabel('Reward', fontsize=14)
    ax2.set_title('Reward vs Training Steps', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis to show thousands for both subplots
    for ax in [ax1, ax2]:
        ax.ticklabel_format(style='plain', axis='x')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
    
    plt.tight_layout()
    
    # Print some statistics
    print(f"Gradient from Q and R:")
    print(f"  Data points: {len(qr_steps)}")
    print(f"  Steps range: {min(qr_steps) if qr_steps else 'N/A'} - {max(qr_steps) if qr_steps else 'N/A'}")
    print(f"  Encoding size range: {min(qr_sizes) if qr_sizes else 'N/A':.3e} - {max(qr_sizes) if qr_sizes else 'N/A':.3e}")
    print(f"  Reward range: {min(qr_rewards) if qr_rewards else 'N/A':.1f} - {max(qr_rewards) if qr_rewards else 'N/A':.1f}")
    
    print(f"\nGradient from policy:")
    print(f"  Data points: {len(policy_steps)}")
    print(f"  Steps range: {min(policy_steps) if policy_steps else 'N/A'} - {max(policy_steps) if policy_steps else 'N/A'}")
    print(f"  Encoding size range: {min(policy_sizes) if policy_sizes else 'N/A':.3e} - {max(policy_sizes) if policy_sizes else 'N/A':.3e}")
    print(f"  Reward range: {min(policy_rewards) if policy_rewards else 'N/A':.1f} - {max(policy_rewards) if policy_rewards else 'N/A':.1f}")
    
    # Save the plot
    plt.savefig('encoding_space_and_reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_encoding_space_comparison() 