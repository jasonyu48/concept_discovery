#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_eval_rewards():
    """
    Plot reward vs step from eval.csv files from three different experiments.
    """
    # Define the experiment directories and their labels
    experiments = [
        {
            'path': './tdmpc2/logs/cheetah-run/1/default_rgb_long_no_linear/eval.csv',
            'label': 'tdmpc2(JEPA sg, grad from Q R)',
            'color': 'g',
            'marker': '^'
        },
        {
            'path': '/home/jyu197/tdmpc2/tdmpc2/logs/cheetah-run/2015/default_rgb_no_linear_seed2015/eval.csv',
            'label': 'tdmpc2(different seed)',
            'color': 'g',
            'marker': 's'
        },
        {
            'path': './tdmpc2/logs/cheetah-run/1/grad_from_policy_011_full_rank_rgb_no_linear_no_simnorm_size_check/eval.csv',
            'label': 'grad from policy',
            'color': 'r',
            'marker': '^'
        },
        {
            'path': './tdmpc2/logs/cheetah-run/1/grad_from_Q_R_011_full_rank_rgb_no_linear_no_simnorm_size_check/eval.csv',
            'label': 'grad from Q R',
            'color': 'b',
            'marker': '^'
        },
        {
            'path': '/home/jyu197/tdmpc2/tdmpc2/logs/cheetah-run/2015/grad_from_Q_R_011_full_rank_rgb_no_linear_no_simnorm_seed2015/eval.csv',
            'label': 'grad from Q R (different seed)',
            'color': 'b',
            'marker': 's'
        },
        {
            'path': '/home/jyu197/tdmpc2/tdmpc2/logs/cheetah-run/2015/grad_from_Q_R_011_rgb_no_linear_no_simnorm_seed2015/eval.csv',
            'label': 'grad from Q R no LU',
            'color': 'b',
            'marker': 'o'
        },
        {
            'path': '/home/jyu197/tdmpc2/tdmpc2/logs/cheetah-run/2015/grad_from_Q_R_1_full_rank_rgb_no_linear_no_simnorm_seed2015/eval.csv',
            'label': 'grad from Q R (different seed, grad weight)',
            'color': 'b',
            'marker': 'p'
        },
    ]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Process each experiment
    for exp in experiments:
        if os.path.exists(exp['path']):
            try:
                # Read the CSV file
                df = pd.read_csv(exp['path'])
                
                # Filter for step <= 350000
                df_filtered = df[df['step'] <= 350000]
                
                # Plot the data
                plt.plot(df_filtered['step'], df_filtered['episode_reward'], 
                        color=exp['color'], marker=exp['marker'], 
                        label=exp['label'], linewidth=2, markersize=6,
                        markevery=max(1, len(df_filtered) // 20))  # Show markers every ~20th point
                
                print(f"Loaded {exp['label']}: {len(df_filtered)} data points")
                print(f"  Step range: {df_filtered['step'].min()} - {df_filtered['step'].max()}")
                print(f"  Reward range: {df_filtered['episode_reward'].min():.1f} - {df_filtered['episode_reward'].max():.1f}")
                
            except Exception as e:
                print(f"Error loading {exp['path']}: {e}")
        else:
            print(f"File not found: {exp['path']}")
    
    # Formatting
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.title('Evaluation Reward vs Training Steps', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show thousands
    plt.ticklabel_format(style='plain', axis='x')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
    
    # Set x-axis limit
    plt.xlim(0, 350000)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('eval_reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_eval_rewards() 