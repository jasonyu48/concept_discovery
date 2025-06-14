#!/usr/bin/env python3
"""
Batch training script for TD-MPC2.

This script automatically runs train.py using all configuration files 
from the specified task directories (fish, hopper, walker, policy).

Usage:
    python batch_train.py [options]

Options:
    --tasks: List of tasks to run (default: fish hopper walker)
    --configs: List of config variants to run (default: config_QR config_Q config_R config_default)
    --parallel: Run configurations in parallel (default: False)
    --max_workers: Maximum number of parallel workers (default: 4)
    --dry_run: Print commands without executing them (default: False)
"""

import os
import sys
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Set
import time

class BatchTrainer:
    def __init__(self, 
                 tasks: List[str] = None,
                 configs: List[str] = None,
                 parallel: str = 'False',
                 max_workers: int = 4,
                 dry_run: str = 'False'):
        
        self.tasks = tasks or ['fish', 'hopper', 'walker', 'policy']
        self.parallel = parallel
        self.max_workers = max_workers
        self.dry_run = dry_run
        
        # Base paths
        self.workspace_path = Path.cwd()
        self.tdmpc2_path = self.workspace_path / 'tdmpc2'
        self.train_script = 'train.py'  # Relative to tdmpc2 directory
        self.config_base_path = self.tdmpc2_path / 'config'
        
        # Validate paths and discover configs
        self._validate_setup()
        
        # If configs are provided, use them; otherwise discover them
        if configs:
            self.configs = configs
            # Validate that provided configs exist
            self._validate_provided_configs()
        else:
            self.configs = self._discover_configs()
            print(f"Auto-discovered configs: {self.configs}")
    
    def _discover_configs(self) -> List[str]:
        """Discover all available config files across all task directories."""
        all_configs = set()
        
        for task in self.tasks:
            task_config_dir = self.config_base_path / task
            if task_config_dir.exists():
                # Find all .yaml files in the task directory
                yaml_files = list(task_config_dir.glob("*.yaml"))
                task_configs = [f.stem for f in yaml_files]  # Remove .yaml extension
                all_configs.update(task_configs)
                print(f"Found configs for {task}: {task_configs}")
        
        configs_list = sorted(list(all_configs))
        
        if not configs_list:
            raise FileNotFoundError("No config files found in any task directories")
        
        return configs_list
    
    def _validate_provided_configs(self):
        """Validate that all provided config files exist for all tasks."""
        for task in self.tasks:
            task_config_dir = self.config_base_path / task
            if not task_config_dir.exists():
                continue
                
            for config in self.configs:
                config_file = task_config_dir / f"{config}.yaml"
                if not config_file.exists():
                    print(f"Warning: Config file not found: {config_file}")
    
    def _validate_setup(self):
        """Validate that all required files and directories exist."""
        if not (self.tdmpc2_path / self.train_script).exists():
            raise FileNotFoundError(f"Training script not found: {self.tdmpc2_path / self.train_script}")
        
        if not self.config_base_path.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_base_path}")
        
        # Check if all specified tasks have config directories
        for task in self.tasks:
            task_config_dir = self.config_base_path / task
            if not task_config_dir.exists():
                raise FileNotFoundError(f"Task config directory not found: {task_config_dir}")
    
    def _run_single_config(self, task: str, config: str) -> tuple:
        """Run training with a single configuration."""
        config_path = self.config_base_path / task
        config_name = config
        
        # Check if this specific config exists for this task
        config_file = config_path / f"{config_name}.yaml"
        if not config_file.exists():
            return task, config, "SKIPPED", f"Config file not found: {config_file}", 0
        
        # Build the command
        cmd = [
            sys.executable, 
            self.train_script,
            '--config-path', str(config_path),
            '--config-name', config_name
        ]
        
        print(f"\n{'='*60}")
        print(f"Running: {task} - {config}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        if self.dry_run == 'True':
            return task, config, "DRY_RUN", "Command printed only", 0
        
        start_time = time.time()
        
        try:
            # Run the training
            result = subprocess.run(
                cmd,
                cwd=self.tdmpc2_path,  # Run from tdmpc2 directory for proper imports
                capture_output=True,
                text=True,
                timeout=None  # No timeout for training
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                status = "SUCCESS"
                message = f"Training completed in {duration:.2f} seconds"
            else:
                status = "FAILED"
                message = f"Training failed after {duration:.2f} seconds\nSTDERR: {result.stderr[-500:]}"  # Last 500 chars of stderr
            
            return task, config, status, message, duration
            
        except subprocess.TimeoutExpired:
            return task, config, "TIMEOUT", "Training timed out", time.time() - start_time
        except Exception as e:
            return task, config, "ERROR", f"Exception: {str(e)}", time.time() - start_time
    
    def run_all(self):
        """Run training for all task-config combinations."""
        # Generate all combinations
        combinations = [(task, config) for task in self.tasks for config in self.configs]
        
        print(f"Planning to run {len(combinations)} training configurations:")
        for i, (task, config) in enumerate(combinations, 1):
            print(f"  {i:2d}. {task:10s} - {config}")
        print()
        
        if self.dry_run == 'False':
            response = input("Continue? (y/N): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
        
        results = []
        
        if self.parallel == 'True':
            print(f"Running configurations in parallel (max_workers={self.max_workers})...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._run_single_config, task, config) 
                          for task, config in combinations]
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    task, config, status, message, duration = result
                    print(f"[{status}] {task}-{config}: {message}")
        else:
            print("Running configurations sequentially...")
            for task, config in combinations:
                result = self._run_single_config(task, config)
                results.append(result)
                task, config, status, message, duration = result
                print(f"[{status}] {task}-{config}: {message}")
        
        # Print summary
        self._print_summary(results)
    
    def _print_summary(self, results: List[tuple]):
        """Print a summary of all training results."""
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        
        successful = [r for r in results if r[2] == "SUCCESS"]
        failed = [r for r in results if r[2] in ["FAILED", "ERROR", "TIMEOUT"]]
        skipped = [r for r in results if r[2] == "SKIPPED"]
        
        print(f"Total configurations: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Skipped: {len(skipped)}")
        
        if successful:
            print(f"\nSuccessful runs:")
            for task, config, status, message, duration in successful:
                print(f"  ✓ {task:10s} - {config:15s} ({duration:.1f}s)")
        
        if failed:
            print(f"\nFailed runs:")
            for task, config, status, message, duration in failed:
                print(f"  ✗ {task:10s} - {config:15s} - {status}")
                if "STDERR:" in message:
                    print(f"    Error: {message.split('STDERR:')[1].strip()}")
        
        if skipped:
            print(f"\nSkipped runs:")
            for task, config, status, message, duration in skipped:
                print(f"  - {task:10s} - {config:15s} - {message}")
        
        total_time = sum(r[4] for r in results if r[4] > 0)
        print(f"\nTotal execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")


def main():
    parser = argparse.ArgumentParser(description='Batch training script for TD-MPC2')
    parser.add_argument('--tasks', nargs='+', default=['policy'],
                       help='List of tasks to run')
    parser.add_argument('--configs', nargs='+', default=None,
                       help='List of config variants to run (if not provided, will auto-discover from task directories)')
    parser.add_argument('--parallel', type=str, default='False', choices=['True', 'False'],
                       help='Run configurations in parallel')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Maximum number of parallel workers')
    parser.add_argument('--dry-run', type=str, default='False', choices=['True', 'False'],
                       help='Print commands without executing them')
    
    args = parser.parse_args()
    
    try:
        trainer = BatchTrainer(
            tasks=args.tasks,
            configs=args.configs,
            parallel=args.parallel,
            max_workers=args.max_workers,
            dry_run=args.dry_run
        )
        trainer.run_all()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 