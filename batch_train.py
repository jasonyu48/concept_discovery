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
import threading
import datetime

class BatchTrainer:
    def __init__(self, 
                 tasks: List[str] = None,
                 configs: List[str] = None,
                 parallel: str = 'False',
                 max_workers: int = 4,
                 dry_run: str = 'False'):
        
        self.tasks = tasks
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
        
        # Create logs directory if it doesn't exist
        logs_dir = self.workspace_path / "batch_train_logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = logs_dir / f"{task}_{config}_{timestamp}.txt"
        
        start_time = time.time()
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                cwd=self.tdmpc2_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Variables for output collection and periodic saving
            output_buffer = []
            buffer_lock = threading.Lock()  # Protect concurrent access to output_buffer
            save_interval = 30  # Save every 30 seconds
            process_finished = threading.Event()
            
            print(f"Saving output to: {log_file_path}")
            print("Training output will be saved every 30 seconds...")
            
            # Thread for periodic saving
            def periodic_save():
                while not process_finished.wait(save_interval):  # Wait 30s or until process finishes
                    with buffer_lock:
                        if output_buffer:  # Only save if there's something to save
                            lines_to_write = output_buffer.copy()
                            output_buffer.clear()
                        else:
                            lines_to_write = None
                    if lines_to_write:
                        self._save_output_to_file(log_file_path, lines_to_write)
            
            # Start the periodic save thread
            save_thread = threading.Thread(target=periodic_save, daemon=True)
            save_thread.start()
            
            # Read output line by line (this blocks efficiently)
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        # Skip progress-bar updates to keep logs clean
                        if self._is_progress_bar_line(line):
                            continue

                        # Add timestamp to each line and store in buffer
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        timestamped_line = f"[{current_time}] {line.rstrip()}\n"
                        with buffer_lock:
                            output_buffer.append(timestamped_line)
                        
                        # Print to console as well for real-time monitoring
                        # print(f"[{task}-{config}] {line.rstrip()}")
            finally:
                # Signal that process output reading is done
                process_finished.set()
            
            # Save any remaining output
            if output_buffer:
                self._save_output_to_file(log_file_path, output_buffer)
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Add final summary to log file
            summary_lines = [
                f"\n{'='*60}\n",
                f"Training completed at: {datetime.datetime.now()}\n",
                f"Duration: {duration:.2f} seconds\n",
                f"Return code: {return_code}\n",
                f"{'='*60}\n"
            ]
            self._save_output_to_file(log_file_path, summary_lines, append=True)
            
            if return_code == 0:
                status = "SUCCESS"
                message = f"Training completed in {duration:.2f} seconds. Log: {log_file_path}"
            else:
                status = "FAILED"
                message = f"Training failed after {duration:.2f} seconds. Log: {log_file_path}"
            
            return task, config, status, message, duration
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            # Save error to log file if it exists
            if 'log_file_path' in locals():
                error_lines = [
                    f"\n{'='*60}\n",
                    f"ERROR occurred at: {datetime.datetime.now()}\n",
                    f"Error: {str(e)}\n",
                    f"Duration before error: {duration:.2f} seconds\n",
                    f"{'='*60}\n"
                ]
                self._save_output_to_file(log_file_path, error_lines, append=True)
                message = f"Exception: {str(e)}. Log: {log_file_path}"
            else:
                message = f"Exception: {str(e)}"
            
            return task, config, "ERROR", message, duration
    
    def _save_output_to_file(self, log_file_path: Path, output_lines: List[str], append: bool = True):
        """Save output lines to the log file."""
        mode = 'a' if append else 'w'
        try:
            with open(log_file_path, mode, encoding='utf-8') as f:
                f.writelines(output_lines)
            if not append:
                output_lines.clear()  # Clear buffer after saving
        except Exception as e:
            print(f"Warning: Could not save to log file {log_file_path}: {e}")
    
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

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _is_progress_bar_line(line: str) -> bool:
        """Return True if the line looks like a tqdm/progress-bar update."""
        return ("it/s" in line)


def main():
    parser = argparse.ArgumentParser(description='Batch training script for TD-MPC2')
    parser.add_argument('--tasks', nargs='+', default=['N_exp'],
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