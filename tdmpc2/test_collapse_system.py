#!/usr/bin/env python3
"""
Test script for encoder collapse monitoring system
Tests the collapse monitor on the fixed regularization system
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the tdmpc2 directory to path
sys.path.append('/root/autodl-tmp/202569/tdmpc2/tdmpc2')

def test_collapse_monitor():
    """Test the collapse monitoring system"""
    print("🔬 Testing Encoder Collapse Monitoring System")
    print("="*60)
    
    try:
        # Import required modules
        from collapse_monitor import CollapseMonitor
        from common.world_model import WorldModel
        from envs import make_env
        import hydra
        from omegaconf import DictConfig
        
        print(" All imports successful")
        
        # Create a minimal config
        cfg = DictConfig({
            'task': 'cheetah-run',
            'obs': 'rgb',
            'latent_dim': 512,
            'model_size': 5,
            'device': 'cuda',
            'exp_name': 'test_collapse_monitor',
            'exist_check_freq': 100,
            'task_dim': 96,
            'multitask': False,
            # Add other required config
            'num_enc_layers': 2,
            'enc_dim': 256,
            'mlp_dim': 512,
            'num_channels': 32,
            'dropout': 0.01,
            'simnorm': False,
            'simnorm_dim': 8,
        })
        
        # Create environment
        print(" Creating test environment...")
        env = make_env(cfg)
        print(f"   Environment created: {type(env)}")
        print(f"   Observation shape: {env.observation_space.shape}")
        print(f"   Action dimension: {env.action_space.shape[0]}")
        
        # Create world model with encoder
        print(" Creating encoder...")
        model = WorldModel(cfg).to('cuda')
        encoder = model._encoder[cfg.obs]
        print(f"   Encoder type: {type(encoder)}")
        
        # Test encoder with a sample observation
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_tensor = torch.from_numpy(obs).to('cuda').float().unsqueeze(0)
        print(f"   Sample observation shape: {obs_tensor.shape}")
        
        with torch.no_grad():
            encoded = encoder(obs_tensor)
            print(f"   Encoded shape: {encoded.shape}")
        
        # Initialize collapse monitor
        print(" Initializing CollapseMonitor...")
        monitor = CollapseMonitor(
            cfg=cfg,
            encoder=encoder,
            env=env,
            device='cuda',
            save_dir="test_collapse_logs"
        )
        print(" CollapseMonitor initialized successfully!")
        
        # Test monitoring at different steps
        print("\n Testing monitoring at different steps...")
        test_steps = [0, 100, 200, 500, 1000]
        
        for step in test_steps:
            print(f"\n--- Step {step} ---")
            metrics = monitor.monitor_step(step)
            if metrics:
                print(" Monitoring metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.6f}")
                    else:
                        print(f"   {key}: {value}")
        
        # Generate test plots
        print("\n Generating test plots...")
        fig = monitor.plot_monitoring_results(save_plot=True)
        if fig:
            print(" Plots generated successfully!")
        
        # Generate test report
        print("\n Generating test report...")
        report = monitor.generate_report()
        print(report)
        
        print("\n All tests passed! Collapse monitoring system is working correctly.")
        return True
        
    except Exception as e:
        print(f" Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regularizations():
    """Test the fixed regularization functions"""
    print("\n Testing Fixed Regularization Functions")
    print("="*60)
    
    try:
        from regularizations import orthogonality_regularization, full_rank_regularization
        from common.world_model import WorldModel
        from omegaconf import DictConfig
        
        # Create minimal config
        cfg = DictConfig({
            'obs': 'rgb',
            'latent_dim': 512,
            'model_size': 5,
            'num_enc_layers': 2,
            'enc_dim': 256,
            'mlp_dim': 512,
            'num_channels': 32,
            'dropout': 0.01,
            'simnorm': False,
            'simnorm_dim': 8,
        })
        
        # Create encoder
        model = WorldModel(cfg).to('cuda')
        encoder = model._encoder[cfg.obs]
        
        # Create test RGB observations (uint8 -> should be converted to float)
        batch_size = 16
        rgb_obs = torch.randint(0, 256, (batch_size, 9, 84, 84), dtype=torch.uint8).to('cuda')
        print(f"Test observations: {rgb_obs.shape}, dtype: {rgb_obs.dtype}")
        
        # Test orthogonality regularization
        print("\n Testing orthogonality regularization...")
        ortho_loss = orthogonality_regularization(
            encoder, rgb_obs, device='cuda', latent_dim=cfg.latent_dim
        )
        print(f" Orthogonality loss: {ortho_loss.item():.6e}")
        
        # Test full rank regularization
        print("\n Testing full rank regularization...")
        fr_loss, abs_det = full_rank_regularization(
            encoder, rgb_obs, device='cuda', latent_dim=cfg.latent_dim
        )
        print(f" Full rank loss: {fr_loss.item():.6e}")
        print(f" Absolute determinant: {abs_det.item():.6e}")
        
        print("\n Regularization tests passed!")
        return True
        
    except Exception as e:
        print(f" Regularization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(" Starting TD-MPC2 Collapse Monitoring Tests")
    print("="*60)
    
    # Set environment variables
    os.environ['MUJOCO_GL'] = 'egl'
    
    # Test regularizations first
    reg_success = test_regularizations()
    
    # Test collapse monitor
    monitor_success = test_collapse_monitor()
    
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    print(f"Regularizations: {' PASS' if reg_success else ' FAIL'}")
    print(f"Collapse Monitor: {' PASS' if monitor_success else ' FAIL'}")
    
    if reg_success and monitor_success:
        print("\n ALL TESTS PASSED! The system is ready for training.")
        print("\n You can now run training with:")
        print("   python train.py task=cheetah-run steps=100000")
        exit(0)
    else:
        print("\n Some tests failed. Please check the errors above.")
        exit(1)
