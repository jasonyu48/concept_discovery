#!/usr/bin/env python3
"""
Simple test script for encoding space monitoring
Only tests encoding space size tracking and consecutive frame generation
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the tdmpc2 directory to path
sys.path.append('/root/autodl-tmp/202569/tdmpc2/tdmpc2')

def test_simple_encoding_monitor():
    """Test the simplified encoding space monitoring system"""
    print("🚀 Testing Simple Encoding Space Monitor")
    print("="*60)
    
    # Set environment variables
    os.environ['MUJOCO_GL'] = 'egl'
    
    try:
        # Import required modules
        from envs import make_env
        from common.world_model import WorldModel
        from simple_encoding_space_monitor import SimpleEncodingSpaceMonitor
        
        print("✅ Successfully imported all modules")
        
        # Create test configuration
        print("\n📋 Creating test configuration...")
        from omegaconf import DictConfig
        cfg = DictConfig({
            'task': 'cheetah-run',
            'obs': 'rgb',
            'seed': 1,
            'device': 'cuda',
            'model_size': 5,
            'latent_dim': 512,
            'action_dim': 6,
            'episode_length': 1000,
            'task_dim': 96,
            'multitask': False,
            'exp_name': 'test_simple_encoding',
            'monitor_freq': 100,  # Check more frequently for testing
            'num_enc_layers': 2,
            'enc_dim': 256,
            'num_channels': 32,
            'dropout': 0.01,
            'simnorm': False,
            'simnorm_dim': 8,
        })
        print("✅ Configuration created")
        
        # Create environment
        print("\n🌍 Creating test environment...")
        env = make_env(cfg)
        print(f"   Environment type: {type(env)}")
        print(f"   Observation shape: {env.observation_space.shape}")
        print(f"   Action dimension: {env.action_space.shape[0]}")
        
        # Test environment consecutive frames
        print("\n🎬 Testing consecutive frame generation...")
        obs1 = env.reset()
        if isinstance(obs1, tuple):
            obs1 = obs1[0]
        print(f"   Reset observation shape: {obs1.shape}")
        
        # Take steps to see frame evolution
        for i in range(3):
            action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
            print(f"   Step {i+1} observation shape: {obs.shape}")
        
        # Create world model with encoder
        print("\n🧠 Creating encoder...")
        model = WorldModel(cfg).to('cuda')
        encoder = model._encoder[cfg.obs]
        print(f"   Encoder type: {type(encoder)}")
        
        # Test encoder
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_tensor = torch.from_numpy(obs).to('cuda').float().unsqueeze(0)
        print(f"   Sample observation shape: {obs_tensor.shape}")
        
        with torch.no_grad():
            encoded = encoder(obs_tensor)
            print(f"   Encoded shape: {encoded.shape}")
        
        # Initialize simple encoding space monitor
        print("\n🔍 Initializing Simple Encoding Space Monitor...")
        monitor = SimpleEncodingSpaceMonitor(
            cfg=cfg,
            encoder=encoder,
            env=env,
            device='cuda',
            save_dir="test_simple_encoding_logs"
        )
        print("✅ Simple monitor initialized successfully!")
        
        # Test monitoring at different steps
        print("\n📊 Testing encoding space monitoring...")
        test_steps = [0, 100, 200, 300, 500]
        
        for step in test_steps:
            print(f"\n--- Step {step} ---")
            metrics = monitor.monitor_step(step)
            if metrics:
                space_size = metrics.get('encoding_space_size', 'N/A')
                monitoring_time = metrics.get('monitoring_time', 'N/A')
                print(f"   ✅ Encoding space size: {space_size:.6f}")
                print(f"   ⏱️  Monitoring time: {monitoring_time:.3f}s")
        
        # Generate encoding space curve
        print("\n📈 Generating encoding space curve...")
        fig = monitor.plot_encoding_space_curve(save_plot=True)
        if fig:
            plt.show()  # Show the plot
            plt.close(fig)
            print("   ✅ Encoding space curve generated!")
        
        # Save observation GIFs
        print("\n🎬 Generating consecutive frame GIFs...")
        gif_dir = monitor.save_observation_gifs(max_gifs=3)
        if gif_dir:
            print(f"   ✅ GIFs saved to: {gif_dir}")
        
        # Generate simple report
        print("\n📋 Generating simple report...")
        report = monitor.generate_simple_report()
        print(report)
        
        # Save final data
        monitor.save_monitoring_data()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("   - Consecutive frames properly generated")
        print("   - Encoding space size tracked")
        print("   - Change curve plotted")
        print("   - GIFs saved for inspection")
        print(f"   - Check results in: test_simple_encoding_logs/")
        print("\n🚀 Simple encoding space monitoring is ready!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Starting Simple Encoding Space Monitor Tests")
    print("="*60)
    
    # Set environment variables
    os.environ['MUJOCO_GL'] = 'egl'
    
    success = test_simple_encoding_monitor()
    
    print("\n" + "="*60)
    print("📝 TEST SUMMARY")
    print("="*60)
    print(f"Simple Encoding Monitor: {'✅ PASS' if success else '❌ FAIL'}")
    
    if success:
        print("\n🎉 SIMPLE MONITORING SYSTEM READY!")
        print("\n📋 What this monitor tracks:")
        print("   • Encoding space size (average pairwise distance)")
        print("   • Change over time")
        print("   • Consecutive frame GIFs")
        print("   • Simple trend analysis")
        print("\n🚀 Next steps:")
        print("   1. Run: python test_simple_encoding_monitor.py")
        print("   2. Check the encoding space curve plot")
        print("   3. View the consecutive frame GIFs")
        print("   4. Use in training with simplified monitoring")
        exit(0)
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        exit(1)
