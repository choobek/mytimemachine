#!/usr/bin/env python3
"""
Simple script to analyze training logs from terminal output
"""
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_logs_from_terminal():
    """Parse logs from terminal output"""
    # You can paste your terminal output here
    log_text = """
    # Paste your terminal output here
    """
    
    # Parse the logs
    steps = []
    losses = []
    id_losses = []
    l2_crop_losses = []
    lpips_crop_losses = []
    
    # Regex pattern to match log lines
    pattern = r'Metrics for train, step (\d+).*?loss =  ([\d.]+)'
    
    for line in log_text.split('\n'):
        if 'Metrics for train, step' in line:
            step_match = re.search(r'step (\d+)', line)
            if step_match:
                step = int(step_match.group(1))
                steps.append(step)
        
        if 'loss = ' in line:
            loss_match = re.search(r'loss =  ([\d.]+)', line)
            if loss_match:
                loss = float(loss_match.group(1))
                losses.append(loss)
        
        if 'loss_id_real = ' in line:
            id_match = re.search(r'loss_id_real =  ([\d.]+)', line)
            if id_match:
                id_loss = float(id_match.group(1))
                id_losses.append(id_loss)
        
        if 'loss_l2_crop = ' in line:
            l2_match = re.search(r'loss_l2_crop =  ([\d.]+)', line)
            if l2_match:
                l2_loss = float(l2_match.group(1))
                l2_crop_losses.append(l2_loss)
        
        if 'loss_lpips_crop = ' in line:
            lpips_match = re.search(r'loss_lpips_crop =  ([\d.]+)', line)
            if lpips_match:
                lpips_loss = float(lpips_match.group(1))
                lpips_crop_losses.append(lpips_loss)
    
    return steps, losses, id_losses, l2_crop_losses, lpips_crop_losses

def analyze_training_stability(steps, losses):
    """Analyze if training is stable"""
    if len(losses) < 10:
        return "Not enough data"
    
    # Calculate rolling average and standard deviation
    window = min(10, len(losses) // 2)
    rolling_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    rolling_std = np.array([np.std(losses[max(0, i-window):i+window]) for i in range(window, len(losses))])
    
    # Check for trends
    recent_losses = losses[-10:] if len(losses) >= 10 else losses
    trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
    
    # Stability analysis
    cv = np.std(losses) / np.mean(losses)  # Coefficient of variation
    
    analysis = {
        'trend': 'improving' if trend < -0.001 else 'stable' if abs(trend) < 0.001 else 'worsening',
        'stability': 'stable' if cv < 0.1 else 'moderate' if cv < 0.2 else 'unstable',
        'recent_avg': np.mean(recent_losses),
        'recent_std': np.std(recent_losses),
        'coefficient_of_variation': cv
    }
    
    return analysis

if __name__ == "__main__":
    print("Training Log Analyzer")
    print("=" * 50)
    print("To use this script:")
    print("1. Copy your terminal output")
    print("2. Paste it into the log_text variable")
    print("3. Run the script to get analysis")
    print("\nOr just share the terminal output directly - that's clearer than screenshots!")
