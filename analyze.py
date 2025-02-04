import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import glob
import subprocess
import sys
import os
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

# Set style
plt.style.use('dark_background')

# Update the common style dictionary with more consistent colors
PLOT_STYLE = {
    'figure.facecolor': '#1a1a1a',
    'axes.facecolor': '#1a1a1a',
    'axes.edgecolor': '#666666',
    'grid.color': '#333333',
    'text.color': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff',
    'figure.dpi': 150,
    'axes.grid': False,  # Changed to False by default
    'grid.alpha': 0.2
}

# Define consistent colors for metrics
METRIC_COLORS = {
    'throughput': '#00FF00',  # Bright Green for all throughput metrics
    'wait': '#FF00FF',        # Bright Magenta for all wait/TTFB metrics
    'secondary': '#4B4BFF'    # Bright Blue for secondary metrics
}

plt.rcParams.update(PLOT_STYLE)

# Update color schemes to be more consistent
CONCURRENCY_COLORS = {
    1: '#4BFF4B',   # Bright Green
    2: '#FF4B4B',   # Bright Red
    4: '#4B4BFF',   # Bright Blue
    8: '#FFFF4B',   # Bright Yellow
    12: '#FF4BFF',  # Bright Magenta
    16: '#4BFFFF',  # Bright Cyan
    24: '#FF8C4B'   # Bright Orange
}

PROCESS_COLORS = {
    92987: '#4BFF4B',  # Bright Green
    93050: '#4B4BFF'   # Bright Blue
}

def print_table(title, headers, rows, widths=None):
    """Helper function to print formatted tables to console"""
    if not widths:
        widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    
    # Print title
    print(f"\n{title}:")
    print("=" * sum(widths + [len(widths) * 3 - 1]))
    
    # Print headers
    header_str = " | ".join(f"{h:>{w}}" for h, w in zip(headers, widths))
    print(header_str)
    print("-" * sum(widths + [len(widths) * 3 - 1]))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(f"{str(cell):>{w}}" for cell, w in zip(row, widths))
        print(row_str)
    
    print("=" * sum(widths + [len(widths) * 3 - 1]))

def create_data_dir(model):
    """Create data directory for model specific outputs"""
    data_dir = f"data/{model}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def load_test_results(model):
    results = {}
    fttb_metrics = {}
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, f'ollama/{model}')
    
    # Load all test results
    for test_dir in glob.glob(f'{base_dir}/test_results_*'):
        try:
            concurrency = int(test_dir.split('_')[-1])
            
            # Load request metrics
            summary_path = os.path.join(test_dir, "request_summary.csv")
            if not os.path.exists(summary_path):
                continue
                
            # Read the summary metrics
            summary_df = pd.read_csv(summary_path)
            
            # Store FTTB metrics
            fttb_metrics[concurrency] = summary_df[['RequestID', 'TTFB']].set_index('RequestID')
            
            # Calculate throughput from summary data
            throughput_data = []
            for _, row in summary_df.iterrows():
                # Calculate tokens per second
                throughput = row['TokenCount'] / row['ResponseDuration'] if row['ResponseDuration'] > 0 else 0
                
                # Create a DataFrame with throughput info
                req_df = pd.DataFrame({
                    'RequestID': [row['RequestID']],
                    'Timestamp': [pd.Timestamp.now()],  # Using current time as placeholder
                    'Throughput': [throughput],
                    'TokenCount': [row['TokenCount']],
                    'WaitingTime': [row['WaitingTime']],
                    'Status': [row['Status']],
                    'ConcurrencyLevel': [concurrency]
                })
                throughput_data.append(req_df)
            
            if throughput_data:
                results[concurrency] = pd.concat(throughput_data, ignore_index=True)
                
        except Exception as e:
            print(f"Error loading results for concurrency {concurrency}: {e}")
            continue
    
    return results, fttb_metrics

def load_process_metrics(model):
    process_metrics = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, f'ollama/{model}')
    
    for test_dir in glob.glob(f'{base_dir}/test_results_*'):
        for metrics_file in glob.glob(f'{test_dir}/process_*_metrics.csv'):
            try:
                pid = int(metrics_file.split('process_')[1].split('_')[0])
                df = pd.read_csv(metrics_file)
                
                # Fix timestamp parsing - handle both cases
                df['Timestamp'] = df['Timestamp'].str.replace('025-', '2025-')
                df['Timestamp'] = df['Timestamp'].str.replace('22025-', '2025-')
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%dT%H:%M:%S%z')
                
                if pid in process_metrics:
                    process_metrics[pid] = pd.concat([process_metrics[pid], df], ignore_index=True)
                else:
                    process_metrics[pid] = df
                    
            except Exception as e:
                print(f"Error loading process metrics from {metrics_file}: {e}")
                continue
    
    return process_metrics

def load_gpu_metrics(model):
    gpu_metrics = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, f'ollama/{model}')
    
    for test_dir in glob.glob(f'{base_dir}/test_results_*'):
        try:
            concurrency = int(test_dir.split('_')[-1])
            gpu_file = os.path.join(test_dir, 'gpu_metrics.csv')
            
            if os.path.exists(gpu_file):
                df = pd.read_csv(gpu_file)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                gpu_metrics[concurrency] = df
                
        except Exception as e:
            print(f"Error loading GPU metrics from {test_dir}: {e}")
            continue
    
    return gpu_metrics

def load_duration_metrics(model):
    duration_metrics = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, f'ollama/{model}')
    
    for test_dir in glob.glob(f'{base_dir}/test_results_*'):
        try:
            concurrency = int(test_dir.split('_')[-1])
            metrics_file = os.path.join(test_dir, 'duration_metrics.csv')
            
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                duration_metrics[concurrency] = df
                
        except Exception as e:
            print(f"Error loading duration metrics from {test_dir}: {e}")
            continue
    
    return duration_metrics

# Add common chart styling function
def apply_chart_style(ax, title, xlabel=None, ylabel=None):
    """Apply consistent styling to chart axes"""
    ax.set_facecolor('#1a1a1a')
    ax.grid(False)  # Ensure grid is off
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('#666666')
    
    if title:
        ax.set_title(title, color='white', pad=20, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, color='white', fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color='white', fontsize=10)
    
    return ax

def create_figure(figsize=(12, 6)):
    """Create a figure with consistent styling"""
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#1a1a1a')
    return fig

def plot_test_durations(results):
    # Add summary table
    print("\nGeneration Speed Analysis Summary")
    headers = ['Concurrency', 'Tokens/s', 'Change %']
    rows = []
    base_throughput = None
    
    for concurrency in sorted(results.keys()):
        df = results[concurrency]
        final_throughputs = []
        for req_id in df['RequestID'].unique():
            req_data = df[df['RequestID'] == req_id]
            if not req_data.empty:
                final_throughputs.append(req_data['Throughput'].iloc[-1])
        avg_throughput = np.mean(final_throughputs)
        
        if base_throughput is None:
            base_throughput = avg_throughput
            change_pct = '-'
        else:
            change_pct = f"{((avg_throughput / base_throughput) - 1) * 100:+.1f}%"
            
        rows.append([concurrency, f"{avg_throughput:.1f}", change_pct])
    
    print_table("Generation Speed Analysis", headers, rows)
    
    fig = create_figure()
    ax = fig.add_subplot(111)
    
    apply_chart_style(
        ax,
        'DeepSeek R1 7B: Generation Speed Analysis',
        'Parallel Requests',
        'Tokens/s'
    )
    
    concurrencies = []
    throughputs = []
    
    # Brighter colors for better visibility
    BRIGHT_COLORS = {
        1: '#FF0000',    # Bright Red
        2: '#00FF00',    # Bright Green
        4: '#0088FF',    # Bright Blue
        8: '#FF00FF',    # Bright Magenta
        12: '#FFFF00',   # Bright Yellow
        16: '#00FFFF',   # Bright Cyan
        24: '#FF8800'    # Bright Orange
    }
    
    for concurrency in sorted(results.keys()):
        df = results[concurrency]
        final_throughputs = []
        for req_id in df['RequestID'].unique():
            req_data = df[df['RequestID'] == req_id]
            if not req_data.empty:
                final_throughputs.append(req_data['Throughput'].iloc[-1])
        avg_throughput = np.mean(final_throughputs)
        
        ax.scatter(concurrency, avg_throughput,
                  label=f'{concurrency} concurrent',
                  color=BRIGHT_COLORS[concurrency],
                  s=100,  # larger markers
                  alpha=0.8,
                  edgecolor='white',
                  linewidth=2)
        
        # Add text annotation for each point
        ax.annotate(f'{avg_throughput:.1f}', 
                   (concurrency, avg_throughput),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center',
                   color='white',
                   bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
    
    ax.set_xlim(0, 25)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

def plot_token_distribution(results):
    # Add summary table
    print("\nToken Distribution Summary")
    headers = ['Concurrency', 'Avg t/s', 'Max t/s', 'Min t/s']
    rows = []
    
    for concurrency in sorted(results.keys()):
        df = results[concurrency]
        if df.empty:
            continue
            
        plot_df = df.copy()
        plot_df = plot_df[plot_df['Throughput'] <= 200]
        
        rows.append([
            concurrency,
            f"{plot_df['Throughput'].mean():.1f}",
            f"{plot_df['Throughput'].max():.1f}",
            f"{plot_df['Throughput'].min():.1f}"
        ])
    
    print_table("Token Distribution", headers, rows)
    
    fig = create_figure()
    ax = fig.add_subplot(111)
    
    apply_chart_style(
        ax,
        'DeepSeek-7B Performance Under Different Load Levels\n(via Ollama)',
        'Time (seconds)',
        'Generation Speed (tokens/s)'
    )
    
    for concurrency in sorted(results.keys()):
        df = results[concurrency]
        
        # Skip if DataFrame is empty
        if df.empty:
            continue
            
        # Create a copy to avoid SettingWithCopyWarning
        plot_df = df.copy()
        
        # Filter out unrealistic values (e.g., above 200 tokens/s)
        plot_df = plot_df[plot_df['Throughput'] <= 200]
        
        start_time = plot_df['Timestamp'].min()
        plot_df['RelativeTime'] = (plot_df['Timestamp'] - start_time).dt.total_seconds()
        
        ax.scatter(plot_df['RelativeTime'], plot_df['Throughput'],
                  label=f'{concurrency} concurrent requests',
                  color=CONCURRENCY_COLORS.get(concurrency, '#FFFFFF'),
                  alpha=0.6,
                  s=30)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_performance_metrics(results, fttb_metrics):
    # Add summary table
    print("\nPerformance Metrics Summary")
    headers = ['Concurrency', 'Throughput', 'TTFB (s)', 'Change %']
    rows = []
    base_throughput = None
    
    for concurrency in sorted(results.keys()):
        df = results[concurrency]
        ttfb_data = fttb_metrics[concurrency]
        
        final_throughputs = []
        for req_id in df['RequestID'].unique():
            req_data = df[df['RequestID'] == req_id]
            if not req_data.empty:
                final_throughputs.append(req_data['Throughput'].iloc[-1])
        
        avg_throughput = np.mean(final_throughputs)
        avg_ttfb = ttfb_data['TTFB'].mean()
        
        if base_throughput is None:
            base_throughput = avg_throughput
            change_pct = '-'
        else:
            change_pct = f"{((avg_throughput / base_throughput) - 1) * 100:+.1f}%"
            
        rows.append([
            concurrency,
            f"{avg_throughput:.1f}",
            f"{avg_ttfb:.2f}",
            change_pct
        ])
    
    print_table("Performance Metrics", headers, rows)
    
    fig = create_figure()
    ax = fig.add_subplot(111)
    
    apply_chart_style(
        ax,
        'Average Throughput by Parallel Requests',
        'Parallel Requests',
        'Tokens/s'
    )
    
    if not results:
        print("No results data available!")
        return
        
    # Calculate metrics
    concurrency_levels = sorted(results.keys())
    throughputs = []
    ttfbs = []
    
    for concurrency in concurrency_levels:
        df = results[concurrency]
        ttfb_data = fttb_metrics[concurrency]
        
        # Calculate final throughput for each request
        final_throughputs = []
        for req_id in df['RequestID'].unique():
            req_data = df[df['RequestID'] == req_id]
            if not req_data.empty:
                final_throughputs.append(req_data['Throughput'].iloc[-1])
        
        throughputs.append(np.mean(final_throughputs))
        ttfbs.append(ttfb_data['TTFB'].mean())
    
    # Plot throughput
    ax.plot(concurrency_levels, throughputs, 'o-', 
            label='Avg Throughput', color='#00FF00', linewidth=2,
            markeredgecolor='white', markeredgewidth=2)
    
    # Add value annotations
    for i, (x, y) in enumerate(zip(concurrency_levels, throughputs)):
        ax.annotate(f'{y:.1f}', (x, y),
                   textcoords="offset points", xytext=(0,10),
                   ha='center', color='white',
                   bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
    
    # Style the plot
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.legend(facecolor='#1a1a1a', labelcolor='white')
    
    plt.tight_layout()
    return fig

def plot_process_metrics(process_metrics):
    # Get all test directories and sort them numerically
    test_dirs = sorted(glob.glob('ollama/deepseek-r1-7b/test_results_*'), 
                      key=lambda x: int(x.split('_')[-1]))
    
    # TODO: Make PIDs configurable via command line or config file
    target_pids = [40163, 42567]  # Focus only on these PIDs - ollama and ollama-cuda processes
    
    # Create separate tables for each PID
    for pid in target_pids:
        headers = ['Concurrency', 'Avg CPU%', 'Max CPU%', 'Avg Mem%', 'Max Mem%', 'Avg Threads', 'Avg FDs', 'Avg RAM(MB)', 'Max RAM(MB)']
        rows = []
        
        for test_dir in test_dirs:
            concurrency = int(test_dir.split('_')[-1])
            metrics_file = f'{test_dir}/process_{pid}_metrics.csv'
            
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                rows.append([
                    concurrency,
                    f"{df['CPU%'].mean():.1f}",
                    f"{df['CPU%'].max():.1f}",
                    f"{df['MEM%'].mean():.1f}",
                    f"{df['MEM%'].max():.1f}",
                    f"{df['Threads'].mean():.1f}",
                    f"{df['FDs'].mean():.1f}",
                    f"{df['RSS'].mean() / 1024:.1f}",  # Convert to MB
                    f"{df['RSS'].max() / 1024:.1f}"    # Convert to MB
                ])
        
        # Sort rows by concurrency
        rows.sort(key=lambda x: x[0])
        print_table(f"Process {pid} Resource Usage", headers, rows)
    
    if not process_metrics:
        print("No process metrics data available!")
        return
        
    # Create a 2x3 grid for 6 metrics
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Add main title with more space
    fig.suptitle('Ollama Process Metrics - DeepSeek R1 7B @ M2 Pro', 
                 color='white', fontsize=16, y=0.98)
    
    # Increase spacing between title and plots
    plt.subplots_adjust(top=0.9)
    
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.grid(False)  # Remove grids
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    colors = ['#4BFF4B', '#FF4B4B']  # One color per PID
    
    # Get all test directories and sort them numerically
    test_dirs = sorted(glob.glob('ollama/deepseek-r1-7b/test_results_*'), 
                      key=lambda x: int(x.split('_')[-1]))
    concurrency_levels = [int(d.split('_')[-1]) for d in test_dirs]
    
    # TODO: Make PIDs configurable via command line or config file
    target_pids = [40163, 42567]  # Focus only on these PIDs - ollama and ollama-cuda processes
    
    for pid_idx, pid in enumerate(target_pids):
        metrics_by_concurrency = {}
        
        # First collect all metrics by concurrency level
        for test_dir in test_dirs:
            concurrency = int(test_dir.split('_')[-1])
            metrics_file = f'{test_dir}/process_{pid}_metrics.csv'
            
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
                metrics_by_concurrency[concurrency] = {
                    'cpu': metrics_df['CPU%'].mean() if 'CPU%' in metrics_df.columns else 0,
                    'mem': metrics_df['MEM%'].mean() if 'MEM%' in metrics_df.columns else 0,
                    'rss': metrics_df['RSS'].mean() / (1024 * 1024) if 'RSS' in metrics_df.columns else 0,  # Convert to MB
                    'threads': metrics_df['Threads'].mean() if 'Threads' in metrics_df.columns else 0,
                    'fds': metrics_df['FDs'].mean() if 'FDs' in metrics_df.columns else 0,
                    'vms': metrics_df['VMS'].mean() / (1024 * 1024) if 'VMS' in metrics_df.columns else metrics_df['RSS'].mean() / (1024 * 1024) if 'RSS' in metrics_df.columns else 0
                }
        
        if not metrics_by_concurrency:
            continue
            
        # Prepare data for plotting
        x_values = sorted(metrics_by_concurrency.keys())
        cpu_values = [metrics_by_concurrency[x]['cpu'] for x in x_values]
        mem_values = [metrics_by_concurrency[x]['mem'] for x in x_values]
        rss_values = [metrics_by_concurrency[x]['rss'] for x in x_values]
        thread_values = [metrics_by_concurrency[x]['threads'] for x in x_values]
        fd_values = [metrics_by_concurrency[x]['fds'] for x in x_values]
        vms_values = [metrics_by_concurrency[x]['vms'] for x in x_values]
        
        color = colors[pid_idx]
        
        # Plot all metrics
        metrics_to_plot = [
            (ax1, cpu_values, 'CPU Usage'),
            (ax2, mem_values, 'Memory Usage'),
            (ax3, rss_values, 'Resident Memory'),
            (ax4, thread_values, 'Thread Count'),
            (ax5, fd_values, 'File Descriptors'),
            (ax6, vms_values, 'Virtual Memory')
        ]
        
        for ax, values, title in metrics_to_plot:
            ax.plot(x_values, values, 'o-', label=f'PID {pid}', color=color, linewidth=2)
            # Add value annotations
            for x, y in zip(x_values, values):
                ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10),
                           ha='center', color='white',
                           bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
    
    # Set labels and titles with units
    ax1.set_ylabel('CPU Usage (%)', color='white')
    ax1.set_title('CPU Usage', color='white', pad=20)
    
    ax2.set_ylabel('Memory Usage (%)', color='white')
    ax2.set_title('Memory Usage', color='white', pad=20)
    
    ax3.set_ylabel('Memory (MB)', color='white')
    ax3.set_title('Resident Memory (RAM) - MB', color='white', pad=20)
    
    ax4.set_ylabel('Count', color='white')
    ax4.set_title('Thread Count', color='white', pad=20)
    
    ax5.set_ylabel('Count', color='white')
    ax5.set_title('File Descriptors', color='white', pad=20)
    
    ax6.set_ylabel('Memory (MB)', color='white')
    ax6.set_title('Virtual Memory Size - MB', color='white', pad=20)
    
    # Add legends and set axis limits
    for ax in axes:
        ax.legend(facecolor='#1a1a1a', labelcolor='white', loc='upper left')
        ax.set_xlim(0, max(concurrency_levels) * 1.1)
        ax.set_xticks(concurrency_levels)
        ax.set_xlabel('Parallel Requests', color='white')
    
    plt.tight_layout()
    return fig

def analyze_gpu_metrics(gpu_metrics):
    summary = {}
    
    # Get global start and end times
    all_timestamps = []
    for df in gpu_metrics.values():
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        all_timestamps.extend(df['Timestamp'].tolist())
    
    if not all_timestamps:
        return {'total': {
            'total_duration_sec': 0,
            'avg_power_w': 0,
            'total_energy_wh': 0,
            'total_energy_kwh': 0,
            'cost_brl': 0
        }}
    
    total_duration_sec = (max(all_timestamps) - min(all_timestamps)).total_seconds()
    total_energy_wh = 0
    total_power_w = 0
    total_samples = 0
    
    for concurrency, df in gpu_metrics.items():
        if df.empty:
            continue
            
        # Calculate time delta between measurements in seconds
        test_duration = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
        
        df['TimeDelta'] = df['Timestamp'].diff().dt.total_seconds()
        
        # Handle first row's time delta
        if len(df) > 1:
            df.loc[df.index[0], 'TimeDelta'] = df['TimeDelta'].iloc[1]
        else:
            df.loc[df.index[0], 'TimeDelta'] = 1  # Default to 1 second if only one row
        
        # Calculate energy used (power * time in seconds = energy in joules)
        df['Energy'] = df['Power(W)'] * df['TimeDelta']
        
        # Calculate average power and total energy
        total_power_w += df['Power(W)'].sum()
        total_samples += len(df)
        energy_wh = df['Energy'].sum() / 3600  # Convert joules to watt-hours
        total_energy_wh += energy_wh
        
        summary[concurrency] = {
            'avg_usage': df['Usage%'].mean(),
            'max_usage': df['Usage%'].max(),
            'avg_power': df['Power(W)'].mean(),
            'max_power': df['Power(W)'].max(),
            'avg_freq': df['Frequency(MHz)'].mean(),
            'max_freq': df['Frequency(MHz)'].max(),
            'energy_wh': energy_wh,
            'duration_sec': test_duration
        }
    
    # Calculate overall average power
    avg_power_w = total_power_w / total_samples if total_samples > 0 else 0
    
    # Add totals to summary
    summary['total'] = {
        'total_duration_sec': total_duration_sec,
        'avg_power_w': avg_power_w,
        'total_energy_wh': total_energy_wh,
        'total_energy_kwh': total_energy_wh / 1000,
        'cost_brl': (total_energy_wh / 1000) * 0.7  # R$ 0.70 per kWh
    }
    
    return summary

def plot_gpu_metrics(gpu_metrics):
    # Add summary table
    print("\nGPU Metrics Summary")
    headers = ['Concurrency', 'Avg Usage%', 'Max Power(W)', 'Avg Freq(MHz)']
    rows = []
    
    for concurrency, df in sorted(gpu_metrics.items()):
        rows.append([
            concurrency,
            f"{df['Usage%'].mean():.1f}",
            f"{df['Power(W)'].max():.1f}",
            f"{df['Frequency(MHz)'].mean():.0f}"
        ])
    
    print_table("GPU Metrics", headers, rows)
    
    if not gpu_metrics:
        print("No GPU metrics data available!")
        return None
        
    # Check if we have any data points
    if not any(len(df) > 0 for df in gpu_metrics.values()):
        print("GPU metrics dataframes are empty!")
        return None
    
    fig = create_figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 4, figure=fig, height_ratios=[1, 2], hspace=0.4)
    
    # Calculate metrics by concurrency level
    concurrency_metrics = {}
    total_energy = 0
    
    # Get global start and end times for total duration
    all_timestamps = []
    for df in gpu_metrics.values():
        if not df.empty:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            all_timestamps.extend(df['Timestamp'].tolist())
    
    if not all_timestamps:
        print("No timestamp data available in GPU metrics!")
        return None
    
    total_duration = (max(all_timestamps) - min(all_timestamps)).total_seconds()
    
    for concurrency, df in sorted(gpu_metrics.items()):
        # Calculate time delta and energy
        test_duration = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
        
        df['TimeDelta'] = df['Timestamp'].diff().dt.total_seconds()
        df.loc[df.index[0], 'TimeDelta'] = df['TimeDelta'].iloc[1]
        df['Energy'] = df['Power(W)'] * df['TimeDelta']
        
        power_w = df['Power(W)']
        energy_wh = df['Energy'].sum() / 3600  # Convert joules to watt-hours
        total_energy += energy_wh
        
        concurrency_metrics[concurrency] = {
            'power_avg': power_w.mean(),
            'power_max': power_w.max(),
            'freq_avg': df['Frequency(MHz)'].mean(),
            'usage_avg': df['Usage%'].mean(),
            'energy_wh': energy_wh,
            'duration_sec': test_duration
        }
    
    # Calculate cost in BRL
    total_kwh = total_energy / 1000
    cost_brl = total_kwh * 0.7
    
    # Format duration using total time span
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    duration_str = f'{hours}h {minutes}m {seconds}s' if hours > 0 else f'{minutes}m {seconds}s'
    
    # Create metric cards
    metrics = [
        ('AVG POWER', f'{np.mean([m["power_avg"] for m in concurrency_metrics.values()]):.1f}W'),
        ('TOTAL ENERGY', f'{total_energy:.2f}Wh'),
        ('ENERGY COST', f'R${cost_brl:.4f}'),
        ('TOTAL TIME', duration_str)
    ]
    
    # Plot metric cards
    for idx, (title, value) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('#2a2a2a')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Value in larger font
        ax.text(0.5, 0.6, value, 
                ha='center', va='center', 
                color='white', 
                fontsize=24, 
                fontweight='bold')
        
        # Title in smaller font above
        ax.text(0.5, 0.25, title,
                ha='center', va='center',
                color='#888888',
                fontsize=12,
                fontweight='bold')
    
    # Create line charts
    ax_power = fig.add_subplot(gs[1, :2])
    ax_freq = fig.add_subplot(gs[1, 2:])
    
    # Style the line chart axes
    for ax in [ax_power, ax_freq]:
        apply_chart_style(
            ax,
            None,
            'Parallel Requests',
            'Power (W)' if ax == ax_power else 'Frequency (MHz)'
        )
    
    # Prepare data for plotting
    concurrency_levels = sorted(concurrency_metrics.keys())
    x = np.arange(len(concurrency_levels))
    
    # Plot power metrics
    power_avg = [concurrency_metrics[c]['power_avg'] for c in concurrency_levels]
    power_max = [concurrency_metrics[c]['power_max'] for c in concurrency_levels]
    
    ax_power.plot(x, power_avg, 'o-', 
                 label='Average', color='#00FF00', linewidth=3, markersize=10,
                 markeredgecolor='white', markeredgewidth=2)
    ax_power.plot(x, power_max, 'o-', 
                 label='Maximum', color='#FF00FF', linewidth=3, markersize=10,
                 markeredgecolor='white', markeredgewidth=2)
    
    # Add value annotations for power
    for i, (avg, max_val) in enumerate(zip(power_avg, power_max)):
        ax_power.annotate(f'{avg:.1f}W', (x[i], avg), 
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', color='white',
                         bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
        ax_power.annotate(f'{max_val:.1f}W', (x[i], max_val), 
                         textcoords="offset points", xytext=(0, -20),
                         ha='center', color='white',
                         bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
    
    # Plot frequency metrics
    freq_avg = [concurrency_metrics[c]['freq_avg'] for c in concurrency_levels]
    freq_p99 = [np.percentile(gpu_metrics[c]['Frequency(MHz)'], 99) for c in concurrency_levels]
    
    ax_freq.plot(x, freq_avg, 'o-', 
                label='Average', color='#00FF00', linewidth=2, markersize=8,
                markeredgecolor='white', markeredgewidth=2)
    ax_freq.plot(x, freq_p99, 'o-', 
                label='P99', color='#FF00FF', linewidth=2, markersize=8,
                markeredgecolor='white', markeredgewidth=2)
    
    # Add frequency annotations
    for i, (avg, p99) in enumerate(zip(freq_avg, freq_p99)):
        ax_freq.annotate(f'{avg:.0f}MHz', (x[i], avg), 
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', color='white',
                        bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
        ax_freq.annotate(f'{p99:.0f}MHz', (x[i], p99), 
                        textcoords="offset points", xytext=(0, -20),
                        ha='center', color='white',
                        bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
    
    # Update labels and titles
    ax_power.set_xlabel('Parallel Requests', color='white')
    ax_power.set_ylabel('Power (W)', color='white')
    ax_power.set_title('Power Usage by Parallel Requests', color='white', pad=20)
    ax_power.legend(facecolor='#1a1a1a', labelcolor='white')
    ax_power.set_xticks(x)
    ax_power.set_xticklabels(concurrency_levels)
    
    ax_freq.set_xlabel('Parallel Requests', color='white')
    ax_freq.set_ylabel('Frequency (MHz)', color='white')
    ax_freq.set_title('GPU Frequency by Parallel Requests', color='white', pad=20)
    ax_freq.legend(facecolor='#1a1a1a', labelcolor='white')
    ax_freq.set_xticks(x)
    ax_freq.set_xticklabels(concurrency_levels)
    
    plt.tight_layout()
    return fig

def analyze_throughput(results, duration_metrics):
    metrics = []
    
    for concurrency in sorted(results.keys()):
        df = results[concurrency]
        duration_df = duration_metrics.get(concurrency)
        
        # Calculate metrics per request first
        req_metrics = []
        for req_id in df['RequestID'].unique():
            req_data = df[df['RequestID'] == req_id]
            if not req_data.empty:
                status = req_data['Status'].iloc[0] if 'Status' in df.columns else 'Unknown'
                is_successful = status == 'Completed'
                
                if is_successful:
                    req_metrics.append({
                        'throughput': req_data['Throughput'].iloc[-1],
                        'waiting_time': req_data['WaitingTime'].iloc[0],
                    })
        
        # Calculate statistics across successful requests
        if req_metrics and duration_df is not None and not duration_df.empty:
            throughputs = [m['throughput'] for m in req_metrics]
            waiting_times = [m['waiting_time'] for m in req_metrics]
            
            total_requests = len(df['RequestID'].unique())
            error_requests = len(df[df['Status'].str.startswith('Error')]['RequestID'].unique()) if 'Status' in df.columns else 0
            error_rate = (error_requests / total_requests) * 100 if total_requests > 0 else 0
            
            metric = {
                'concurrency': concurrency,
                'avg_throughput': np.mean(throughputs),
                'p95_throughput': np.percentile(throughputs, 95),
                'p99_throughput': np.percentile(throughputs, 99),
                'avg_waiting_time': np.mean(waiting_times),
                'error_rate': error_rate,
                'avg_duration': duration_df['AvgDuration'].iloc[0],
                'p99_duration': duration_df['P99Duration'].iloc[0],
                'total_tokens': df[df['Status'] == 'Completed']['TokenCount'].sum() if 'Status' in df.columns else df['TokenCount'].sum()
            }
        else:
            metric = {
                'concurrency': concurrency,
                'avg_throughput': 0,
                'p95_throughput': 0,
                'p99_throughput': 0,
                'avg_waiting_time': 0,
                'error_rate': 100,
                'avg_duration': 0,
                'p99_duration': 0,
                'total_tokens': 0
            }
            
        metrics.append(metric)
    
    return pd.DataFrame(metrics)

def analyze_process_metrics(process_metrics):
    summary = {}
    
    for pid, df in process_metrics.items():
        summary[pid] = {
            'avg_cpu': df['CPU%'].mean(),
            'max_cpu': df['CPU%'].max(),
            'avg_mem': df['MEM%'].mean(),
            'max_mem': df['MEM%'].max(),
            'avg_threads': df['Threads'].mean(),
            'max_rss': df['RSS'].max()
        }
    
    return summary

def generate_markdown_report(model, results, process_metrics, gpu_metrics, duration_metrics):
    df_metrics = analyze_throughput(results, duration_metrics)
    process_summary = analyze_process_metrics(process_metrics)
    gpu_summary = analyze_gpu_metrics(gpu_metrics)
    
    # Extract and remove the total from gpu_summary
    gpu_totals = gpu_summary.pop('total')
    
    report = []
    
    # Header
    report.append("# DeepSeek R1 7B Performance Analysis\n")
    
    # Add GPU Energy Analysis section
    report.append("## GPU Energy Analysis\n")
    hours = int(gpu_totals['total_duration_sec'] // 3600)
    minutes = int((gpu_totals['total_duration_sec'] % 3600) // 60)
    seconds = int(gpu_totals['total_duration_sec'] % 60)
    duration_str = f'{hours}h {minutes}m {seconds}s' if hours > 0 else f'{minutes}m {seconds}s'
    
    report.append(f"- **Total Test Duration:** {duration_str}")
    report.append(f"- **Average Power:** {gpu_totals['avg_power_w']:.1f}W")
    report.append(f"- **Total Energy:** {gpu_totals['total_energy_wh']:.2f}Wh ({gpu_totals['total_energy_kwh']:.4f}kWh)")
    report.append(f"- **Energy Cost:** R${gpu_totals['cost_brl']:.4f}\n")
    
    # GPU Metrics Table
    report.append("### GPU Metrics by Parallel Request Level\n")
    report.append("| Parallel Requests | Avg Power (W) | Energy (Wh) | Duration | Avg GPU Usage % |")
    report.append("|-------------------|---------------|-------------|-----------|----------------|")
    
    for concurrency, metrics in sorted(gpu_summary.items()):
        duration = f"{metrics['duration_sec']:.1f}s"
        report.append(
            f"| {concurrency:>17} | {metrics['avg_power']:>11.1f} | "
            f"{metrics['energy_wh']:>10.2f} | {duration:>9} | "
            f"{metrics['avg_usage']:>14.1f} |"
        )
    
    # Rest of the report sections
    report.append("\n## Performance\n")
    report.append("![Performance Metrics](./performance_metrics.png)\n")
    report.append("This chart shows the throughput scaling and test duration across different parallel request levels.\n")
    
    report.append("## Token Generation Distribution\n")
    report.append("![Token Distribution](./token_distribution.png)\n")
    report.append("This visualization shows the token generation speed over time for different concurrency levels.\n")
    
    report.append("## Process Resource Usage\n")
    report.append("![Process Metrics](./process_metrics.png)\n")
    report.append("This chart shows CPU, Memory, and Thread usage of the Ollama processes during the test.\n")
    
    # Detailed Metrics Table
    report.append("## Detailed Metrics\n")
    report.append("| Concurrency | Avg Tokens/s | P95 Tokens/s | P99 Tokens/s | Avg Waiting Time | Error Rate | Duration | P99 Duration | Total Tokens |")
    report.append("|------------|--------------|--------------|--------------|-----------------|------------|-----------|--------------|--------------|")
    
    for _, row in df_metrics.iterrows():
        avg_duration = format_duration(row['avg_duration'])
        p99_duration = format_duration(row['p99_duration'])
        
        report.append(
            f"| {row['concurrency']:>10} | {row['avg_throughput']:>11.1f} | "
            f"{row['p95_throughput']:>11.1f} | {row['p99_throughput']:>11.1f} | "
            f"{row['avg_waiting_time']:>14.2f} | {row['error_rate']:>10.1f} | "
            f"{avg_duration:>8} | {p99_duration:>11} | {row['total_tokens']:>11,.0f} |"
        )
    
    # Key Findings
    report.append("\n## Key Findings\n")
    
    best_throughput = df_metrics.loc[df_metrics['avg_throughput'].idxmax()]
    report.append(f"- **Optimal Concurrency:** {best_throughput['concurrency']} concurrent requests")
    report.append(f"- **Peak Performance:** {best_throughput['avg_throughput']:.1f} tokens/s average")
    report.append(f"- **Scaling Factor:** {best_throughput['avg_throughput'] / df_metrics.iloc[0]['avg_throughput']:.1f}x speedup from single request\n")
    
    # Process Analysis
    report.append("## Process Resource Details\n")
    report.append("| Process ID | Avg CPU % | Max CPU % | Avg Memory % | Max Memory % | Avg Threads | Max RSS (MB) |")
    report.append("|------------|-----------|-----------|--------------|--------------|-------------|--------------|")
    
    for pid, metrics in process_summary.items():
        report.append(
            f"| {pid:>10} | {metrics['avg_cpu']:>8.1f} | {metrics['max_cpu']:>8.1f} | "
            f"{metrics['avg_mem']:>11.1f} | {metrics['max_mem']:>11.1f} | "
            f"{metrics['avg_threads']:>10.1f} | {metrics['max_rss'] / 1024:.1f} |"
        )
    
    # GPU Analysis
    report.append("\n## GPU Resource Usage\n")
    report.append("![GPU Metrics](./gpu_metrics.png)\n")
    report.append("This chart shows GPU usage, power consumption, and frequency across different concurrency levels.\n")
    
    # Add GPU metrics table
    report.append("\n## GPU Metrics Details\n")
    report.append("| Concurrency | Avg Usage % | Max Usage % | Avg Power (W) | Max Power (W) | Avg Freq (MHz) | Max Freq (MHz) |")
    report.append("|------------|-------------|-------------|---------------|---------------|----------------|----------------|")
    
    for concurrency, metrics in sorted(gpu_summary.items()):
        report.append(
            f"| {concurrency:>10} | {metrics['avg_usage']:>10.1f} | {metrics['max_usage']:>10.1f} | "
            f"{metrics['avg_power']:>12.1f} | {metrics['max_power']:>12.1f} | "
            f"{metrics['avg_freq']:>13.0f} | {metrics['max_freq']:>13.0f} |"
        )
    
    # Write report
    with open(f'performance_report_{model}.md', 'w') as f:
        f.write('\n'.join(report))

def format_duration(seconds):
    """Helper function to format duration in mm:ss"""
    if seconds < 1:
        return f"00:{seconds:.2f}"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02.2f}"

def plot_duration_metrics(duration_metrics):
    # Add summary table
    print("\nDuration Metrics Summary")
    headers = ['Concurrency', 'Avg (s)', 'Min (s)', 'Max (s)', 'P99 (s)']
    rows = []
    
    for concurrency, df in sorted(duration_metrics.items()):
        rows.append([
            concurrency,
            f"{df['AvgDuration'].iloc[0]:.2f}",
            f"{df['MinDuration'].iloc[0]:.2f}",
            f"{df['MaxDuration'].iloc[0]:.2f}",
            f"{df['P99Duration'].iloc[0]:.2f}"
        ])
    
    print_table("Duration Metrics", headers, rows)
    
    def create_duration_plot(concurrency_range, title_suffix):
        fig = create_figure()
        ax = fig.add_subplot(111)
        
        apply_chart_style(
            ax,
            f'Request Duration Analysis {title_suffix}',
            'Parallel Requests',
            'Duration (mm:ss)'
        )
        
        # Filter concurrency levels based on range
        concurrency_levels = sorted([c for c in duration_metrics.keys() if c <= concurrency_range])
        
        # Prepare data
        max_durations = [duration_metrics[c]['MaxDuration'].iloc[0] for c in concurrency_levels]
        min_durations = [duration_metrics[c]['MinDuration'].iloc[0] for c in concurrency_levels]
        avg_durations = [duration_metrics[c]['AvgDuration'].iloc[0] for c in concurrency_levels]
        
        # Plot average line first (so it appears behind the points)
        ax.plot(concurrency_levels, avg_durations, '-',
               color=METRIC_COLORS['wait'],
               linewidth=2,
               alpha=0.5)
        
        # Plot average points
        ax.scatter(concurrency_levels, avg_durations, 
                  color=METRIC_COLORS['wait'],
                  s=100,  # larger markers
                  alpha=0.8,
                  edgecolor='white',
                  linewidth=2,
                  label='Avg')
        
        # Plot metrics with consistent colors
        for i, (c, min_d, avg_d, max_d) in enumerate(zip(
            concurrency_levels, min_durations, avg_durations, max_durations
        )):
            # Vertical line from min to max
            ax.vlines(c, min_d, max_d, color='#666666', linewidth=2)
            
            # Box for average
            box_width = 0.5
            ax.hlines(avg_d, c - box_width/2, c + box_width/2, color=METRIC_COLORS['wait'], linewidth=2)
            
            # Horizontal lines for min/max
            ax.hlines(min_d, c - box_width/2, c + box_width/2, color=METRIC_COLORS['secondary'], linewidth=2)
            ax.hlines(max_d, c - box_width/2, c + box_width/2, color=METRIC_COLORS['secondary'], linewidth=2)
            
            # Add annotations with min:seconds format
            ax.annotate(format_duration(min_d), (c, min_d), 
                       textcoords="offset points",
                       xytext=(0, -15), 
                       ha='center', 
                       color=METRIC_COLORS['secondary'],
                       bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
            
            ax.annotate(format_duration(avg_d), (c, avg_d), 
                       textcoords="offset points",
                       xytext=(0, -15), 
                       ha='center', 
                       color=METRIC_COLORS['wait'],
                       bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
            
            ax.annotate(format_duration(max_d), (c, max_d), 
                       textcoords="offset points",
                       xytext=(0, 10), 
                       ha='center', 
                       color=METRIC_COLORS['secondary'],
                       bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
        
        # Format y-axis ticks to show mm:ss
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_duration(x)))
        
        # Update legend with consistent colors
        legend_elements = [
            Patch(facecolor=METRIC_COLORS['secondary'], edgecolor='none', label='Min/Max', alpha=0.7),
            plt.scatter([], [], color=METRIC_COLORS['wait'], s=100, edgecolor='white', linewidth=2, label='Avg')
        ]
        
        ax.legend(handles=legend_elements, 
                 facecolor='#1a1a1a', 
                 labelcolor='white',
                 loc='upper left')
        
        plt.tight_layout()
        return fig
    
    # Create two plots with different ranges
    fig1 = create_duration_plot(32, '(1-32 Parallel)')
    fig2 = create_duration_plot(256, '(1-256 Parallel)')
    
    return [
        (fig1, 'duration_metrics_32.png'),
        (fig2, 'duration_metrics_256.png')
    ]

def plot_throughput_and_wait(results):
    # Add summary table
    print("\nThroughput vs Wait Time Summary")
    headers = ['Concurrency', 'Throughput', 'Avg Wait(s)', 'P99 Wait(s)']
    rows = []
    
    concurrencies = []
    throughputs = []
    waits = []
    
    for concurrency in sorted(results.keys()):
        df = results[concurrency]
        final_throughputs = []
        wait_times = []
        
        for req_id in df['RequestID'].unique():
            req_data = df[df['RequestID'] == req_id]
            if not req_data.empty:
                final_throughputs.append(req_data['Throughput'].iloc[-1])
                wait_times.append(req_data['WaitingTime'].iloc[0])
        
        avg_throughput = np.mean(final_throughputs)
        avg_wait = np.mean(wait_times)
        p99_wait = np.percentile(wait_times, 99)
        
        rows.append([
            concurrency,
            f"{avg_throughput:.1f}",
            f"{avg_wait:.2f}",
            f"{p99_wait:.2f}"
        ])
        
        concurrencies.append(concurrency)
        throughputs.append(avg_throughput)
        waits.append(avg_wait)
    
    print_table("Throughput vs Wait Time", headers, rows)
    
    fig = create_figure()
    ax = fig.add_subplot(111)
    
    apply_chart_style(
        ax,
        'DeepSeek R1 7B - Throughput vs Wait Time',
        'Parallel Requests',
        'Tokens/s'
    )
    
    # Plot throughput vs concurrency
    ax.plot(concurrencies, throughputs, 'o-', 
            label='Throughput', 
            color=METRIC_COLORS['throughput'], 
            linewidth=2, 
            markersize=8,
            markeredgecolor='white', 
            markeredgewidth=2)
    
    # Create second y-axis for wait times
    ax2 = ax.twinx()
    ax2.plot(concurrencies, waits, 's-',
            label='Wait Time',
            color=METRIC_COLORS['wait'],
            linewidth=2,
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=2)
    
    # Style axes
    ax.set_xlabel('Parallel Requests', color='white')
    ax.set_ylabel('Tokens/s', color=METRIC_COLORS['throughput'])
    ax2.set_ylabel('Wait Time (s)', color=METRIC_COLORS['wait'])
    
    ax.tick_params(axis='y', labelcolor=METRIC_COLORS['throughput'])
    ax2.tick_params(axis='y', labelcolor=METRIC_COLORS['wait'])
    
    # Add annotations
    for i, (x, y1, y2) in enumerate(zip(concurrencies, throughputs, waits)):
        ax.annotate(f'{y1:.1f}', (x, y1),
                   textcoords="offset points", 
                   xytext=(0,10),
                   ha='center', 
                   color=METRIC_COLORS['throughput'],
                   bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
        
        ax2.annotate(f'{y2:.1f}s', (x, y2),
                    textcoords="offset points", 
                    xytext=(0,-15),
                    ha='center', 
                    color=METRIC_COLORS['wait'],
                    bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, 
             facecolor='#1a1a1a', 
             labelcolor='white',
             loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_ttfb_analysis(fttb_metrics):
    def create_ttfb_plot(concurrency_range, title_suffix):
        fig = create_figure()
        ax = fig.add_subplot(111)
        
        apply_chart_style(
            ax,
            f'Time to First Byte Analysis {title_suffix}',
            'Time (mm:ss)',
            'Parallel Requests'
        )
        
        concurrencies = []
        avg_ttfb = []
        p95_ttfb = []
        p99_ttfb = []
        
        # Filter concurrency levels based on range
        for concurrency in sorted(fttb_metrics.keys()):
            if concurrency <= concurrency_range:
                df = fttb_metrics[concurrency]
                if not df.empty:
                    concurrencies.append(concurrency)
                    ttfb_values = df['TTFB'].values
                    
                    avg_ttfb.append(np.mean(ttfb_values))
                    p95_ttfb.append(np.percentile(ttfb_values, 95))
                    p99_ttfb.append(np.percentile(ttfb_values, 99))
        
        # Plot lines connecting same metric points
        ax.plot(avg_ttfb, concurrencies, 'o-',
               label='Average TTFB',
               color='#00FF00',
               linewidth=2,
               markersize=8,
               markeredgecolor='white',
               markeredgewidth=2)
        
        ax.plot(p95_ttfb, concurrencies, 's-',
               label='P95 TTFB',
               color='#FF00FF',
               linewidth=2,
               markersize=8,
               markeredgecolor='white',
               markeredgewidth=2)
        
        ax.plot(p99_ttfb, concurrencies, '^-',
               label='P99 TTFB',
               color='#FFFF00',
               linewidth=2,
               markersize=8,
               markeredgecolor='white',
               markeredgewidth=2)
        
        # Add horizontal lines connecting the points
        for i, concurrency in enumerate(concurrencies):
            ax.hlines(concurrency, avg_ttfb[i], p99_ttfb[i],
                     color='#666666',
                     linewidth=1,
                     alpha=0.5)
        
        # Format x-axis ticks to show mm:ss
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_duration(x)))
        
        ax.set_ylim(0, concurrency_range * 1.1)
        ax.set_xlim(left=0)
        ax.legend(facecolor='#1a1a1a', labelcolor='white', loc='upper right')
        
        plt.tight_layout()
        return fig
    
    # Create two plots with different ranges
    fig1 = create_ttfb_plot(32, '(1-32 Parallel)')
    fig2 = create_ttfb_plot(256, '(1-256 Parallel)')
    
    return [
        (fig1, 'ttfb_analysis_32.png'),
        (fig2, 'ttfb_analysis_256.png')
    ]

def plot_average_throughput(results, duration_metrics):
    fig = create_figure()
    ax = fig.add_subplot(111)
    
    apply_chart_style(
        ax,
        'DeepSeek R1 7B: Average Throughput Analysis',
        'Parallel Requests',
        'Tokens/s'
    )
    
    concurrencies = []
    throughputs = []
    
    # Calculate metrics
    for concurrency in sorted(results.keys()):
        df = results[concurrency]
        if df.empty:
            continue
            
        final_throughputs = []
        for req_id in df['RequestID'].unique():
            req_data = df[df['RequestID'] == req_id]
            if not req_data.empty:
                final_throughputs.append(req_data['Throughput'].iloc[-1])
        
        avg_throughput = np.mean(final_throughputs)
        
        concurrencies.append(concurrency)
        throughputs.append(avg_throughput)
    
    # Plot average throughput
    ax.plot(concurrencies, throughputs, 'o-',
            color='#FF0000',
            linewidth=2,
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=2,
            label='Average Throughput')
    
    ax.set_ylim(bottom=0)
    ax.legend(facecolor='#1a1a1a', labelcolor='white', loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_combined_metrics(results, duration_metrics):
    def create_combined_plot(concurrency_range, title_suffix):
        # Create figure with extra space for legend
        fig = create_figure(figsize=(13, 6))  # Wider figure to accommodate legend
        ax = fig.add_subplot(111)
        
        apply_chart_style(
            ax,
            f'DeepSeek R1 7B: Combined Performance Metrics {title_suffix}',
            'Parallel Requests',
            'Tokens/s'
        )
        
        concurrencies = []
        throughputs = []
        wait_times = []
        durations = []
        
        # Calculate metrics
        for concurrency in sorted(results.keys()):
            if concurrency > concurrency_range:
                continue
                
            df = results[concurrency]
            duration_df = duration_metrics.get(concurrency)
            
            if duration_df is None or df.empty:
                continue
                
            # Calculate average throughput
            final_throughputs = []
            wait_time_values = []
            for req_id in df['RequestID'].unique():
                req_data = df[df['RequestID'] == req_id]
                if not req_data.empty:
                    final_throughputs.append(req_data['Throughput'].iloc[-1])
                    wait_time_values.append(req_data['WaitingTime'].iloc[0])
            
            avg_throughput = np.mean(final_throughputs)
            avg_wait = np.mean(wait_time_values)
            avg_duration = duration_df['AvgDuration'].iloc[0]
            
            concurrencies.append(concurrency)
            throughputs.append(avg_throughput)
            wait_times.append(avg_wait)
            durations.append(avg_duration)
        
        # Plot throughput on primary y-axis
        ax.plot(concurrencies, throughputs, 'o-',
                color='#FF0000',
                linewidth=2,
                markersize=8,
                markeredgecolor='white',
                markeredgewidth=2,
                label='Throughput (t/s)')
        
        # Create second y-axis for times
        ax2 = ax.twinx()
        
        # Plot wait time
        ax2.plot(concurrencies, wait_times, 's-',
                 color='#00FF00',
                 linewidth=2,
                 markersize=8,
                 markeredgecolor='white',
                 markeredgewidth=2,
                 label='Wait Time')
        
        # Plot duration
        ax2.plot(concurrencies, durations, '^-',
                 color='#0088FF',
                 linewidth=2,
                 markersize=8,
                 markeredgecolor='white',
                 markeredgewidth=2,
                 label='Duration')
        
        # Style the second y-axis
        ax2.set_ylabel('Time (mm:ss)', color='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_duration(x)))
        
        # Combine legends and place outside
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                 facecolor='#1a1a1a',
                 labelcolor='white',
                 bbox_to_anchor=(1.15, 1),
                 loc='upper left')
        
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        ax.set_xlim(0, concurrency_range * 1.1)
        
        plt.tight_layout()  # Adjust layout to prevent legend cutoff
        return fig
    
    # Create two plots with different ranges
    fig1 = create_combined_plot(32, '(1-32 Parallel)')
    fig2 = create_combined_plot(256, '(1-256 Parallel)')
    
    return [
        (fig1, 'combined_metrics_32.png'),
        (fig2, 'combined_metrics.png')
    ]

def plot_multi_model_throughput(models_data):
    # Add summary table
    print("\nComparative Model Throughput Summary")
    headers = ['Model', 'Concurrency', 'Tokens/s', 'Change %']
    rows = []
    
    for model, data in models_data.items():
        base_throughput = None
        model_name = model.replace('-', ' ').title()
        
        for concurrency, df in sorted(data.items()):
            final_throughputs = []
            for req_id in df['RequestID'].unique():
                req_data = df[df['RequestID'] == req_id]
                if not req_data.empty:
                    final_throughputs.append(req_data['Throughput'].iloc[-1])
            
            avg_throughput = np.mean(final_throughputs)
            
            if base_throughput is None:
                base_throughput = avg_throughput
                change_pct = '-'
            else:
                change_pct = f"{((avg_throughput / base_throughput) - 1) * 100:+.1f}%"
                
            rows.append([
                model_name,
                concurrency,
                f"{avg_throughput:.1f}",
                change_pct
            ])
    
    print_table("Comparative Model Throughput", headers, rows)
    
    fig = create_figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    apply_chart_style(
        ax,
        'Ollama + DeepSeek R1 1.5, 7b, 8b - Tokens/s',
        'Parallel Requests',
        'Generation Speed (tokens/s)'
    )
    
    # Define a color palette for different models
    model_colors = {
        'deepseek-r1-1.5b': '#4B4BFF',       # Bright Blue
        'deepseek-r1-7b': '#00FF00',  # Bright Green
        'deepseek-r1-8b': '#FF00FF',      # Bright Magenta
    }
    
    # Plot each model's data
    for model, data in models_data.items():
        concurrencies = []
        throughputs = []
        
        # Calculate average throughput for each concurrency level
        for concurrency, df in sorted(data.items()):
            final_throughputs = []
            for req_id in df['RequestID'].unique():
                req_data = df[df['RequestID'] == req_id]
                if not req_data.empty:
                    final_throughputs.append(req_data['Throughput'].iloc[-1])
            
            if final_throughputs:
                concurrencies.append(concurrency)
                throughputs.append(np.mean(final_throughputs))
        
        # Plot with larger markers and lines
        ax.plot(concurrencies, throughputs, 'o-',
                label=model.replace('-', ' ').title(),
                color=model_colors.get(model, '#FFFFFF'),
                linewidth=3,
                markersize=10,
                markeredgecolor='white',
                markeredgewidth=2)
        
        # Add value annotations for key points
        for i, (x, y) in enumerate(zip(concurrencies, throughputs)):
            if x in [1, 19, 256]:  # Annotate only specific points
                ax.annotate(f'{y:.1f}', (x, y),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           color='white',
                           bbox=dict(facecolor='#1a1a1a', edgecolor='none', alpha=0.7))

    # Add a grid for better readability
    ax.grid(True, color='#333333', alpha=0.3)
    
    # Add legend with custom styling
    ax.legend(facecolor='#1a1a1a',
             labelcolor='white',
             loc='upper right',
             framealpha=0.9,
             edgecolor='#666666')
    
    # Add methodology note
    ax.text(0.5, -0.15,
            'Methodology: Tests conducted using Ollama on MacBook Pro M2 (19 GPU cores). ' +
            'Each data point represents the average of multiple inference runs.',
            transform=ax.transAxes,
            ha='center',
            color='#888888',
            fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <model_name> [model_name2 ...]")
        sys.exit(1)
    
    models = sys.argv[1:]
    models_data = {}
    
    # Load data for each model
    for model in models:
        data_dir = create_data_dir(model)
        results, fttb_metrics = load_test_results(model)
        
        if results:
            models_data[model] = results
            print(f"Loaded data for {model}")
            
            # Individual model analysis
            process_metrics = load_process_metrics(model)
            gpu_metrics = load_gpu_metrics(model)
            duration_metrics = load_duration_metrics(model)
            
            if not results:
                print(f"No test results found for {model}!")
                continue
            
            print(f"Generating charts for {model}...")
            
            # Define plot functions with their arguments and filenames
            plot_configs = [
                (plot_token_distribution, (results,), 'token_distribution.png'),
                (plot_process_metrics, (process_metrics,), 'process_metrics.png'),
                (plot_gpu_metrics, (gpu_metrics,), 'gpu_metrics.png'),
                (plot_average_throughput, (results, duration_metrics), 'average_throughput.png'),
                (plot_combined_metrics, (results, duration_metrics), None)  # None because it returns multiple files
            ]
            
            # Handle regular plots
            for plot_func, args, filename in plot_configs:
                try:
                    result = plot_func(*args)
                    if isinstance(result, list):
                        # Handle functions that return multiple plots
                        for fig, fname in result:
                            if fig is not None:
                                fig.savefig(
                                    os.path.join(data_dir, fname),
                                    bbox_inches='tight',
                                    facecolor='#1a1a1a',
                                    dpi=150
                                )
                                plt.close(fig)
                    elif result is not None and filename:
                        # Handle single plot functions
                        result.savefig(
                            os.path.join(data_dir, filename),
                            bbox_inches='tight',
                            facecolor='#1a1a1a',
                            dpi=150
                        )
                        plt.close(result)
                except Exception as e:
                    print(f"Error generating {filename}: {e}")
            
            # Handle throughput and wait plots
            try:
                throughput_plots = plot_throughput_and_wait(results)
                for fig, filename in throughput_plots:
                    fig.savefig(
                        os.path.join(data_dir, filename),
                        bbox_inches='tight',
                        facecolor='#1a1a1a',
                        dpi=150
                    )
                    plt.close(fig)
            except Exception as e:
                print(f"Error generating throughput and wait plots: {e}")
            
            # Handle duration metrics plots
            try:
                duration_plots = plot_duration_metrics(duration_metrics)
                for fig, filename in duration_plots:
                    fig.savefig(
                        os.path.join(data_dir, filename),
                        bbox_inches='tight',
                        facecolor='#1a1a1a',
                        dpi=150
                    )
                    plt.close(fig)
            except Exception as e:
                print(f"Error generating duration metrics plots: {e}")
            
            # Handle TTFB plots
            try:
                ttfb_plots = plot_ttfb_analysis(fttb_metrics)
                for fig, filename in ttfb_plots:
                    fig.savefig(
                        os.path.join(data_dir, filename),
                        bbox_inches='tight',
                        facecolor='#1a1a1a',
                        dpi=150
                    )
                    plt.close(fig)
            except Exception as e:
                print(f"Error generating TTFB plots: {e}")
            
            # Generate report
            generate_markdown_report(model, results, process_metrics, gpu_metrics, duration_metrics)
            print(f"Analysis completed for {model}")
        else:
            print(f"No data found for {model}")
    
    if len(models_data) > 1:
        # Generate comparative chart if we have multiple models
        print("Generating comparative analysis chart...")
        fig = plot_multi_model_throughput(models_data)
        fig.savefig(
            'data/comparative_throughput_analysis.png',
            bbox_inches='tight',
            facecolor='#1a1a1a',
            dpi=150
        )
        plt.close(fig)
        print("Comparative analysis chart generated!")

if __name__ == "__main__":
    main() 