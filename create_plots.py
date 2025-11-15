import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib import rc
from IPython.display import HTML
from matplotlib import rcParams
import seaborn as sns
from textwrap import wrap
from tqdm import tqdm
import pandas as pd
import re
import math
import gc
import json
import argparse

def replace_tuples_to_lists(str_obj):
    str_obj = str_obj.replace(',)', ')')
    # find tuples and replace parentheses with brackets
    pattern = r'\(([^()]+?)\)'
    while re.search(pattern, str_obj):
        str_obj = re.sub(pattern, lambda m: '[' + m.group(1) + ']', str_obj)
    return str_obj

def read_dfs(dirname):
    with open(f'{dirname}/logs/training_log_group.log', 'r') as f:
        logs = f.read().replace("'", '"').replace('nan', 'null')
        logs = logs.replace('tensor(', '').replace(', device="cuda:0", grad_fn=<NegBackward0>)', '').replace(', device="cuda:0", grad_fn=<MeanBackward0>)', '').replace('.,', '.0,')
        logs = replace_tuples_to_lists(logs)
        logs = logs.splitlines()
        logs = [json.loads(line) for line in logs if line.strip()]

    G = max(len(logs[i]['logps']) for i in range(len(logs)))

    data = []
    for i, log in enumerate(logs):
        for key in ['rewards', 'normalized_rewards', 'advantages', 'normalized_advantages','logps']:
            for g in range(G):
                if key in log:
                    data.append({
                        'log_idx': i,
                        'epoch': log['epoch'],
                        't': log['t'],
                        'prompt_idx': log['prompt_idx'],
                        'seed_idx': log['seed_idx'],
                        'G_idx': g,
                        'metric': key,
                        'value': log[key][g]
                    })
        data.append({
            'log_idx': i,
            'epoch': log['epoch'],
            't': log['t'],
            'prompt_idx': log['prompt_idx'],
            'seed_idx': log['seed_idx'],
            'G_idx': None,
            'metric': 'loss',
            'value': log['loss']
        })
        if 'ref_reward' in log:
            data.append({
                'log_idx': i,
                'epoch': log['epoch'],
                't': log['t'],
                'prompt_idx': log['prompt_idx'],
                'seed_idx': log['seed_idx'],
                'G_idx': None,
                'metric': 'rewards',
                'value': log['ref_reward']
            })

    df = pd.DataFrame(data)

    metrics = df['metric'].unique()
    n_metrics = len(metrics)
    n_epochs = df['epoch'].nunique()
    n_logs = df['log_idx'].nunique()
    G = df['G_idx'].nunique()  # exclude None

    with open(f'{dirname}/logs/policy_stats.log', 'r') as f:
        policy_logs = f.read().replace("'", '"').replace('nan', 'null').splitlines()
        policy_logs = [json.loads(line) for line in policy_logs if line.strip()]

    for log in policy_logs:
        tag = log['tag']
        m = re.match(r'ep(\d+)_t(\d+)_p(\d+)_s(\d+)_g(\d+)', tag)
        if m:
            log['epoch'] = int(m.group(1))
            log['t'] = int(m.group(2))
            log['prompt_idx'] = int(m.group(3))
            log['seed_idx'] = int(m.group(4))
            log['G_idx'] = int(m.group(5))  
        else:
            log['epoch'] = None
            log['t'] = None
            log['prompt_idx'] = None
            log['seed_idx'] = None
            log['G_idx'] = None
            
        stats_after = log['stats_after']
        stats_before = log['stats_before']
        log['latent_before_min'] = stats_before['min']
        log['latent_before_max'] = stats_before['max']
        log['latent_before_mean'] = stats_before['mean']
        log['latent_before_std'] = stats_before['std']
        log['latent_after_min'] = stats_after['min']
        log['latent_after_max'] = stats_after['max']
        log['latent_after_mean'] = stats_after['mean']
        log['latent_after_std'] = stats_after['std']

    policy_df = pd.DataFrame(policy_logs)

    return df, policy_df

def create_metrics_plot(dirname, dfs, plots_dir):
    df, policy_df = dfs

    metrics = df['metric'].unique()
    n_metrics = len(metrics)
    n_epochs = df['epoch'].nunique()
    n_logs = df['log_idx'].nunique()
    G = df['G_idx'].nunique()  # exclude None

    ncols = 2
    nrows = math.ceil(n_metrics / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    for i, metric in enumerate(metrics):
        r = i // ncols
        c = i % ncols
        ax = axs[r, c] if nrows > 1 else axs[c]
        if metric == 'loss':
            sns.lineplot(x='log_idx', y='value', data=df[df['metric'] == metric], ax=ax)
        else:
            sns.boxplot(x='log_idx', y='value', data=df[df['metric'] == metric], ax=ax)
            # also add a lineplot of the mean value per epoch
            mean_values = df[df['metric'] == metric].groupby('log_idx')['value'].mean().reset_index()
            sns.lineplot(x='log_idx', y='value', data=mean_values, ax=ax, color='red', label='Mean', marker='o')
            # and another line for moving average of mean
            mean_values['moving_avg'] = mean_values['value'].rolling(window=10, min_periods=1).mean()
            sns.lineplot(x='log_idx', y='moving_avg', data=mean_values, ax=ax, color='green', label='Moving Avg', marker='o')
        ax.set_title('\n'.join(wrap(f'{metric} over Epochs', 40)))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
    plt.suptitle(f'Metrics for {dirname}', fontsize=16)
    plt.tight_layout()

    output_filename = f'{plots_dir}/{dirname}_metrics.pdf'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, bbox_inches='tight', dpi=300, format='pdf')

def plot_image_grid(dirname, dfs, plots_dir):
    df, policy_df = dfs
    
    metrics = df['metric'].unique()
    n_metrics = len(metrics)
    n_epochs = df['epoch'].nunique()
    n_logs = df['log_idx'].nunique()
    G = df['G_idx'].nunique()  # exclude None

    # plot grid of images, epoch rows, G cols. The images are in 'dirname/ep{epoch}_t000_p00_s00_g{G_idx}.png'. For each image, write its reward value from the logs above.
    fig, axs = plt.subplots(n_logs, G+1, figsize=(G*3, n_logs*3))
    # for epoch in range(1, n_epochs+1):
    for log_idx in range(0, n_logs):
        epoch = df[df['log_idx'] == log_idx]['epoch'].values[0]
        t = df[df['log_idx'] == log_idx]['t'].values[0]
        prompt_idx = df[df['log_idx'] == log_idx]['prompt_idx'].values[0]
        seed_idx = df[df['log_idx'] == log_idx]['seed_idx'].values[0]
        image_paths = [ f'{dirname}/imgs/ep{epoch:02d}_t{t:03d}_p{prompt_idx:02d}_s{seed_idx:02d}_ref.png' ] + [ f'{dirname}/imgs/ep{epoch:02d}_t{t:03d}_p{prompt_idx:02d}_s{seed_idx:02d}_g{g:02d}.png' for g in range(0, G) ]
        advantages = [ df[(df['epoch'] == epoch) & (df['log_idx'] == log_idx) & (df['G_idx'] == g) & (df['metric'] == 'advantages')]['value'].values[0] for g in range(0, G) ]
        ref_reward = df[(df['epoch'] == epoch) & (df['log_idx'] == log_idx) & (df['metric'] == 'rewards') & (df['G_idx'].isnull())]['value'].values[0]

        titles = [f'Reference Reward = {ref_reward:.3f}'] + [f'Advantage = {advantages[g]:.3f}' for g in range(0, G)]

        for g in range(0,G+1):
            ax = axs[log_idx, g] if n_logs > 1 else axs[g]
            ax.imshow(plt.imread(image_paths[g]))
            ax.set_title(titles[g])
            ax.set_xticks([])
            ax.set_yticks([])
            if g == 0:
                ax.set_ylabel(f'Epoch {epoch}', fontsize=14)
            else:
                ax.set_ylabel('')
    plt.suptitle(f'Image Grid for {dirname}', fontsize=16)
    plt.tight_layout()

    output_filename = f'{plots_dir}/{dirname}_image_grid.pdf'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, bbox_inches='tight', dpi=300, format='pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dirs", type=str, nargs='+')
    parser.add_argument("--plots_dir", type=str, default='plots')
    args = parser.parse_args()

    experiment_dirs = []
    for base_dir in args.base_dirs:
        experiment_dirs.extend([ os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) ])

    os.makedirs(f'{args.plots_dir}', exist_ok=True)

    for dirname in experiment_dirs:
        print(f'Processing {dirname}...')
        try:
            dfs = read_dfs(dirname)
            create_metrics_plot(dirname, dfs, args.plots_dir)
            plot_image_grid(dirname, dfs, args.plots_dir)
            gc.collect()
        except Exception as e:
            print(f'Error processing {dirname}: {e}')