import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
from scipy import stats
import json


def determine_door_opening(episode_df,
                           agent0_reward_col, agent1_reward_col,
                           agent0_action_col, agent1_action_col):
    """
    Determine who opened which door in an episode.
    Returns: (outcome, who) where outcome is one of:
    - "solo": one agent opened both doors
    - "cooperative": different agents opened different doors
    - "failed: blue opened first": blue door was opened before red
    - "incomplete": episode didn't complete successfully
    """
    red_opener = None
    blue_opener = None
    red_opened = False

    if episode_df.empty:
        return "incomplete", None

    for i, row in episode_df.iterrows():
        r0 = row[agent0_reward_col]
        r1 = row[agent1_reward_col]

        if not red_opened and (r0 == 0.5 or r1 == 0.5):
            red_opener = "agent_0" if r0 == 0.5 else "agent_1"
            red_opened = True

        if red_opened and r0 == 1.0 and r1 == 1.0:
            # safe reverse scan of all prior steps including current
            prior_steps = episode_df.iloc[:i+1][::-1]

            for _, rev_row in prior_steps.iterrows():
                if rev_row[agent0_action_col] == 4:
                    blue_opener = "agent_0"
                    break
                if rev_row[agent1_action_col] == 4:
                    blue_opener = "agent_1"
                    break
            break

        if r0 == -1.0 and r1 == -1.0:
            return "failed: blue opened first", None

    if red_opener and blue_opener:
        if red_opener == blue_opener:
            return "solo", red_opener
        else:
            return "cooperative", (red_opener, blue_opener)
    elif red_opener and not blue_opener:
        return "incomplete (only red opened)", red_opener
    else:
        return "incomplete", None


def count_who_secured_success(df, team_name,
                               agent0_name, agent1_name,
                               agent0_reward_col, agent1_reward_col,
                               agent0_action_col, agent1_action_col):
    """
    Count success patterns for a team across all episodes.
    """
    success_counts = {
        "team": team_name,
        f"{agent0_name}_solo": 0,
        f"{agent1_name}_solo": 0,
        f"{agent0_name}_in_coop": 0,
        f"{agent1_name}_in_coop": 0,
        "cooperative": 0,
        "solo": 0,
        "failed": 0,
        "incomplete": 0
    }

    grouped = df.groupby(["seed", "episode"])

    for (seed, episode), episode_df in grouped:
        outcome, who = determine_door_opening(
            episode_df,
            agent0_reward_col=agent0_reward_col,
            agent1_reward_col=agent1_reward_col,
            agent0_action_col=agent0_action_col,
            agent1_action_col=agent1_action_col
        )

        if outcome == "solo":
            success_counts["solo"] += 1
            if who == "agent_0":
                success_counts[f"{agent0_name}_solo"] += 1
            elif who == "agent_1":
                success_counts[f"{agent1_name}_solo"] += 1

        elif outcome == "cooperative":
            success_counts["cooperative"] += 1
            red, blue = who
            if red == "agent_0":
                success_counts[f"{agent0_name}_in_coop"] += 1
            elif red == "agent_1":
                success_counts[f"{agent1_name}_in_coop"] += 1

        elif outcome.startswith("failed"):
            success_counts["failed"] += 1
        else:
            success_counts["incomplete"] += 1

    return success_counts


def analyze_team_door_opening_patterns(logs_path: str, NUM_SEEDS: int, print_individual_seeds: bool = False):
    """
    Analyze door opening patterns for all teams across all seeds.
    Returns a dictionary with average counts for each team.
    """
    teams_config = {
        "aif_aif": {
            "agent0_name": "aif1",
            "agent1_name": "aif2", 
            "agent0_reward_col": "aif1_reward",
            "agent1_reward_col": "aif2_reward",
            "agent0_action_col": "aif1_action",
            "agent1_action_col": "aif2_action"
        },
        "aif_ql": {
            "agent0_name": "aif",
            "agent1_name": "ql",
            "agent0_reward_col": "aif_reward", 
            "agent1_reward_col": "ql_reward",
            "agent0_action_col": "aif_action",
            "agent1_action_col": "ql_action"
        },
        "aif_rand": {
            "agent0_name": "aif",
            "agent1_name": "random",
            "agent0_reward_col": "aif_reward",
            "agent1_reward_col": "rand_reward", 
            "agent0_action_col": "aif_action",
            "agent1_action_col": "rand_action"
        },
        "ql_ql": {
            "agent0_name": "ql1",
            "agent1_name": "ql2",
            "agent0_reward_col": "ql1_reward",
            "agent1_reward_col": "ql2_reward",
            "agent0_action_col": "ql1_action", 
            "agent1_action_col": "ql2_action"
        },
        "ql_rand": {
            "agent0_name": "ql",
            "agent1_name": "random",
            "agent0_reward_col": "ql_reward",
            "agent1_reward_col": "random_reward",
            "agent0_action_col": "ql_action",
            "agent1_action_col": "random_action"
        },
        "rand_rand": {
            "agent0_name": "rand1", 
            "agent1_name": "rand2",
            "agent0_reward_col": "rand1_reward",
            "agent1_reward_col": "rand2_reward",
            "agent0_action_col": "rand1_action",
            "agent1_action_col": "rand2_action"
        }
    }
    
    all_team_results = {}
    
    for team_name, config in teams_config.items():
        print(f"\n{'='*50}")
        print(f"Analyzing team: {team_name}")
        print(f"{'='*50}")
        
        aggregate_counts = defaultdict(int)
        
        for seed in range(NUM_SEEDS):
            fname = f"{team_name}_log_seed_{seed}.csv"
            path = os.path.join(logs_path, fname)
            
            try:
                seed_df = pd.read_csv(path)
                
                counts = count_who_secured_success(
                    df=seed_df,
                    team_name=team_name,
                    agent0_name=config["agent0_name"],
                    agent1_name=config["agent1_name"],
                    agent0_reward_col=config["agent0_reward_col"],
                    agent1_reward_col=config["agent1_reward_col"],
                    agent0_action_col=config["agent0_action_col"],
                    agent1_action_col=config["agent1_action_col"]
                )
                
                if print_individual_seeds:
                    print(f"Seed {seed}:", counts)
                
                for k, v in counts.items():
                    if k != "team":
                        aggregate_counts[k] += v
                        
            except FileNotFoundError:
                print(f"Warning: File not found for {team_name} seed {seed}")
                continue
        
        # Calculate averages
        avg_counts = {k: v / NUM_SEEDS for k, v in aggregate_counts.items()}
        avg_counts["team"] = team_name
        
        print(f"\nAverage across {NUM_SEEDS} seeds:")
        for k, v in avg_counts.items():
            if k != "team":
                print(f"  {k}: {v:.2f}")
        
        all_team_results[team_name] = avg_counts
    
    return all_team_results


def plot_average_episode_return_across_seeds(
    log_paths,
    NUM_SEEDS,
    prefix,
    window=50,
    agent_names=['aif_reward', 'rand_reward'],
    k=20,
    save_dir="/Users/el/dev/AIF_RedBlueDoors/debug/results/average_episode_return_across_seeds",

):

    dfs = []
    for seed in range(NUM_SEEDS):

        fname = f"{prefix}_log_seed_{seed}.csv"

        path = os.path.join(log_paths, fname)
        df = pd.read_csv(path)
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    episode_returns = (
        all_data
        .groupby(['seed', 'episode'])[[agent_names[0], agent_names[1]]]
        .mean()
        .reset_index()
    )

    mean_returns = episode_returns.groupby('episode')[[agent_names[0], agent_names[1]]].mean()
    sem_returns  = episode_returns.groupby('episode')[[agent_names[0], agent_names[1]]].sem()
    ci95_returns = 1.96 * sem_returns

    for agent in agent_names:
        mean_returns[f"{agent}_smooth"]   = mean_returns[agent].rolling(window, min_periods=1).mean()
        ci95_returns[f"{agent}_smooth"]   = ci95_returns[agent].rolling(window, min_periods=1).mean()

    episodes = mean_returns.index.values
    plt.figure(figsize=(12, 6))
    for idx, agent in enumerate(agent_names):
        sm = mean_returns[f"{agent}_smooth"]
        ci = ci95_returns[f"{agent}_smooth"]
        color = f"C{idx}"
        plt.plot(episodes, sm, label=f"{agent} Return", color=color)
        plt.fill_between(
            episodes,
            np.float64(sm - ci),
            np.float64(sm + ci),
            color=color,
            alpha=0.2
        )

    plt.xlabel('Episode')
    plt.ylabel('Average Reward (per episode)')
    plt.title(f'Average Returns per Episode ({prefix}) with 95% CI')
    plt.legend()
    plt.grid(True)

    for ep in range(0, int(episodes.max()) + 1, k):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(save_path)



def load_and_concat_team(
    logs_path: str,
    prefix: str,
    NUM_SEEDS: int,
    reward_cols: tuple
) -> pd.DataFrame:
    """
    Loads and concatenates logs for a team across seeds, and computes team_reward as the sum of the two reward columns.
    """
    dfs = []
    for seed in range(NUM_SEEDS):
        fname = f"{prefix}_log_seed_{seed}.csv"
        path = os.path.join(logs_path, fname)
        df = pd.read_csv(path)
        df['seed'] = seed
        df['team_reward'] = df[reward_cols[0]] + df[reward_cols[1]]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compute_episode_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a concatenated DataFrame (columns include ['seed','episode','team_reward']),
    return a DataFrame indexed by episode with columns:
      - mean_return   = mean of (sum of team_reward per episode) across seeds
      - sem_return    = SEM of (same) across seeds
      - ci95_return   = 1.96 * sem_return
    """
    ep_returns = (
        df.groupby(['seed','episode'])['team_reward']
          .mean()
          .reset_index(name='episode_return')
    )
    grouped = ep_returns.groupby('episode')['episode_return']
    mean_ret = grouped.mean().rename('mean_return')
    sem_ret  = grouped.sem().rename('sem_return')
    ci95_ret = (1.96 * sem_ret).rename('ci95_return')
    stats = pd.concat([mean_ret, sem_ret, ci95_ret], axis=1)
    return stats  # index = episode

def plot_team_with_ci(
    stats: pd.DataFrame,
    prefix: str,
    window: int = 50,
    save_dir: str = "/Users/el/dev/AIF_RedBlueDoors/debug/results/team_stats_and_plots"
):
    """
    Given a stats DataFrame (with index=episode and columns
      ['mean_return','sem_return','ci95_return']),
    smooth with rolling(window) and plot mean ± CI95 band.
    """
    stats['mean_smooth'] = stats['mean_return'].rolling(window, min_periods=1).mean()
    stats['ci95_smooth'] = stats['ci95_return'].rolling(window, min_periods=1).mean()
    episodes = stats.index.values
    plt.figure(figsize=(10, 5))
    plt.plot(
        episodes,
        stats['mean_smooth'],
        label=f"{prefix} Mean Return (smoothed)",
        color='C0'
    )
    plt.fill_between(
        episodes,
        np.float64(stats['mean_smooth'] - stats['ci95_smooth']),
        np.float64(stats['mean_smooth'] + stats['ci95_smooth']),
        color='C0',
        alpha=0.3,
        label="95% CI (smoothed)"
    )
    plt.xlabel("Episode")
    plt.ylabel("Team Return")
    plt.title(f"{prefix}: Average Team Return per Episode (window={window})")
    plt.legend()
    plt.grid(True)
    for ep in range(0, 1000 + 1, 50):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{prefix}_mean_return_ci.png")
    plt.tight_layout()
    plt.savefig(out_path)

def run_team_stats_and_plots(logs_path: str, NUM_SEEDS: int, save_dir: str = "/Users/el/dev/AIF_RedBlueDoors/debug/results/team_stats_and_plots"):
    """
    Run the complete team stats and plotting pipeline for all teams.
    """
    teams = {
        "aif_aif":    ("aif1_reward",   "aif2_reward"),
        "aif_rand":   ("aif_reward",    "rand_reward"),
        "aif_ql":     ("aif_reward",    "ql_reward"),
        "ql_rand":    ("ql_reward",     "random_reward"),
        "ql_ql":      ("ql1_reward",    "ql2_reward"),
        "rand_rand":  ("rand1_reward",  "rand2_reward")
    }
    stats_by_team = {}
    for prefix, (col1, col2) in teams.items():
        df_team = load_and_concat_team(
            logs_path   = logs_path,
            prefix      = prefix,
            NUM_SEEDS   = NUM_SEEDS,
            reward_cols = (col1, col2)
        )
        stats = compute_episode_stats(df_team)
        stats_by_team[prefix] = stats
        plot_team_with_ci(
            stats    = stats,
            prefix   = prefix,
            window   = 50,
            save_dir = save_dir
        )
    # All teams comparison plot
    plt.figure(figsize=(12, 6))
    for idx, prefix in enumerate(stats_by_team):
        stats = stats_by_team[prefix]
        mean_smooth = stats['mean_return'].rolling(50, min_periods=1).mean()
        plt.plot(
            stats.index.values,
            mean_smooth,
            label=prefix,
            color=f"C{idx}"
        )
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Mean Team Return")
    plt.title("Comparison of All Teams (window=50 smoothing)")
    plt.legend()
    plt.grid(True)
    for ep in range(0, 1000 + 1, 50):
        plt.axvline(x=ep, color='grey', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "all_teams_comparison.png"))


def plot_door_opening_statistics(door_results: dict, save_dir: str = "/Users/el/dev/AIF_RedBlueDoors/debug/results/door_opening_analysis"):
    """
    Create a comprehensive bar plot comparing door opening statistics across all teams.
    
    Args:
        door_results: Dictionary returned by analyze_team_door_opening_patterns
        save_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define the metrics we want to plot
    metrics = ['cooperative', 'solo', 'failed', 'incomplete']
    
    # Extract team names and data
    team_names = list(door_results.keys())
    team_data = {metric: [door_results[team][metric] for team in team_names] for metric in metrics}
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Success vs Failure rates
    x = np.arange(len(team_names))
    width = 0.35
    
    # Success metrics (cooperative + solo)
    success_rates = [team_data['cooperative'][i] + team_data['solo'][i] for i in range(len(team_names))]
    failure_rates = team_data['failed']
    incomplete_rates = team_data['incomplete']
    
    bars1 = ax1.bar(x - width/2, success_rates, width, label='Success (Cooperative + Solo)', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, failure_rates, width, label='Failed', color='red', alpha=0.7)
    bars3 = ax1.bar(x + width/2, incomplete_rates, width, bottom=failure_rates, label='Incomplete', color='orange', alpha=0.7)
    
    ax1.set_xlabel('Teams')
    ax1.set_ylabel('Average Count per Seed')
    ax1.set_title('Door Opening Success vs Failure Rates by Team')
    ax1.set_xticks(x)
    ax1.set_xticklabels(team_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Cooperative vs Solo breakdown
    cooperative_rates = team_data['cooperative']
    solo_rates = team_data['solo']
    
    bars4 = ax2.bar(x - width/2, cooperative_rates, width, label='Cooperative', color='blue', alpha=0.7)
    bars5 = ax2.bar(x + width/2, solo_rates, width, label='Solo', color='purple', alpha=0.7)
    
    ax2.set_xlabel('Teams')
    ax2.set_ylabel('Average Count per Seed')
    ax2.set_title('Cooperative vs Solo Success Breakdown by Team')
    ax2.set_xticks(x)
    ax2.set_xticklabels(team_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars5:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Create directory and save
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'door_opening_statistics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Team', 'Cooperative', 'Solo', 'Failed', 'Incomplete', 'Total Success']
    
    for team in team_names:
        data = door_results[team]
        total_success = data['cooperative'] + data['solo']
        table_data.append([
            team,
            f"{data['cooperative']:.2f}",
            f"{data['solo']:.2f}",
            f"{data['failed']:.2f}",
            f"{data['incomplete']:.2f}",
            f"{total_success:.2f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 0:  # Team name column
                table[(i, j)].set_facecolor('#E3F2FD')
            elif j in [1, 2, 5]:  # Success metrics
                table[(i, j)].set_facecolor('#C8E6C9')
            elif j == 3:  # Failed
                table[(i, j)].set_facecolor('#FFCDD2')
            else:  # Incomplete
                table[(i, j)].set_facecolor('#FFE0B2')
    
    ax.set_title('Door Opening Statistics Summary Table', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(save_dir, 'door_opening_statistics_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Door opening statistics plots saved to: {save_dir}")


def analyze_ql_behavior_comparison(logs_path: str, NUM_SEEDS: int, save_dir: str = "/Users/el/dev/AIF_RedBlueDoors/debug/results/ql_behavior_analysis"):
    """
    Analyze how QL agents behave differently when playing with QL vs AIF partners.
    Focuses on action patterns, learning curves, and cooperation strategies.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"\n{'='*60}")
    print("ANALYZING QL AGENT BEHAVIOR COMPARISON")
    print(f"{'='*60}")
    
    # Define the comparisons we want to make
    comparisons = {
        "ql_ql": {
            "description": "QL playing with QL",
            "ql_reward_col": "ql1_reward",
            "ql_action_col": "ql1_action", 
            "partner_reward_col": "ql2_reward",
            "partner_action_col": "ql2_action"
        },
        "aif_ql": {
            "description": "QL playing with AIF",
            "ql_reward_col": "ql_reward",
            "ql_action_col": "ql_action",
            "partner_reward_col": "aif_reward", 
            "partner_action_col": "aif_action"
        }
    }
    
    results = {}
    
    for team_name, config in comparisons.items():
        print(f"\nAnalyzing {config['description']}...")
        
        all_ql_actions = []
        all_ql_rewards = []
        all_partner_actions = []
        all_partner_rewards = []
        all_episodes = []
        all_seeds = []
        
        for seed in range(NUM_SEEDS):
            fname = f"{team_name}_log_seed_{seed}.csv"
            path = os.path.join(logs_path, fname)
            
            try:
                df = pd.read_csv(path)
                
                # Group by episode to analyze behavior
                for episode, episode_df in df.groupby('episode'):
                    all_ql_actions.extend(episode_df[config['ql_action_col']].tolist())
                    all_ql_rewards.extend(episode_df[config['ql_reward_col']].tolist())
                    all_partner_actions.extend(episode_df[config['partner_action_col']].tolist())
                    all_partner_rewards.extend(episode_df[config['partner_reward_col']].tolist())
                    all_episodes.extend([episode] * len(episode_df))
                    all_seeds.extend([seed] * len(episode_df))
                    
            except FileNotFoundError:
                print(f"Warning: File not found for {team_name} seed {seed}")
                continue
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'seed': all_seeds,
            'episode': all_episodes,
            'ql_action': all_ql_actions,
            'ql_reward': all_ql_rewards,
            'partner_action': all_partner_actions,
            'partner_reward': all_partner_rewards
        })
        
        results[team_name] = {
            'data': analysis_df,
            'config': config
        }
    
    # Create comprehensive comparison plots
    create_ql_behavior_plots(results, save_dir)
    
    return results


def create_ql_behavior_plots(results: dict, save_dir = "/Users/el/dev/AIF_RedBlueDoors/debug/results/ql_behavior_analysis"):
    """
    Create detailed plots comparing QL behavior in different partnerships.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Action Distribution Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = {'ql_ql': 'blue', 'aif_ql': 'red'}
    
    for i, (team_name, team_data) in enumerate(results.items()):
        df = team_data['data']
        config = team_data['config']
        
        # QL Action Distribution
        ql_actions = df['ql_action'].value_counts().sort_index()
        ax1.bar(ql_actions.index + i*0.3, ql_actions.values, width=0.3, 
                alpha=0.7, label=config['description'], color=colors[team_name])
        
        # Partner Action Distribution  
        partner_actions = df['partner_action'].value_counts().sort_index()
        ax2.bar(partner_actions.index + i*0.3, partner_actions.values, width=0.3,
                alpha=0.7, label=config['description'], color=colors[team_name])
        
        # QL Reward Distribution
        ql_rewards = df['ql_reward'].value_counts().sort_index()
        ax3.bar(ql_rewards.index + i*0.3, ql_rewards.values, width=0.3,
                alpha=0.7, label=config['description'], color=colors[team_name])
        
        # Partner Reward Distribution
        partner_rewards = df['partner_reward'].value_counts().sort_index()
        ax4.bar(partner_rewards.index + i*0.3, partner_rewards.values, width=0.3,
                alpha=0.7, label=config['description'], color=colors[team_name])
    
    ax1.set_title('QL Agent Action Distribution')
    ax1.set_xlabel('Action')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Partner Agent Action Distribution')
    ax2.set_xlabel('Action')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('QL Agent Reward Distribution')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Partner Agent Reward Distribution')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ql_behavior_action_reward_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning Curves (Average Rewards per Episode)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for team_name, team_data in results.items():
        df = team_data['data']
        config = team_data['config']
        
        # Calculate average QL reward per episode
        episode_rewards = df.groupby('episode')['ql_reward'].mean()
        episode_rewards_smooth = episode_rewards.rolling(window=10, min_periods=1).mean()
        
        ax1.plot(episode_rewards_smooth.index, episode_rewards_smooth.values, 
                label=config['description'], color=colors[team_name], linewidth=2)
        
        # Calculate average partner reward per episode
        partner_episode_rewards = df.groupby('episode')['partner_reward'].mean()
        partner_episode_rewards_smooth = partner_episode_rewards.rolling(window=10, min_periods=1).mean()
        
        ax2.plot(partner_episode_rewards_smooth.index, partner_episode_rewards_smooth.values,
                label=config['description'], color=colors[team_name], linewidth=2)
    
    ax1.set_title('QL Agent Learning Curve (Average Reward per Episode)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Partner Agent Learning Curve (Average Reward per Episode)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ql_behavior_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cooperation Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    for team_name, team_data in results.items():
        df = team_data['data']
        config = team_data['config']
        
        # Action coordination (same actions in same timestep)
        same_actions = (df['ql_action'] == df['partner_action']).astype(int)
        df['same_actions'] = same_actions  # Add as a column to the dataframe
        episode_coordination = df.groupby('episode')['same_actions'].mean()
        episode_coordination_smooth = episode_coordination.rolling(window=10, min_periods=1).mean()
        
        ax1.plot(episode_coordination_smooth.index, episode_coordination_smooth.values,
                label=config['description'], color=colors[team_name], linewidth=2)
        
        # Reward correlation
        episode_ql_rewards = df.groupby('episode')['ql_reward'].mean()
        episode_partner_rewards = df.groupby('episode')['partner_reward'].mean()
        
        # Calculate rolling correlation
        window = 20
        correlations = []
        episodes = []
        
        for i in range(window, len(episode_ql_rewards)):
            corr = np.corrcoef(episode_ql_rewards.iloc[i-window:i], 
                              episode_partner_rewards.iloc[i-window:i])[0,1]
            correlations.append(corr)
            episodes.append(episode_ql_rewards.index[i])
        
        ax2.plot(episodes, correlations, label=config['description'], 
                color=colors[team_name], linewidth=2)
        
        # Action diversity (number of unique actions per episode)
        action_diversity = df.groupby('episode')['ql_action'].nunique()
        action_diversity_smooth = action_diversity.rolling(window=10, min_periods=1).mean()
        
        ax3.plot(action_diversity_smooth.index, action_diversity_smooth.values,
                label=config['description'], color=colors[team_name], linewidth=2)
        
        # Success rate (episodes with positive total reward)
        episode_total_reward = df.groupby('episode')[['ql_reward', 'partner_reward']].sum().sum(axis=1)
        success_rate = (episode_total_reward > 0).rolling(window=20, min_periods=1).mean()
        
        ax4.plot(success_rate.index, success_rate.values,
                label=config['description'], color=colors[team_name], linewidth=2)
    
    ax1.set_title('Action Coordination (Same Actions)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Proportion of Coordinated Actions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Reward Correlation (Rolling Window)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax3.set_title('QL Action Diversity per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Unique Actions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax4.set_title('Success Rate (Positive Total Reward)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Success Rate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ql_behavior_cooperation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary Statistics Table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Metric', 'QL+QL', 'AIF+QL', 'Difference']
    
    for team_name, team_data in results.items():
        df = team_data['data']
        
        # Calculate key metrics
        avg_ql_reward = df['ql_reward'].mean()
        avg_partner_reward = df['partner_reward'].mean()
        action_coordination = (df['ql_action'] == df['partner_action']).mean()
        ql_action_diversity = df.groupby('episode')['ql_action'].nunique().mean()
        success_rate = (df.groupby('episode')[['ql_reward', 'partner_reward']].sum().sum(axis=1) > 0).mean()
        
        if team_name == 'ql_ql':
            ql_ql_metrics = [avg_ql_reward, avg_partner_reward, action_coordination, ql_action_diversity, success_rate]
        else:
            aif_ql_metrics = [avg_ql_reward, avg_partner_reward, action_coordination, ql_action_diversity, success_rate]
    
    # Calculate differences
    differences = [aif_ql_metrics[i] - ql_ql_metrics[i] for i in range(len(ql_ql_metrics))]
    
    metrics_names = ['Avg QL Reward', 'Avg Partner Reward', 'Action Coordination', 'QL Action Diversity', 'Success Rate']
    
    for i, metric in enumerate(metrics_names):
        table_data.append([
            metric,
            f"{ql_ql_metrics[i]:.3f}",
            f"{aif_ql_metrics[i]:.3f}",
            f"{differences[i]:+.3f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 0:  # Metric column
                table[(i, j)].set_facecolor('#E3F2FD')
            elif j == 1:  # QL+QL column
                table[(i, j)].set_facecolor('#C8E6C9')
            elif j == 2:  # AIF+QL column
                table[(i, j)].set_facecolor('#FFCDD2')
            elif j == 3:  # Difference
                if float(table_data[i-1][j]) > 0:
                    table[(i, j)].set_facecolor('#C8E6C9')
                else:
                    table[(i, j)].set_facecolor('#FFCDD2')
            else:  # Interpretation
                table[(i, j)].set_facecolor('#FFF3E0')
    
    ax.set_title('QL Agent Behavior Comparison Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(save_dir, 'ql_behavior_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"QL behavior analysis plots saved to: {save_dir}")


def analyze_behavior_correlations(results: dict, save_dir: str = "/Users/el/dev/AIF_RedBlueDoors/debug/results/ql_behavior_analysis"):
    """
    Analyze correlations between QL and partner behaviors across different partnerships.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    print(f"\n{'='*60}")
    print("ANALYZING BEHAVIOR CORRELATIONS")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Action-Action Correlation Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = {'ql_ql': 'blue', 'aif_ql': 'red'}
    
    for team_name, team_data in results.items():
        df = team_data['data']
        config = team_data['config']
        
        # Calculate action correlation per episode
        episode_correlations = []
        episodes = []
        
        for episode, episode_df in df.groupby('episode'):
            if len(episode_df) > 1:  # Need at least 2 data points for correlation
                corr, p_value = stats.pearsonr(episode_df['ql_action'], episode_df['partner_action'])
                episode_correlations.append(corr)
                episodes.append(episode)
        
        # Smooth the correlation over episodes
        if len(episode_correlations) > 10:
            episode_correlations_smooth = pd.Series(episode_correlations).rolling(window=10, min_periods=1).mean()
        else:
            episode_correlations_smooth = episode_correlations
            
        ax1.plot(episodes, episode_correlations_smooth, 
                label=config['description'], color=colors[team_name], linewidth=2, alpha=0.8)
        
        # Calculate reward correlation per episode
        reward_correlations = []
        reward_episodes = []
        
        for episode, episode_df in df.groupby('episode'):
            if len(episode_df) > 1:
                corr, p_value = stats.pearsonr(episode_df['ql_reward'], episode_df['partner_reward'])
                reward_correlations.append(corr)
                reward_episodes.append(episode)
        
        if len(reward_correlations) > 10:
            reward_correlations_smooth = pd.Series(reward_correlations).rolling(window=10, min_periods=1).mean()
        else:
            reward_correlations_smooth = reward_correlations
            
        ax2.plot(reward_episodes, reward_correlations_smooth,
                label=config['description'], color=colors[team_name], linewidth=2, alpha=0.8)
        
        # Action synchrony (lag-1 correlation)
        action_synchrony = []
        sync_episodes = []
        
        for episode, episode_df in df.groupby('episode'):
            if len(episode_df) > 2:
                # Calculate lag-1 correlation (current action vs previous partner action)
                ql_actions = episode_df['ql_action'].values
                partner_actions = episode_df['partner_action'].values
                
                if len(ql_actions) > 1 and len(partner_actions) > 1:
                    # Lag-1 correlation: QL action at t vs partner action at t-1
                    corr, p_value = stats.pearsonr(ql_actions[1:], partner_actions[:-1])
                    action_synchrony.append(corr)
                    sync_episodes.append(episode)
        
        if len(action_synchrony) > 10:
            action_synchrony_smooth = pd.Series(action_synchrony).rolling(window=10, min_periods=1).mean()
        else:
            action_synchrony_smooth = action_synchrony
            
        ax3.plot(sync_episodes, action_synchrony_smooth,
                label=config['description'], color=colors[team_name], linewidth=2, alpha=0.8)
        
        # Behavioral entropy correlation
        entropy_correlations = []
        entropy_episodes = []
        
        for episode, episode_df in df.groupby('episode'):
            if len(episode_df) > 5:  # Need enough data for entropy calculation
                # Calculate action entropy for both agents
                ql_entropy = stats.entropy(episode_df['ql_action'].value_counts())
                partner_entropy = stats.entropy(episode_df['partner_action'].value_counts())
                
                # Calculate correlation between entropies across episodes
                entropy_correlations.append(ql_entropy - partner_entropy)  # Difference in entropy
                entropy_episodes.append(episode)
        
        if len(entropy_correlations) > 10:
            entropy_correlations_smooth = pd.Series(entropy_correlations).rolling(window=10, min_periods=1).mean()
        else:
            entropy_correlations_smooth = entropy_correlations
            
        ax4.plot(entropy_episodes, entropy_correlations_smooth,
                label=config['description'], color=colors[team_name], linewidth=2, alpha=0.8)
    
    ax1.set_title('Action-Action Correlation Over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_title('Reward-Reward Correlation Over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax3.set_title('Action Synchrony (Lag-1 Correlation)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax4.set_title('Behavioral Entropy Difference (QL - Partner)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Entropy Difference')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'behavior_correlations_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cross-correlation analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for team_name, team_data in results.items():
        df = team_data['data']
        config = team_data['config']
        
        # Calculate cross-correlation for actions
        all_ql_actions = df['ql_action'].values
        all_partner_actions = df['partner_action'].values
        
        # Normalize the data
        ql_norm = (all_ql_actions - np.mean(all_ql_actions)) / np.std(all_ql_actions)
        partner_norm = (all_partner_actions - np.mean(all_partner_actions)) / np.std(all_partner_actions)
        
        # Calculate cross-correlation
        max_lag = min(50, len(ql_norm) // 4)  # Limit lag to reasonable range
        lags = np.arange(-max_lag, max_lag + 1)
        cross_corr = []
        
        for lag in lags:
            if lag < 0:
                # QL leads partner
                corr = np.corrcoef(ql_norm[-lag:], partner_norm[:lag])[0, 1] if len(ql_norm[-lag:]) > 10 else 0
            elif lag > 0:
                # Partner leads QL
                corr = np.corrcoef(ql_norm[:-lag], partner_norm[lag:])[0, 1] if len(ql_norm[:-lag]) > 10 else 0
            else:
                # No lag
                corr = np.corrcoef(ql_norm, partner_norm)[0, 1]
            cross_corr.append(corr)
        
        ax1.plot(lags, cross_corr, label=config['description'], color=colors[team_name], linewidth=2)
        
        # Calculate cross-correlation for rewards
        all_ql_rewards = df['ql_reward'].values
        all_partner_rewards = df['partner_reward'].values
        
        ql_reward_norm = (all_ql_rewards - np.mean(all_ql_rewards)) / np.std(all_ql_rewards)
        partner_reward_norm = (all_partner_rewards - np.mean(all_partner_rewards)) / np.std(all_partner_rewards)
        
        cross_corr_reward = []
        for lag in lags:
            if lag < 0:
                corr = np.corrcoef(ql_reward_norm[-lag:], partner_reward_norm[:lag])[0, 1] if len(ql_reward_norm[-lag:]) > 10 else 0
            elif lag > 0:
                corr = np.corrcoef(ql_reward_norm[:-lag], partner_reward_norm[lag:])[0, 1] if len(ql_reward_norm[:-lag]) > 10 else 0
            else:
                corr = np.corrcoef(ql_reward_norm, partner_reward_norm)[0, 1]
            cross_corr_reward.append(corr)
        
        ax2.plot(lags, cross_corr_reward, label=config['description'], color=colors[team_name], linewidth=2)
    
    ax1.set_title('Action Cross-Correlation (Lag Analysis)')
    ax1.set_xlabel('Lag (QL leads if negative, Partner leads if positive)')
    ax1.set_ylabel('Cross-Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_title('Reward Cross-Correlation (Lag Analysis)')
    ax2.set_xlabel('Lag (QL leads if negative, Partner leads if positive)')
    ax2.set_ylabel('Cross-Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cross_correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Correlation Type', 'QL+QL', 'AIF+QL', 'Difference', 'Interpretation']
    
    for team_name, team_data in results.items():
        df = team_data['data']
        
        # Overall action correlation
        action_corr, action_p = stats.pearsonr(df['ql_action'], df['partner_action'])
        
        # Overall reward correlation
        reward_corr, reward_p = stats.pearsonr(df['ql_reward'], df['partner_reward'])
        
        # Action synchrony (lag-1)
        ql_actions = df['ql_action'].values
        partner_actions = df['partner_action'].values
        if len(ql_actions) > 1 and len(partner_actions) > 1:
            sync_corr, sync_p = stats.pearsonr(ql_actions[1:], partner_actions[:-1])
        else:
            sync_corr = 0
        
        # Behavioral entropy
        ql_entropy = stats.entropy(df['ql_action'].value_counts())
        partner_entropy = stats.entropy(df['partner_action'].value_counts())
        entropy_diff = ql_entropy - partner_entropy
        
        if team_name == 'ql_ql':
            ql_ql_corrs = [action_corr, reward_corr, sync_corr, entropy_diff]
        else:
            aif_ql_corrs = [action_corr, reward_corr, sync_corr, entropy_diff]
    
    # Calculate differences
    differences = [aif_ql_corrs[i] - ql_ql_corrs[i] for i in range(len(ql_ql_corrs))]
    
    corr_names = ['Action Correlation', 'Reward Correlation', 'Action Synchrony', 'Entropy Difference']
    interpretations = [
        'How coordinated their actions are',
        'How synchronized their rewards are', 
        'How much QL follows partner\'s previous action',
        'How much more/less predictable QL is vs partner'
    ]
    
    for i, corr_name in enumerate(corr_names):
        table_data.append([
            corr_name,
            f"{ql_ql_corrs[i]:.3f}",
            f"{aif_ql_corrs[i]:.3f}",
            f"{differences[i]:+.3f}",
            interpretations[i]
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 0:  # Metric column
                table[(i, j)].set_facecolor('#E3F2FD')
            elif j == 1:  # QL+QL column
                table[(i, j)].set_facecolor('#C8E6C9')
            elif j == 2:  # AIF+QL column
                table[(i, j)].set_facecolor('#FFCDD2')
            elif j == 3:  # Difference
                if float(table_data[i-1][j]) > 0:
                    table[(i, j)].set_facecolor('#C8E6C9')
                else:
                    table[(i, j)].set_facecolor('#FFCDD2')
            elif j == 4:  # Interpretation
                table[(i, j)].set_facecolor('#FFF3E0')
    
    ax.set_title('Behavior Correlation Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(save_dir, 'correlation_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Behavior correlation analysis saved to: {save_dir}")


def analyze_episode_lengths(logs_path: str, NUM_SEEDS: int, save_dir: str = "/Users/el/dev/AIF_RedBlueDoors/debug/results/episode_length_analysis"):
    """
    Analyze episode lengths for successful episodes across all teams.
    Shows how efficiently different teams complete the task.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    print(f"\n{'='*60}")
    print("ANALYZING EPISODE LENGTHS FOR SUCCESSFUL EPISODES")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Define all teams
    teams_config = {
        "aif_aif": {
            "description": "AIF + AIF",
            "reward_cols": ["aif1_reward", "aif2_reward"]
        },
        "aif_ql": {
            "description": "AIF + QL", 
            "reward_cols": ["aif_reward", "ql_reward"]
        },
        "aif_rand": {
            "description": "AIF + Random",
            "reward_cols": ["aif_reward", "rand_reward"]
        },
        "ql_ql": {
            "description": "QL + QL",
            "reward_cols": ["ql1_reward", "ql2_reward"]
        },
        "ql_rand": {
            "description": "QL + Random",
            "reward_cols": ["ql_reward", "random_reward"]
        },
        "rand_rand": {
            "description": "Random + Random",
            "reward_cols": ["rand1_reward", "rand2_reward"]
        }
    }
    
    all_episode_lengths = {}
    all_success_rates = {}
    
    for team_name, config in teams_config.items():
        print(f"\nAnalyzing {config['description']}...")
        
        team_episode_lengths = []
        team_success_rates = []
        
        for seed in range(NUM_SEEDS):
            fname = f"{team_name}_log_seed_{seed}.csv"
            path = os.path.join(logs_path, fname)
            
            try:
                df = pd.read_csv(path)
                
                # Group by episode
                for episode, episode_df in df.groupby('episode'):
                    episode_length = len(episode_df)
                    last_step = episode_df.iloc[-1]
                    if (last_step[config['reward_cols']] == 1).any():
                        team_episode_lengths.append(episode_length)
                    
                    # Calculate success rate for this seed
                    if episode == df['episode'].max():  # Last episode
                        all_episodes = df.groupby('episode').apply(lambda edf: (edf.iloc[-1][config['reward_cols']] == 1).any())
                        success_rate = all_episodes.mean()
                        team_success_rates.append(success_rate)
                        
            except FileNotFoundError:
                print(f"Warning: File not found for {team_name} seed {seed}")
                continue
        
        all_episode_lengths[team_name] = team_episode_lengths
        all_success_rates[team_name] = team_success_rates
    
    # Create comprehensive plots
    create_episode_length_plots(all_episode_lengths, all_success_rates, teams_config, save_dir, logs_path, NUM_SEEDS)
    
    return all_episode_lengths, all_success_rates


def create_episode_length_plots(episode_lengths: dict, success_rates: dict, teams_config: dict, save_dir: str, logs_path: str, NUM_SEEDS: int):
    """
    Create detailed plots for episode length analysis.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # 1. Box plot of episode lengths
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data for box plot
    team_names = list(episode_lengths.keys())
    team_labels = [teams_config[team]['description'] for team in team_names]
    lengths_data = [episode_lengths[team] for team in team_names]
    
    # Filter out empty lists
    valid_data = [(label, data) for label, data in zip(team_labels, lengths_data) if len(data) > 0]
    if valid_data:
        labels, data = zip(*valid_data)
        
        bp = ax1.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax1.set_title('Episode Lengths for Successful Episodes')
        ax1.set_ylabel('Episode Length (timesteps)')
        ax1.set_xlabel('Team')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # 2. Success rate vs average episode length
    avg_lengths = []
    avg_success_rates = []
    team_names_plot = []
    
    for team_name in team_names:
        if len(episode_lengths[team_name]) > 0 and len(success_rates[team_name]) > 0:
            avg_length = np.mean(episode_lengths[team_name])
            avg_success = np.mean(success_rates[team_name])
            avg_lengths.append(avg_length)
            avg_success_rates.append(avg_success)
            team_names_plot.append(teams_config[team_name]['description'])
    
    if len(avg_lengths) > 0:
        scatter = ax2.scatter(avg_lengths, avg_success_rates, s=100, alpha=0.7)
        
        # Add team labels
        for i, team in enumerate(team_names_plot):
            ax2.annotate(team, (avg_lengths[i], avg_success_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add trend line
        if len(avg_lengths) > 1:
            z = np.polyfit(avg_lengths, avg_success_rates, 1)
            p = np.poly1d(z)
            ax2.plot(avg_lengths, p(avg_lengths), "r--", alpha=0.8)
            
            # Calculate correlation
            corr, p_value = stats.pearsonr(avg_lengths, avg_success_rates)
            ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {p_value:.3f}', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax2.set_title('Success Rate vs Average Episode Length')
        ax2.set_xlabel('Average Episode Length (timesteps)')
        ax2.set_ylabel('Success Rate')
        ax2.grid(True, alpha=0.3)
    
    # 3. Histogram of episode lengths
    ax3.hist(lengths_data, bins=20, alpha=0.7, label=team_labels[:len(lengths_data)])
    ax3.set_title('Distribution of Episode Lengths')
    ax3.set_xlabel('Episode Length (timesteps)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Success rate comparison
    success_data = [success_rates[team] for team in team_names if len(success_rates[team]) > 0]
    success_labels = [teams_config[team]['description'] for team in team_names if len(success_rates[team]) > 0]
    
    if success_data:
        bp2 = ax4.boxplot(success_data, labels=success_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
            patch.set_facecolor(color)
        
        ax4.set_title('Success Rates Across Seeds')
        ax4.set_ylabel('Success Rate')
        ax4.set_xlabel('Team')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'episode_length_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed statistics table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Team', 'Successful Episodes', 'Avg Length', 'Std Length', 'Min Length', 'Max Length', 'Success Rate', 'Efficiency Score']
    
    for team_name in team_names:
        if len(episode_lengths[team_name]) > 0:
            lengths = episode_lengths[team_name]
            success_rate = np.mean(success_rates[team_name]) if len(success_rates[team_name]) > 0 else 0
            
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            min_length = np.min(lengths)
            max_length = np.max(lengths)
            
            # Efficiency score: success rate / average length (higher is better)
            efficiency = success_rate / avg_length if avg_length > 0 else 0
            
            table_data.append([
                teams_config[team_name]['description'],
                len(lengths),
                f"{avg_length:.1f}",
                f"{std_length:.1f}",
                min_length,
                max_length,
                f"{success_rate:.3f}",
                f"{efficiency:.4f}"
            ])
    
    # Sort by efficiency score
    table_data.sort(key=lambda x: float(x[7]), reverse=True)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 0:  # Team name column
                table[(i, j)].set_facecolor('#E3F2FD')
            elif j == 1:  # Successful episodes
                table[(i, j)].set_facecolor('#C8E6C9')
            elif j == 6:  # Success rate
                success_rate = float(table_data[i-1][j])
                if success_rate > 0.5:
                    table[(i, j)].set_facecolor('#C8E6C9')  # Green for high success
                else:
                    table[(i, j)].set_facecolor('#FFCDD2')  # Red for low success
            elif j == 7:  # Efficiency score
                efficiency = float(table_data[i-1][j])
                if efficiency > 0.01:
                    table[(i, j)].set_facecolor('#C8E6C9')  # Green for high efficiency
                else:
                    table[(i, j)].set_facecolor('#FFCDD2')  # Red for low efficiency
            else:
                table[(i, j)].set_facecolor('#FFF3E0')
    
    ax.set_title('Episode Length Statistics for Successful Episodes', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(save_dir, 'episode_length_statistics_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Learning progression plot (separate for each team)
    for i, team_name in enumerate(team_names):
        if len(episode_lengths[team_name]) > 0:
            # Load data to get episode progression
            all_lengths_by_episode = []
            
            for seed in range(NUM_SEEDS):
                fname = f"{team_name}_log_seed_{seed}.csv"
                path = os.path.join(logs_path, fname)
                
                try:
                    df = pd.read_csv(path)
                    config = teams_config[team_name]
                    
                    for episode, episode_df in df.groupby('episode'):
                        episode_length = len(episode_df)
                        last_step = episode_df.iloc[-1]
                        if (last_step[config['reward_cols']] == 1).any():
                            all_lengths_by_episode.append((episode, episode_length))
                            
                except FileNotFoundError:
                    continue
            
            if all_lengths_by_episode:
                episodes, lengths = zip(*all_lengths_by_episode)
                df_temp = pd.DataFrame({'episode': episodes, 'length': lengths})
                df_temp = df_temp.sort_values('episode')
                rolling_avg = df_temp['length'].rolling(window=10, min_periods=1).mean()
                
                # Create a separate plot for this team
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_temp['episode'], rolling_avg, color='blue', linewidth=2, alpha=0.8)
                ax.set_title(f"{teams_config[team_name]['description']} - Episode Length Learning Progression")
                ax.set_xlabel('Episode')
                ax.set_ylabel('Average Episode Length (timesteps)')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{team_name}_episode_length_learning_progression.png'), dpi=300, bbox_inches='tight')
                plt.close()


def plot_agent_location_heatmaps(logs_path, NUM_SEEDS, save_dir="/Users/el/dev/AIF_RedBlueDoors/debug/results/location_heatmaps"):
    """
    For each team, for each config (config.json, config2.json), reconstruct agent trajectories from actions and create a heatmap of all visited locations for each agent, averaged across seeds.
    The config alternates every 50 episodes.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # Load configs and parse map, walls, doors, and starting positions
    config_files = [
        ("config.json", "config1"),
        ("config2.json", "config2")
    ]
    config_dir = os.path.join(os.path.dirname(__file__), "..", "envs", "redbluedoors_env", "configs")
    config_data = []  # [(agent0_start, agent1_start, walls, red_door, blue_door, width, height, label)]
    for fname, label in config_files:
        with open(os.path.join(config_dir, fname), "r") as f:
            config = json.load(f)
            # Parse map for walls, doors, agent starts
            map_data = config["map"]
            width = len(map_data[0].split())
            height = len(map_data)
            walls = set()
            agent0_start = agent1_start = None
            red_door = blue_door = None
            for y, row in enumerate(map_data):
                row = row.split()
                for x, char in enumerate(row):
                    if char == "#":
                        walls.add((x, y))
                    elif char == "R":
                        red_door = (x, y)
                    elif char == "B":
                        blue_door = (x, y)
                    elif char == "0":
                        agent0_start = (x, y)
                    elif char == "1":
                        agent1_start = (x, y)
            config_data.append((agent0_start, agent1_start, walls, red_door, blue_door, width, height, label))

    # Action mapping
    ACTIONS = {
        0: (0, -1),  # up (y-1)
        1: (0, 1),   # down (y+1)
        2: (-1, 0),  # left (x-1)
        3: (1, 0),   # right (x+1)
        4: (0, 0),   # open (no move)
    }

    # Team agent action columns
    team_agent_cols = {
        "aif_aif":   ["aif1_action", "aif2_action"],
        "aif_ql":    ["aif_action", "ql_action"],
        "aif_rand":  ["aif_action", "rand_action"],
        "ql_ql":     ["ql1_action", "ql2_action"],
        "ql_rand":   ["ql_action", "random_action"],
        "rand_rand": ["rand1_action", "rand2_action"],
    }
    teams = list(team_agent_cols.keys())
    episodes_per_config = 50

    for team_name in teams:
        agent_cols = team_agent_cols[team_name]
        # Prepare heatmap arrays: [config][agent][row][col][seed]
        team_heatmaps = [
            [[], []]  # agent 0, agent 1
            for _ in range(2)
        ]
        for cidx in range(2):
            for aidx in range(2):
                # One heatmap per seed, to average later
                team_heatmaps[cidx][aidx] = [np.zeros((config_data[cidx][6], config_data[cidx][5]), dtype=int) for _ in range(NUM_SEEDS)]

        for seed in range(NUM_SEEDS):
            fname = f"{team_name}_log_seed_{seed}.csv"
            path = os.path.join(logs_path, fname)
            try:
                df = pd.read_csv(path)
                for episode, ep_df in df.groupby("episode"):
                    config_idx = (episode // episodes_per_config) % 2
                    agent_starts = [config_data[config_idx][0], config_data[config_idx][1]]
                    walls = config_data[config_idx][2]
                    red_door = config_data[config_idx][3]
                    blue_door = config_data[config_idx][4]
                    width = config_data[config_idx][5]
                    height = config_data[config_idx][6]
                    # For each agent, reconstruct trajectory
                    for agent_id, action_col in enumerate(agent_cols):
                        pos = agent_starts[agent_id]
                        for idx, row in ep_df.iterrows():
                            # Mark visit
                            if 0 <= pos[0] < width and 0 <= pos[1] < height:
                                team_heatmaps[config_idx][agent_id][seed][pos[1], pos[0]] += 1
                            action = row[action_col]
                            try:
                                action = int(action)
                            except:
                                continue
                            dx, dy = ACTIONS.get(action, (0, 0))
                            new_pos = (pos[0] + dx, pos[1] + dy)
                            # Check move validity (cannot move into wall, door, or out of bounds)
                            if (0 <= new_pos[0] < width and 0 <= new_pos[1] < height and
                                new_pos not in walls and new_pos != red_door and new_pos != blue_door):
                                pos = new_pos
            except Exception as e:
                print(f"Could not process {fname}: {e}")

        # Average heatmaps across seeds and plot
        for config_idx, (_, _, _, _, _, width, height, label) in enumerate(config_data):
            for agent_id in [0, 1]:
                avg_heatmap = np.mean(team_heatmaps[config_idx][agent_id], axis=0)
                plt.figure(figsize=(6, 5))
                plt.imshow(avg_heatmap, origin='upper', cmap='hot', interpolation='nearest', aspect='auto')
                plt.title(f"{team_name} - Agent {agent_id} - {label} Visitation Heatmap (avg across seeds)")
                plt.xlabel('X (col)')
                plt.ylabel('Y (row)')
                plt.colorbar(label='Visit Count (avg)')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{team_name}_agent{agent_id}_{label}_visitation_heatmap.png"))
                plt.close()
    print(f"Location visitation heatmaps saved to: {save_dir}")


