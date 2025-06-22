import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from collections import defaultdict


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


