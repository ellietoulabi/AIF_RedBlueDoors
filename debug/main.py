from utils import run_team_stats_and_plots
from utils import plot_average_episode_return_across_seeds
from utils import plot_team_with_ci
from utils import analyze_team_door_opening_patterns
from utils import plot_door_opening_statistics

if __name__ == "__main__":
    NUM_SEEDS = 5
    logs_path = "/Users/el/dev/AIF_RedBlueDoors/logs/cc_logs copy"
    
    print("="*80)
    print("RUNNING COMPREHENSIVE TEAM ANALYSIS")
    print("="*80)
    
    # 1. Door Opening Pattern Analysis
    print("\n" + "="*80)
    print("1. DOOR OPENING PATTERN ANALYSIS")
    print("="*80)
    door_results = analyze_team_door_opening_patterns(logs_path, NUM_SEEDS, print_individual_seeds=False)
    
    # 1.5. Create Door Opening Statistics Plots
    print("\n" + "="*80)
    print("1.5. CREATING DOOR OPENING STATISTICS PLOTS")
    print("="*80)
    plot_door_opening_statistics(door_results)
    
    # 2. Episode Return Plots
    print("\n" + "="*80)
    print("2. GENERATING EPISODE RETURN PLOTS")
    print("="*80)
    plot_average_episode_return_across_seeds(logs_path, NUM_SEEDS, prefix="aif_aif", k=50,agent_names=['aif1_reward', 'aif2_reward'])
    plot_average_episode_return_across_seeds(logs_path, NUM_SEEDS, prefix="aif_ql", k=50,agent_names=['aif_reward', 'ql_reward'])
    plot_average_episode_return_across_seeds(logs_path, NUM_SEEDS, prefix="aif_rand", k=50,agent_names=['aif_reward', 'rand_reward'])
    plot_average_episode_return_across_seeds(logs_path, NUM_SEEDS, prefix="ql_ql", k=50,agent_names=['ql1_reward', 'ql2_reward'])
    plot_average_episode_return_across_seeds(logs_path, NUM_SEEDS, prefix="ql_rand", k=50,agent_names=['ql_reward', 'random_reward'])
    plot_average_episode_return_across_seeds(logs_path, NUM_SEEDS, prefix="rand_rand", k=50,agent_names=['rand1_reward', 'rand2_reward'])
    
    # 3. Team Stats and Plots
    print("\n" + "="*80)
    print("3. GENERATING TEAM STATS AND PLOTS")
    print("="*80)
    run_team_stats_and_plots(logs_path, NUM_SEEDS)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Results saved to:")
    print("- Door opening analysis: /Users/el/dev/AIF_RedBlueDoors/debug/results/door_opening_analysis/")
    print("- Episode return plots: /Users/el/dev/AIF_RedBlueDoors/debug/results/average_episode_return_across_seeds/")
    print("- Team stats plots: /Users/el/dev/AIF_RedBlueDoors/debug/results/team_stats_and_plots/")
    print("="*80)

