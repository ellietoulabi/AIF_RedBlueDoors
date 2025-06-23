from utils import analyze_ql_behavior_comparison
from utils import analyze_behavior_correlations
from utils import analyze_episode_lengths
from utils import plot_agent_location_heatmaps


if __name__ == "__main__":
    NUM_SEEDS = 5
    logs_path = "/Users/el/dev/AIF_RedBlueDoors/logs/cc_logs copy"

    print("\n" + "="*80)
    print("1.7. QL BEHAVIOR COMPARISON ANALYSIS")
    print("="*80)
    # ql_behavior_results = analyze_ql_behavior_comparison(logs_path, NUM_SEEDS)
    
    print("\n" + "="*80)
    print("1.8. BEHAVIOR CORRELATION ANALYSIS")
    print("="*80)
    # analyze_behavior_correlations(ql_behavior_results)
    
    print("\n" + "="*80)
    print("1.9. EPISODE LENGTH ANALYSIS")
    print("="*80)
    # episode_length_results = analyze_episode_lengths(logs_path, NUM_SEEDS)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Results saved to:")
    print("- QL behavior analysis: /Users/el/dev/AIF_RedBlueDoors/debug/results/ql_behavior_analysis/")
    print("- Episode length analysis: /Users/el/dev/AIF_RedBlueDoors/debug/results/episode_length_analysis/")
    print("="*80)
    plot_agent_location_heatmaps(logs_path, NUM_SEEDS)
    