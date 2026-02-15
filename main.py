"""
FPL Prediction Tool — Main Pipeline
====================================
Ties together: Harvester → Scoring → Prediction → Squad Optimisation

Usage:
    python main.py                          # Full pipeline (needs prices CSV)
    python main.py --score-only             # Just score historical performance
    python main.py --predict-only           # Score + predict (no squad optimisation)
    python main.py --prices prices.csv      # Specify prices file
    python main.py --budget 103             # Set budget (default 103)
    python main.py --gameweeks 3            # Predict N gameweeks ahead
"""

import pandas as pd
import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.scoring.fpl_scoring import score_all_players, safe_float
from src.models.points_predictor import (
    calculate_team_strength, build_player_profiles,
    predict_all_players, parse_future_fixtures,
)
from src.optimisation.squad_optimizer import optimize_squad

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "fbref")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DEFAULT_PRICES = os.path.join(PROJECT_ROOT, "data", "prices.csv")


def load_fbref_data():
    """Load all FBref CSVs from the harvested data directory."""
    data = {}
    files = {
        "standard": "player_standard_stats.csv",
        "defense": "player_defense_stats.csv",
        "keepers": "goalkeeper_stats.csv",
        "shooting": "player_shooting_stats.csv",
        "fixtures": "fixtures_and_results.csv",
        "misc": "player_misc_stats.csv",
    }
    for key, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            print(f"  Loaded {key}: {len(data[key])} rows")
        else:
            print(f"  Missing: {filename}")
            data[key] = None

    return data


def run_scoring(data, min_minutes=270):
    """Step 1: Calculate historical FPL points from FBref data."""
    print("\n" + "=" * 60)
    print("STEP 1: SCORING — Converting FBref stats to FPL points")
    print("=" * 60)

    scored = score_all_players(
        data["standard"],
        data.get("defense"),
        data.get("keepers"),
        min_minutes=min_minutes,
    )

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "player_fpl_scores.csv")
    scored.to_csv(out_path, index=False)
    print(f"\n  Scored {len(scored)} players (min {min_minutes} minutes)")
    print(f"  Saved to: {out_path}")

    print(f"\n  Top 15 by estimated FPL points:")
    print(scored.head(15)[["player", "team", "position", "total_points", "per_match",
                           "goals_pts", "assists_pts", "cs_pts", "defcon_pts"]].to_string(index=False))

    return scored


def run_predictions(data, scored, gameweeks=3):
    """Step 2: Predict expected points for future gameweeks."""
    print("\n" + "=" * 60)
    print(f"STEP 2: PREDICTING — Expected points for next {gameweeks} GWs")
    print("=" * 60)

    # Calculate team strengths from fixtures
    team_strengths = calculate_team_strength(data.get("fixtures"))
    print(f"  Calculated strength for {len(team_strengths)} teams")

    # Build player profiles (per-90 rates)
    profiles = build_player_profiles(
        scored, data["standard"],
        data.get("defense"), data.get("keepers")
    )
    print(f"  Built profiles for {len(profiles)} players")

    # Parse future fixtures
    future_fixtures = parse_future_fixtures(data.get("fixtures"), num_gws=gameweeks)
    print(f"  Found {len(future_fixtures)} future fixture slots")

    # Predict
    predictions = predict_all_players(profiles, future_fixtures, team_strengths, gameweeks)

    # Save
    pred_path = os.path.join(PROCESSED_DIR, "player_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    print(f"\n  Saved predictions to: {pred_path}")

    gw_cols = [c for c in predictions.columns if c.startswith("gw")]
    display_cols = ["player", "team", "position", "total_expected_3gw"] + gw_cols

    print(f"\n  Top 20 predicted players (next {gameweeks} GWs):")
    print(predictions.head(20)[display_cols].to_string(index=False))

    return predictions


def run_optimization(predictions, prices_path, budget=103.0):
    """Step 3: Build optimal squad under budget."""
    print("\n" + "=" * 60)
    print(f"STEP 3: OPTIMISING — Best squad under £{budget}m")
    print("=" * 60)

    if not os.path.exists(prices_path):
        print(f"\n  ERROR: Prices file not found: {prices_path}")
        print("  Please provide a CSV with columns: player_name, position, team, price")
        print("  Place it at: data/prices.csv")
        return None

    prices = pd.read_csv(prices_path)
    print(f"  Loaded {len(prices)} player prices")

    result = optimize_squad(predictions, prices, budget)

    if result is None:
        print("  Optimization failed.")
        return None

    # Save
    squad_path = os.path.join(PROCESSED_DIR, "optimal_squad.csv")
    result["squad"].to_csv(squad_path, index=False)

    print(f"\n  {'='*50}")
    print(f"  OPTIMAL SQUAD (£{result['total_cost']}m / £{budget}m)")
    print(f"  Budget remaining: £{result['budget_remaining']}m")
    print(f"  {'='*50}")

    print(f"\n  STARTING XI (Expected: {result['starting_xi_points']} pts):")
    print(result["starting_xi"][["player", "team", "position", "price", "expected_pts"]].to_string(index=False))

    print(f"\n  BENCH:")
    print(result["bench"][["player", "team", "position", "price", "expected_pts"]].to_string(index=False))

    return result


def main():
    parser = argparse.ArgumentParser(description="FPL Prediction Tool")
    parser.add_argument("--score-only", action="store_true", help="Only run scoring")
    parser.add_argument("--predict-only", action="store_true", help="Score + predict, skip optimization")
    parser.add_argument("--prices", type=str, default=DEFAULT_PRICES, help="Path to prices CSV")
    parser.add_argument("--budget", type=float, default=103.0, help="Squad budget in millions")
    parser.add_argument("--gameweeks", type=int, default=3, help="Number of future GWs to predict")
    parser.add_argument("--min-minutes", type=int, default=270, help="Min minutes for player inclusion")
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("#  FPL PREDICTION TOOL")
    print(f"#  Budget: £{args.budget}m | GWs: {args.gameweeks} | Min mins: {args.min_minutes}")
    print("#" * 60)

    # Load data
    print("\nLoading FBref data...")
    data = load_fbref_data()

    if data["standard"] is None:
        print("\nERROR: No standard stats found. Run the harvester first:")
        print("  python Segments/fbref_harvester.py --all-teams")
        return

    # Step 1: Score
    scored = run_scoring(data, args.min_minutes)

    if args.score_only:
        print("\n  Done (score-only mode).")
        return

    # Step 2: Predict
    predictions = run_predictions(data, scored, args.gameweeks)

    if args.predict_only:
        print("\n  Done (predict-only mode).")
        return

    # Step 3: Optimize
    run_optimization(predictions, args.prices, args.budget)

    print("\n" + "#" * 60)
    print("#  PIPELINE COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()
