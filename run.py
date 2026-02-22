#!/usr/bin/env python3
"""
FPL Enhanced Model — Underlying Stats Optimizer
================================================
Combines FBref detailed per-90 stats with FPL API data to find undervalued
players based on underlying numbers rather than raw FPL points.

HOW THE SCORING MODEL WORKS (calculate_enhanced_scores):
=========================================================
Each player's expected FPL points per gameweek is built from 7 signals:

  1. HISTORICAL POINTS RATE (40% weight in final blend)
     - Actual FPL points ÷ games played = points-per-game (ppg)
     - Anchors predictions to real FPL output

  2. UNDERLYING QUALITY (60% weight, from FBref per-90 data)
     - Goals/90 × FPL goal points (6 for DEF, 5 for MID, 4 for FWD)
     - Assists/90 × 3 (FPL assist points)
     - Appearance points (2 if avg mins ≥ 60, else 1)
     - Card deductions (yellows × −1, reds × −3) per match
     - Bonus estimate (position-dependent, boosted for high G+A/90)
     → The 40/60 blend catches players whose FPL points trail their
       true quality — the core "buy low" signal.

  3. xG REGRESSION (multiplicative adjustment, up to ±20%)
     - If actual goals < xG → player is "unlucky", boost expected output
     - If actual goals > xG → slightly drag expected output
     - Same logic for xA (assists vs expected assists)
     → Key insight: xG underperformers tend to regress upward

  4. FORM MOMENTUM (multiplicative, ±15%)
     - Recent FPL form vs season-average ppg
     - Hot streak = small boost; cold streak = small drag

  5. FIXTURE DIFFICULTY (per-gameweek multiplier)
     - FDR 1–5 scale derived from team attack/defence stats
     - Home advantage (+10%) vs away penalty (−8%)
     - Playing probability from starts/matches ratio
     → Each of the next N gameweeks gets its own adjusted prediction

  6. CAPTAIN SELECTION
     - Highest GW1 xPts player in XI = captain (double points)
     - Second highest = vice-captain (backup)

  7. VALUE & DIFFERENTIAL METRICS
     - value_score = total_expected ÷ price
     - Low ownership (<5%) → 20% boost to surface hidden gems

SQUAD OPTIMISATION (two-tier LP with PuLP):
  - Binary variables: x[i] = in 15-man squad, s[i] = in starting XI
  - XI (11) maximises raw expected points; bench (4) optimises reliability
  - Constraints: 2GK/5DEF/5MID/3FWD, max 3 per team, budget cap
  - XI formation: 1GK, 3–5 DEF, 2–5 MID, 1–3 FWD

Usage:
    python run.py
    python run.py --budget 100 --gameweeks 5
"""

import pandas as pd
import numpy as np
import os
import argparse

from config import OUTPUT_DIR
from data_loader import load_all_fbref_data, load_fpl_api_data, build_unified_database
from scoring import calculate_enhanced_scores
from optimizer import optimize_squad, print_squad, print_value_picks, print_regression_candidates, print_differentials, generate_html_lineup
from transfers import run_transfer_analysis, MY_TEAM_DIR


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(budget=103.0, gameweeks=5, min_minutes=180):
    """Main pipeline: load data → score → optimize squad → export results."""
    print("\n" + "█" * 80)
    print("█  FPL ENHANCED MODEL — Underlying Stats Optimizer")
    print(f"█  Budget: £{budget}m | Lookahead: {gameweeks} GWs | Min minutes: {min_minutes}")
    print("█" * 80)

    print(f"\n{'='*60}\n  STEP 1: Loading FBref data from team folders\n{'='*60}")
    fbref_df = load_all_fbref_data()

    print(f"\n{'='*60}\n  STEP 2: Loading FPL API data\n{'='*60}")
    fpl_data = load_fpl_api_data()

    print(f"\n{'='*60}\n  STEP 3: Building unified player database\n{'='*60}")
    unified = build_unified_database(fbref_df, fpl_data)

    print(f"\n{'='*60}\n  STEP 4: Running enhanced scoring model\n{'='*60}")
    predictions = calculate_enhanced_scores(unified, fpl_data, gameweeks=gameweeks)
    print(f"  Scored {len(predictions)} players")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pred_path = os.path.join(OUTPUT_DIR, "enhanced_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    print(f"  Saved predictions to: {pred_path}")

    print_value_picks(predictions)
    print_regression_candidates(predictions)
    print_differentials(predictions)

    print(f"\n{'='*60}\n  STEP 5: Optimizing squad selection\n{'='*60}")
    result = optimize_squad(predictions, budget)
    if result:
        print_squad(result, gameweeks)
        squad_path = os.path.join(OUTPUT_DIR, "enhanced_squad.csv")
        result["squad"].to_csv(squad_path, index=False)
        print(f"\n  Squad saved to: {squad_path}")
        html = generate_html_lineup(result, predictions, gameweeks)
        html_path = os.path.join(OUTPUT_DIR, "squad_lineup.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML lineup saved to: {html_path}")

    print(f"\n{'█'*80}\n█  PIPELINE COMPLETE\n{'█'*80}\n")
    return predictions, result


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPL Enhanced Model")
    parser.add_argument("--budget", type=float, default=103.0)
    parser.add_argument("--gameweeks", type=int, default=5)
    parser.add_argument("--min-minutes", type=int, default=180)
    args = parser.parse_args()
    predictions, result = main(budget=args.budget, gameweeks=args.gameweeks, min_minutes=args.min_minutes)

    # Run transfer analysis if team data exists
    if os.path.exists(os.path.join(MY_TEAM_DIR, "my_team.csv")):
        print(f"\n{'='*60}\n  STEP 6: Transfer Recommendations for YOUR Team\n{'='*60}")
        run_transfer_analysis(predictions)
