"""
FPL API Data Adapter
====================
Converts FPL API data into the same format expected by the scoring/prediction pipeline.
Used when FBref data is unavailable (e.g. blocked from server environments).

The FPL API provides:
  - Player stats (goals, assists, xG, xA, clean sheets, saves, cards)
  - Current prices
  - Fixture difficulty ratings
  - Historical points
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scoring.fpl_scoring import SCORING, DEFCON, safe_float


def load_fpl_data(data_dir):
    """Load FPL API CSV files."""
    stats = pd.read_csv(os.path.join(data_dir, "fpl_player_stats.csv"))
    prices = pd.read_csv(os.path.join(data_dir, "prices.csv"))
    fixtures = pd.read_csv(os.path.join(data_dir, "fixtures_future.csv"))
    return stats, prices, fixtures


def score_from_fpl_stats(stats_df, min_minutes=270):
    """
    Calculate estimated FPL points from FPL API stats.

    The FPL API gives us actual FPL total_points, but we also reconstruct
    the breakdown to understand WHERE points come from (for prediction).
    """
    df = stats_df[stats_df["minutes"] >= min_minutes].copy()

    results = []
    for _, p in df.iterrows():
        pos = p["position"]
        minutes = safe_float(p["minutes"])
        starts = safe_float(p.get("starts", 0))
        matches = starts + max(0, (minutes - starts * 70) / 25) if starts > 0 else minutes / 60
        matches = max(1, matches)

        # Reconstruct points breakdown
        full_games = starts
        partial = max(0, matches - starts)

        appearance_pts = full_games * 2 + partial * 1
        goal_pts = safe_float(p["goals_scored"]) * SCORING["goal"].get(pos, 4)
        assist_pts = safe_float(p["assists"]) * SCORING["assist"]
        cs_pts = safe_float(p["clean_sheets"]) * SCORING["clean_sheet"].get(pos, 0)

        gc_pts = 0
        if pos in ("GK", "DEF"):
            gc_pts = -(safe_float(p["goals_conceded"]) // 2)

        save_pts = 0
        pk_save_pts = 0
        if pos == "GK":
            save_pts = (safe_float(p["saves"]) // 3)
            pk_save_pts = safe_float(p.get("penalties_saved", 0)) * 5

        card_pts = safe_float(p["yellow_cards"]) * -1 + safe_float(p["red_cards"]) * -3

        # We don't have per-match defensive actions from FPL API,
        # so DefCon is estimated from position and general defensive contribution
        defcon_pts = 0

        # Use actual FPL total_points as ground truth
        actual_total = safe_float(p["total_points"])
        reconstructed = appearance_pts + goal_pts + assist_pts + cs_pts + gc_pts + save_pts + pk_save_pts + card_pts

        # The difference is bonus + any rounding
        bonus_and_other = actual_total - reconstructed

        results.append({
            "player": p["player_name"],
            "full_name": p.get("full_name", p["player_name"]),
            "team": p["team"],
            "position": pos,
            "price": safe_float(p.get("price", 0)),
            "minutes": minutes,
            "matches": round(matches, 1),
            "total_points": actual_total,
            "per_match": round(actual_total / matches, 2) if matches > 0 else 0,
            "goals_pts": goal_pts,
            "assists_pts": assist_pts,
            "cs_pts": cs_pts,
            "appearance_pts": appearance_pts,
            "cards_pts": card_pts,
            "saves_pts": save_pts + pk_save_pts,
            "gc_pts": gc_pts,
            "bonus_pts": safe_float(p.get("bonus", 0)),
            "defcon_pts": defcon_pts,
            "form": safe_float(p.get("form", 0)),
            "xG": safe_float(p.get("expected_goals", 0)),
            "xA": safe_float(p.get("expected_assists", 0)),
            "ict_index": safe_float(p.get("ict_index", 0)),
            "selected_pct": safe_float(p.get("selected_by_percent", 0)),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("total_points", ascending=False).reset_index(drop=True)
    return result_df


def build_fpl_predictions(scored_df, fixtures_df, gameweeks=3):
    """
    Predict expected points using FPL stats + fixture difficulty.

    Method:
      1. Base rate = per-match average (actual FPL points / matches)
      2. Weight recent form (FPL 'form' field = last 30 days avg)
      3. Adjust for fixture difficulty (FDR: 1=easy, 5=hardest)
      4. Adjust for home/away
    """
    # Get next N gameweek numbers
    future_gws = sorted(fixtures_df["gameweek"].unique())[:gameweeks]

    results = []
    for _, player in scored_df.iterrows():
        team = player["team"]
        position = player["position"]

        # Base expected points per match
        ppm = player["per_match"]
        form = player["form"]  # FPL's own form metric (recent avg)

        # Blend historical average with recent form (60/40 split)
        if form > 0:
            base_expected = ppm * 0.6 + form * 0.4
        else:
            base_expected = ppm

        # xG/xA boost: if xG+xA underperformance suggests upside
        xg = player.get("xG", 0)
        xa = player.get("xA", 0)
        minutes = player["minutes"]
        actual_goals = player["goals_pts"] / SCORING["goal"].get(position, 4) if SCORING["goal"].get(position, 4) != 0 else 0
        actual_assists = player["assists_pts"] / SCORING["assist"] if SCORING["assist"] != 0 else 0

        # If xG > actual goals, player is "due" (positive regression signal)
        if xg > actual_goals and actual_goals > 0:
            xg_boost = min(0.3, (xg - actual_goals) / actual_goals * 0.1)
        else:
            xg_boost = 0

        gw_predictions = {}
        total_expected = 0

        for i, gw in enumerate(future_gws):
            # Find fixture for this team in this GW
            home_fix = fixtures_df[(fixtures_df["gameweek"] == gw) &
                                   (fixtures_df["home_team"] == team)]
            away_fix = fixtures_df[(fixtures_df["gameweek"] == gw) &
                                   (fixtures_df["away_team"] == team)]

            if len(home_fix) > 0:
                fdr = safe_float(home_fix.iloc[0]["home_difficulty"])
                is_home = True
                opponent = home_fix.iloc[0]["away_team"]
            elif len(away_fix) > 0:
                fdr = safe_float(away_fix.iloc[0]["away_difficulty"])
                is_home = False
                opponent = away_fix.iloc[0]["home_team"]
            else:
                # No fixture (blank GW)
                gw_predictions[f"gw{i+1}_exp"] = 0
                gw_predictions[f"gw{i+1}_opp"] = "BLANK"
                gw_predictions[f"gw{i+1}_fdr"] = 0
                continue

            # Fixture difficulty adjustment
            # FDR: 1=very easy, 2=easy, 3=medium, 4=hard, 5=very hard
            # Convert to multiplier: FDR 2 → 1.15x, FDR 3 → 1.0x, FDR 5 → 0.7x
            fdr_multiplier = 1.0 + (3 - fdr) * 0.15

            # Home/away adjustment
            home_multiplier = 1.08 if is_home else 0.92

            # Final expected points for this GW
            gw_expected = base_expected * fdr_multiplier * home_multiplier * (1 + xg_boost)
            gw_expected = max(0, round(gw_expected, 2))

            gw_predictions[f"gw{i+1}_exp"] = gw_expected
            gw_predictions[f"gw{i+1}_opp"] = opponent
            gw_predictions[f"gw{i+1}_fdr"] = int(fdr)
            gw_predictions[f"gw{i+1}_home"] = "H" if is_home else "A"
            total_expected += gw_expected

        results.append({
            "player": player["player"],
            "team": team,
            "position": position,
            "price": player.get("price", 0),
            "total_expected_3gw": round(total_expected, 2),
            "avg_expected_per_gw": round(total_expected / gameweeks, 2),
            "historical_ppm": ppm,
            "form": form,
            "xg_boost": round(xg_boost, 3),
            **gw_predictions,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("total_expected_3gw", ascending=False).reset_index(drop=True)
    return result_df


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    stats, prices, fixtures = load_fpl_data(data_dir)
    scored = score_from_fpl_stats(stats)
    print(f"Scored {len(scored)} players")
    print(scored.head(10).to_string())
