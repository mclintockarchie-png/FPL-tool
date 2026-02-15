#!/usr/bin/env python3
"""
FPL Enhanced Model — Underlying Stats Optimizer
================================================
Combines FBref detailed stats with FPL API data to find
undervalued players based on underlying numbers.

Key innovations over the basic model:
  1. Loads ALL team fbref CSVs (handles both multi-header and clean formats)
  2. Merges with FPL API data for prices, actual points, form, ownership
  3. xG/xA regression model — finds players due a points surge
  4. Per-90 normalization across all metrics
  5. "Value Score" = expected output per £ spent
  6. Differential detector — low ownership + strong underlying stats
  7. Full squad optimization with PuLP LP solver

Usage:
    python run_enhanced_model.py
    python run_enhanced_model.py --budget 100 --gameweeks 5
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FBREF_DIR = os.path.join(BASE_DIR, "data", "Harvested Data", "PL teams data 2025-2026")
FPL_API_DIR = os.path.join(BASE_DIR, "data", "Harvested Data", "FPL API data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Results")

# FPL scoring rules
SCORING = {
    "appearance_60_plus": 2,
    "appearance_1_59": 1,
    "goal": {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": 3,
    "clean_sheet": {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "gc_per_2": -1,
    "saves_per_3": 1,
    "penalty_save": 5,
    "yellow": -1,
    "red": -3,
    "bonus_avg": {"GK": 0.15, "DEF": 0.18, "MID": 0.22, "FWD": 0.25},
}

# Position mapping
POS_MAP = {
    "FW": "FWD", "FW,MF": "FWD", "MF,FW": "MID", "FW,DF": "FWD",
    "MF": "MID", "MF,DF": "MID", "DF,MF": "DEF",
    "DF": "DEF", "DF,FW": "DEF",
    "GK": "GK",
}

# Team name normalization (fbref folder → FPL API team name)
TEAM_NAME_MAP = {
    "Arsenal": "Arsenal",
    "Aston_Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal_Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich_Town": "Ipswich",
    "Leeds": "Leeds",
    "Leicester_City": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester_City": "Man City",
    "Manchester_Utd": "Man Utd",
    "Newcastle_Utd": "Newcastle",
    "Nottham_Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Sunderland ": "Sunderland",  # trailing space in folder name
    "Tottenham": "Spurs",
    "West_Ham": "West Ham",
    "Wolves": "Wolves",
}


# ══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD ALL FBREF DATA
# ══════════════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    """Safely convert to float, handling commas and special chars."""
    if pd.isna(val):
        return default
    try:
        v = float(str(val).replace(",", "").strip())
        return v if not np.isnan(v) else default
    except (ValueError, TypeError):
        return default


def load_single_team_csv(filepath, team_name):
    """
    Load a single team's fbref CSV, handling both formats:
      Format A (multi-header): Has a category row before actual headers
      Format B (clean): Single header row with clean column names
    """
    try:
        # Read first two lines to detect format
        with open(filepath, "r", encoding="utf-8-sig") as f:
            line1 = f.readline().strip()
            line2 = f.readline().strip()

        # Detect format: if line1 starts with empty/category fields, it's multi-header
        first_fields = line1.split(",")
        is_multi_header = first_fields[0].strip() in ("", "Playing Time", "Performance")

        if is_multi_header:
            # Format A: skip the category row, use row 2 as header
            df = pd.read_csv(filepath, header=1, encoding="utf-8-sig")
        else:
            # Format B: clean single header
            df = pd.read_csv(filepath, header=0, encoding="utf-8-sig")

        # Standardize column names
        df = _standardize_columns(df)

        # Add team name
        df["team"] = team_name

        # Filter out empty/summary rows
        if "Player" in df.columns:
            df = df[df["Player"].notna() & (df["Player"] != "")].copy()
            # Remove "Matches" entries that fbref sometimes puts in
            df = df[~df["Player"].str.contains("Squad Total|Opponent Total", na=False)].copy()

        return df

    except Exception as e:
        print(f"  WARNING: Could not load {filepath}: {e}")
        return pd.DataFrame()


def _standardize_columns(df):
    """Map various fbref column name formats to a standard set."""
    rename_map = {}
    col_list = list(df.columns)

    # Handle duplicate column names (e.g., two "Gls" columns — season total and per-90)
    # The per-90 versions come after the absolute versions in fbref exports
    seen = {}
    new_cols = []
    for col in col_list:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_per90" if seen[col] == 1 else f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols

    # Standard renames
    standard_renames = {
        "Position": "Pos",
        "Minutes": "Min",
        "Goals": "Gls",
        "Assists": "Ast",
        "Yellow Cards": "CrdY",
        "Red Cards": "CrdR",
    }
    for old, new in standard_renames.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def load_all_fbref_data():
    """Load fbref CSVs from all team folders."""
    all_players = []
    teams_loaded = []

    if not os.path.exists(FBREF_DIR):
        print(f"  ERROR: fbref data directory not found: {FBREF_DIR}")
        return pd.DataFrame()

    for folder_name in sorted(os.listdir(FBREF_DIR)):
        folder_path = os.path.join(FBREF_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Find CSV files in this team folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            continue

        team_fpl_name = TEAM_NAME_MAP.get(folder_name, folder_name.replace("_", " "))

        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            df = load_single_team_csv(csv_path, team_fpl_name)
            if not df.empty:
                all_players.append(df)
                if team_fpl_name not in teams_loaded:
                    teams_loaded.append(team_fpl_name)

    if not all_players:
        print("  ERROR: No fbref data loaded from any team.")
        return pd.DataFrame()

    combined = pd.concat(all_players, ignore_index=True)
    print(f"  Loaded fbref data: {len(combined)} players from {len(teams_loaded)} teams")
    print(f"  Teams: {', '.join(teams_loaded)}")
    return combined


# ══════════════════════════════════════════════════════════════════════
# STEP 2: LOAD FPL API DATA
# ══════════════════════════════════════════════════════════════════════

def load_fpl_api_data():
    """Load FPL API stats, prices, and fixtures."""
    stats_path = os.path.join(FPL_API_DIR, "fpl_player_stats.csv")
    prices_path = os.path.join(FPL_API_DIR, "prices.csv")
    fixtures_path = os.path.join(FPL_API_DIR, "fixtures_future.csv")

    data = {}
    for name, path in [("stats", stats_path), ("prices", prices_path), ("fixtures", fixtures_path)]:
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
            print(f"  Loaded FPL {name}: {len(data[name])} rows")
        else:
            print(f"  Missing FPL {name}: {path}")
            data[name] = pd.DataFrame()

    return data


# ══════════════════════════════════════════════════════════════════════
# STEP 3: BUILD UNIFIED PLAYER DATABASE
# ══════════════════════════════════════════════════════════════════════

def map_position(pos_str):
    """Map fbref position to FPL position."""
    if pd.isna(pos_str):
        return "MID"
    pos = str(pos_str).strip()
    # Handle slash-separated positions (e.g. "MF/DF")
    pos = pos.replace("/", ",")
    return POS_MAP.get(pos, POS_MAP.get(pos.split(",")[0], "MID"))


def build_unified_database(fbref_df, fpl_data):
    """
    Merge fbref underlying stats with FPL API data.
    fbref gives us: per-90 rates, detailed performance metrics
    FPL API gives us: prices, actual FPL points, form, ownership, xG, xA
    """
    fpl_stats = fpl_data["stats"]
    fpl_prices = fpl_data["prices"]

    # ── Process fbref data ────────────────────────────────────────────
    players = []
    for _, row in fbref_df.iterrows():
        player_name = str(row.get("Player", "")).strip()
        if not player_name or player_name == "nan":
            continue

        pos = map_position(row.get("Pos", "MID"))
        minutes = safe_float(row.get("Min", 0))
        matches = safe_float(row.get("MP", 0))
        starts = safe_float(row.get("Starts", 0))
        nineties = safe_float(row.get("90s", 0))

        if minutes < 90 or matches < 1:
            continue

        # Core stats
        goals = safe_float(row.get("Gls", 0))
        assists = safe_float(row.get("Ast", 0))
        g_plus_a = safe_float(row.get("G+A", goals + assists))
        goals_npk = safe_float(row.get("G-PK", goals))
        pks = safe_float(row.get("PK", 0))
        pk_att = safe_float(row.get("PKatt", 0))
        yellows = safe_float(row.get("CrdY", 0))
        reds = safe_float(row.get("CrdR", 0))

        # Per-90 rates (use fbref per-90 if available, else calculate)
        if nineties > 0:
            gls_per90 = goals / nineties
            ast_per90 = assists / nineties
            ga_per90 = g_plus_a / nineties
            npg_per90 = goals_npk / nineties
        else:
            gls_per90 = safe_float(row.get("Gls_per90", 0))
            ast_per90 = safe_float(row.get("Ast_per90", 0))
            ga_per90 = safe_float(row.get("G+A_per90", 0))
            npg_per90 = safe_float(row.get("G-PK_per90", 0))

        players.append({
            "player_fbref": player_name,
            "team": row.get("team", "Unknown"),
            "position": pos,
            "age": str(row.get("Age", "")),
            "minutes": minutes,
            "matches": matches,
            "starts": starts,
            "nineties": nineties if nineties > 0 else minutes / 90,
            "goals": goals,
            "assists": assists,
            "g_plus_a": g_plus_a,
            "goals_npk": goals_npk,
            "pks": pks,
            "pk_att": pk_att,
            "yellows": yellows,
            "reds": reds,
            "gls_per90": round(gls_per90, 3),
            "ast_per90": round(ast_per90, 3),
            "ga_per90": round(ga_per90, 3),
            "npg_per90": round(npg_per90, 3),
        })

    fbref_processed = pd.DataFrame(players)
    print(f"  Processed {len(fbref_processed)} fbref players (90+ mins)")

    # ── Merge with FPL API data ───────────────────────────────────────
    # We need: price, actual FPL points, form, ownership%, xG, xA from FPL API
    if fpl_stats.empty or fpl_prices.empty:
        print("  WARNING: No FPL API data available. Using fbref data only.")
        fbref_processed["price"] = 5.0  # default
        fbref_processed["fpl_points"] = 0
        fbref_processed["form"] = 0
        fbref_processed["selected_pct"] = 0
        fbref_processed["fpl_xG"] = 0
        fbref_processed["fpl_xA"] = 0
        fbref_processed["bonus"] = 0
        fbref_processed["ict_index"] = 0
        return fbref_processed

    # Build FPL lookup — normalize names for matching
    fpl_merged = fpl_stats.merge(
        fpl_prices[["player_name", "price"]].drop_duplicates(),
        on="player_name", how="left", suffixes=("", "_price")
    )
    # Use price from prices.csv if available, else from stats
    if "price_price" in fpl_merged.columns:
        fpl_merged["price"] = fpl_merged["price_price"].fillna(fpl_merged["price"])

    fpl_merged["name_lower"] = fpl_merged["player_name"].str.lower().str.strip()
    fpl_merged["name_last"] = fpl_merged["name_lower"].str.split().str[-1]
    # Also build full_name lookup for better matching
    if "full_name" in fpl_merged.columns:
        fpl_merged["full_lower"] = fpl_merged["full_name"].str.lower().str.strip()
    else:
        fpl_merged["full_lower"] = fpl_merged["name_lower"]

    fbref_processed["name_lower"] = fbref_processed["player_fbref"].str.lower().str.strip()
    fbref_processed["name_last"] = fbref_processed["name_lower"].str.split().str[-1]
    # Also extract first name for combined matching
    fbref_processed["name_first"] = fbref_processed["name_lower"].str.split().str[0]

    # ── Match Strategy (improved, multi-pass) ─────────────────────────
    # Pass 1: exact name match
    # Pass 2: fbref name contained in FPL full_name (same team)
    # Pass 3: FPL player_name contained in fbref name (same team)
    # Pass 4: last name + team match (excluding ambiguous common surnames)
    # Pass 5: first name + team match for single-name players
    # Critically: ALWAYS require team match for ambiguous names

    AMBIGUOUS_LASTNAMES = {
        "james", "wilson", "johnson", "king", "anderson", "gray", "white",
        "martinez", "neto", "silva", "barnes", "armstrong", "harrison",
        "o'brien", "patterson", "mosquera", "onana", "ward", "smith",
        "taylor", "brown", "jones", "williams", "thomas", "moore",
        "davis", "roberts", "walker", "young", "wood", "jackson",
    }

    matched = []
    unmatched_fbref = []
    used_fpl_indices = set()  # track which FPL rows have been matched

    for _, fb_row in fbref_processed.iterrows():
        fb_name = fb_row["name_lower"]
        fb_last = fb_row["name_last"]
        fb_first = fb_row.get("name_first", "")
        fb_team = fb_row["team"]
        best_match = None

        # Pass 1: exact name match on player_name
        exact = fpl_merged[fpl_merged["name_lower"] == fb_name]
        if len(exact) == 1:
            best_match = exact.iloc[0]
        elif len(exact) > 1:
            # Multiple exact matches — prefer same team
            team_exact = exact[exact["team"] == fb_team]
            if len(team_exact) >= 1:
                best_match = team_exact.iloc[0]

        # Pass 2: fbref full name contained in FPL full_name (same team)
        if best_match is None:
            contained = fpl_merged[
                fpl_merged["full_lower"].str.contains(fb_name.replace(".", ""), case=False, na=False, regex=False) &
                (fpl_merged["team"] == fb_team)
            ]
            if len(contained) == 1:
                best_match = contained.iloc[0]

        # Pass 3: FPL player_name contained in fbref name (same team)
        if best_match is None:
            same_team_fpl = fpl_merged[fpl_merged["team"] == fb_team]
            for fpl_idx, fpl_row in same_team_fpl.iterrows():
                fpl_name = fpl_row["name_lower"]
                if len(fpl_name) >= 4 and fpl_name in fb_name:
                    best_match = fpl_row
                    break

        # Pass 4: last name + team match (require team for all names)
        if best_match is None:
            team_last = fpl_merged[
                (fpl_merged["name_last"] == fb_last) &
                (fpl_merged["team"] == fb_team)
            ]
            if len(team_last) == 1:
                best_match = team_last.iloc[0]
            elif len(team_last) > 1:
                # Multiple on same team with same last name — try first name too
                first_match = team_last[team_last["name_lower"].str.contains(fb_first, na=False)]
                if len(first_match) == 1:
                    best_match = first_match.iloc[0]

        # Pass 5: last name only (if unique globally AND not ambiguous)
        if best_match is None and fb_last not in AMBIGUOUS_LASTNAMES:
            last_only = fpl_merged[fpl_merged["name_last"] == fb_last]
            if len(last_only) == 1:
                best_match = last_only.iloc[0]

        # Pass 6: check if fbref first name matches FPL player_name (e.g. "Salah" == "salah")
        if best_match is None:
            for part in fb_name.split():
                if len(part) >= 4:
                    part_match = fpl_merged[
                        (fpl_merged["name_lower"] == part) &
                        (fpl_merged["team"] == fb_team)
                    ]
                    if len(part_match) == 1:
                        best_match = part_match.iloc[0]
                        break

        # Pass 7: check FPL full_name contains fbref last name (same team)
        if best_match is None and fb_last not in AMBIGUOUS_LASTNAMES:
            full_contains = fpl_merged[
                fpl_merged["full_lower"].str.contains(fb_last, case=False, na=False, regex=False) &
                (fpl_merged["team"] == fb_team)
            ]
            if len(full_contains) == 1:
                best_match = full_contains.iloc[0]

        if best_match is not None:
            matched.append(_merge_row(fb_row, best_match))
            if hasattr(best_match, 'name'):
                used_fpl_indices.add(best_match.name)
        else:
            # Unmatched — still include but flag it
            row_dict = fb_row.to_dict()
            row_dict.update({
                "price": 0,  # Use 0 instead of 5.0 so we can filter these out
                "fpl_points": 0,
                "form": 0,
                "selected_pct": 0,
                "fpl_xG": 0,
                "fpl_xA": 0,
                "bonus": 0,
                "ict_index": 0,
                "points_per_game": 0,
                "_unmatched": True,
            })
            matched.append(row_dict)
            unmatched_fbref.append(f"{fb_row['player_fbref']} ({fb_team})")

    # Also add FPL-only players (no fbref data — mostly from teams without fbref data)
    matched_players = set()
    for m in matched:
        matched_players.add((m.get("player_fbref", "").lower(), m.get("team", "")))
        if "fpl_name" in m:
            matched_players.add((m.get("fpl_name", "").lower(), m.get("team", "")))

    for idx, fpl_row in fpl_merged.iterrows():
        # Skip if already matched
        if idx in used_fpl_indices:
            continue
        key = (fpl_row["name_lower"], fpl_row["team"])
        if key in matched_players:
            continue
        if safe_float(fpl_row.get("minutes", 0)) < 90:
            continue

        pos = fpl_row.get("position", "MID")
        minutes = safe_float(fpl_row.get("minutes", 0))
        starts = safe_float(fpl_row.get("starts", 0))
        matches_est = starts + max(0, (minutes - starts * 70) / 25) if starts > 0 else max(1, minutes / 60)
        nineties = minutes / 90

        matched.append({
            "player_fbref": fpl_row["player_name"],
            "team": fpl_row["team"],
            "position": pos,
            "age": "",
            "minutes": minutes,
            "matches": round(matches_est, 1),
            "starts": starts,
            "nineties": round(nineties, 1),
            "goals": safe_float(fpl_row.get("goals_scored", 0)),
            "assists": safe_float(fpl_row.get("assists", 0)),
            "g_plus_a": safe_float(fpl_row.get("goals_scored", 0)) + safe_float(fpl_row.get("assists", 0)),
            "goals_npk": safe_float(fpl_row.get("goals_scored", 0)),
            "pks": 0,
            "pk_att": 0,
            "yellows": safe_float(fpl_row.get("yellow_cards", 0)),
            "reds": safe_float(fpl_row.get("red_cards", 0)),
            "gls_per90": round(safe_float(fpl_row.get("goals_scored", 0)) / nineties, 3) if nineties > 0 else 0,
            "ast_per90": round(safe_float(fpl_row.get("assists", 0)) / nineties, 3) if nineties > 0 else 0,
            "ga_per90": round((safe_float(fpl_row.get("goals_scored", 0)) + safe_float(fpl_row.get("assists", 0))) / nineties, 3) if nineties > 0 else 0,
            "npg_per90": round(safe_float(fpl_row.get("goals_scored", 0)) / nineties, 3) if nineties > 0 else 0,
            "price": safe_float(fpl_row.get("price", 5.0)),
            "fpl_points": safe_float(fpl_row.get("total_points", 0)),
            "form": safe_float(fpl_row.get("form", 0)),
            "selected_pct": safe_float(fpl_row.get("selected_by_percent", 0)),
            "fpl_xG": safe_float(fpl_row.get("expected_goals", 0)),
            "fpl_xA": safe_float(fpl_row.get("expected_assists", 0)),
            "bonus": safe_float(fpl_row.get("bonus", 0)),
            "ict_index": safe_float(fpl_row.get("ict_index", 0)),
            "points_per_game": safe_float(fpl_row.get("points_per_game", 0)),
            "name_lower": fpl_row["name_lower"],
            "name_last": fpl_row["name_last"],
        })
        matched_players.add(key)

    unified = pd.DataFrame(matched)

    # ── Remove unmatched fbref players (no price = can't be in squad) ──
    unmatched_count = len(unified[unified.get("price", 0) == 0]) if "price" in unified.columns else 0
    unified = unified[unified["price"] > 0].copy()

    # ── Deduplicate globally by player name ─────────────────────────────
    # A player may appear twice if they have fbref data from both old and new clubs
    # (mid-season transfers). Keep the version with the most data (highest ga_per90 + fpl_points).
    # First dedup by (name, team) — exact duplicates
    unified["_dedup_key"] = unified["player_fbref"].str.lower().str.strip() + "|" + unified["team"]
    unified["_dedup_score"] = unified["fpl_points"].fillna(0) + unified["ga_per90"].fillna(0) * 100 + unified["minutes"].fillna(0) * 0.01
    unified = unified.sort_values("_dedup_score", ascending=False).drop_duplicates(subset="_dedup_key", keep="first")

    # Then dedup globally by player name — handles transfers (same player, different teams)
    # Keep the version with highest _dedup_score (most data + most minutes)
    unified["_global_key"] = unified["player_fbref"].str.lower().str.strip()
    unified = unified.sort_values("_dedup_score", ascending=False).drop_duplicates(subset="_global_key", keep="first")
    unified = unified.drop(columns=["_dedup_key", "_dedup_score", "_global_key"], errors="ignore")
    unified = unified.reset_index(drop=True)

    if unmatched_fbref:
        print(f"  Note: {len(unmatched_fbref)} fbref players had no FPL match (excluded)")
        if len(unmatched_fbref) <= 20:
            for name in unmatched_fbref:
                print(f"    - {name}")

    print(f"  Unified database: {len(unified)} players total")
    has_fpl = len(unified[unified["fpl_points"] > 0])
    has_fbref_detail = len(unified[unified["ga_per90"] > 0])
    print(f"  With FPL points data: {has_fpl}")
    print(f"  With fbref per-90 data: {has_fbref_detail}")

    return unified


def _merge_row(fb_row, fpl_row):
    """Merge a single fbref row with its FPL API match.
    IMPORTANT: Use FPL team as authoritative (handles mid-season transfers)."""
    row_dict = fb_row.to_dict()
    fpl_team = fpl_row.get("team", row_dict.get("team", "Unknown"))
    row_dict.update({
        "team": fpl_team,  # FPL team is the CURRENT team (post-transfer)
        "price": safe_float(fpl_row.get("price", 5.0)),
        "fpl_points": safe_float(fpl_row.get("total_points", 0)),
        "form": safe_float(fpl_row.get("form", 0)),
        "selected_pct": safe_float(fpl_row.get("selected_by_percent", 0)),
        "fpl_xG": safe_float(fpl_row.get("expected_goals", 0)),
        "fpl_xA": safe_float(fpl_row.get("expected_assists", 0)),
        "bonus": safe_float(fpl_row.get("bonus", 0)),
        "ict_index": safe_float(fpl_row.get("ict_index", 0)),
        "points_per_game": safe_float(fpl_row.get("points_per_game", 0)),
        "fpl_name": fpl_row.get("player_name", ""),
        "position": fpl_row.get("position", row_dict.get("position", "MID")),  # FPL position is authoritative
    })
    return row_dict


# ══════════════════════════════════════════════════════════════════════
# STEP 4: ENHANCED SCORING MODEL
# ══════════════════════════════════════════════════════════════════════

def calculate_enhanced_scores(unified_df, fixtures_df, gameweeks=5):
    """
    Build the enhanced expected-points model.

    Key signals:
      1. Historical per-match rate (actual FPL points / games)
      2. xG regression: if xG > actual goals → player is "due"
      3. Form momentum: recent form vs season average
      4. Per-90 underlying quality (fbref G+A per 90)
      5. Fixture difficulty for next N gameweeks
      6. Clean sheet probability for DEF/GK
      7. Differential bonus: low ownership amplifier
    """
    results = []

    # ── Pre-compute fixture difficulties ──────────────────────────────
    fixture_schedule = _parse_fixtures(fixtures_df, gameweeks)

    for _, p in unified_df.iterrows():
        player = p["player_fbref"]
        team = p["team"]
        pos = p["position"]
        price = p["price"]
        minutes = p["minutes"]
        matches = p["matches"]
        nineties = p.get("nineties", minutes / 90)
        starts = p.get("starts", matches)

        if minutes < 180 or price <= 0:
            continue

        # ── 1. Historical Points Rate ─────────────────────────────────
        fpl_pts = p.get("fpl_points", 0)
        ppg = p.get("points_per_game", 0)
        if ppg == 0 and matches > 0:
            ppg = fpl_pts / matches

        # ── 2. Underlying Quality Score (per 90) ─────────────────────
        ga_per90 = p.get("ga_per90", 0)
        gls_per90 = p.get("gls_per90", 0)
        ast_per90 = p.get("ast_per90", 0)
        npg_per90 = p.get("npg_per90", 0)

        # Convert per-90 rates into expected FPL points per match
        exp_goal_pts_per90 = gls_per90 * SCORING["goal"].get(pos, 4)
        exp_assist_pts_per90 = ast_per90 * SCORING["assist"]
        exp_attacking_per90 = exp_goal_pts_per90 + exp_assist_pts_per90

        # Appearance points (estimated)
        avg_mins = minutes / matches if matches > 0 else 0
        if avg_mins >= 60:
            exp_appearance = SCORING["appearance_60_plus"]
        elif avg_mins >= 30:
            exp_appearance = (SCORING["appearance_60_plus"] * 0.6 +
                            SCORING["appearance_1_59"] * 0.4)
        else:
            exp_appearance = SCORING["appearance_1_59"]

        # Card deductions per match
        card_rate = 0
        if matches > 0:
            card_rate = (p["yellows"] * SCORING["yellow"] +
                        p["reds"] * SCORING["red"]) / matches

        # Bonus estimate
        bonus_rate = SCORING["bonus_avg"].get(pos, 0.2)
        if ga_per90 > 0.5:
            bonus_rate *= 1.3  # high involvement → more bonus
        elif ga_per90 > 0.3:
            bonus_rate *= 1.15

        # ── 3. xG Regression Signal ──────────────────────────────────
        # If xG > actual goals, player is underperforming luck and is due points
        fpl_xG = p.get("fpl_xG", 0)
        fpl_xA = p.get("fpl_xA", 0)
        actual_goals = p.get("goals", 0)
        actual_assists = p.get("assists", 0)

        xg_overperformance = actual_goals - fpl_xG  # positive = overperforming
        xa_overperformance = actual_assists - fpl_xA

        # Regression boost: underperformers get a boost, overperformers get a slight drag
        xg_regression_factor = 1.0
        if fpl_xG > 0 and nineties > 5:
            # xG delta as fraction of xG
            xg_ratio = actual_goals / fpl_xG if fpl_xG > 0 else 1.0
            if xg_ratio < 0.8:
                # Significantly underperforming xG — strong buy signal
                xg_regression_factor = 1.0 + min(0.20, (1.0 - xg_ratio) * 0.3)
            elif xg_ratio > 1.3:
                # Overperforming xG — slight sell signal
                xg_regression_factor = 1.0 - min(0.10, (xg_ratio - 1.0) * 0.1)

        xa_regression_factor = 1.0
        if fpl_xA > 0 and nineties > 5:
            xa_ratio = actual_assists / fpl_xA if fpl_xA > 0 else 1.0
            if xa_ratio < 0.8:
                xa_regression_factor = 1.0 + min(0.15, (1.0 - xa_ratio) * 0.25)
            elif xa_ratio > 1.3:
                xa_regression_factor = 1.0 - min(0.08, (xa_ratio - 1.0) * 0.08)

        # ── 4. Form Momentum ─────────────────────────────────────────
        form = p.get("form", 0)
        form_factor = 1.0
        if ppg > 0 and form > 0:
            form_ratio = form / ppg
            if form_ratio > 1.3:
                form_factor = 1.0 + min(0.15, (form_ratio - 1.0) * 0.15)
            elif form_ratio < 0.7:
                form_factor = 1.0 - min(0.10, (1.0 - form_ratio) * 0.10)

        # ── 5. Build Base Expected Points Per Match ───────────────────
        # Blend historical actuals with underlying rates
        # Weight: 40% actual points rate, 60% underlying quality
        underlying_per_match = (exp_attacking_per90 + exp_appearance +
                               card_rate + bonus_rate)

        if ppg > 0:
            base_expected = ppg * 0.40 + underlying_per_match * 0.60
        else:
            base_expected = underlying_per_match

        # Apply regression and form adjustments
        adjusted_expected = (base_expected *
                           xg_regression_factor *
                           xa_regression_factor *
                           form_factor)

        # ── 6. Fixture Difficulty Adjustment ──────────────────────────
        team_fixtures = fixture_schedule.get(team, [])
        gw_expectations = []

        for gw_info in team_fixtures[:gameweeks]:
            fdr = gw_info.get("fdr", 3)
            is_home = gw_info.get("is_home", True)
            opponent = gw_info.get("opponent", "Average")

            # FDR multiplier: 1=very easy(+30%), 2=easy(+15%), 3=neutral, 4=hard(-15%), 5=very hard(-25%)
            fdr_mult = 1.0 + (3 - fdr) * 0.15
            if fdr == 1:
                fdr_mult = 1.30
            elif fdr == 5:
                fdr_mult = 0.75

            # Home/away
            home_mult = 1.10 if is_home else 0.92

            gw_exp = adjusted_expected * fdr_mult * home_mult
            # Playing probability (based on starts/matches ratio)
            play_prob = min(1.0, starts / matches) if matches > 0 else 0.5
            gw_exp *= play_prob

            gw_expectations.append({
                "expected": round(max(0, gw_exp), 2),
                "opponent": opponent,
                "fdr": fdr,
                "is_home": is_home,
            })

        # Pad if fewer fixtures than requested
        while len(gw_expectations) < gameweeks:
            gw_expectations.append({
                "expected": round(adjusted_expected * 0.85, 2),  # slightly conservative default
                "opponent": "TBD",
                "fdr": 3,
                "is_home": True,
            })

        total_expected = sum(gw["expected"] for gw in gw_expectations)

        # ── 7. Value & Differential Metrics ───────────────────────────
        value_score = total_expected / price if price > 0 else 0
        selected_pct = p.get("selected_pct", 50)

        # Differential bonus: multiply value by ownership inverse
        # Low owned + high value = hidden gem
        if selected_pct < 5:
            differential_mult = 1.20
        elif selected_pct < 10:
            differential_mult = 1.10
        elif selected_pct < 20:
            differential_mult = 1.05
        else:
            differential_mult = 1.00

        # ── Compile Result ────────────────────────────────────────────
        result = {
            "player": player,
            "team": team,
            "position": pos,
            "price": price,
            "minutes": minutes,
            "matches": matches,
            "starts": starts,

            # Raw stats
            "goals": int(p.get("goals", 0)),
            "assists": int(p.get("assists", 0)),
            "g_plus_a": int(p.get("g_plus_a", 0)),

            # Per-90 rates
            "gls_per90": gls_per90,
            "ast_per90": ast_per90,
            "ga_per90": ga_per90,
            "npg_per90": npg_per90,

            # FPL data
            "fpl_points": fpl_pts,
            "ppg": round(ppg, 2),
            "form": form,
            "selected_pct": selected_pct,
            "bonus": p.get("bonus", 0),

            # xG analysis
            "fpl_xG": fpl_xG,
            "fpl_xA": fpl_xA,
            "xG_delta": round(actual_goals - fpl_xG, 2),  # negative = underperforming = buy
            "xA_delta": round(actual_assists - fpl_xA, 2),
            "xg_regression_factor": round(xg_regression_factor, 3),

            # Model outputs
            "underlying_per_match": round(underlying_per_match, 2),
            "base_expected_per_match": round(base_expected, 2),
            "adjusted_expected_per_match": round(adjusted_expected, 2),
            "total_expected_Ngw": round(total_expected, 2),
            "avg_expected_per_gw": round(total_expected / gameweeks, 2),
            "value_score": round(value_score, 2),
            "differential_mult": differential_mult,
            "enhanced_value": round(value_score * differential_mult, 2),
        }

        # Add per-GW breakdown
        for i, gw in enumerate(gw_expectations):
            result[f"gw{i+1}_exp"] = gw["expected"]
            result[f"gw{i+1}_opp"] = gw["opponent"]
            result[f"gw{i+1}_fdr"] = gw["fdr"]
            result[f"gw{i+1}_home"] = "H" if gw["is_home"] else "A"

        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("total_expected_Ngw", ascending=False).reset_index(drop=True)
    return df


def _parse_fixtures(fixtures_df, gameweeks):
    """Parse future fixtures into per-team schedule."""
    schedule = {}
    if fixtures_df is None or fixtures_df.empty:
        return schedule

    # Sort by gameweek
    fixtures_df = fixtures_df.sort_values("gameweek")

    # Find future (unfinished) gameweeks
    future = fixtures_df[fixtures_df["finished"] != True].copy()
    if future.empty:
        # All marked as finished — use latest GWs anyway
        future = fixtures_df.tail(gameweeks * 10)

    future_gws = sorted(future["gameweek"].unique())[:gameweeks]

    for _, row in future.iterrows():
        gw = row["gameweek"]
        if gw not in future_gws:
            continue

        home_team = row["home_team"]
        away_team = row["away_team"]
        home_fdr = safe_float(row.get("home_difficulty", 3))
        away_fdr = safe_float(row.get("away_difficulty", 3))

        if home_team not in schedule:
            schedule[home_team] = []
        if away_team not in schedule:
            schedule[away_team] = []

        schedule[home_team].append({
            "gw": gw,
            "opponent": away_team,
            "fdr": int(home_fdr),
            "is_home": True,
        })
        schedule[away_team].append({
            "gw": gw,
            "opponent": home_team,
            "fdr": int(away_fdr),
            "is_home": False,
        })

    return schedule


# ══════════════════════════════════════════════════════════════════════
# STEP 5: SQUAD OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════

SQUAD_RULES = {
    "total_players": 15,
    "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
    "max_per_team": 3,
}


def optimize_squad(predictions_df, budget=103.0):
    """
    Build optimal 15-man FPL squad using linear programming.
    Objective: maximize total expected points over N gameweeks.
    """
    df = predictions_df.copy()
    df = df[df["price"] > 0].reset_index(drop=True)

    # Use total_expected_Ngw as the primary objective
    df["opt_score"] = df["total_expected_Ngw"]

    print(f"\n  Optimizing from {len(df)} players | Budget: £{budget}m")

    if HAS_PULP:
        result = _lp_optimize(df, budget)
    else:
        print("  PuLP not available — using greedy fallback")
        result = _greedy_optimize(df, budget)

    return result


def _lp_optimize(df, budget):
    """Linear programming optimization with PuLP."""
    try:
        prob = LpProblem("FPL_Enhanced", LpMaximize)
        players = df.index.tolist()
        x = LpVariable.dicts("p", players, cat="Binary")

        # Objective: maximize expected points
        prob += lpSum(df.loc[i, "opt_score"] * x[i] for i in players)

        # Squad size = 15
        prob += lpSum(x[i] for i in players) == SQUAD_RULES["total_players"]

        # Position constraints
        for pos, count in SQUAD_RULES["positions"].items():
            pos_idx = df[df["position"] == pos].index.tolist()
            prob += lpSum(x[i] for i in pos_idx) == count

        # Budget
        prob += lpSum(df.loc[i, "price"] * x[i] for i in players) <= budget

        # Max 3 per team
        for team in df["team"].unique():
            team_idx = df[df["team"] == team].index.tolist()
            prob += lpSum(x[i] for i in team_idx) <= SQUAD_RULES["max_per_team"]

        # Solve (suppress output)
        prob.solve()

        if LpStatus[prob.status] != "Optimal":
            print(f"  Solver status: {LpStatus[prob.status]} — falling back to greedy")
            return _greedy_optimize(df, budget)

        selected = [i for i in players if value(x[i]) == 1]
        squad = df.loc[selected].copy()
        return _format_result(squad, budget)
    except (OSError, Exception) as e:
        print(f"  Solver error ({type(e).__name__}) — using greedy optimization")
        return _greedy_optimize(df, budget)


def _greedy_optimize(df, budget):
    """Greedy fallback if PuLP unavailable."""
    df = df.copy()
    df["value_ratio"] = df["opt_score"] / df["price"]
    df = df.sort_values("value_ratio", ascending=False)

    squad = []
    pos_count = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    team_count = {}
    total_cost = 0

    for _, player in df.iterrows():
        pos = player["position"]
        team = player["team"]
        if pos_count.get(pos, 0) >= SQUAD_RULES["positions"].get(pos, 0):
            continue
        if team_count.get(team, 0) >= SQUAD_RULES["max_per_team"]:
            continue
        if total_cost + player["price"] > budget:
            continue

        squad.append(player)
        pos_count[pos] = pos_count.get(pos, 0) + 1
        team_count[team] = team_count.get(team, 0) + 1
        total_cost += player["price"]

        if len(squad) == SQUAD_RULES["total_players"]:
            break

    return _format_result(pd.DataFrame(squad), budget)


def _format_result(squad, budget):
    """Format squad result with starting XI and bench."""
    squad = squad.sort_values(["position", "total_expected_Ngw"], ascending=[True, False])
    total_cost = squad["price"].sum()

    # Pick starting XI
    xi = _pick_starting_xi(squad)
    bench = squad[~squad.index.isin(xi.index)]

    return {
        "squad": squad,
        "starting_xi": xi,
        "bench": bench,
        "total_cost": round(total_cost, 1),
        "expected_points": round(squad["total_expected_Ngw"].sum(), 1),
        "xi_expected": round(xi["total_expected_Ngw"].sum(), 1),
        "budget_remaining": round(budget - total_cost, 1),
    }


def _pick_starting_xi(squad):
    """Pick best starting XI from 15-man squad."""
    gks = squad[squad["position"] == "GK"].sort_values("total_expected_Ngw", ascending=False)
    defs = squad[squad["position"] == "DEF"].sort_values("total_expected_Ngw", ascending=False)
    mids = squad[squad["position"] == "MID"].sort_values("total_expected_Ngw", ascending=False)
    fwds = squad[squad["position"] == "FWD"].sort_values("total_expected_Ngw", ascending=False)

    # Start with minimums: 1 GK, 3 DEF, 2 MID, 1 FWD = 7
    xi = pd.concat([gks.head(1), defs.head(3), mids.head(2), fwds.head(1)])

    # Fill remaining 4 from highest expected outfield
    remaining = squad[~squad.index.isin(xi.index)]
    remaining = remaining[remaining["position"] != "GK"].sort_values("total_expected_Ngw", ascending=False)
    xi = pd.concat([xi, remaining.head(4)])

    return xi


# ══════════════════════════════════════════════════════════════════════
# STEP 6: REPORTING
# ══════════════════════════════════════════════════════════════════════

def print_value_picks(predictions_df, n=15):
    """Print top value/differential picks."""
    print(f"\n{'='*80}")
    print("  TOP VALUE PICKS (best underlying stats per £)")
    print(f"{'='*80}")

    # Filter players with enough minutes
    df = predictions_df[predictions_df["minutes"] >= 450].copy()

    # By position
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_df = df[df["position"] == pos].sort_values("enhanced_value", ascending=False).head(8)
        print(f"\n  {'─'*60}")
        print(f"  {pos} — Top Value Picks")
        print(f"  {'─'*60}")
        for _, p in pos_df.iterrows():
            xg_flag = ""
            if p.get("xG_delta", 0) < -1:
                xg_flag = " ⬆ xG UNDERPERFORMER"
            elif p.get("xG_delta", 0) > 2:
                xg_flag = " ⬇ xG overperformer"

            diff_flag = ""
            if p.get("selected_pct", 50) < 10:
                diff_flag = f" [DIFF {p['selected_pct']}%]"

            form_str = f"form:{p['form']}" if p.get("form", 0) > 0 else ""

            print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
                  f"exp:{p['total_expected_Ngw']:5.1f}pts | val:{p['enhanced_value']:4.2f} | "
                  f"G+A/90:{p['ga_per90']:.2f} | {form_str}{xg_flag}{diff_flag}")


def print_regression_candidates(predictions_df, n=10):
    """Find players whose actual output trails their xG/xA significantly."""
    print(f"\n{'='*80}")
    print("  xG REGRESSION CANDIDATES (due a points surge)")
    print(f"{'='*80}")

    df = predictions_df[
        (predictions_df["minutes"] >= 900) &
        (predictions_df["fpl_xG"] > 0)
    ].copy()

    df["xGI_delta"] = df["xG_delta"] + df["xA_delta"]
    underperformers = df[df["xGI_delta"] < -1].sort_values("xGI_delta").head(n)

    if len(underperformers) == 0:
        print("  No significant xG underperformers found (data may be limited)")
        return

    for _, p in underperformers.iterrows():
        print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
              f"Goals:{p['goals']} vs xG:{p['fpl_xG']:.1f} (delta:{p['xG_delta']:+.1f}) | "
              f"Assists:{p['assists']} vs xA:{p['fpl_xA']:.1f} (delta:{p['xA_delta']:+.1f}) | "
              f"Owned:{p['selected_pct']:.1f}%")


def print_differentials(predictions_df, n=15):
    """Find low-ownership players with strong underlying stats."""
    print(f"\n{'='*80}")
    print("  DIFFERENTIAL PICKS (low ownership, strong underlying)")
    print(f"{'='*80}")

    df = predictions_df[
        (predictions_df["minutes"] >= 900) &
        (predictions_df["selected_pct"] > 0) &
        (predictions_df["selected_pct"] < 15)
    ].copy()

    df = df.sort_values("enhanced_value", ascending=False).head(n)

    for _, p in df.iterrows():
        print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
              f"Owned:{p['selected_pct']:5.1f}% | exp:{p['total_expected_Ngw']:5.1f}pts | "
              f"G+A/90:{p['ga_per90']:.2f} | form:{p['form']}")


def print_squad(result, gameweeks):
    """Print the final optimized squad."""
    print(f"\n{'#'*80}")
    print(f"#  OPTIMAL SQUAD — £{result['total_cost']}m / £103m "
          f"(£{result['budget_remaining']}m remaining)")
    print(f"#  Expected points (next {gameweeks} GWs): {result['expected_points']:.1f}")
    print(f"#  Starting XI expected: {result['xi_expected']:.1f}")
    print(f"{'#'*80}")

    print(f"\n  STARTING XI:")
    print(f"  {'─'*75}")
    xi = result["starting_xi"].sort_values(
        ["position", "total_expected_Ngw"],
        ascending=[True, False],
        key=lambda x: x.map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}) if x.name == "position" else x
    )

    for _, p in xi.iterrows():
        gw_str = ""
        for i in range(1, gameweeks + 1):
            opp = p.get(f"gw{i}_opp", "?")
            ha = p.get(f"gw{i}_home", "?")
            exp = p.get(f"gw{i}_exp", 0)
            gw_str += f" {opp}({ha}):{exp:.1f}"

        diff_tag = f" DIFF({p['selected_pct']:.0f}%)" if p.get("selected_pct", 50) < 10 else ""
        xg_tag = " xG↑" if p.get("xG_delta", 0) < -1 else ""

        print(f"  {p['position']:3s} {p['player']:25s} {p['team']:15s} "
              f"£{p['price']:4.1f}m | exp:{p['total_expected_Ngw']:5.1f}{diff_tag}{xg_tag}")

    print(f"\n  BENCH:")
    print(f"  {'─'*75}")
    for _, p in result["bench"].iterrows():
        print(f"  {p['position']:3s} {p['player']:25s} {p['team']:15s} "
              f"£{p['price']:4.1f}m | exp:{p['total_expected_Ngw']:5.1f}")


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main(budget=103.0, gameweeks=5, min_minutes=180):
    print("\n" + "█" * 80)
    print("█  FPL ENHANCED MODEL — Underlying Stats Optimizer")
    print(f"█  Budget: £{budget}m | Lookahead: {gameweeks} GWs | Min minutes: {min_minutes}")
    print("█" * 80)

    # Step 1: Load fbref data
    print(f"\n{'='*60}")
    print("  STEP 1: Loading FBref data from team folders")
    print(f"{'='*60}")
    fbref_df = load_all_fbref_data()

    # Step 2: Load FPL API data
    print(f"\n{'='*60}")
    print("  STEP 2: Loading FPL API data")
    print(f"{'='*60}")
    fpl_data = load_fpl_api_data()

    # Step 3: Build unified database
    print(f"\n{'='*60}")
    print("  STEP 3: Building unified player database")
    print(f"{'='*60}")
    unified = build_unified_database(fbref_df, fpl_data)

    # Step 4: Enhanced scoring
    print(f"\n{'='*60}")
    print("  STEP 4: Running enhanced scoring model")
    print(f"{'='*60}")
    predictions = calculate_enhanced_scores(
        unified,
        fpl_data.get("fixtures", pd.DataFrame()),
        gameweeks=gameweeks,
    )
    print(f"  Scored {len(predictions)} players")

    # Save predictions
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pred_path = os.path.join(OUTPUT_DIR, "enhanced_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    print(f"  Saved predictions to: {pred_path}")

    # Step 5: Print insights
    print_value_picks(predictions)
    print_regression_candidates(predictions)
    print_differentials(predictions)

    # Step 6: Optimize squad
    print(f"\n{'='*60}")
    print("  STEP 5: Optimizing squad selection")
    print(f"{'='*60}")
    result = optimize_squad(predictions, budget)

    if result:
        print_squad(result, gameweeks)

        # Save squad
        squad_path = os.path.join(OUTPUT_DIR, "enhanced_squad.csv")
        result["squad"].to_csv(squad_path, index=False)
        print(f"\n  Squad saved to: {squad_path}")

    print(f"\n{'█'*80}")
    print("█  PIPELINE COMPLETE")
    print(f"{'█'*80}\n")

    return predictions, result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FPL Enhanced Model")
    parser.add_argument("--budget", type=float, default=103.0)
    parser.add_argument("--gameweeks", type=int, default=5)
    parser.add_argument("--min-minutes", type=int, default=180)
    args = parser.parse_args()

    main(budget=args.budget, gameweeks=args.gameweeks, min_minutes=args.min_minutes)
