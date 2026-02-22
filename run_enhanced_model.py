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
    python run_enhanced_model.py
    python run_enhanced_model.py --budget 100 --gameweeks 5
"""

import pandas as pd
import numpy as np
import os
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

# BASE_DIR points to the project root (parent of Code/ folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FBREF_DIR = os.path.join(BASE_DIR, "data", "Harvested Data", "PL teams data 2025-2026")
FPL_API_DIR = os.path.join(BASE_DIR, "data", "Harvested Data", "FPL API data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Results")

# FPL scoring rules (points per action)
SCORING = {
    "appearance_60_plus": 2, "appearance_1_59": 1,
    "goal": {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": 3,
    "clean_sheet": {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "gc_per_2": -1, "saves_per_3": 1, "penalty_save": 5,
    "yellow": -1, "red": -3,
    "bonus_avg": {"GK": 0.15, "DEF": 0.18, "MID": 0.22, "FWD": 0.25},
}

# FBref position codes → FPL positions
POS_MAP = {
    "FW": "FWD", "FW,MF": "FWD", "MF,FW": "MID", "FW,DF": "FWD",
    "MF": "MID", "MF,DF": "MID", "DF,MF": "DEF",
    "DF": "DEF", "DF,FW": "DEF", "GK": "GK",
}

# FBref folder names → FPL team names
TEAM_NAME_MAP = {
    "Arsenal": "Arsenal", "Aston_Villa": "Aston Villa",
    "Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Brighton": "Brighton", "Burnley": "Burnley",
    "Chelsea": "Chelsea", "Crystal_Palace": "Crystal Palace",
    "Everton": "Everton", "Fulham": "Fulham",
    "Leeds": "Leeds", "Leeds United": "Leeds",
    "Liverpool": "Liverpool", "Manchester_City": "Man City",
    "Manchester_Utd": "Man Utd", "Newcastle_Utd": "Newcastle",
    "Nottham_Forest": "Nott'm Forest", "Nott'ham_Forest": "Nott'm Forest",
    "Sunderland": "Sunderland", "Sunderland ": "Sunderland",
    "Tottenham": "Spurs", "West_Ham": "West Ham", "Wolves": "Wolves",
}

# Fixture CSV full names → FPL short names (only non-trivial mappings needed,
# but we include all 20 so any CSV format works without error)
FIXTURE_TEAM_MAP = {
    "Manchester City": "Man City", "Manchester Utd": "Man Utd",
    "Manchester United": "Man Utd", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Spurs",
    "Tottenham": "Spurs", "West Ham United": "West Ham",
    "West Ham": "West Ham", "Leeds United": "Leeds",
    "Crystal Palace": "Crystal Palace", "Aston Villa": "Aston Villa",
    "Arsenal": "Arsenal", "Chelsea": "Chelsea", "Liverpool": "Liverpool",
    "Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Brighton": "Brighton", "Burnley": "Burnley", "Everton": "Everton",
    "Fulham": "Fulham", "Sunderland": "Sunderland", "Wolves": "Wolves",
}

# Explicit name aliases for known tricky FBref→FPL mismatches.
# Add new entries here when a player's FBref name doesn't match FPL.
NAME_ALIASES = {
    "gabriel jesus": "g.jesus",
    "gabriel magalhães": "gabriel",
    "gabriel martinelli": "martinelli",
    "bernardo silva": "bernardo",
}

# Last names shared by multiple PL players — always require team match
AMBIGUOUS_LASTNAMES = {
    "james", "wilson", "johnson", "king", "anderson", "gray", "white",
    "martinez", "neto", "silva", "barnes", "armstrong", "harrison",
    "o'brien", "patterson", "mosquera", "onana", "ward", "smith",
    "taylor", "brown", "jones", "williams", "thomas", "moore",
    "davis", "roberts", "walker", "young", "wood", "jackson",
}

SQUAD_RULES = {"total_players": 15, "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, "max_per_team": 3}

# Team shirt colours for HTML: (primary, secondary)
TEAM_COLORS = {
    "Arsenal": ("#EF0107", "#FFFFFF"), "Aston Villa": ("#670E36", "#95BFE5"),
    "Bournemouth": ("#DA291C", "#000000"), "Brentford": ("#E30613", "#FFFFFF"),
    "Brighton": ("#0057B8", "#FFFFFF"), "Burnley": ("#6C1D45", "#99D6EA"),
    "Chelsea": ("#034694", "#FFFFFF"), "Crystal Palace": ("#1B458F", "#C4122E"),
    "Everton": ("#003399", "#FFFFFF"), "Fulham": ("#000000", "#FFFFFF"),
    "Leeds": ("#FFFFFF", "#1D428A"), "Liverpool": ("#C8102E", "#FFFFFF"),
    "Man City": ("#6CABDD", "#FFFFFF"), "Man Utd": ("#DA291C", "#FFFFFF"),
    "Newcastle": ("#241F20", "#FFFFFF"), "Nott'm Forest": ("#DD0000", "#FFFFFF"),
    "Sunderland": ("#EB172B", "#FFFFFF"), "Spurs": ("#FFFFFF", "#132257"),
    "West Ham": ("#7A263A", "#1BB1E7"), "Wolves": ("#FDB913", "#231F20"),
}


# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    if pd.isna(val):
        return default
    try:
        v = float(str(val).replace(",", "").strip())
        return v if not np.isnan(v) else default
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD FBREF DATA
# ══════════════════════════════════════════════════════════════════════

def _standardize_columns(df):
    """Handle duplicate column names and map alternatives to standard FBref names."""
    seen, new_cols = {}, []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_per90" if seen[col] == 1 else f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    for old, new in {"Position": "Pos", "Minutes": "Min", "Goals": "Gls",
                     "Assists": "Ast", "Yellow Cards": "CrdY", "Red Cards": "CrdR"}.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


def load_single_team_csv(filepath, team_name):
    """Load a team's FBref CSV, auto-detecting multi-header vs clean format."""
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            line1 = f.readline().strip()
        is_multi = line1.split(",")[0].strip() in ("", "Playing Time", "Performance")
        df = pd.read_csv(filepath, header=1 if is_multi else 0, encoding="utf-8-sig")
        df = _standardize_columns(df)
        df["team"] = team_name
        if "Player" in df.columns:
            df = df[df["Player"].notna() & (df["Player"] != "")].copy()
            df = df[~df["Player"].str.contains("Squad Total|Opponent Total", na=False)].copy()
        return df
    except Exception as e:
        print(f"  WARNING: Could not load {filepath}: {e}")
        return pd.DataFrame()


def load_all_fbref_data():
    """Load FBref CSVs from all team folders."""
    all_players, teams_loaded = [], []
    if not os.path.exists(FBREF_DIR):
        print(f"  ERROR: fbref directory not found: {FBREF_DIR}")
        return pd.DataFrame()
    for folder in sorted(os.listdir(FBREF_DIR)):
        path = os.path.join(FBREF_DIR, folder)
        if not os.path.isdir(path):
            continue
        csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csvs:
            continue
        team = TEAM_NAME_MAP.get(folder, folder.replace("_", " "))
        for csv_file in csvs:
            df = load_single_team_csv(os.path.join(path, csv_file), team)
            if not df.empty:
                all_players.append(df)
                if team not in teams_loaded:
                    teams_loaded.append(team)
    if not all_players:
        print("  ERROR: No fbref data loaded.")
        return pd.DataFrame()
    combined = pd.concat(all_players, ignore_index=True)
    print(f"  Loaded fbref data: {len(combined)} players from {len(teams_loaded)} teams")
    print(f"  Teams: {', '.join(teams_loaded)}")
    return combined


# ══════════════════════════════════════════════════════════════════════
# STEP 2: LOAD FPL API DATA
# ══════════════════════════════════════════════════════════════════════

def load_fpl_api_data():
    """Load all FPL API CSVs: stats, prices, fixtures, team-level data."""
    files = {
        "stats": "fpl_player_stats.csv", "prices": "prices.csv",
        "prices_updated": "fpl_players_Prices and points 2025-26.csv",
        "injuries": "pl_injuries_2025-26.csv", "fixtures": "fixtures_future.csv",
        "team_attack": "AllTeamsStats - attacking .csv",
        "team_defense": "AllTeamsStats - defensive .csv",
        "team_attack_detail": "PremierLeague_AttackingStats.csv",
        "team_gk": "PremierLeague_GoalkeeperStats.csv",
        "past_results": "PremierLeague_PastResults_All.csv",
        "upcoming": "PremierLeague_UpcomingMatches.csv",
    }
    data = {}
    for name, filename in files.items():
        path = os.path.join(FPL_API_DIR, filename)
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
            print(f"  Loaded {name}: {len(data[name])} rows")
        else:
            data[name] = pd.DataFrame()
    return data


# ══════════════════════════════════════════════════════════════════════
# STEP 3: BUILD UNIFIED PLAYER DATABASE
# Merges FBref per-90 stats with FPL API data via multi-pass name matching
# ══════════════════════════════════════════════════════════════════════

def map_position(pos_str):
    if pd.isna(pos_str):
        return "MID"
    pos = str(pos_str).strip().replace("/", ",")
    return POS_MAP.get(pos, POS_MAP.get(pos.split(",")[0], "MID"))


def _merge_row(fb_row, fpl_row, updated_price_map=None, updated_points_map=None, injury_map=None):
    """Merge one FBref row with its FPL match. FPL team is authoritative."""
    d = fb_row.to_dict()
    fn = str(fpl_row.get("player_name", "")).strip()
    price = safe_float(fpl_row.get("price", 5.0))
    if updated_price_map and fn.lower() in updated_price_map:
        price = updated_price_map[fn.lower()]
    fpl_pts = safe_float(fpl_row.get("total_points", 0))
    if updated_points_map and fn.lower() in updated_points_map:
        p = updated_points_map[fn.lower()]
        if p > 0:
            fpl_pts = p
    inj, inj_info, inj_ret = False, "", ""
    if injury_map:
        i = injury_map.get(fn.lower()) or injury_map.get(d.get("player_fbref", "").lower())
        if i:
            inj, inj_info, inj_ret = True, i.get("injury", ""), i.get("return_date", "")
    d.update({
        "team": fpl_row.get("team", d.get("team", "Unknown")),
        "price": price, "fpl_points": fpl_pts,
        "form": safe_float(fpl_row.get("form", 0)),
        "selected_pct": safe_float(fpl_row.get("selected_by_percent", 0)),
        "fpl_xG": safe_float(fpl_row.get("expected_goals", 0)),
        "fpl_xA": safe_float(fpl_row.get("expected_assists", 0)),
        "bonus": safe_float(fpl_row.get("bonus", 0)),
        "ict_index": safe_float(fpl_row.get("ict_index", 0)),
        "points_per_game": safe_float(fpl_row.get("points_per_game", 0)),
        "fpl_name": fn,
        "position": fpl_row.get("position", d.get("position", "MID")),
        "is_injured": inj, "injury_info": inj_info, "injury_return": inj_ret,
    })
    return d


def build_unified_database(fbref_df, fpl_data):
    """Merge FBref stats with FPL API via 8-pass name matching."""
    fpl_stats, fpl_prices = fpl_data["stats"], fpl_data["prices"]

    # Updated prices & injuries
    prices_updated = fpl_data.get("prices_updated", pd.DataFrame())
    up_price, up_pts = {}, {}
    if not prices_updated.empty and "Player" in prices_updated.columns:
        for _, r in prices_updated.iterrows():
            nm = str(r["Player"]).strip().lower()
            up_price[nm] = safe_float(str(r.get("Cost", "0")).replace("£", "").replace("m", "").strip())
            up_pts[nm] = safe_float(r.get("Points", 0))
        print(f"  Updated prices loaded: {len(up_price)} players")

    injuries = fpl_data.get("injuries", pd.DataFrame())
    inj_map = {}
    if not injuries.empty and "Player" in injuries.columns:
        for _, r in injuries.iterrows():
            inj_map[str(r["Player"]).strip().lower()] = {
                "injury": str(r.get("Injury", "")),
                "return_date": str(r.get("Expected Return", "")),
            }
        print(f"  Injury list loaded: {len(inj_map)} players")

    # Extract per-90 rates from FBref
    players = []
    for _, row in fbref_df.iterrows():
        name = str(row.get("Player", "")).strip()
        if not name or name == "nan":
            continue
        mins = safe_float(row.get("Min", 0))
        mp = safe_float(row.get("MP", 0))
        if mins < 90 or mp < 1:
            continue
        g = safe_float(row.get("Gls", 0))
        a = safe_float(row.get("Ast", 0))
        ga = safe_float(row.get("G+A", g + a))
        gnpk = safe_float(row.get("G-PK", g))
        n90 = safe_float(row.get("90s", 0)) or mins / 90
        players.append({
            "player_fbref": name, "team": row.get("team", "Unknown"),
            "position": map_position(row.get("Pos", "MID")),
            "age": str(row.get("Age", "")),
            "minutes": mins, "matches": mp,
            "starts": safe_float(row.get("Starts", 0)), "nineties": n90,
            "goals": g, "assists": a, "g_plus_a": ga, "goals_npk": gnpk,
            "pks": safe_float(row.get("PK", 0)), "pk_att": safe_float(row.get("PKatt", 0)),
            "yellows": safe_float(row.get("CrdY", 0)), "reds": safe_float(row.get("CrdR", 0)),
            "gls_per90": round(g / n90, 3), "ast_per90": round(a / n90, 3),
            "ga_per90": round(ga / n90, 3), "npg_per90": round(gnpk / n90, 3),
        })

    fb = pd.DataFrame(players)
    print(f"  Processed {len(fb)} fbref players (90+ mins)")

    if fpl_stats.empty or fpl_prices.empty:
        print("  WARNING: No FPL API data. Using fbref only.")
        for c in ["price", "fpl_points", "form", "selected_pct", "fpl_xG", "fpl_xA", "bonus", "ict_index"]:
            fb[c] = 5.0 if c == "price" else 0
        return fb

    # Build FPL lookup
    fpl = fpl_stats.merge(fpl_prices[["player_name", "price"]].drop_duplicates(),
                          on="player_name", how="left", suffixes=("", "_p"))
    if "price_p" in fpl.columns:
        fpl["price"] = fpl["price_p"].fillna(fpl["price"])
    fpl["name_lower"] = fpl["player_name"].str.lower().str.strip()
    fpl["name_last"] = fpl["name_lower"].str.split().str[-1]
    fpl["full_lower"] = (fpl["full_name"].str.lower().str.strip()
                         if "full_name" in fpl.columns else fpl["name_lower"])
    fb["name_lower"] = fb["player_fbref"].str.lower().str.strip()
    fb["name_last"] = fb["name_lower"].str.split().str[-1]

    # 8-pass name matching
    matched, unmatched = [], []
    used_idx = set()

    for _, fr in fb.iterrows():
        fn, fl, ft = fr["name_lower"], fr["name_last"], fr["team"]
        ff = fn.split()[0] if " " in fn else ""
        best = None

        # Pass 0: alias
        at = NAME_ALIASES.get(fn)
        if at:
            m = fpl[(fpl["name_lower"] == at) & (fpl["team"] == ft)]
            if len(m) == 1: best = m.iloc[0]
            elif len(m) == 0:
                m = fpl[fpl["name_lower"] == at]
                if len(m) == 1: best = m.iloc[0]

        # Pass 1: exact
        if best is None:
            m = fpl[fpl["name_lower"] == fn]
            if len(m) == 1: best = m.iloc[0]
            elif len(m) > 1:
                mt = m[m["team"] == ft]
                if len(mt) >= 1: best = mt.iloc[0]

        # Pass 2: fbref name in FPL full_name (same team)
        if best is None:
            m = fpl[fpl["full_lower"].str.contains(fn.replace(".", ""), case=False, na=False, regex=False)
                     & (fpl["team"] == ft)]
            if len(m) == 1: best = m.iloc[0]

        # Pass 3: FPL name in fbref name (same team)
        if best is None:
            for _, r in fpl[fpl["team"] == ft].iterrows():
                if len(r["name_lower"]) >= 4 and r["name_lower"] in fn:
                    best = r; break

        # Pass 4: last name + team
        if best is None:
            m = fpl[(fpl["name_last"] == fl) & (fpl["team"] == ft)]
            if len(m) == 1: best = m.iloc[0]
            elif len(m) > 1 and ff:
                m2 = m[m["name_lower"].str.contains(ff, na=False)]
                if len(m2) == 1: best = m2.iloc[0]

        # Pass 5: last name globally (if unique + not ambiguous)
        if best is None and fl not in AMBIGUOUS_LASTNAMES:
            m = fpl[fpl["name_last"] == fl]
            if len(m) == 1: best = m.iloc[0]

        # Pass 6: name parts (same team)
        if best is None:
            for part in fn.split():
                if len(part) >= 4:
                    m = fpl[(fpl["name_lower"] == part) & (fpl["team"] == ft)]
                    if len(m) == 1: best = m.iloc[0]; break

        # Pass 7: FPL full_name contains last name (same team)
        if best is None and fl not in AMBIGUOUS_LASTNAMES:
            m = fpl[fpl["full_lower"].str.contains(fl, case=False, na=False, regex=False)
                     & (fpl["team"] == ft)]
            if len(m) == 1: best = m.iloc[0]

        if best is not None:
            matched.append(_merge_row(fr, best, up_price, up_pts, inj_map))
            if hasattr(best, 'name'): used_idx.add(best.name)
        else:
            d = fr.to_dict()
            d.update({"price": 0, "fpl_points": 0, "form": 0, "selected_pct": 0,
                       "fpl_xG": 0, "fpl_xA": 0, "bonus": 0, "ict_index": 0,
                       "points_per_game": 0})
            matched.append(d)
            unmatched.append(f"{fr['player_fbref']} ({ft})")

    # Add FPL-only players (no FBref data)
    mk = set()
    for m in matched:
        mk.add((m.get("player_fbref", "").lower(), m.get("team", "")))
        if "fpl_name" in m:
            mk.add((m.get("fpl_name", "").lower(), m.get("team", "")))
    for idx, r in fpl.iterrows():
        if idx in used_idx: continue
        key = (r["name_lower"], r["team"])
        if key in mk: continue
        mins = safe_float(r.get("minutes", 0))
        if mins < 90: continue
        starts = safe_float(r.get("starts", 0))
        n90 = mins / 90
        g, a = safe_float(r.get("goals_scored", 0)), safe_float(r.get("assists", 0))
        matched.append({
            "player_fbref": r["player_name"], "team": r["team"],
            "position": r.get("position", "MID"), "age": "",
            "minutes": mins,
            "matches": round(starts + max(0, (mins - starts * 70) / 25) if starts > 0 else max(1, mins / 60), 1),
            "starts": starts, "nineties": round(n90, 1),
            "goals": g, "assists": a, "g_plus_a": g + a, "goals_npk": g,
            "pks": 0, "pk_att": 0,
            "yellows": safe_float(r.get("yellow_cards", 0)),
            "reds": safe_float(r.get("red_cards", 0)),
            "gls_per90": round(g / n90, 3) if n90 > 0 else 0,
            "ast_per90": round(a / n90, 3) if n90 > 0 else 0,
            "ga_per90": round((g + a) / n90, 3) if n90 > 0 else 0,
            "npg_per90": round(g / n90, 3) if n90 > 0 else 0,
            "price": safe_float(r.get("price", 5.0)),
            "fpl_points": safe_float(r.get("total_points", 0)),
            "form": safe_float(r.get("form", 0)),
            "selected_pct": safe_float(r.get("selected_by_percent", 0)),
            "fpl_xG": safe_float(r.get("expected_goals", 0)),
            "fpl_xA": safe_float(r.get("expected_assists", 0)),
            "bonus": safe_float(r.get("bonus", 0)),
            "ict_index": safe_float(r.get("ict_index", 0)),
            "points_per_game": safe_float(r.get("points_per_game", 0)),
            "name_lower": r["name_lower"], "name_last": r["name_last"],
        })
        mk.add(key)

    u = pd.DataFrame(matched)
    u = u[u["price"] > 0].copy()

    # Deduplicate by (name, team) then globally by name
    u["_ds"] = u["fpl_points"].fillna(0) + u["ga_per90"].fillna(0) * 100 + u["minutes"].fillna(0) * 0.01
    u = u.sort_values("_ds", ascending=False)
    u["_dk1"] = u["player_fbref"].str.lower().str.strip() + "|" + u["team"]
    u = u.drop_duplicates(subset=["_dk1"], keep="first")
    u["_dk2"] = u["player_fbref"].str.lower().str.strip()
    u = u.drop_duplicates(subset=["_dk2"], keep="first")
    u = u.drop(columns=["_ds", "_dk1", "_dk2"], errors="ignore").reset_index(drop=True)

    # Apply updated prices
    if up_price:
        for idx, row in u.iterrows():
            nk = str(row.get("fpl_name", row.get("player_fbref", ""))).strip().lower()
            if nk in up_price and up_price[nk] > 0: u.at[idx, "price"] = up_price[nk]
            if nk in up_pts and up_pts[nk] > 0: u.at[idx, "fpl_points"] = up_pts[nk]

    # Injury flags
    if inj_map and "is_injured" not in u.columns:
        u["is_injured"], u["injury_info"], u["injury_return"] = False, "", ""
    if inj_map:
        for idx, row in u.iterrows():
            if row.get("is_injured"): continue
            nk = str(row.get("fpl_name", row.get("player_fbref", ""))).strip().lower()
            i = inj_map.get(nk)
            if i:
                u.at[idx, "is_injured"] = True
                u.at[idx, "injury_info"] = i.get("injury", "")
                u.at[idx, "injury_return"] = i.get("return_date", "")

    # Remove 0-point players
    b4 = len(u)
    u = u[u["fpl_points"] > 0].copy()
    print(f"  Removed {b4 - len(u)} players with 0 FPL points")

    # 60% minutes filter (with exemptions for quality / injured / high-ownership)
    MAX_MINS = 26 * 90
    b4 = len(u)
    keep = pd.Series(True, index=u.index)
    for idx, row in u.iterrows():
        if row.get("minutes", 0) / MAX_MINS > 0.60: continue
        ppg = row.get("points_per_game", 0)
        if ppg == 0 and row.get("fpl_points", 0) > 0 and row.get("matches", 1) > 0:
            ppg = row["fpl_points"] / max(1, row["matches"])
        if ppg >= 4.0: continue
        if row.get("is_injured") and row.get("injury_return", ""):
            try:
                if pd.to_datetime(row["injury_return"]) <= pd.Timestamp("2026-03-15"): continue
            except (ValueError, TypeError): pass
        if row.get("selected_pct", 0) >= 10: continue
        keep[idx] = False
    u = u[keep].copy().reset_index(drop=True)
    print(f"  Minutes filter (60%): {b4 - len(u)} players removed, {len(u)} remain")
    if unmatched:
        print(f"  Note: {len(unmatched)} fbref players had no FPL match (excluded)")
    print(f"  Injured players tracked: {int(u['is_injured'].sum()) if 'is_injured' in u.columns else 0}")
    return u


# ══════════════════════════════════════════════════════════════════════
# STEP 4: ENHANCED SCORING MODEL
# Full methodology documented in module docstring at top of file.
# ══════════════════════════════════════════════════════════════════════

def _build_home_away_profiles(fpl_data):
    """Build per-team home/away performance profiles from past results.

    Returns dict: team_name → {
        home_gf_per_match, home_ga_per_match, home_cs_pct, home_matches,
        away_gf_per_match, away_ga_per_match, away_cs_pct, away_matches,
        home_attack_mult, away_attack_mult  ← relative to league average
    }

    These replace the flat +10%/-8% home/away adjustment with team-specific
    multipliers derived from actual results. A team like Sunderland (strong
    at home, weak away) will get a much bigger home boost than Brighton.
    """
    import re
    past = fpl_data.get("past_results", pd.DataFrame())
    if past.empty or "Home Team" not in past.columns:
        return {}

    records = {}
    for _, row in past.iterrows():
        score = str(row.get("Score", ""))
        m = re.match(r"(\d+)\s*[–−-]\s*(\d+)", score)
        if not m:
            continue
        hg, ag = int(m.group(1)), int(m.group(2))
        ht = FIXTURE_TEAM_MAP.get(str(row["Home Team"]).strip(), str(row["Home Team"]).strip())
        at = FIXTURE_TEAM_MAP.get(str(row["Away Team"]).strip(), str(row["Away Team"]).strip())

        for team, is_home, gf, ga in [(ht, True, hg, ag), (at, False, ag, hg)]:
            if team not in records:
                records[team] = dict(h_gf=0, h_ga=0, h_cs=0, h_mp=0,
                                     a_gf=0, a_ga=0, a_cs=0, a_mp=0)
            p = "h" if is_home else "a"
            records[team][f"{p}_mp"] += 1
            records[team][f"{p}_gf"] += gf
            records[team][f"{p}_ga"] += ga
            if ga == 0:
                records[team][f"{p}_cs"] += 1

    # Calculate per-match rates and league averages
    profiles = {}
    all_home_gf, all_away_gf = [], []
    for team, r in records.items():
        hm = max(r["h_mp"], 1)
        am = max(r["a_mp"], 1)
        profiles[team] = {
            "home_gf_pm": r["h_gf"] / hm, "home_ga_pm": r["h_ga"] / hm,
            "home_cs_pct": r["h_cs"] / hm, "home_mp": r["h_mp"],
            "away_gf_pm": r["a_gf"] / am, "away_ga_pm": r["a_ga"] / am,
            "away_cs_pct": r["a_cs"] / am, "away_mp": r["a_mp"],
        }
        all_home_gf.append(profiles[team]["home_gf_pm"])
        all_away_gf.append(profiles[team]["away_gf_pm"])

    # Team-specific attack multipliers relative to league average
    avg_home_gf = np.mean(all_home_gf) if all_home_gf else 1.5
    avg_away_gf = np.mean(all_away_gf) if all_away_gf else 1.2
    for team in profiles:
        p = profiles[team]
        # How much better/worse this team attacks at home vs avg home team
        p["home_attack_mult"] = p["home_gf_pm"] / avg_home_gf if avg_home_gf > 0 else 1.0
        p["away_attack_mult"] = p["away_gf_pm"] / avg_away_gf if avg_away_gf > 0 else 1.0

    return profiles


def _calc_cs_prob(defending_team, opponent, is_home, profiles, gk_data):
    """Estimate clean sheet probability for a specific fixture.

    Combines: (a) team's own CS rate (home or away), (b) opponent's
    attacking output per match, (c) GK save percentage.
    Returns float 0.0–1.0.

    This is the key missing piece for DEF/GK scoring — a defender playing
    for Arsenal at home vs Wolves has a ~55% CS chance, while a West Ham
    defender at home has ~5%. The old model ignored this entirely.
    """
    dp = profiles.get(defending_team, {})
    op = profiles.get(opponent, {})

    # Base CS rate for this team at home or away
    if is_home:
        team_cs = dp.get("home_cs_pct", 0.25)
        opp_gf = op.get("away_gf_pm", 1.2)  # opponent's away scoring
    else:
        team_cs = dp.get("away_cs_pct", 0.20)
        opp_gf = op.get("home_gf_pm", 1.5)  # opponent's home scoring

    # League average goals per match ~ 1.35
    avg_gpm = 1.35
    # Opponent attack factor: >1 means more dangerous than average
    opp_factor = opp_gf / avg_gpm if avg_gpm > 0 else 1.0

    # Adjust team CS rate by opponent strength
    # If opponent scores 2x league average, CS becomes much less likely
    adjusted_cs = team_cs / max(0.5, opp_factor)

    # Blend with GK save% as a secondary signal
    gk_cs = gk_data.get(defending_team, {}).get("cs_pct", 0.25)
    # 70% team record, 30% GK baseline (avoid double counting)
    blended = adjusted_cs * 0.70 + gk_cs * 0.30

    return max(0.02, min(0.65, blended))  # clamp to reasonable range


def calculate_enhanced_scores(unified_df, fpl_data, gameweeks=5):
    """Score each player via the enhanced model with clean sheet probability,
    team-specific home/away multipliers, and improved DEF/GK scoring.
    Returns DataFrame sorted by total xPts."""
    results = []
    fixtures = _parse_fixtures(fpl_data, gameweeks)
    profiles = _build_home_away_profiles(fpl_data)

    # Build GK lookup for CS probability calc
    gk_data = {}
    tg = fpl_data.get("team_gk", pd.DataFrame())
    if not tg.empty and "Squad" in tg.columns:
        for _, r in tg.iterrows():
            gk_data[str(r["Squad"]).strip()] = {
                "cs_pct": safe_float(r.get("CS%", 25)) / 100,
                "ga90": safe_float(r.get("GA90", 1.3)),
                "save_pct": safe_float(r.get("Save%", 67)) / 100,
            }

    for _, p in unified_df.iterrows():
        player, team, pos, price = p["player_fbref"], p["team"], p["position"], p["price"]
        mins, mp = p["minutes"], p["matches"]
        n90 = p.get("nineties", mins / 90)
        starts = p.get("starts", mp)
        if mins < 180 or price <= 0: continue

        # Signal 1: historical ppg
        fpl_pts = p.get("fpl_points", 0)
        ppg = p.get("points_per_game", 0) or (fpl_pts / mp if mp > 0 else 0)

        # Signal 2: underlying per-90 quality → expected FPL pts per match
        ga90 = p.get("ga_per90", 0)
        g90, a90 = p.get("gls_per90", 0), p.get("ast_per90", 0)
        exp_goals = g90 * SCORING["goal"].get(pos, 4)
        exp_assists = a90 * SCORING["assist"]
        avg_mins = mins / mp if mp > 0 else 0
        exp_app = SCORING["appearance_60_plus"] if avg_mins >= 60 else (
            SCORING["appearance_60_plus"] * 0.6 + SCORING["appearance_1_59"] * 0.4 if avg_mins >= 30
            else SCORING["appearance_1_59"])
        cards = ((p["yellows"] * SCORING["yellow"] + p["reds"] * SCORING["red"]) / mp) if mp > 0 else 0
        bonus = SCORING["bonus_avg"].get(pos, 0.2) * (1.3 if ga90 > 0.5 else 1.15 if ga90 > 0.3 else 1.0)
        underlying = exp_goals + exp_assists + exp_app + cards + bonus

        # Blend: 40% actual + 60% underlying
        base = ppg * 0.40 + underlying * 0.60 if ppg > 0 else underlying

        # Signal 3: xG/xA regression
        xG, xA = p.get("fpl_xG", 0), p.get("fpl_xA", 0)
        ag, aa = p.get("goals", 0), p.get("assists", 0)
        xg_f = 1.0
        if xG > 0 and n90 > 5:
            r = ag / xG
            xg_f = 1.0 + min(0.20, (1.0 - r) * 0.3) if r < 0.8 else (1.0 - min(0.10, (r - 1.0) * 0.1) if r > 1.3 else 1.0)
        xa_f = 1.0
        if xA > 0 and n90 > 5:
            r = aa / xA
            xa_f = 1.0 + min(0.15, (1.0 - r) * 0.25) if r < 0.8 else (1.0 - min(0.08, (r - 1.0) * 0.08) if r > 1.3 else 1.0)

        # Signal 4: form momentum
        form = p.get("form", 0)
        ff = 1.0
        if ppg > 0 and form > 0:
            fr = form / ppg
            ff = 1.0 + min(0.15, (fr - 1.0) * 0.15) if fr > 1.3 else (1.0 - min(0.10, (1.0 - fr) * 0.10) if fr < 0.7 else 1.0)

        adj = base * xg_f * xa_f * ff

        # Signal 5: fixture difficulty per GW with team-specific home/away
        # and clean sheet probability for DEF/GK
        gw_exp = []
        tp = profiles.get(team, {})
        for gw in fixtures.get(team, [])[:gameweeks]:
            fdr = gw.get("fdr", 3)
            fdr_m = {1: 1.30, 5: 0.75}.get(fdr, 1.0 + (3 - fdr) * 0.15)
            is_home = gw.get("is_home", True)
            opponent = gw.get("opponent", "Average")

            # Team-specific home/away multiplier (replaces flat +10%/-8%)
            # Uses actual team attacking output at home vs away relative to league avg
            if is_home:
                home_m = 0.85 + tp.get("home_attack_mult", 1.0) * 0.25  # range ~0.9–1.5
            else:
                home_m = 0.80 + tp.get("away_attack_mult", 1.0) * 0.20  # range ~0.82–1.2

            play_p = min(1.0, starts / mp) if mp > 0 else 0.5
            base_gw = max(0, adj * fdr_m * home_m * play_p)

            # Clean sheet points for DEF and GK
            cs_pts = 0
            if pos in ("DEF", "GK"):
                cs_prob = _calc_cs_prob(team, opponent, is_home, profiles, gk_data)
                cs_reward = SCORING["clean_sheet"].get(pos, 0)  # 4 for DEF/GK
                cs_pts = cs_prob * cs_reward * play_p
                # Also estimate goals-conceded penalty for GK (−1 per 2 goals conceded)
                if pos == "GK":
                    opp_p = profiles.get(opponent, {})
                    opp_scoring = opp_p.get("home_gf_pm", 1.5) if not is_home else opp_p.get("away_gf_pm", 1.2)
                    gc_penalty = (opp_scoring / 2) * SCORING["gc_per_2"] * play_p  # negative
                    cs_pts += gc_penalty
                    # Save points estimate
                    gk_info = gk_data.get(team, {})
                    save_rate = gk_info.get("save_pct", 0.67)
                    shots_faced = opp_scoring * 3.5  # rough shots on target from goals
                    saves_est = shots_faced * save_rate
                    cs_pts += (saves_est / 3) * SCORING["saves_per_3"] * play_p

            gw_exp.append({"expected": round(base_gw + cs_pts, 2),
                           "cs_pts": round(cs_pts, 2),
                           "opponent": opponent, "fdr": fdr, "is_home": is_home})
        while len(gw_exp) < gameweeks:
            gw_exp.append({"expected": round(adj * 0.85, 2), "cs_pts": 0,
                           "opponent": "TBD", "fdr": 3, "is_home": True})

        total = sum(g["expected"] for g in gw_exp)
        val = total / price if price > 0 else 0
        sel = p.get("selected_pct", 50)
        dm = 1.20 if sel < 5 else 1.10 if sel < 10 else 1.05 if sel < 20 else 1.00

        res = {
            "player": player, "team": team, "position": pos, "price": price,
            "minutes": mins, "matches": mp, "starts": starts,
            "goals": int(ag), "assists": int(aa), "g_plus_a": int(p.get("g_plus_a", 0)),
            "gls_per90": g90, "ast_per90": a90, "ga_per90": ga90, "npg_per90": p.get("npg_per90", 0),
            "fpl_points": fpl_pts, "ppg": round(ppg, 2), "form": form,
            "selected_pct": sel, "bonus": p.get("bonus", 0),
            "fpl_xG": xG, "fpl_xA": xA,
            "xG_delta": round(ag - xG, 2), "xA_delta": round(aa - xA, 2),
            "xg_regression_factor": round(xg_f, 3),
            "underlying_per_match": round(underlying, 2),
            "base_expected_per_match": round(base, 2),
            "adjusted_expected_per_match": round(adj, 2),
            "total_expected_Ngw": round(total, 2),
            "avg_expected_per_gw": round(total / gameweeks, 2),
            "value_score": round(val, 2), "differential_mult": dm,
            "enhanced_value": round(val * dm, 2),
        }
        for i, g in enumerate(gw_exp):
            res[f"gw{i+1}_exp"] = g["expected"]
            res[f"gw{i+1}_opp"] = g["opponent"]
            res[f"gw{i+1}_fdr"] = g["fdr"]
            res[f"gw{i+1}_home"] = "H" if g["is_home"] else "A"
        results.append(res)

    return pd.DataFrame(results).sort_values("total_expected_Ngw", ascending=False).reset_index(drop=True)


def _parse_fixtures(fpl_data, gameweeks):
    """Per-team chronological fixtures. Skips matches ≤2 days from today (current GW)."""
    from datetime import date, timedelta
    schedule = {}
    today = date.today()
    cutoff = today + timedelta(days=2)
    upcoming = fpl_data.get("upcoming", pd.DataFrame())
    if not upcoming.empty and "Home Team" in upcoming.columns:
        upcoming = upcoming.dropna(subset=["Home Team", "Away Team"])
        if "Date" in upcoming.columns:
            upcoming["_date"] = pd.to_datetime(upcoming["Date"], errors="coerce")
            upcoming = upcoming.dropna(subset=["_date"]).sort_values("_date")
        skipped = upcoming[upcoming["_date"].dt.date <= cutoff]
        future = upcoming[upcoming["_date"].dt.date > cutoff].copy()
        if len(skipped) > 0:
            st = set()
            for _, r in skipped.iterrows():
                st.add(FIXTURE_TEAM_MAP.get(str(r["Home Team"]).strip(), str(r["Home Team"]).strip()))
                st.add(FIXTURE_TEAM_MAP.get(str(r["Away Team"]).strip(), str(r["Away Team"]).strip()))
            print(f"  Current GW: skipping {len(skipped)} match(es) for {', '.join(sorted(st))}")
        print(f"  Future fixtures: {len(future)} matches")
        for _, row in future.iterrows():
            h = FIXTURE_TEAM_MAP.get(str(row["Home Team"]).strip(), str(row["Home Team"]).strip())
            a = FIXTURE_TEAM_MAP.get(str(row["Away Team"]).strip(), str(row["Away Team"]).strip())
            schedule.setdefault(h, []).append({"opponent": a, "fdr": 3, "is_home": True, "date": str(row["_date"].date())})
            schedule.setdefault(a, []).append({"opponent": h, "fdr": 3, "is_home": False, "date": str(row["_date"].date())})
        for t in schedule:
            for i, f in enumerate(schedule[t]):
                f["gw"] = i + 1
    _enrich_fdr(schedule, fpl_data)
    return schedule


def _enrich_fdr(schedule, fpl_data):
    """Replace default FDR with data-driven ratings from team attack/defence stats."""
    ta = fpl_data.get("team_attack", pd.DataFrame())
    td = fpl_data.get("team_defense", pd.DataFrame())
    tg = fpl_data.get("team_gk", pd.DataFrame())
    if ta.empty and td.empty: return
    st = {}
    if not ta.empty and "Squad" in ta.columns:
        for _, r in ta.iterrows():
            n = safe_float(r.get("90s", 1))
            st.setdefault(str(r["Squad"]).strip(), {})["gp90"] = safe_float(r.get("Gls", 0)) / n if n > 0 else 0
    if not td.empty and "Squad" in td.columns:
        for _, r in td.iterrows():
            n = safe_float(r.get("90s", 1))
            st.setdefault(str(r["Squad"]).strip().replace("vs ", ""), {})["gap90"] = safe_float(r.get("Gls", 0)) / n if n > 0 else 0
    if not tg.empty and "Squad" in tg.columns:
        for _, r in tg.iterrows():
            s = st.setdefault(str(r["Squad"]).strip(), {})
            s["cs_pct"] = safe_float(r.get("CS%", 0)) / 100
    if not st: return
    avg_g = np.mean([s.get("gp90", 1.4) for s in st.values()]) or 1.4
    avg_ga = np.mean([s.get("gap90", 1.4) for s in st.values()]) or 1.4
    for team, fixes in schedule.items():
        for f in fixes:
            os_ = st.get(f["opponent"], {})
            if os_:
                oa = os_.get("gp90", avg_g) / avg_g
                od = os_.get("gap90", avg_ga) / avg_ga
                df = max(1, min(5, round((2.0 - od + oa) / 2 * 3)))
                f["fdr"] = round(f["fdr"] * 0.4 + df * 0.6)


# ══════════════════════════════════════════════════════════════════════
# STEP 5: SQUAD OPTIMISATION
# ══════════════════════════════════════════════════════════════════════

def optimize_squad(predictions_df, budget=103.0):
    """Two-tier: XI maximises xPts, bench optimises reliability + fixture ease."""
    df = predictions_df[predictions_df["price"] > 0].copy().reset_index(drop=True)
    df["opt_score"] = df["total_expected_Ngw"]
    df["mins_reliability"] = (df["starts"] / df["matches"].clip(lower=1)).clip(0, 1)
    fdr_cols = [c for c in df.columns if c.endswith("_fdr")]
    df["fixture_ease"] = ((5 - df[fdr_cols].mean(axis=1).fillna(3)) / 4).clip(0, 1) if fdr_cols else 0.5
    mx = df["opt_score"].max() or 1
    df["bench_score"] = ((df["opt_score"] / mx) * 0.30 + df["mins_reliability"] * 0.35 + df["fixture_ease"] * 0.35) * mx
    print(f"\n  Optimizing from {len(df)} players | Budget: £{budget}m")
    print(f"  Strategy: Two-tier — XI maximises expected points, bench optimises reliability + fixtures")
    return _lp_optimize(df, budget) if HAS_PULP else _greedy_optimize(df, budget)


def _lp_optimize(df, budget):
    """Two-tier LP: squad(x) + XI(s) binary variables, s[i] ≤ x[i]."""
    try:
        from pulp import PULP_CBC_CMD
        prob = LpProblem("FPL", LpMaximize)
        P = df.index.tolist()
        x = LpVariable.dicts("sq", P, cat="Binary")
        s = LpVariable.dicts("xi", P, cat="Binary")
        for i in P: prob += s[i] <= x[i]
        prob += lpSum(x[i] for i in P) == 15
        prob += lpSum(s[i] for i in P) == 11
        pi = {p: df[df["position"] == p].index.tolist() for p in ["GK", "DEF", "MID", "FWD"]}
        prob += lpSum(x[i] for i in pi["GK"]) == 2
        prob += lpSum(x[i] for i in pi["DEF"]) == 5
        prob += lpSum(x[i] for i in pi["MID"]) == 5
        prob += lpSum(x[i] for i in pi["FWD"]) == 3
        prob += lpSum(s[i] for i in pi["GK"]) == 1
        for p, lo, hi in [("DEF", 3, 5), ("MID", 2, 5), ("FWD", 1, 3)]:
            prob += lpSum(s[i] for i in pi[p]) >= lo
            prob += lpSum(s[i] for i in pi[p]) <= hi
        prob += lpSum(df.loc[i, "price"] * x[i] for i in P) <= budget
        for t in df["team"].unique():
            prob += lpSum(x[i] for i in df[df["team"] == t].index.tolist()) <= 3
        prob += lpSum(s[i] * df.loc[i, "opt_score"] + (x[i] - s[i]) * df.loc[i, "bench_score"] * 0.10 for i in P)
        prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[prob.status] != "Optimal":
            return _greedy_optimize(df, budget)
        si = [i for i in P if value(x[i]) == 1]
        xi = [i for i in P if value(s[i]) == 1]
        return _finalize(df.loc[si].copy(), df.loc[xi].copy(),
                         df.loc[[i for i in si if i not in xi]].copy(),
                         df.loc[si, "price"].sum(), budget)
    except Exception as e:
        print(f"  Solver error: {e}")
        return _greedy_optimize(df, budget)


def _greedy_optimize(df, budget):
    """Greedy fallback when PuLP unavailable."""
    df = df.copy()
    df["vr"] = df["opt_score"] / df["price"].clip(lower=0.1)
    xi, tc, cost, used = [], {}, 0, set()
    xp = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    xmin = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    xmax = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        for idx, p in df[df["position"] == pos].sort_values("vr", ascending=False).iterrows():
            if xp[pos] >= xmin[pos]: break
            if tc.get(p["team"], 0) >= 3 or cost + p["price"] > budget - 16: continue
            xi.append(p); used.add(idx); xp[pos] += 1
            tc[p["team"]] = tc.get(p["team"], 0) + 1; cost += p["price"]
    for idx, p in df[~df.index.isin(used)][df["position"] != "GK"].sort_values("vr", ascending=False).iterrows():
        if len(xi) >= 11: break
        pos = p["position"]
        if xp.get(pos, 0) >= xmax.get(pos, 0) or tc.get(p["team"], 0) >= 3: continue
        if cost + p["price"] > budget - 16: continue
        xi.append(p); used.add(idx); xp[pos] = xp.get(pos, 0) + 1
        tc[p["team"]] = tc.get(p["team"], 0) + 1; cost += p["price"]
    needed = {pos: SQUAD_RULES["positions"][pos] - xp.get(pos, 0) for pos in SQUAD_RULES["positions"]}
    bench = []
    bp = df[~df.index.isin(used)].copy()
    bp["bv"] = bp["bench_score"] / bp["price"].clip(lower=0.1)
    bp = bp.sort_values("bv", ascending=False)
    for pos, n in needed.items():
        for idx, p in bp[bp["position"] == pos].iterrows():
            if n <= 0: break
            if tc.get(p["team"], 0) >= 3 or cost + p["price"] > budget: continue
            bench.append(p); used.add(idx); tc[p["team"]] = tc.get(p["team"], 0) + 1
            cost += p["price"]; n -= 1
    bp = bp[~bp.index.isin(used)]
    for idx, p in bp.iterrows():
        if len(bench) + len(xi) >= 15: break
        pos = p["position"]
        if sum(1 for q in xi + bench if q["position"] == pos) >= SQUAD_RULES["positions"].get(pos, 0): continue
        if tc.get(p["team"], 0) >= 3 or cost + p["price"] > budget: continue
        bench.append(p); tc[p["team"]] = tc.get(p["team"], 0) + 1; cost += p["price"]
    xi_df, bench_df = pd.DataFrame(xi), pd.DataFrame(bench)
    return _finalize(pd.concat([xi_df, bench_df], ignore_index=True), xi_df, bench_df, cost, budget)


def _finalize(squad, xi, bench, total_cost, budget):
    """Pick captain (highest GW1 xPts in XI) and build result dict."""
    ci, cn, cg = None, "", 0
    vi, vn, vg = None, "", 0
    if not xi.empty and "gw1_exp" in xi.columns:
        xs = xi.sort_values("gw1_exp", ascending=False)
        if len(xs) >= 1: ci, cn, cg = xs.index[0], xs.iloc[0].get("player", "?"), xs.iloc[0].get("gw1_exp", 0)
        if len(xs) >= 2: vi, vn, vg = xs.index[1], xs.iloc[1].get("player", "?"), xs.iloc[1].get("gw1_exp", 0)
    gw1_total = (xi["gw1_exp"].sum() if "gw1_exp" in xi.columns else 0) + cg
    print(f"  Captain: {cn} ({cg:.1f} pts × 2 = {cg*2:.1f} pts)")
    if vn: print(f"  Vice-Captain: {vn} ({vg:.1f} pts)")
    return {
        "squad": squad, "starting_xi": xi, "bench": bench,
        "total_cost": round(total_cost, 1),
        "expected_points": round(squad["opt_score"].sum(), 1) if not squad.empty else 0,
        "xi_expected": round(xi["opt_score"].sum(), 1) if not xi.empty else 0,
        "budget_remaining": round(budget - total_cost, 1),
        "captain_idx": ci, "captain_name": cn, "captain_gw1": cg,
        "vc_idx": vi, "vc_name": vn,
        "xi_gw1_with_captain": round(gw1_total, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 6: CONSOLE REPORTING
# ══════════════════════════════════════════════════════════════════════

def print_value_picks(df):
    print(f"\n{'='*80}\n  TOP VALUE PICKS (best underlying stats per £)\n{'='*80}")
    d = df[df["minutes"] >= 450].copy()
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pd_ = d[d["position"] == pos].sort_values("enhanced_value", ascending=False).head(8)
        print(f"\n  {'─'*60}\n  {pos} — Top Value Picks\n  {'─'*60}")
        for _, p in pd_.iterrows():
            xf = " ⬆ xG UNDERPERFORMER" if p.get("xG_delta", 0) < -1 else " ⬇ xG overperformer" if p.get("xG_delta", 0) > 2 else ""
            df_ = f" [DIFF {p['selected_pct']}%]" if p.get("selected_pct", 50) < 10 else ""
            fm = f"form:{p['form']}" if p.get("form", 0) > 0 else ""
            print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
                  f"exp:{p['total_expected_Ngw']:5.1f}pts | val:{p['enhanced_value']:4.2f} | "
                  f"G+A/90:{p['ga_per90']:.2f} | {fm}{xf}{df_}")


def print_regression_candidates(df):
    print(f"\n{'='*80}\n  xG REGRESSION CANDIDATES (due a points surge)\n{'='*80}")
    d = df[(df["minutes"] >= 900) & (df["fpl_xG"] > 0)].copy()
    d["xGI_delta"] = d["xG_delta"] + d["xA_delta"]
    for _, p in d[d["xGI_delta"] < -1].sort_values("xGI_delta").head(10).iterrows():
        print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
              f"Goals:{p['goals']} vs xG:{p['fpl_xG']:.1f} (delta:{p['xG_delta']:+.1f}) | "
              f"Assists:{p['assists']} vs xA:{p['fpl_xA']:.1f} (delta:{p['xA_delta']:+.1f}) | "
              f"Owned:{p['selected_pct']:.1f}%")


def print_differentials(df):
    print(f"\n{'='*80}\n  DIFFERENTIAL PICKS (low ownership, strong underlying)\n{'='*80}")
    d = df[(df["minutes"] >= 900) & (df["selected_pct"] > 0) & (df["selected_pct"] < 15)]
    for _, p in d.sort_values("enhanced_value", ascending=False).head(15).iterrows():
        print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
              f"Owned:{p['selected_pct']:5.1f}% | exp:{p['total_expected_Ngw']:5.1f}pts | "
              f"G+A/90:{p['ga_per90']:.2f} | form:{p['form']}")


def print_squad(result, gameweeks):
    """Console pitch-style formation display with captain tags."""
    xi, bench = result["starting_xi"].copy(), result["bench"].copy()
    W = 90
    po = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi["_po"] = xi["position"].map(po)
    xi = xi.sort_values(["_po", "total_expected_Ngw"], ascending=[True, False])
    nd, nm, nf = len(xi[xi["position"]=="DEF"]), len(xi[xi["position"]=="MID"]), len(xi[xi["position"]=="FWD"])
    fm = f"{nd}-{nm}-{nf}"
    ci, vi = result.get("captain_idx"), result.get("vc_idx")
    g1 = result.get("xi_gw1_with_captain", 0)

    def _cl(t):
        pad = W - 4 - len(t); l = max(0, pad // 2)
        print(f"│ {' '*l}{t}{' '*max(0,pad-l)} │")

    print("\n┌" + "─"*(W-2) + "┐")
    for t in [f"OPTIMAL SQUAD  |  {fm}", f"£{result['total_cost']}m / £103.0m  (£{result['budget_remaining']}m ITB)",
              f"GW1 Predicted: {g1:.1f} pts (incl. captain)",
              f"Captain: {result.get('captain_name','')} ({result.get('captain_gw1',0):.1f} x2 = {result.get('captain_gw1',0)*2:.1f} pts)",
              f"Next {gameweeks} GWs Total: {result['xi_expected']:.1f} pts"]:
        _cl(t)
    print("├" + "─"*(W-2) + "┤")

    def _card(p, cw, idx=None):
        name = p.get("player", "?"); price = f"£{p.get('price',0):.1f}m"
        tag = "(C)" if idx == ci else "(VC)" if idx == vi else ""
        parts = name.split(); short = parts[-1] if len(parts) >= 2 else name
        if len(short) <= 3 and len(parts) >= 2: short = parts[0][0] + ". " + short
        mx = cw - len(price) - 2
        if tag:
            nm = mx - len(tag) - 1
            short = (short[:nm-1]+"." if len(short)>nm else short) + " " + tag
        elif len(short) > mx:
            short = short[:mx-1] + "."
        opp = str(p.get("gw1_opp","TBD")); ha = str(p.get("gw1_home",""))
        if opp in ("nan",""): opp = "TBD"
        ms = f"{opp} ({ha})" if ha and ha != "nan" else opp
        g = p.get("gw1_exp", 0)
        pts = f"{g*2:.1f} pts (x2)" if idx == ci else f"{g:.1f} pts"
        return (f"{short} {price}".center(cw), ms.center(cw), pts.center(cw))

    for pc in ["GK","DEF","MID","FWD"]:
        pp = xi[xi["position"]==pc]
        if pp.empty: continue
        n = len(pp); cw = min(18, (W-4)//n - 1); tw = n*cw+(n-1); pl = (W-2-tw)//2
        cards = [_card(p, cw, idx=idx) for idx, p in pp.iterrows()]
        for ri in range(3):
            ln = " "*pl + " ".join(cards[i][ri] for i in range(n))
            print(f"│{ln[:W-2].ljust(W-2)}│")
        print(f"│{' '*(W-2)}│")

    print("├" + "─"*(W-2) + "┤"); _cl("BENCH")
    if "_po" not in bench.columns: bench["_po"] = bench["position"].map(po)
    bench = bench.sort_values(["_po","total_expected_Ngw"], ascending=[True,False])
    nb = len(bench)
    if nb > 0:
        cw = min(18, (W-4)//nb - 1); tw = nb*cw+(nb-1); pl = (W-2-tw)//2
        cards = [_card(p, cw) for _, p in bench.iterrows()]
        for ri in range(3):
            ln = " "*pl + " ".join(cards[i][ri] for i in range(nb))
            print(f"│{ln[:W-2].ljust(W-2)}│")
    print("└" + "─"*(W-2) + "┘")


# ══════════════════════════════════════════════════════════════════════
# STEP 7: HTML PITCH VISUAL
# ══════════════════════════════════════════════════════════════════════

def generate_html_lineup(result, predictions_df, gameweeks):
    """Full HTML page: pitch with team shirts, captain badge, FDR colours, bench, mentions."""
    xi, bench, squad = result["starting_xi"].copy(), result["bench"].copy(), result["squad"].copy()
    po = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi["_po"] = xi["position"].map(po)
    xi = xi.sort_values(["_po", "total_expected_Ngw"], ascending=[True, False])
    nd, nm, nf = len(xi[xi["position"]=="DEF"]), len(xi[xi["position"]=="MID"]), len(xi[xi["position"]=="FWD"])
    fm = f"{nd}-{nm}-{nf}"
    ci, vi = result.get("captain_idx"), result.get("vc_idx")
    cn, cg = result.get("captain_name", ""), result.get("captain_gw1", 0)
    g1 = result.get("xi_gw1_with_captain", 0)

    mentions = predictions_df[~predictions_df["player"].str.lower().isin(set(squad["player"].str.lower()))].sort_values("total_expected_Ngw", ascending=False).head(7)

    def _svg(team, sz=50, cap=False):
        bg, tx = TEAM_COLORS.get(team, ("#666","#FFF"))
        st = "#333" if bg.upper() in ("#FFFFFF","#FFF","#FDB913") else bg
        cb = (f'<circle cx="48" cy="10" r="9" fill="#FFD700" stroke="#000" stroke-width="1"/>'
              f'<text x="48" y="14" text-anchor="middle" font-size="11" font-weight="bold" fill="#000">C</text>') if cap else ""
        return (f'<svg viewBox="0 0 60 55" width="{sz}" height="{sz}"><path d="M15,5 L5,15 L12,20 L12,50 L48,50 L48,20 L55,15 L45,5 L38,10 Q30,15 22,10 Z" fill="{bg}" stroke="{st}" stroke-width="2"/>{cb}</svg>')

    def _ph(p, rel=False, idx=None):
        nm = p.get("player","?"); parts = nm.split()
        short = parts[-1] if len(parts) >= 2 else nm
        if len(short) <= 3 and len(parts) >= 2: short = parts[0][0] + ". " + short
        opp = str(p.get("gw1_opp","TBD")); ha = str(p.get("gw1_home",""))
        if opp in ("nan",""): opp = "TBD"
        fix = f"{opp} ({ha})" if ha and ha != "nan" else opp
        g1e = p.get("gw1_exp", 0)
        fdr = f"fdr-{min(5, max(1, int(p.get('gw1_fdr', 3))))}"
        ic = idx is not None and idx == ci
        iv = idx is not None and idx == vi
        ct = ' <span style="color:#FFD700;font-weight:bold">(C)</span>' if ic else (' <span style="color:#C0C0C0;font-weight:bold">(VC)</span>' if iv else "")
        pts = f"{g1e*2:.1f} pts (x2)" if ic else f"{g1e:.1f} pts"
        ex = f'<div class="reliability">Mins: {p.get("mins_reliability",0):.0%}</div>' if rel else ""
        return (f'<div class="player-card">{_svg(p.get("team","?"), cap=ic)}'
                f'<div class="player-name">{short}{ct}</div><div class="player-price">£{p.get("price",0):.1f}m</div>'
                f'<div class="fixture {fdr}">{fix}</div><div class="xpts">{pts}</div>{ex}</div>')

    gk = "".join(_ph(p, idx=i) for i, p in xi[xi["position"]=="GK"].iterrows())
    df_ = "".join(_ph(p, idx=i) for i, p in xi[xi["position"]=="DEF"].iterrows())
    mf = "".join(_ph(p, idx=i) for i, p in xi[xi["position"]=="MID"].iterrows())
    fw = "".join(_ph(p, idx=i) for i, p in xi[xi["position"]=="FWD"].iterrows())
    if "_po" not in bench.columns: bench["_po"] = bench["position"].map(po)
    bench = bench.sort_values(["_po","total_expected_Ngw"], ascending=[True,False])
    bh = "".join(_ph(p, rel=True) for _, p in bench.iterrows())

    mr = ""
    for _, m in mentions.iterrows():
        opp = str(m.get("gw1_opp","TBD")); ha = str(m.get("gw1_home",""))
        if opp in ("nan",""): opp = "TBD"
        fs = f"{opp} ({ha})" if ha and ha != "nan" else opp
        bg, tx = TEAM_COLORS.get(m["team"], ("#666","#FFF"))
        mr += (f'<tr><td><span class="color-dot" style="background:{bg};border:1px solid {tx}"></span>{m["player"]}</td>'
               f'<td>{m["team"]}</td><td>{m["position"]}</td><td>£{m["price"]:.1f}m</td><td>{fs}</td>'
               f'<td><b>{m.get("gw1_exp",0):.1f}</b></td><td>{m["total_expected_Ngw"]:.1f}</td>'
               f'<td>{m.get("selected_pct",0):.1f}%</td></tr>')

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>FPL Optimal Squad</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:#1a1a2e;color:#fff}}
.container{{max-width:800px;margin:0 auto;padding:20px}}
.header{{text-align:center;padding:20px 0 10px}}
.header h1{{font-size:22px;color:#00ff87;letter-spacing:1px}}
.header .formation{{font-size:16px;color:#ccc;margin-top:4px}}
.header .predicted{{font-size:28px;font-weight:700;color:#fff;margin:10px 0 4px}}
.header .predicted span{{color:#00ff87}}
.header .budget{{font-size:13px;color:#888}}
.pitch{{background:linear-gradient(180deg,#2d8a4e 0%,#34a058 8%,#2d8a4e 8%,#2d8a4e 16%,#34a058 16%,#34a058 24%,#2d8a4e 24%,#2d8a4e 32%,#34a058 32%,#34a058 40%,#2d8a4e 40%,#2d8a4e 48%,#34a058 48%,#34a058 56%,#2d8a4e 56%,#2d8a4e 64%,#34a058 64%,#34a058 72%,#2d8a4e 72%,#2d8a4e 80%,#34a058 80%,#34a058 88%,#2d8a4e 88%,#2d8a4e 96%,#34a058 96%);border-radius:12px;padding:25px 10px 15px;position:relative;border:2px solid #fff3}}
.pitch::before{{content:'';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:100px;height:100px;border:2px solid #fff3;border-radius:50%}}
.position-row{{display:flex;justify-content:center;gap:8px;margin-bottom:12px}}
.player-card{{text-align:center;width:90px;flex-shrink:0}}
.player-card svg{{display:block;margin:0 auto 2px;filter:drop-shadow(0 2px 3px rgba(0,0,0,0.4))}}
.player-name{{font-size:11px;font-weight:700;color:#fff;background:#1a1a2e;border-radius:4px;padding:2px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.player-price{{font-size:10px;color:#aaa;margin-top:1px}}
.fixture{{font-size:10px;font-weight:600;margin-top:3px;padding:2px 6px;border-radius:3px;display:inline-block}}
.fdr-1{{background:#375523;color:#00ff87}}.fdr-2{{background:#01fc7a33;color:#00ff87}}
.fdr-3{{background:#e7e7e733;color:#eee}}.fdr-4{{background:#ff474733;color:#ff6b6b}}
.fdr-5{{background:#8b000033;color:#ff4747}}
.xpts{{font-size:13px;font-weight:700;color:#00ff87;margin-top:2px}}
.reliability{{font-size:9px;color:#888;margin-top:1px}}
.bench-section{{background:#16213e;border-radius:10px;padding:15px 10px;margin-top:12px;text-align:center}}
.bench-section h3{{font-size:14px;color:#888;margin-bottom:10px;letter-spacing:2px}}
.mentions{{background:#16213e;border-radius:10px;padding:15px 18px;margin-top:12px}}
.mentions h3{{font-size:14px;color:#00ff87;margin-bottom:10px}}
.mentions table{{width:100%;border-collapse:collapse;font-size:12px}}
.mentions th{{text-align:left;color:#888;padding:4px 6px;border-bottom:1px solid #333}}
.mentions td{{padding:5px 6px;border-bottom:1px solid #222}}
.mentions tr:hover{{background:#1a1a3e}}
.color-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}}
</style></head><body>
<div class="container">
<div class="header"><h1>OPTIMAL SQUAD</h1>
<div class="formation">Formation: {fm} &nbsp;|&nbsp; £{result['total_cost']}m / £103.0m &nbsp;(£{result['budget_remaining']}m ITB)</div>
<div class="predicted">Next GW: <span>{g1:.1f} pts</span> (incl. captain)</div>
<div class="budget">Captain: {cn} ({cg:.1f} x2 = {cg*2:.1f} pts)</div>
<div class="budget">Next {gameweeks} GWs projected: {result['xi_expected']:.1f} pts</div></div>
<div class="pitch"><div class="position-row">{gk}</div><div class="position-row">{df_}</div>
<div class="position-row">{mf}</div><div class="position-row">{fw}</div></div>
<div class="bench-section"><h3>BENCH</h3><div class="position-row">{bh}</div></div>
<div class="mentions"><h3>HONOURABLE MENTIONS</h3>
<table><thead><tr><th>Player</th><th>Team</th><th>Pos</th><th>Price</th>
<th>Next</th><th>GW xPts</th><th>{gameweeks}GW xPts</th><th>Owned</th></tr></thead>
<tbody>{mr}</tbody></table></div></div></body></html>"""


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(budget=103.0, gameweeks=5, min_minutes=180):
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
# TRANSFER RECOMMENDER
# ══════════════════════════════════════════════════════════════════════
# Reads your current squad from a CSV, then finds the best transfers
# using the same scoring model. Respects budget, free transfers, and
# the -4 hit penalty. Shows net xPts gain per transfer.
#
# CSV format (my_team.csv):
#   player, position, team, price
#   Haaland, FWD, Man City, 14.9
#   Saka, MID, Arsenal, 10.5
#   ...
#
# Also reads a my_team_settings.csv (optional):
#   setting, value
#   bank, 1.3          ← money in the bank (£m)
#   free_transfers, 1  ← available free transfers
#   gameweeks, 5       ← lookahead for evaluation

MY_TEAM_DIR = os.path.join(BASE_DIR, "data", "My Team")

def load_my_team():
    """Load current squad and settings from My Team folder."""
    team_path = os.path.join(MY_TEAM_DIR, "my_team.csv")
    settings_path = os.path.join(MY_TEAM_DIR, "my_team_settings.csv")

    if not os.path.exists(team_path):
        print(f"\n  ⚠ No team file found at: {team_path}")
        print(f"    Create a CSV with columns: player, position, team, price")
        return None, {}

    team_df = pd.read_csv(team_path)
    # Normalise column names
    team_df.columns = [c.strip().lower() for c in team_df.columns]
    for col in ["player", "position", "team"]:
        if col in team_df.columns:
            team_df[col] = team_df[col].str.strip()
    if "price" in team_df.columns:
        team_df["price"] = pd.to_numeric(team_df["price"], errors="coerce")

    print(f"  Loaded your squad: {len(team_df)} players")

    # Load settings
    settings = {"bank": 0.0, "free_transfers": 1, "gameweeks": 5}
    if os.path.exists(settings_path):
        sdf = pd.read_csv(settings_path)
        sdf.columns = [c.strip().lower() for c in sdf.columns]
        for _, row in sdf.iterrows():
            k = str(row.get("setting", "")).strip().lower()
            v = row.get("value", "")
            if k in settings:
                settings[k] = float(v) if k == "bank" else int(float(v))
        print(f"  Settings: £{settings['bank']}m ITB, {settings['free_transfers']} free transfer(s)")
    else:
        print(f"  No settings file found — using defaults (£0m ITB, 1 FT)")

    return team_df, settings


def match_my_team_to_predictions(my_team, predictions):
    """Match user's squad names to prediction database using multi-pass matching.
    Handles FPL short names (B.Fernandes), nicknames (Kroupi.Jr), accents, etc."""
    matched = []
    pred_lower = predictions.copy()
    pred_lower["_name_lower"] = pred_lower["player"].str.lower().str.strip()
    pred_lower["_fbref_lower"] = pred_lower.get("player_fbref", pred_lower["player"]).str.lower().str.strip()
    # Clean versions: strip dots, hyphens, accents for fuzzy matching
    import re, unicodedata
    def _clean(s):
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # strip accents
        return re.sub(r"[.\-'\s]+", " ", s).strip().lower()
    pred_lower["_clean"] = pred_lower["player"].apply(lambda x: _clean(str(x)))
    pred_lower["_clean_fbref"] = pred_lower.get("player_fbref", pred_lower["player"]).apply(lambda x: _clean(str(x)))

    for _, row in my_team.iterrows():
        name = str(row["player"]).strip().lower()
        name_clean = _clean(str(row["player"]))
        pos = str(row.get("position", "")).strip().upper()
        team = str(row.get("team", "")).strip()
        sell_price = row.get("price", 0)

        m = None
        def _pick(candidates):
            """From candidates df, pick best match (position+team filtered)."""
            if len(candidates) == 0: return None
            if len(candidates) == 1: return candidates.iloc[0]
            if pos:
                sub = candidates[candidates["position"] == pos]
                if len(sub) > 0: candidates = sub
            if team:
                sub = candidates[candidates["team"].str.lower().str.contains(team.lower(), na=False)]
                if len(sub) > 0: candidates = sub
            return candidates.iloc[0]

        # Pass 1: Exact match on player display name
        m = _pick(pred_lower[pred_lower["_name_lower"] == name])

        # Pass 2: Exact match on fbref name
        if m is None:
            m = _pick(pred_lower[pred_lower["_fbref_lower"] == name])

        # Pass 3: Cleaned name match (strips dots, accents, hyphens)
        if m is None:
            m = _pick(pred_lower[pred_lower["_clean"] == name_clean])
        if m is None:
            m = _pick(pred_lower[pred_lower["_clean_fbref"] == name_clean])

        # Pass 4: Last name contained in prediction name (+ team/pos filter)
        if m is None:
            # Extract last meaningful part: "B.Fernandes" → "fernandes", "Kroupi.Jr" → "kroupi"
            parts = re.sub(r"[.\-]+", " ", name).split()
            # Remove common suffixes like "jr", "sr", "ii", "iii"
            meaningful = [p for p in parts if p not in ("jr", "sr", "ii", "iii", "b", "j", "m", "d", "a")]
            last = meaningful[-1] if meaningful else parts[-1] if parts else ""
            if len(last) >= 3:  # avoid matching single letters
                candidates = pred_lower[pred_lower["_name_lower"].str.contains(last, na=False)]
                m = _pick(candidates)
            # Also try on fbref names
            if m is None and len(last) >= 3:
                candidates = pred_lower[pred_lower["_fbref_lower"].str.contains(last, na=False)]
                m = _pick(candidates)

        # Pass 5: First name initial + last name (e.g. "B.Fernandes" → first_initial=B, last=Fernandes)
        if m is None and "." in str(row["player"]):
            parts = str(row["player"]).replace(".", " ").split()
            if len(parts) >= 2 and len(parts[0]) == 1:
                initial = parts[0].lower()
                surname = parts[-1].lower()
                candidates = pred_lower[
                    (pred_lower["_name_lower"].str.contains(surname, na=False)) &
                    (pred_lower["_name_lower"].str.startswith(initial) |
                     pred_lower["_name_lower"].str.contains(f" {initial}", na=False) |
                     pred_lower["_fbref_lower"].str.startswith(initial))
                ]
                m = _pick(candidates)

        if m is not None:
            matched.append({
                "player": m["player"],
                "position": m["position"],
                "team": m["team"],
                "sell_price": sell_price,
                "current_price": m["price"],
                "total_expected_Ngw": m["total_expected_Ngw"],
                "gw1_exp": m.get("gw1_exp", 0),
                "value_score": m.get("value_score", 0),
                "form": m.get("form", 0),
                "pred_idx": m.name,
            })
        else:
            print(f"    ⚠ Could not match: {row['player']} ({pos}, {team})")
            matched.append({
                "player": row["player"], "position": pos, "team": team,
                "sell_price": sell_price, "current_price": sell_price,
                "total_expected_Ngw": 0, "gw1_exp": 0, "value_score": 0,
                "form": 0, "pred_idx": -1,
            })

    return pd.DataFrame(matched)


def recommend_transfers(my_squad_matched, predictions, settings):
    """
    Find the best transfers by comparing every possible out→in swap.

    For each player in the squad, finds the best replacement at the same position
    that: (a) isn't already in the squad, (b) respects the 3-per-team rule,
    (c) fits within budget (sell price + bank).

    Ranks by net xPts gained over the gameweek lookahead.
    A -4 hit is applied to transfers beyond the free allocation.
    """
    bank = settings.get("bank", 0.0)
    free_transfers = settings.get("free_transfers", 1)
    hit_penalty = 4  # FPL hit cost

    squad_names = set(my_squad_matched["player"].str.lower())
    # Count players per team in current squad
    team_counts = my_squad_matched["team"].value_counts().to_dict()

    # Available players = all predictions not in my squad
    available = predictions[~predictions["player"].str.lower().isin(squad_names)].copy()

    all_swaps = []

    for _, out_player in my_squad_matched.iterrows():
        pos = out_player["position"]
        out_name = out_player["player"]
        out_xpts = out_player["total_expected_Ngw"]
        out_sell = out_player["sell_price"]
        out_team = out_player["team"]

        # Budget available if we sell this player
        transfer_budget = out_sell + bank

        # Team count if we remove this player
        adj_team_counts = team_counts.copy()
        adj_team_counts[out_team] = adj_team_counts.get(out_team, 1) - 1

        # Find same-position replacements within budget
        candidates = available[
            (available["position"] == pos) &
            (available["price"] <= transfer_budget)
        ].copy()

        # Enforce 3-per-team rule
        valid_idx = []
        for idx, cand in candidates.iterrows():
            cand_team = cand["team"]
            if adj_team_counts.get(cand_team, 0) < 3:
                valid_idx.append(idx)
        candidates = candidates.loc[valid_idx]

        if candidates.empty:
            continue

        # Calculate net gain for each candidate
        for idx, inp in candidates.iterrows():
            net_xpts = inp["total_expected_Ngw"] - out_xpts
            cost_change = inp["price"] - out_sell  # negative = saves money
            all_swaps.append({
                "out_player": out_name,
                "out_team": out_team,
                "out_xpts": round(out_xpts, 1),
                "out_price": out_sell,
                "in_player": inp["player"],
                "in_team": inp["team"],
                "in_xpts": round(inp["total_expected_Ngw"], 1),
                "in_price": inp["price"],
                "in_gw1": round(inp.get("gw1_exp", 0), 1),
                "in_form": round(inp.get("form", 0), 1),
                "position": pos,
                "net_xpts_gain": round(net_xpts, 1),
                "cost_change": round(cost_change, 1),
                "new_bank": round(bank - cost_change, 1),
            })

    if not all_swaps:
        print("  No valid transfers found!")
        return pd.DataFrame()

    swaps_df = pd.DataFrame(all_swaps).sort_values("net_xpts_gain", ascending=False)

    # ── Print best single transfers ──
    print(f"\n{'='*80}")
    print(f"  TRANSFER RECOMMENDATIONS (next {settings.get('gameweeks', 5)} GWs)")
    print(f"  Bank: £{bank}m | Free transfers: {free_transfers}")
    print(f"{'='*80}")

    # Best transfer per position
    seen_out = set()
    top_transfers = []
    for _, swap in swaps_df.iterrows():
        if swap["out_player"] in seen_out:
            continue
        seen_out.add(swap["out_player"])
        top_transfers.append(swap)
        if len(top_transfers) >= 15:
            break

    print(f"\n  {'─'*76}")
    print(f"  TOP SINGLE TRANSFERS (ranked by xPts gained over {settings.get('gameweeks', 5)} GWs)")
    print(f"  {'─'*76}")

    for i, t in enumerate(top_transfers):
        hit_str = ""
        if i >= free_transfers:
            adjusted = t["net_xpts_gain"] - hit_penalty
            hit_str = f" (-4 hit → net {adjusted:+.1f})"
            # Skip if hit wipes out the gain
            if adjusted <= 0 and i >= free_transfers:
                hit_str = f" (-4 hit → net {adjusted:+.1f}) ⚠ NOT WORTH IT"

        arrow = "⬆" if t["net_xpts_gain"] > 0 else "⬇"
        free_tag = "FREE" if i < free_transfers else "HIT"
        print(f"\n  {i+1}. [{free_tag}] {t['out_player']} ({t['out_team']}, £{t['out_price']}m, {t['out_xpts']}pts)")
        print(f"     → {t['in_player']} ({t['in_team']}, £{t['in_price']}m, {t['in_xpts']}pts)")
        print(f"     {arrow} Net gain: {t['net_xpts_gain']:+.1f} xPts | GW1: {t['in_gw1']}pts | Form: {t['in_form']}{hit_str}")
        if t["cost_change"] != 0:
            print(f"     💰 Cost: {t['cost_change']:+.1f}m → £{t['new_bank']}m ITB")

    # ── Best combo of 2 transfers ──
    if free_transfers >= 2 or len(top_transfers) >= 2:
        print(f"\n  {'─'*76}")
        print(f"  BEST 2-TRANSFER COMBOS")
        print(f"  {'─'*76}")

        # Find top combos (different positions ideally)
        best_combos = []
        for i, t1 in enumerate(top_transfers[:8]):
            for j, t2 in enumerate(top_transfers[:8]):
                if i >= j: continue
                if t1["out_player"] == t2["out_player"]: continue
                if t1["in_player"] == t2["in_player"]: continue
                # Check team limits
                combo_in_teams = [t1["in_team"], t2["in_team"]]
                combo_out_teams = [t1["out_team"], t2["out_team"]]
                tc = team_counts.copy()
                for ot in combo_out_teams: tc[ot] = tc.get(ot, 1) - 1
                valid = True
                for it in combo_in_teams:
                    tc[it] = tc.get(it, 0) + 1
                    if tc[it] > 3: valid = False
                if not valid: continue
                # Check combined budget
                total_sell = t1["out_price"] + t2["out_price"]
                total_buy = t1["in_price"] + t2["in_price"]
                if total_buy > total_sell + bank: continue
                total_gain = t1["net_xpts_gain"] + t2["net_xpts_gain"]
                hits = max(0, 2 - free_transfers) * hit_penalty
                best_combos.append({
                    "t1": t1, "t2": t2,
                    "total_gain": total_gain,
                    "net_after_hits": total_gain - hits,
                    "hits": hits,
                    "new_bank": round(bank - (total_buy - total_sell), 1),
                })
        best_combos.sort(key=lambda x: x["net_after_hits"], reverse=True)

        for k, combo in enumerate(best_combos[:3]):
            t1, t2 = combo["t1"], combo["t2"]
            hit_note = f" (after -{combo['hits']}pt hit)" if combo["hits"] > 0 else ""
            worth = "✅" if combo["net_after_hits"] > 0 else "⚠ MARGINAL"
            print(f"\n  Combo {k+1}: {worth} Net {combo['net_after_hits']:+.1f} xPts{hit_note} | £{combo['new_bank']}m ITB")
            print(f"    OUT: {t1['out_player']} (£{t1['out_price']}m) + {t2['out_player']} (£{t2['out_price']}m)")
            print(f"    IN:  {t1['in_player']} (£{t1['in_price']}m, {t1['in_xpts']}pts) + {t2['in_player']} (£{t2['in_price']}m, {t2['in_xpts']}pts)")

    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, "transfer_recommendations.csv")
    swaps_df.head(50).to_csv(out_path, index=False)
    print(f"\n  Full recommendations saved to: {out_path}")

    return swaps_df


def run_transfer_analysis(predictions):
    """Entry point for transfer recommender — called from main if team data exists."""
    my_team, settings = load_my_team()
    if my_team is None:
        return

    print(f"\n  Matching your squad to prediction database...")
    matched = match_my_team_to_predictions(my_team, predictions)
    unmatched = matched[matched["pred_idx"] == -1]
    if len(unmatched) > 0:
        print(f"    ⚠ {len(unmatched)} player(s) could not be matched — their xPts will be 0")

    squad_xpts = matched["total_expected_Ngw"].sum()
    squad_cost = matched["sell_price"].sum()
    print(f"  Your squad: {len(matched)} players | Total value: £{squad_cost:.1f}m | Expected: {squad_xpts:.1f} xPts")

    recommend_transfers(matched, predictions, settings)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
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
