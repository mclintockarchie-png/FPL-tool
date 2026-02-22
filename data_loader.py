"""
FPL Enhanced Model — Data Loader Module
========================================
Loads FBref and FPL API data, merges them via intelligent name matching.
"""

import pandas as pd
import numpy as np
import os
from config import (
    FBREF_DIR, FPL_API_DIR, TEAM_NAME_MAP, POS_MAP,
    NAME_ALIASES, AMBIGUOUS_LASTNAMES, safe_float, FIXTURE_TEAM_MAP
)

# ══════════════════════════════════════════════════════════════════════
# FBREF DATA LOADING
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
# FPL API DATA LOADING
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
# NAME MATCHING HELPERS
# ══════════════════════════════════════════════════════════════════════

def map_position(pos_str):
    """Convert FBref position string to FPL position."""
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


def _match_players(fbref_df, fpl_stats, fpl_prices, updated_price_map, updated_points_map, injury_map):
    """Execute 8-pass name matching between FBref and FPL data."""
    matched, unmatched = [], []
    used_idx = set()

    # Build FPL lookup
    fpl = fpl_stats.merge(fpl_prices[["player_name", "price"]].drop_duplicates(),
                          on="player_name", how="left", suffixes=("", "_p"))
    if "price_p" in fpl.columns:
        fpl["price"] = fpl["price_p"].fillna(fpl["price"])
    fpl["name_lower"] = fpl["player_name"].str.lower().str.strip()
    fpl["name_last"] = fpl["name_lower"].str.split().str[-1]
    fpl["full_lower"] = (fpl["full_name"].str.lower().str.strip()
                         if "full_name" in fpl.columns else fpl["name_lower"])
    fbref_df["name_lower"] = fbref_df["player_fbref"].str.lower().str.strip()
    fbref_df["name_last"] = fbref_df["name_lower"].str.split().str[-1]

    for _, fr in fbref_df.iterrows():
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
            matched.append(_merge_row(fr, best, updated_price_map, updated_points_map, injury_map))
            if hasattr(best, 'name'): used_idx.add(best.name)
        else:
            d = fr.to_dict()
            d.update({"price": 0, "fpl_points": 0, "form": 0, "selected_pct": 0,
                       "fpl_xG": 0, "fpl_xA": 0, "bonus": 0, "ict_index": 0,
                       "points_per_game": 0})
            matched.append(d)
            unmatched.append(f"{fr['player_fbref']} ({ft})")

    return matched, unmatched, used_idx, fpl


# ══════════════════════════════════════════════════════════════════════
# BUILD UNIFIED DATABASE
# ══════════════════════════════════════════════════════════════════════

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

    # Run 8-pass name matching
    matched, unmatched, used_idx, fpl = _match_players(fb, fpl_stats, fpl_prices, up_price, up_pts, inj_map)

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
