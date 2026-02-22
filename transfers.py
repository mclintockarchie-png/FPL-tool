"""
FPL Enhanced Model — Transfers Module
======================================
Transfer recommendation engine that matches your squad and suggests improvements.
"""

import pandas as pd
import numpy as np
import os
import re
import unicodedata
from config import MY_TEAM_DIR, OUTPUT_DIR, safe_float


# ══════════════════════════════════════════════════════════════════════
# LOAD MY TEAM
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# MATCH SQUAD TO PREDICTIONS
# ══════════════════════════════════════════════════════════════════════

def match_my_team_to_predictions(my_team, predictions):
    """Match user's squad names to prediction database using multi-pass matching.
    Handles FPL short names (B.Fernandes), nicknames (Kroupi.Jr), accents, etc."""
    matched = []
    pred_lower = predictions.copy()
    pred_lower["_name_lower"] = pred_lower["player"].str.lower().str.strip()
    pred_lower["_fbref_lower"] = pred_lower.get("player_fbref", pred_lower["player"]).str.lower().str.strip()

    # Clean versions: strip dots, hyphens, accents for fuzzy matching
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


# ══════════════════════════════════════════════════════════════════════
# RECOMMEND TRANSFERS
# ══════════════════════════════════════════════════════════════════════

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
            print(f"     Cost: {t['cost_change']:+.1f}m → £{t['new_bank']}m ITB")

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


# ══════════════════════════════════════════════════════════════════════
# RUN TRANSFER ANALYSIS
# ══════════════════════════════════════════════════════════════════════

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
