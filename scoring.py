"""
FPL Enhanced Model — Scoring Module
====================================
Calculates enhanced xPts via multi-signal model with clean sheet probability.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from config import SCORING, FIXTURE_TEAM_MAP, safe_float


# ══════════════════════════════════════════════════════════════════════
# HOME/AWAY PROFILE BUILDING
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


# ══════════════════════════════════════════════════════════════════════
# CLEAN SHEET PROBABILITY
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# FIXTURE PARSING
# ══════════════════════════════════════════════════════════════════════

def _parse_fixtures(fpl_data, gameweeks):
    """Per-team chronological fixtures. Skips matches ≤2 days from today (current GW)."""
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


# ══════════════════════════════════════════════════════════════════════
# FDR ENRICHMENT
# ══════════════════════════════════════════════════════════════════════

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
# ENHANCED SCORING MODEL
# ══════════════════════════════════════════════════════════════════════

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
