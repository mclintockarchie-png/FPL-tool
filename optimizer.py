"""
FPL Enhanced Model — Optimizer Module
======================================
Squad optimization via LP solver (PuLP) with greedy fallback.
"""

import pandas as pd
import os
from config import SQUAD_RULES, TEAM_COLORS, HAS_PULP, OUTPUT_DIR

if HAS_PULP:
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value


# ══════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION ENTRY POINT
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


# ══════════════════════════════════════════════════════════════════════
# LP OPTIMIZATION (PuLP)
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# GREEDY OPTIMIZATION (Fallback)
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# FINALIZE RESULT
# ══════════════════════════════════════════════════════════════════════

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
# CONSOLE REPORTING
# ══════════════════════════════════════════════════════════════════════

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


def print_value_picks(df):
    """Print top value picks by position."""
    print(f"\n{'='*80}\n  TOP VALUE PICKS (best underlying stats per £)\n{'='*80}")
    d = df[df["minutes"] >= 450].copy()
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pd_ = d[d["position"] == pos].sort_values("enhanced_value", ascending=False).head(8)
        print(f"\n  {'─'*60}\n  {pos} — Top Value Picks\n  {'─'*60}")
        for _, p in pd_.iterrows():
            xf = " xG UNDERPERFORMER" if p.get("xG_delta", 0) < -1 else " xG overperformer" if p.get("xG_delta", 0) > 2 else ""
            df_ = f" [DIFF {p['selected_pct']}%]" if p.get("selected_pct", 50) < 10 else ""
            fm = f"form:{p['form']}" if p.get("form", 0) > 0 else ""
            print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
                  f"exp:{p['total_expected_Ngw']:5.1f}pts | val:{p['enhanced_value']:4.2f} | "
                  f"G+A/90:{p['ga_per90']:.2f} | {fm}{xf}{df_}")


def print_regression_candidates(df):
    """Print xG underperformers due for regression."""
    print(f"\n{'='*80}\n  xG REGRESSION CANDIDATES (due a points surge)\n{'='*80}")
    d = df[(df["minutes"] >= 900) & (df["fpl_xG"] > 0)].copy()
    d["xGI_delta"] = d["xG_delta"] + d["xA_delta"]
    for _, p in d[d["xGI_delta"] < -1].sort_values("xGI_delta").head(10).iterrows():
        print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
              f"Goals:{p['goals']} vs xG:{p['fpl_xG']:.1f} (delta:{p['xG_delta']:+.1f}) | "
              f"Assists:{p['assists']} vs xA:{p['fpl_xA']:.1f} (delta:{p['xA_delta']:+.1f}) | "
              f"Owned:{p['selected_pct']:.1f}%")


def print_differentials(df):
    """Print low-ownership, high-quality differential picks."""
    print(f"\n{'='*80}\n  DIFFERENTIAL PICKS (low ownership, strong underlying)\n{'='*80}")
    d = df[(df["minutes"] >= 900) & (df["selected_pct"] > 0) & (df["selected_pct"] < 15)]
    for _, p in d.sort_values("enhanced_value", ascending=False).head(15).iterrows():
        print(f"  {p['player']:25s} {p['team']:15s} £{p['price']:4.1f}m | "
              f"Owned:{p['selected_pct']:5.1f}% | exp:{p['total_expected_Ngw']:5.1f}pts | "
              f"G+A/90:{p['ga_per90']:.2f} | form:{p['form']}")


# ══════════════════════════════════════════════════════════════════════
# HTML VISUALIZATION
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
