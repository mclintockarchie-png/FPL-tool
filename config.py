"""
FPL Enhanced Model — Configuration Module
==========================================
All constants, imports, paths, and configuration shared across modules.
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
# PATHS
# ══════════════════════════════════════════════════════════════════════

# BASE_DIR points to the project root (parent of Code/ folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FBREF_DIR = os.path.join(BASE_DIR, "data", "Harvested Data", "PL teams data 2025-2026")
FPL_API_DIR = os.path.join(BASE_DIR, "data", "Harvested Data", "FPL API data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Results")
MY_TEAM_DIR = os.path.join(BASE_DIR, "data", "My Team")

# ══════════════════════════════════════════════════════════════════════
# FPL SCORING RULES
# ══════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════
# POSITION AND TEAM MAPPINGS
# ══════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════
# NAME MATCHING
# ══════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════
# SQUAD RULES
# ══════════════════════════════════════════════════════════════════════

SQUAD_RULES = {
    "total_players": 15,
    "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
    "max_per_team": 3
}

# ══════════════════════════════════════════════════════════════════════
# TEAM COLORS (for HTML visualization)
# ══════════════════════════════════════════════════════════════════════

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
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    """Safely convert a value to float, handling NaN and format issues."""
    if pd.isna(val):
        return default
    try:
        v = float(str(val).replace(",", "").strip())
        return v if not np.isnan(v) else default
    except (ValueError, TypeError):
        return default
