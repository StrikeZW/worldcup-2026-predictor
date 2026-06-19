#!/usr/bin/env python3
import json
import math
import shutil
import time
import urllib.request
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
WORLDCUP_JSON_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"

FIXTURES_FILE = "worldcup_group_stage.csv"
OUTPUT_DIR = Path("output")
SIMULATIONS = 20000
TRAINING_YEARS_BACK = 25
DECAY_HALF_LIFE_YEARS = 3
USE_TIME_DECAY = True
ELO_BASE_K = 20

ELO_TOURNAMENT_K = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "Copa América": 50,
    "AFC Asian Cup": 45,
    "African Cup of Nations": 45,
    "CONCACAF Gold Cup": 40,
    "Friendly": 20,
}

FEATURES = [
    "elo_diff",
    "form_points_diff",
    "goals_for_diff",
    "goals_against_diff",
    "h2h",
    "neutral",
    "tournament_weight",
]

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 5,
    "FIFA World Cup qualification": 3,
    "UEFA Euro": 4,
    "Copa América": 4,
    "AFC Asian Cup": 4,
    "African Cup of Nations": 4,
    "CONCACAF Gold Cup": 4,
    "Friendly": 1,
}

ROUND_OF_32_TEMPLATE = [
    ("A1", "T8"),
    ("B1", "T7"),
    ("C1", "T6"),
    ("D1", "T5"),
    ("E1", "T4"),
    ("F1", "T3"),
    ("G1", "T2"),
    ("H1", "T1"),
    ("I1", "J2"),
    ("J1", "I2"),
    ("K1", "L2"),
    ("L1", "K2"),
    ("A2", "B2"),
    ("C2", "D2"),
    ("E2", "F2"),
    ("G2", "H2"),
]



START = time.time()

def goal_difference_multiplier(home_score, away_score):
    gd = abs(home_score - away_score)

    if gd <= 1:
        return 1.0

    if gd == 2:
        return 1.5

    return 1.75 + (gd - 3) / 8

def elo_k_factor(tournament, home_score, away_score):
    base_k = ELO_TOURNAMENT_K.get(str(tournament), ELO_BASE_K)
    return base_k * goal_difference_multiplier(home_score, away_score)

def log(msg):
    print(f"[{time.time() - START:8.1f}s] {msg}", flush=True)

def print_match_predictions(prediction_cache):
    print("\nEinzelspiel-Prognosen:")
    print("=" * 80)

    for p in prediction_cache.values():
        print(
            f"{p['home_team']} vs {p['away_team']}: "
            f"xG {p['home_xg']:.2f}:{p['away_xg']:.2f} | "
            f"Sieg {p['home_team']} {p['home_win']:.1%} | "
            f"Remis {p['draw']:.1%} | "
            f"Sieg {p['away_team']} {p['away_win']:.1%}"
        )

def safe_mean(values, default=0.0):
    return float(np.mean(values)) if values else default


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(home_elo, away_elo, home_score, away_score, k):
    exp_home = expected_score(home_elo, away_elo)

    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5

    return (
        home_elo + k * (actual_home - exp_home),
        away_elo + k * ((1 - actual_home) - (1 - exp_home)),
    )


def h2h_key(a, b):
    return tuple(sorted([a, b]))


def get_h2h(h2h_stats, home, away):
    stats = h2h_stats[h2h_key(home, away)]
    if stats["total"] == 0:
        return 0.0

    return (
        stats["wins"].get(home, 0)
        - stats["wins"].get(away, 0)
    ) / stats["total"]


def tournament_weight(name):
    return TOURNAMENT_WEIGHT.get(str(name), 2)


def update_worldcup_fixtures_csv():
    log("Aktualisiere World-Cup-Fixtures...")

    req = urllib.request.Request(
        WORLDCUP_JSON_URL,
        headers={"User-Agent": "worldcup-predictor/4.0"},
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = []

    for match in payload.get("matches", []):
        group_raw = match.get("group")

        if not group_raw or not str(group_raw).startswith("Group "):
            continue

        home = match.get("team1")
        away = match.get("team2")

        if not home or not away:
            continue

        score = match.get("score", {})
        ft = score.get("ft") if isinstance(score, dict) else None

        if isinstance(ft, list) and len(ft) == 2:
            home_score, away_score = ft
        else:
            home_score, away_score = "", ""

        rows.append({
            "group": str(group_raw).replace("Group ", "").strip(),
            "date": match.get("date", ""),
            "home_team": home,
            "away_team": away,
            "neutral": True,
            "home_score": home_score,
            "away_score": away_score,
        })

    if not rows:
        raise RuntimeError("Keine Gruppenspiele gefunden.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["group", "date", "home_team", "away_team"])
    df.to_csv(FIXTURES_FILE, index=False)

    played = df["home_score"].ne("").sum()
    log(f"{FIXTURES_FILE} geschrieben: {len(df)} Spiele, {played} mit Ergebnis")


def build_training_data(results):
    log("Erzeuge Trainingsdaten...")

    results = results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_score", "away_score"])
    results = results.sort_values("date").reset_index(drop=True)

    elo = defaultdict(lambda: 1500.0)
    form_points = defaultdict(lambda: deque(maxlen=10))
    goals_for = defaultdict(lambda: deque(maxlen=10))
    goals_against = defaultdict(lambda: deque(maxlen=10))
    h2h_stats = defaultdict(lambda: {"total": 0, "wins": defaultdict(int)})

    rows = []

    for idx, r in results.iterrows():
        if idx and idx % 1000 == 0:
            log(f"{idx:,}/{len(results):,} historische Spiele verarbeitet")

        home = r["home_team"]
        away = r["away_team"]
        hs = int(r["home_score"])
        aw = int(r["away_score"])

        rows.append({
            "date": r["date"],
            "elo_diff": elo[home] - elo[away],
            "form_points_diff": safe_mean(form_points[home]) - safe_mean(form_points[away]),
            "goals_for_diff": safe_mean(goals_for[home]) - safe_mean(goals_for[away]),
            "goals_against_diff": safe_mean(goals_against[home]) - safe_mean(goals_against[away]),
            "h2h": get_h2h(h2h_stats, home, away),
            "neutral": int(bool(r["neutral"])),
            "tournament_weight": tournament_weight(r["tournament"]),
            "home_goals": hs,
            "away_goals": aw,
        })

        k = elo_k_factor(r["tournament"], hs, aw)
        elo[home], elo[away] = update_elo(
            elo[home],
            elo[away],
            hs,
            aw,
            k=k,
        )

        if hs > aw:
            hp, ap = 3, 0
            h2h_stats[h2h_key(home, away)]["wins"][home] += 1
        elif hs < aw:
            hp, ap = 0, 3
            h2h_stats[h2h_key(home, away)]["wins"][away] += 1
        else:
            hp, ap = 1, 1

        form_points[home].append(hp)
        form_points[away].append(ap)
        goals_for[home].append(hs)
        goals_for[away].append(aw)
        goals_against[home].append(aw)
        goals_against[away].append(hs)
        h2h_stats[h2h_key(home, away)]["total"] += 1

    state = {
        "elo": dict(elo),
        "form_points": form_points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "h2h_stats": h2h_stats,
    }

    log("Trainingsdaten fertig")
    return pd.DataFrame(rows), state


def train_models():
    log("Lade historische Ergebnisse...")
    results = pd.read_csv(RESULTS_URL)
    log(f"{len(results):,} historische Zeilen geladen")

    train_df, state = build_training_data(results)

    latest_date = train_df["date"].max()
    cutoff_date = latest_date - pd.DateOffset(years=TRAINING_YEARS_BACK)

    before_count = len(train_df)
    train_df = train_df[train_df["date"] >= cutoff_date].reset_index(drop=True)

    log(
        f"Trainings-Cutoff: letzte {TRAINING_YEARS_BACK} Jahre "
        f"({before_count:,} -> {len(train_df):,} Spiele)"
    )

    if USE_TIME_DECAY:
        age_days = (latest_date - train_df["date"]).dt.days
        half_life_days = DECAY_HALF_LIFE_YEARS * 365.25
        sample_weight = 0.5 ** (age_days / half_life_days)
        sample_weight = pd.Series(sample_weight, index=train_df.index)

        log(f"Zeitgewichtung aktiv: Half-Life {DECAY_HALF_LIFE_YEARS} Jahre")
    else:
        sample_weight = None

    X = train_df[FEATURES]
    yh = train_df["home_goals"]
    ya = train_df["away_goals"]

    split = int(len(train_df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    yh_train, yh_test = yh.iloc[:split], yh.iloc[split:]
    ya_train, ya_test = ya.iloc[:split], ya.iloc[split:]

    if sample_weight is not None:
        w_train = sample_weight.iloc[:split]
    else:
        w_train = None

    log("Trainiere Heimtor-Modell...")
    home_model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.04,
        l2_regularization=0.1,
        random_state=42,
    )
    home_model.fit(X_train, yh_train, sample_weight=w_train)

    log("Trainiere Auswärtstor-Modell...")
    away_model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.04,
        l2_regularization=0.1,
        random_state=43,
    )
    away_model.fit(X_train, ya_train, sample_weight=w_train)

    pred_h = np.maximum(home_model.predict(X_test), 0.05)
    pred_a = np.maximum(away_model.predict(X_test), 0.05)

    log(f"MAE Heimtore: {mean_absolute_error(yh_test, pred_h):.3f}")
    log(f"MAE Auswärtstore: {mean_absolute_error(ya_test, pred_a):.3f}")

    return home_model, away_model, state

def make_feature_row(state, home, away, neutral=True):
    return pd.DataFrame([{
        "elo_diff": state["elo"].get(home, 1500.0) - state["elo"].get(away, 1500.0),
        "form_points_diff": safe_mean(state["form_points"][home]) - safe_mean(state["form_points"][away]),
        "goals_for_diff": safe_mean(state["goals_for"][home]) - safe_mean(state["goals_for"][away]),
        "goals_against_diff": safe_mean(state["goals_against"][home]) - safe_mean(state["goals_against"][away]),
        "h2h": get_h2h(state["h2h_stats"], home, away),
        "neutral": int(bool(neutral)),
        "tournament_weight": 5,
    }])


def predict_expected_goals(home_model, away_model, state, home, away, neutral=True):
    row = make_feature_row(state, home, away, neutral)
    home_xg = max(float(home_model.predict(row[FEATURES])[0]), 0.05)
    away_xg = max(float(away_model.predict(row[FEATURES])[0]), 0.05)
    return home_xg, away_xg


def poisson_goal_distribution(xg, max_goals=8):
    probs = np.array([
        np.exp(-xg) * xg ** i / math.factorial(i)
        for i in range(max_goals + 1)
    ])
    return probs / probs.sum()


def poisson_match_probs(home_probs, away_probs):
    home_win = draw = away_win = 0.0

    for h, hp in enumerate(home_probs):
        for a, ap in enumerate(away_probs):
            p = hp * ap
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

    return home_win, draw, away_win


def build_match_prediction_cache(fixtures, home_model, away_model, state):
    log("Baue Match-Prediction-Cache...")

    cache = {}

    for _, r in fixtures.iterrows():
        home = r["home_team"]
        away = r["away_team"]

        hxg, axg = predict_expected_goals(
            home_model,
            away_model,
            state,
            home,
            away,
            bool(r["neutral"]),
        )

        home_probs = poisson_goal_distribution(hxg)
        away_probs = poisson_goal_distribution(axg)
        home_win, draw, away_win = poisson_match_probs(home_probs, away_probs)

        cache[(home, away)] = {
            "home_team": home,
            "away_team": away,
            "home_xg": hxg,
            "away_xg": axg,
            "home_goal_probs": home_probs,
            "away_goal_probs": away_probs,
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }

    log(f"Match-Prediction-Cache enthält {len(cache)} Spiele")
    return cache

def save_match_predictions(cache):
    rows = []

    for p in cache.values():
        rows.append({
            "home_team": p["home_team"],
            "away_team": p["away_team"],
            "home_xg": p["home_xg"],
            "away_xg": p["away_xg"],
            "home_win": p["home_win"],
            "draw": p["draw"],
            "away_win": p["away_win"],
        })

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "match_predictions.csv", index=False)

def save_frozen_match_predictions(prediction_cache, fixtures):
    """
    Speichert Pre-Match-Prognosen dauerhaft.

    Wichtig:
    - Bereits vorhandene Prognosen werden NICHT überschrieben.
    - Neue Spiele werden ergänzt.
    - Dadurch bleibt die ursprüngliche Vorhersage für spätere Vergleiche erhalten.
    """

    file = OUTPUT_DIR / "frozen_match_predictions.csv"

    if file.exists():
        frozen_df = pd.read_csv(file)
    else:
        frozen_df = pd.DataFrame(columns=[
            "home_team",
            "away_team",
            "prediction_created_at",
            "home_xg",
            "away_xg",
            "home_win",
            "draw",
            "away_win",
        ])

    existing_keys = set(
        zip(
            frozen_df["home_team"],
            frozen_df["away_team"],
        )
    )

    new_rows = []
    now = datetime.now().isoformat(timespec="seconds")

    for _, r in fixtures.iterrows():
        home = r["home_team"]
        away = r["away_team"]
        key = (home, away)

        if key in existing_keys:
            continue

        p = prediction_cache.get(key)

        if not p:
            continue

        new_rows.append({
            "home_team": home,
            "away_team": away,
            "prediction_created_at": now,
            "home_xg": p["home_xg"],
            "away_xg": p["away_xg"],
            "home_win": p["home_win"],
            "draw": p["draw"],
            "away_win": p["away_win"],
        })

    if new_rows:
        frozen_df = pd.concat(
            [frozen_df, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

        frozen_df.to_csv(file, index=False)
        log(f"Frozen Match Predictions ergänzt: {len(new_rows)} neue Spiele")
    else:
        log("Frozen Match Predictions unverändert")

def prepare_groups(fixtures, prediction_cache):
    prepared = {}

    for group, group_fixtures in fixtures.groupby("group"):
        teams = set()
        matches = []

        for _, r in group_fixtures.iterrows():
            home = r["home_team"]
            away = r["away_team"]

            teams.add(home)
            teams.add(away)

            if pd.notna(r["home_score"]) and pd.notna(r["away_score"]):
                matches.append({
                    "home": home,
                    "away": away,
                    "played": True,
                    "home_score": int(r["home_score"]),
                    "away_score": int(r["away_score"]),
                })
            else:
                p = prediction_cache[(home, away)]
                matches.append({
                    "home": home,
                    "away": away,
                    "played": False,
                    "home_goal_probs": p["home_goal_probs"],
                    "away_goal_probs": p["away_goal_probs"],
                })

        prepared[str(group)] = {
            "teams": sorted(teams),
            "matches": matches,
        }

    return prepared


def simulate_group_once(group, group_data):
    table = {
        team: {
            "team": team,
            "group": group,
            "points": 0,
            "gf": 0,
            "ga": 0,
        }
        for team in group_data["teams"]
    }

    for m in group_data["matches"]:
        home = m["home"]
        away = m["away"]

        if m["played"]:
            hs = m["home_score"]
            aw = m["away_score"]
        else:
            hs = np.random.choice(len(m["home_goal_probs"]), p=m["home_goal_probs"])
            aw = np.random.choice(len(m["away_goal_probs"]), p=m["away_goal_probs"])

        table[home]["gf"] += hs
        table[home]["ga"] += aw
        table[away]["gf"] += aw
        table[away]["ga"] += hs

        if hs > aw:
            table[home]["points"] += 3
        elif aw > hs:
            table[away]["points"] += 3
        else:
            table[home]["points"] += 1
            table[away]["points"] += 1

    standings = []

    for row in table.values():
        row = row.copy()
        row["gd"] = row["gf"] - row["ga"]
        standings.append(row)

    standings.sort(
        key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"])
    )

    return standings


def build_round_of_32_slots(group_tables):
    slots = {}
    third_placed = []

    for group, table in group_tables.items():
        slots[f"{group}1"] = table[0]["team"]
        slots[f"{group}2"] = table[1]["team"]

        third = table[2].copy()
        third["group"] = group
        third_placed.append(third)

    third_placed.sort(
        key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"])
    )

    qualified_thirds = third_placed[:8]

    for idx, row in enumerate(qualified_thirds, start=1):
        slots[f"T{idx}"] = row["team"]

    return slots, qualified_thirds


def simulate_knockout_match(team_a, team_b, knockout_cache, home_model, away_model, state):
    key = tuple(sorted([team_a, team_b]))

    if key not in knockout_cache:
        hxg, axg = predict_expected_goals(
            home_model,
            away_model,
            state,
            team_a,
            team_b,
            neutral=True,
        )
        knockout_cache[key] = {
            team_a: hxg,
            team_b: axg,
        }

    xg_a = knockout_cache[key][team_a]
    xg_b = knockout_cache[key][team_b]

    goals_a = np.random.poisson(xg_a)
    goals_b = np.random.poisson(xg_b)

    if goals_a > goals_b:
        return team_a
    if goals_b > goals_a:
        return team_b

    p_a = xg_a / (xg_a + xg_b)
    return np.random.choice([team_a, team_b], p=[p_a, 1 - p_a])


def simulate_tournament_once(prepared_groups, home_model, away_model, state, knockout_cache):
    group_tables = {}

    for group, group_data in prepared_groups.items():
        group_tables[group] = simulate_group_once(group, group_data)

    slots, qualified_thirds = build_round_of_32_slots(group_tables)

    reached = defaultdict(set)
    bracket_nodes = []

    for group, table in group_tables.items():
        for rank, row in enumerate(table, start=1):
            reached[row["team"]].add(f"GROUP_RANK_{rank}")
            reached[row["team"]].add("GROUP_STAGE")

    for row in qualified_thirds:
        reached[row["team"]].add("ADVANCE_AS_THIRD")

    for team in slots.values():
        reached[team].add("R32")

    current_round = [
        (slots[left], slots[right])
        for left, right in ROUND_OF_32_TEMPLATE
    ]

    rounds = [
        ("R32", "R16"),
        ("R16", "QF"),
        ("QF", "SF"),
        ("SF", "FINAL"),
        ("FINAL", "WINNER"),
    ]

    for current_round_name, next_stage in rounds:
        winners = []

        for match_idx, (team_a, team_b) in enumerate(current_round, start=1):
            bracket_nodes.append({
                "round": current_round_name,
                "match": match_idx,
                "slot": "left",
                "team": team_a,
            })
            bracket_nodes.append({
                "round": current_round_name,
                "match": match_idx,
                "slot": "right",
                "team": team_b,
            })

            winner = simulate_knockout_match(
                team_a,
                team_b,
                knockout_cache,
                home_model,
                away_model,
                state,
            )

            winners.append(winner)
            reached[winner].add(next_stage)

        if next_stage == "WINNER":
            break

        current_round = [
            (winners[i], winners[i + 1])
            for i in range(0, len(winners), 2)
        ]

    return group_tables, reached, bracket_nodes


def simulate_world_cup(fixtures, prediction_cache, home_model, away_model, state, sims):
    log("Bereite Gruppen für konsistente Turniersimulation vor...")
    prepared_groups = prepare_groups(fixtures, prediction_cache)

    all_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))

    group_counts = {
        team: {
            "expected_points": 0.0,
            "rank1": 0,
            "rank2": 0,
            "rank3": 0,
            "rank4": 0,
            "advance": 0,
            "advance_as_third": 0,
        }
        for team in all_teams
    }

    knockout_counts = {
        team: {
            "R32": 0,
            "R16": 0,
            "QF": 0,
            "SF": 0,
            "FINAL": 0,
            "WINNER": 0,
        }
        for team in all_teams
    }

    bracket_counts = defaultdict(int)
    bracket_matchup_counts = defaultdict(int)
    knockout_cache = {}

    progress_interval = max(1, sims // 10)

    for i in range(sims):
        if i and i % progress_interval == 0:
            log(f"{i:,}/{sims:,} komplette Turniere simuliert")

        group_tables, reached, bracket_nodes = simulate_tournament_once(
            prepared_groups,
            home_model,
            away_model,
            state,
            knockout_cache,
        )

        for group, table in group_tables.items():
            for rank, row in enumerate(table, start=1):
                team = row["team"]
                group_counts[team]["expected_points"] += row["points"]
                group_counts[team][f"rank{rank}"] += 1

        for team, stages in reached.items():
            if "R32" in stages:
                group_counts[team]["advance"] += 1
            if "ADVANCE_AS_THIRD" in stages:
                group_counts[team]["advance_as_third"] += 1

            for stage in knockout_counts[team]:
                if stage in stages:
                    knockout_counts[team][stage] += 1

        for node in bracket_nodes:
            bracket_counts[
                (node["round"], node["match"], node["slot"], node["team"])
            ] += 1

        for left, right in zip(bracket_nodes[0::2], bracket_nodes[1::2]):
            if left["round"] != right["round"] or left["match"] != right["match"]:
                continue

            bracket_matchup_counts[
                (
                    left["round"],
                    left["match"],
                    left["team"],
                    right["team"],
                 )
            ] += 1

    log(f"{sims:,}/{sims:,} komplette Turniere simuliert")

    group_rows = []
    team_to_group = {}

    for group, group_data in prepared_groups.items():
        for team in group_data["teams"]:
            team_to_group[team] = group

    for team, c in group_counts.items():
        group_rows.append({
            "group": team_to_group.get(team, ""),
            "team": team,
            "expected_points": c["expected_points"] / sims,
            "p_rank1": c["rank1"] / sims,
            "p_rank2": c["rank2"] / sims,
            "p_rank3": c["rank3"] / sims,
            "p_rank4": c["rank4"] / sims,
            "p_top2": (c["rank1"] + c["rank2"]) / sims,
            "p_advance_as_third": c["advance_as_third"] / sims,
            "p_advance": c["advance"] / sims,
        })

    knockout_rows = []

    for team, c in knockout_counts.items():
        knockout_rows.append({
            "team": team,
            "p_r32": c["R32"] / sims,
            "p_r16": c["R16"] / sims,
            "p_qf": c["QF"] / sims,
            "p_sf": c["SF"] / sims,
            "p_final": c["FINAL"] / sims,
            "p_winner": c["WINNER"] / sims,
        })

    bracket_rows = []

    for (round_name, match_idx, slot, team), count in bracket_counts.items():
        bracket_rows.append({
            "round": round_name,
          "match": match_idx,
           "slot": slot,
           "team": team,
           "probability": count / sims,
        })

    matchup_rows = []

    for (round_name, match_idx, team_a, team_b), count in bracket_matchup_counts.items():
        matchup_rows.append({
            "round": round_name,
            "match": match_idx,
            "team_a": team_a,
            "team_b": team_b,
            "probability": count / sims,
        })

    group_df = pd.DataFrame(group_rows).sort_values(
        ["group", "expected_points", "p_advance"],
        ascending=[True, False, False],
    )

    knockout_df = pd.DataFrame(knockout_rows).sort_values(
        ["p_winner", "p_final", "p_sf"],
        ascending=[False, False, False],
    )

    bracket_df = pd.DataFrame(bracket_rows).sort_values(
     ["round", "match", "slot", "probability"],
     ascending=[True, True, True, False],
    )

    matchup_df = pd.DataFrame(matchup_rows).sort_values(
      ["round", "match", "probability"],
      ascending=[True, True, False],
    )

    return group_df, knockout_df, bracket_df, matchup_df


def save_group_outputs(group_df):
    for group, df in group_df.groupby("group"):
        df = df.sort_values(
            ["expected_points", "p_advance"],
            ascending=[False, False],
        )
        df.drop(columns=["group"]).to_csv(
            OUTPUT_DIR / f"group_{group}.csv",
            index=False,
        )


def save_snapshot(fixtures, group_df, knockout_df, bracket_df, matchup_df):
    snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = OUTPUT_DIR / "snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    fixtures.to_csv(snapshot_dir / "fixtures.csv", index=False)
    knockout_df.to_csv(snapshot_dir / "knockout.csv", index=False)
    bracket_df.to_csv(snapshot_dir / "bracket.csv", index=False)
    matchup_df.to_csv(snapshot_dir / "bracket_matchups.csv", index=False)

    match_predictions = OUTPUT_DIR / "match_predictions.csv"
    if match_predictions.exists():
        shutil.copy(match_predictions, snapshot_dir / "match_predictions.csv")

    for file in OUTPUT_DIR.glob("group_*.csv"):
        shutil.copy(file, snapshot_dir / file.name)

    (OUTPUT_DIR / "latest_snapshot.txt").write_text(snapshot_id, encoding="utf-8")
    log(f"Snapshot gespeichert: {snapshot_dir}")


def print_group_summary(group_df):
    print("\nGruppenprognose:")
    print("=" * 80)

    for group, df in group_df.groupby("group"):
        print()
        print(f"Gruppe {group}")
        print(
            df.drop(columns=["group"]).assign(
                expected_points=lambda d: d["expected_points"].round(2),
                p_rank1=lambda d: (d["p_rank1"] * 100).round(1),
                p_rank2=lambda d: (d["p_rank2"] * 100).round(1),
                p_rank3=lambda d: (d["p_rank3"] * 100).round(1),
                p_rank4=lambda d: (d["p_rank4"] * 100).round(1),
                p_top2=lambda d: (d["p_top2"] * 100).round(1),
                p_advance_as_third=lambda d: (d["p_advance_as_third"] * 100).round(1),
                p_advance=lambda d: (d["p_advance"] * 100).round(1),
            ).to_string(index=False)
        )


def print_knockout_summary(knockout_df):
    print("\nFinalrunden-Prognose:")
    print("=" * 80)

    print(
        knockout_df.assign(
            p_r32=lambda d: (d["p_r32"] * 100).round(1),
            p_r16=lambda d: (d["p_r16"] * 100).round(1),
            p_qf=lambda d: (d["p_qf"] * 100).round(1),
            p_sf=lambda d: (d["p_sf"] * 100).round(1),
            p_final=lambda d: (d["p_final"] * 100).round(1),
            p_winner=lambda d: (d["p_winner"] * 100).round(2),
        ).to_string(index=False)
    )


def main():
    np.random.seed(42)
    OUTPUT_DIR.mkdir(exist_ok=True)

    update_worldcup_fixtures_csv()

    home_model, away_model, state = train_models()

    fixtures = pd.read_csv(FIXTURES_FILE)
    fixtures.to_csv(OUTPUT_DIR / "fixtures.csv", index=False)

    prediction_cache = build_match_prediction_cache(
        fixtures,
        home_model,
        away_model,
        state,
    )

    save_match_predictions(prediction_cache)
    save_frozen_match_predictions(prediction_cache, fixtures)
    print_match_predictions(prediction_cache)
    
    group_df, knockout_df, bracket_df, matchup_df  = simulate_world_cup(
        fixtures,
        prediction_cache,
        home_model,
        away_model,
        state,
        SIMULATIONS,
    )

    save_group_outputs(group_df)

    knockout_df.to_csv(OUTPUT_DIR / "knockout.csv", index=False)
    bracket_df.to_csv(OUTPUT_DIR / "bracket.csv", index=False)
    matchup_df.to_csv(OUTPUT_DIR / "bracket_matchups.csv", index=False)
    print_group_summary(group_df)
    print_knockout_summary(knockout_df)

    save_snapshot(fixtures, group_df, knockout_df, bracket_df, matchup_df)

    log("Fertig")


if __name__ == "__main__":
    main()