#!/usr/bin/env python3
from pathlib import Path
import html
import pandas as pd

OUTPUT_DIR = Path("output")
REPORT_FILE = OUTPUT_DIR / "worldcup_report.html"

def load_bracket_matchups():
    file = OUTPUT_DIR / "bracket_matchups.csv"

    if not file.exists():
        return pd.DataFrame()

    return pd.read_csv(file)

def load_match_predictions():
    file = OUTPUT_DIR / "frozen_match_predictions.csv"

    if not file.exists():
        file = OUTPUT_DIR / "match_predictions.csv"

    if not file.exists():
        return {}

    df = pd.read_csv(file)
    predictions = {}

    for _, r in df.iterrows():
        predictions[(r["home_team"], r["away_team"])] = r

    return predictions

def render_bracket_tree():
    bracket_file = OUTPUT_DIR / "bracket.csv"
    matchup_file = OUTPUT_DIR / "bracket_matchups.csv"

    if not bracket_file.exists():
        return ""

    bracket_df = pd.read_csv(bracket_file)

    if matchup_file.exists():
        matchup_df = pd.read_csv(matchup_file)
    else:
        matchup_df = pd.DataFrame()

    round_order = ["R32", "R16", "QF", "SF", "FINAL"]
    round_titles = {
        "R32": "Round of 32",
        "R16": "Round of 16",
        "QF": "Quarter-finals",
        "SF": "Semi-finals",
        "FINAL": "Final",
    }

    columns = []

    for round_name in round_order:
        rdf = bracket_df[bracket_df["round"] == round_name]
        matches_html = []

        for match_idx in sorted(rdf["match"].unique()):
            left_df = rdf[
                (rdf["match"] == match_idx) &
                (rdf["slot"] == "left")
            ].sort_values("probability", ascending=False).head(4)

            right_df = rdf[
                (rdf["match"] == match_idx) &
                (rdf["slot"] == "right")
            ].sort_values("probability", ascending=False).head(4)

            def render_slot(title, sdf):
                teams = []

                for _, r in sdf.iterrows():
                    teams.append(f"""
                    <div class="bracket-team">
                        <span>{esc(r["team"])}</span>
                        <strong>{float(r["probability"]) * 100:.1f}%</strong>
                    </div>
                    """)

                return f"""
                <div class="bracket-slot">
                    <div class="bracket-slot-title">{title}</div>
                    {''.join(teams)}
                </div>
                """

            matchup_html = ""

            if not matchup_df.empty:
                mdf = matchup_df[
                    (matchup_df["round"] == round_name) &
                    (matchup_df["match"] == match_idx)
                ].sort_values("probability", ascending=False).head(3)

                if not mdf.empty:
                    rows = []

                    for _, r in mdf.iterrows():
                        rows.append(f"""
                        <div class="bracket-team matchup">
                            <span>{esc(r["team_a"])} vs {esc(r["team_b"])}</span>
                            <strong>{float(r["probability"]) * 100:.1f}%</strong>
                        </div>
                        """)

                    matchup_html = f"""
                    <div class="bracket-matchups">
                        <div class="bracket-slot-title">Most likely matchup</div>
                        {''.join(rows)}
                    </div>
                    """

            matches_html.append(f"""
            <div class="bracket-match">
                <div class="bracket-match-title">Match {int(match_idx)}</div>

                {render_slot("Left slot", left_df)}

                <div class="versus">VS</div>

                {render_slot("Right slot", right_df)}

                {matchup_html}
            </div>
            """)

        columns.append(f"""
        <div class="bracket-column">
            <h3>{round_titles[round_name]}</h3>
            {''.join(matches_html)}
        </div>
        """)

    return f"""
    <section class="card wide">
        <h2 data-i18n="bracket">Prognose-Turnierbaum</h2>
        <p class="hint">
            Der Baum zeigt je K.-o.-Runden-Match die wahrscheinlichsten Teams pro Slot
            sowie die wahrscheinlichsten direkten Paarungen.
        </p>
        <div class="bracket">
            {''.join(columns)}
        </div>
    </section>
    """

def esc(value):
    return html.escape(str(value))



def pct(value, decimals=1):
    return f"{float(value) * 100:.{decimals}f}%"


def latest_two_snapshots():
    snap_dir = OUTPUT_DIR / "snapshots"
    if not snap_dir.exists():
        return None, None

    snaps = sorted([p for p in snap_dir.iterdir() if p.is_dir()])
    if len(snaps) == 0:
        return None, None
    if len(snaps) == 1:
        return snaps[-1], None

    return snaps[-1], snaps[-2]


def delta_html(current, previous, multiplier=100, decimals=1):
    if previous is None or pd.isna(previous):
        return ""

    delta = (float(current) - float(previous)) * multiplier

    if abs(delta) < 0.05:
        return '<span class="delta neutral">±0.0</span>'

    if delta > 0:
        return f'<span class="delta up">▲ +{delta:.{decimals}f}</span>'

    return f'<span class="delta down">▼ {delta:.{decimals}f}</span>'


def load_group_tables():
    groups = []
    for file in sorted(OUTPUT_DIR.glob("group_*.csv")):
        group = file.stem.replace("group_", "")
        groups.append((group, pd.read_csv(file)))
    return groups


def load_previous_group_table(group):
    _, previous = latest_two_snapshots()
    if previous is None:
        return None

    file = previous / f"group_{group}.csv"
    if not file.exists():
        return None

    return pd.read_csv(file).set_index("team")


def load_fixtures():
    file = OUTPUT_DIR / "fixtures.csv"
    columns = ["group", "date", "home_team", "away_team", "neutral", "home_score", "away_score"]

    if not file.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(file)
    missing = set(columns) - set(df.columns)

    if missing:
        raise RuntimeError(f"output/fixtures.csv fehlen Spalten: {sorted(missing)}")

    return df


def current_group_table(fixtures):
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))

    table = {
        team: {
            "team": team,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "points": 0,
        }
        for team in teams
    }

    for _, r in fixtures.iterrows():
        if pd.isna(r["home_score"]) or pd.isna(r["away_score"]):
            continue

        home = r["home_team"]
        away = r["away_team"]
        hs = int(r["home_score"])
        aw = int(r["away_score"])

        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["gf"] += hs
        table[home]["ga"] += aw
        table[away]["gf"] += aw
        table[away]["ga"] += hs

        if hs > aw:
            table[home]["wins"] += 1
            table[away]["losses"] += 1
            table[home]["points"] += 3
        elif aw > hs:
            table[away]["wins"] += 1
            table[home]["losses"] += 1
            table[away]["points"] += 3
        else:
            table[home]["draws"] += 1
            table[away]["draws"] += 1
            table[home]["points"] += 1
            table[away]["points"] += 1

    df = pd.DataFrame(table.values())
    df["gd"] = df["gf"] - df["ga"]

    return df.sort_values(
        ["points", "gd", "gf", "team"],
        ascending=[False, False, False, True],
    )


def render_current_table(fixtures):
    if fixtures.empty:
        return "<p>Keine Fixtures vorhanden.</p>"

    df = current_group_table(fixtures)
    rows = []

    for _, r in df.iterrows():
        rows.append(f"""
        <tr>
            <td class="team">{esc(r["team"])}</td>
            <td>{int(r["played"])}</td>
            <td>{int(r["wins"])}</td>
            <td>{int(r["draws"])}</td>
            <td>{int(r["losses"])}</td>
            <td>{int(r["gf"])}:{int(r["ga"])}</td>
            <td>{int(r["gd"])}</td>
            <td><strong>{int(r["points"])}</strong></td>
        </tr>
        """)

    return f"""
    <h3 data-i18n="current_table">Aktuelle Tabelle</h3>
    <table>
        <thead>
            <tr>
                <th data-i18n="team">Team</th>
                <th data-i18n="played">Sp</th>
                <th data-i18n="wins">S</th>
                <th data-i18n="draws">U</th>
                <th data-i18n="losses">N</th>
                <th data-i18n="goals">Tore</th>
                <th data-i18n="gd">TD</th>
                <th data-i18n="points">Pkt</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """


def render_matches(fixtures, predictions):
    rows = []

    for _, r in fixtures.sort_values("date").iterrows():
        home = r["home_team"]
        away = r["away_team"]
        pred = predictions.get((home, away))

        if pd.notna(r["home_score"]) and pd.notna(r["away_score"]):
            result = f"{int(r['home_score'])}:{int(r['away_score'])}"
            status = "Gespielt"
            status_key = "played_status"
            status_class = "played"
        else:
            result = "offen"
            status = "Offen"
            status_key = "open"
            status_class = "open"

        if pred is not None:
            prognosis = (
                f"{float(pred['home_win']) * 100:.1f}% / "
                f"{float(pred['draw']) * 100:.1f}% / "
                f"{float(pred['away_win']) * 100:.1f}%"
            )
            xg = f"{float(pred['home_xg']):.2f}:{float(pred['away_xg']):.2f}"
        else:
            prognosis = "-"
            xg = "-"

        rows.append(f"""
        <tr>
            <td>{esc(r.get("date", ""))}</td>
            <td class="team">{esc(home)}</td>
            <td class="score">{esc(result)}</td>
            <td class="team">{esc(away)}</td>
            <td>{esc(xg)}</td>
            <td>{esc(prognosis)}</td>
            <td><span class="badge {status_class}" data-i18n="{status_key}">{status}</span></td>
        </tr>
        """)

    return f"""
    <h3 data-i18n="matches">Spiele</h3>
    <table>
        <thead>
            <tr>
                <th data-i18n="date">Datum</th>
                <th data-i18n="team1">Team 1</th>
                <th data-i18n="result">Ergebnis</th>
                <th data-i18n="team2">Team 2</th>
                <th>xG</th>
                <th data-i18n="prediction">Prognose</th>
                <th data-i18n="status">Status</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """
def render_simulation_table(group, df):
    prev_df = load_previous_group_table(group)
    rows = []

    for _, r in df.iterrows():
        team = r["team"]

        prev_rank1 = prev_top2 = prev_points = None
        if prev_df is not None and team in prev_df.index:
            prev_rank1 = prev_df.loc[team, "p_rank1"]
            prev_top2 = prev_df.loc[team, "p_top2"]
            prev_points = prev_df.loc[team, "expected_points"]

        rows.append(f"""
        <tr>
            <td class="team">{esc(team)}</td>
            <td>
                {float(r["expected_points"]):.2f}
                {delta_html(r["expected_points"], prev_points, multiplier=1)}
            </td>
            <td>
                {pct(r["p_rank1"])}
                {delta_html(r["p_rank1"], prev_rank1)}
            </td>
            <td>{pct(r["p_rank2"])}</td>
            <td>{pct(r["p_rank3"])}</td>
            <td>{pct(r["p_rank4"])}</td>
            <td>
                <strong>{pct(r["p_top2"])}</strong>
                {delta_html(r["p_top2"], prev_top2)}
            </td>
        </tr>
        """)

    return f"""
    <h3 data-i18n="simulation">Simulation</h3>
    <table>
        <thead>
            <tr>
                <th data-i18n="team">Team</th>
                <th data-i18n="expected_points">Erwartete Punkte</th>
                <th data-i18n="rank1">Platz 1</th>
                <th data-i18n="rank2">Platz 2</th>
                <th data-i18n="rank3">Platz 3</th>
                <th data-i18n="rank4">Platz 4</th>
                <th data-i18n="top2">Top 2</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """

def render_group_card(group, sim_df, fixtures, predictions):
    group_fixtures = fixtures[fixtures["group"].astype(str) == str(group)]

    return f"""
    <section class="card wide">
        <h2><span data-i18n="group">Gruppe</span> {esc(group)}</h2>
        <div class="grid-two">
            <div>{render_current_table(group_fixtures)}</div>
            <div>{render_matches(group_fixtures, predictions)}</div>
        </div>
        {render_simulation_table(group, sim_df)}
    </section>
    """


def load_previous_knockout():
    _, previous = latest_two_snapshots()
    if previous is None:
        return None

    file = previous / "knockout.csv"
    if not file.exists():
        return None

    return pd.read_csv(file).set_index("team")


def render_knockout_table():
    file = OUTPUT_DIR / "knockout.csv"

    if not file.exists():
        return ""

    df = pd.read_csv(file).sort_values(
        ["p_winner", "p_final", "p_sf"],
        ascending=[False, False, False],
    )

    prev_df = load_previous_knockout()
    rows = []

    for _, r in df.iterrows():
        team = r["team"]

        prev_winner = prev_final = prev_sf = None
        if prev_df is not None and team in prev_df.index:
            prev_winner = prev_df.loc[team, "p_winner"]
            prev_final = prev_df.loc[team, "p_final"]
            prev_sf = prev_df.loc[team, "p_sf"]

        rows.append(f"""
        <tr>
            <td class="team">{esc(team)}</td>
            <td>{pct(r["p_r32"])}</td>
            <td>{pct(r["p_r16"])}</td>
            <td>{pct(r["p_qf"])}</td>
            <td>
                {pct(r["p_sf"])}
                {delta_html(r["p_sf"], prev_sf)}
            </td>
            <td>
                {pct(r["p_final"])}
                {delta_html(r["p_final"], prev_final)}
            </td>
            <td>
                <strong>{pct(r["p_winner"], decimals=2)}</strong>
                {delta_html(r["p_winner"], prev_winner)}
            </td>
        </tr>
        """)

    return f"""
    <section class="card wide">
        <h2 data-i18n="knockout">Finalrunden-Prognose</h2>
        <table>
            <thead>
                <tr>
                    <th data-i18n="team">Team</th>
                    <th data-i18n="r32">R32</th>
                    <th data-i18n="r16">Achtelfinale</th>
                    <th data-i18n="qf">Viertelfinale</th>
                    <th data-i18n="sf">Halbfinale</th>
                    <th data-i18n="final">Finale</th>
                    <th data-i18n="winner">Weltmeister</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </section>
    """


def render_snapshot_info():
    latest, previous = latest_two_snapshots()

    latest_txt = latest.name if latest else "kein Snapshot"
    previous_txt = previous.name if previous else "kein vorheriger Snapshot"

    return f"""
    <section class="card wide meta-card">
        <h2 data-i18n="changes">Veränderungen</h2>
        <p>
            <span data-i18n="latest_snapshot">Aktueller Snapshot</span>:
            <strong>{esc(latest_txt)}</strong>
        </p>
        <p>
            <span data-i18n="previous_snapshot">Vergleich mit</span>:
            <strong>{esc(previous_txt)}</strong>
        </p>
        <p class="legend">
            <span class="delta up">▲</span> verbessert,
            <span class="delta down">▼</span> verschlechtert,
            <span class="delta neutral">±</span> unverändert
        </p>
    </section>
    """


LANG_SCRIPT = """
<script>
const translations = {
    de: {
        subtitle: "Aktuelle Gruppentabellen, Spiele und Monte-Carlo-Prognosen",
        group: "Gruppe",
        current_table: "Aktuelle Tabelle",
        matches: "Spiele",
        prediction: "Prognose",
        simulation: "Simulation",
        knockout: "Finalrunden-Prognose",
        changes: "Veränderungen",
        latest_snapshot: "Aktueller Snapshot",
        previous_snapshot: "Vergleich mit",
        team: "Team",
        prediction: "Prognose",
        played: "Sp",
        wins: "S",
        draws: "U",
        losses: "N",
        goals: "Tore",
        gd: "TD",
        points: "Pkt",
        date: "Datum",
        team1: "Team 1",
        result: "Ergebnis",
        team2: "Team 2",
        status: "Status",
        open: "Offen",
        played_status: "Gespielt",
        expected_points: "Erwartete Punkte",
        rank1: "Platz 1",
        rank2: "Platz 2",
        rank3: "Platz 3",
        rank4: "Platz 4",
        top2: "Top 2",
        r32: "R32",
        r16: "Achtelfinale",
        qf: "Viertelfinale",
        sf: "Halbfinale",
        final: "Finale",
        winner: "Weltmeister",
        footer: "Generiert aus lokalen Simulationsergebnissen. Datenquelle: https://github.com/martj42/international_results"
    },
    en: {
        subtitle: "Current group tables, fixtures and Monte Carlo predictions",
        group: "Group",
        current_table: "Current table",
        matches: "Fixtures",
        prediction: "Prediction",
        simulation: "Simulation",
        knockout: "Knockout stage prediction",
        changes: "Changes",
        latest_snapshot: "Current snapshot",
        previous_snapshot: "Compared with",
        team: "Team",
        prediction: "Prediction",
        played: "P",
        wins: "W",
        draws: "D",
        losses: "L",
        goals: "Goals",
        gd: "GD",
        points: "Pts",
        date: "Date",
        team1: "Team 1",
        result: "Result",
        team2: "Team 2",
        status: "Status",
        open: "Open",
        played_status: "Played",
        expected_points: "Expected points",
        rank1: "Rank 1",
        rank2: "Rank 2",
        rank3: "Rank 3",
        rank4: "Rank 4",
        top2: "Top 2",
        r32: "R32",
        r16: "Round of 16",
        qf: "Quarter-final",
        sf: "Semi-final",
        final: "Final",
        winner: "Champion",
        footer: "Generated from local simulation results. Data source: https://github.com/martj42/international_results"
    }
};

function setLang(lang) {
    document.documentElement.lang = lang;
    localStorage.setItem("reportLang", lang);

    document.querySelectorAll("[data-i18n]").forEach(function(el) {
        const key = el.getAttribute("data-i18n");
        if (translations[lang] && translations[lang][key]) {
            el.textContent = translations[lang][key];
        }
    });

    document.getElementById("btn-de").classList.toggle("active", lang === "de");
    document.getElementById("btn-en").classList.toggle("active", lang === "en");
}

setLang(localStorage.getItem("reportLang") || "de");
</script>
"""


STYLE = """
<style>
body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f4f6f8;
    color: #1f2937;
}

header {
    background: #111827;
    color: white;
    padding: 32px 48px;
}

.header-row {
    display: flex;
    justify-content: space-between;
    gap: 24px;
    align-items: center;
}

header h1 {
    margin: 0;
    font-size: 32px;
}

header p {
    margin: 8px 0 0;
    color: #d1d5db;
}

.lang-switch {
    display: flex;
    gap: 8px;
}
.prediction {
    font-size: 12px;
    line-height: 1.4;
    color: #374151;
    text-align: left;
}

.lang-switch button {
    border: 1px solid #6b7280;
    background: #1f2937;
    color: white;
    border-radius: 999px;
    padding: 8px 14px;
    cursor: pointer;
    font-weight: 700;
}

.lang-switch button.active {
    background: white;
    color: #111827;
}

main {
    padding: 32px 48px;
    display: grid;
    grid-template-columns: 1fr;
    gap: 24px;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    overflow-x: auto;
}

.wide {
    grid-column: 1 / -1;
}

.meta-card {
    background: #eef2ff;
}

.bracket {
    display: grid;
    grid-template-columns: repeat(5, minmax(220px, 1fr));
    gap: 20px;
    align-items: start;
    overflow-x: auto;
}

.bracket-column h3 {
    margin-top: 0;
    font-size: 16px;
    color: #111827;
}

.bracket-match {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 14px;
}

.bracket-match-title {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 8px;
    font-weight: 700;
}

.bracket-team {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 5px 0;
    border-top: 1px solid #e5e7eb;
    font-size: 13px;
}

.bracket-team:first-of-type {
    border-top: none;
}

.bracket-team span {
    font-weight: 600;
}

.bracket-team strong {
    color: #111827;
}

.hint {
    color: #6b7280;
    font-size: 14px;
}

.grid-two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
}

h2 {
    margin-top: 0;
    font-size: 26px;
}

h3 {
    margin-top: 12px;
    font-size: 18px;
    color: #374151;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    margin-bottom: 24px;
}

th {
    text-align: right;
    background: #f3f4f6;
    padding: 10px;
    border-bottom: 1px solid #e5e7eb;
    white-space: nowrap;
}

th:first-child {
    text-align: left;
}

td {
    text-align: right;
    padding: 10px;
    border-bottom: 1px solid #e5e7eb;
    white-space: nowrap;
}

td.team {
    text-align: left;
    font-weight: 600;
}

td.score {
    font-weight: 700;
    text-align: center;
}

tr:hover {
    background: #f9fafb;
}

.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

.badge.played {
    background: #dcfce7;
    color: #166534;
}

.badge.open {
    background: #e0f2fe;
    color: #075985;
}

.delta {
    display: inline-block;
    margin-left: 6px;
    font-size: 12px;
    font-weight: 800;
}

.delta.up {
    color: #15803d;
}

.delta.down {
    color: #b91c1c;
}

.disclaimer {
    font-size: 14px;
    line-height: 1.7;
}

.disclaimer p {
    margin-bottom: 12px;
}

.disclaimer a {
    color: #2563eb;
    text-decoration: none;
}

.disclaimer a:hover {
    text-decoration: underline;
}

.delta.neutral {
    color: #6b7280;
}

.legend {
    color: #4b5563;
}

footer {
    padding: 24px 48px;
    color: #6b7280;
    font-size: 13px;
}

.bracket-slot {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 8px;
    margin-bottom: 8px;
}

.bracket-slot-title {
    font-size: 11px;
    text-transform: uppercase;
    color: #6b7280;
    font-weight: 800;
    margin-bottom: 6px;
}

.versus {
    text-align: center;
    font-size: 11px;
    font-weight: 900;
    color: #6b7280;
    margin: 6px 0;
}

.bracket-matchups {
    background: #eef2ff;
    border-radius: 10px;
    padding: 8px;
    margin-top: 10px;
}

.bracket-team.matchup span {
    font-weight: 500;
}

@media (max-width: 1000px) {
    .grid-two {
        grid-template-columns: 1fr;
    }

    header, main, footer {
        padding-left: 20px;
        padding-right: 20px;
    }

    .header-row {
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>
"""


def build_html():
    groups = load_group_tables()
    fixtures = load_fixtures()
    predictions = load_match_predictions()

    if not groups:
        raise RuntimeError("Keine group_*.csv Dateien im output/-Verzeichnis gefunden.")

    group_sections = "\n".join(
        render_group_card(group, df, fixtures, predictions)
        for group, df in groups
    )

    return f"""<!doctype html>
<html lang="de">
<head>
    <meta charset="utf-8">
    <title>World Cup Predictor Report</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {STYLE}
</head>
<body>
    <header>
        <div class="header-row">
            <div>
                <h1>World Cup Predictor</h1>
                <p data-i18n="subtitle">Aktuelle Gruppentabellen, Spiele und Monte-Carlo-Prognosen</p>
            </div>
            <div class="lang-switch">
                <button onclick="setLang('de')" id="btn-de">DE</button>
                <button onclick="setLang('en')" id="btn-en">EN</button>
            </div>
        </div>
    </header>

    <main>
        {render_snapshot_info()}
        {group_sections}
        {render_bracket_tree()}
        {render_knockout_table()}
    </main>

    <section class="card wide disclaimer">
    <h2>Disclaimer</h2>

    <p>
        This website is a personal hobby project focused on football analytics,
        machine learning and statistical modelling.
    </p>

    <p>
        All predictions shown on this site are generated automatically using
        historical football data, Elo ratings, machine learning models,
        Poisson goal simulations and Monte Carlo tournament simulations.
    </p>

    <p>
        The results are intended solely for educational and entertainment
        purposes and must not be interpreted as betting advice, gambling advice,
        financial advice or professional sports forecasts.
    </p>
    <p>
    This website does not use cookies, tracking technologies,
    analytics services, user accounts or advertising networks.
    No personal data is collected, stored or processed.
    </p>
    <p>
        No guarantee is made regarding the accuracy, completeness or correctness
        of any prediction.
    </p>
</section>

<section class="card wide disclaimer">
    <h2>Impressum</h2>

    <p>
        Betreiber dieser Webseite:
    </p>

    <p>
        Stefan Loewe<br>
        Deutschland
    </p>

    <p>
        Kontakt:<br>
        GitHub:
        <a href="https://github.com/StrikeZW/worldcup-2026-predictor"
           target="_blank">
           worldcup-2026-predictor
        </a>
    </p>

    <p>
        Dies ist ein nicht-kommerzielles Hobbyprojekt.
    </p>
</section>

<footer data-i18n="footer">
    Generated from local simulation results.<br>
    Data sources:
    <a href="https://github.com/martj42/international_results">
        international_results
    </a>
    &
    <a href="https://github.com/openfootball/worldcup.json">
        worldcup.json
    </a>
</footer>

    {LANG_SCRIPT}
</body>
</html>
"""


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_FILE.write_text(build_html(), encoding="utf-8")
    print(f"Report geschrieben: {REPORT_FILE}")


if __name__ == "__main__":
    main()