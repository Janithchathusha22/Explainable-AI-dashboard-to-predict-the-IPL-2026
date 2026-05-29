import pandas as pd
import numpy as np
import os
import sys

# Standardize outputs to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

TEAM_MAPPING = {
    'Royal Challengers Bangalore': 'RCB',
    'Royal Challengers Bengaluru': 'RCB',
    'Gujarat Titans': 'GT',
    'Sunrisers Hyderabad': 'SRH',
    'Rajasthan Royals': 'RR',
    'Chennai Super Kings': 'CSK',
    'Mumbai Indians': 'MI',
    'Punjab Kings': 'PBKS',
    'Kings XI Punjab': 'PBKS',
    'Delhi Capitals': 'DC',
    'Delhi Daredevils': 'DC',
    'Kolkata Knight Riders': 'KKR',
    'Lucknow Super Giants': 'LSG',
    'Rising Pune Supergiant': 'RPS',
    'Rising Pune Supergiants': 'RPS',
    'Gujarat Lions': 'GL',
    'Kochi Tuskers Kerala': 'KTK',
    'Deccan Chargers': 'DEC',
    'Pune Warriors': 'PWI'
}

PREDICTION_TARGET_DATE = pd.Timestamp("2026-05-30")
FINAL_MATCH_DATE = pd.Timestamp("2026-05-31")
WEATHER_FILE = "ipl_playoffs_live_weather.csv"
RIGHT_LEFT_FILE = "right_left.csv"
Q1_SCORECARD_FILE = "ipl_2026_q1_rcb_vs_gt.csv"
PLAYOFF_RESULTS_FILE = "ipl_playoffs_2026.csv"


def normalize_team_name(value):
    return TEAM_MAPPING.get(str(value).strip(), str(value).strip())

def normalize_toss_decision(value):
    text = str(value).strip().lower()
    if text in {"bowl", "bowling", "field", "fielding", "field first", "bowl first"}:
        return "field"
    if text in {"bat", "batting", "bat first"}:
        return "bat"
    return text


def venue_key(value):
    text = str(value).lower()
    if "ahmedabad" in text or "narendra modi" in text or "nms" in text:
        return "ahmedabad"
    if "dharamshala" in text or "hpca" in text:
        return "dharamshala"
    if "chandigarh" in text or "yadavindra" in text or "mys" in text:
        return "chandigarh"
    return text.split(",")[0].strip()


def load_weather_summary():
    if not os.path.exists(WEATHER_FILE):
        return {}

    df = pd.read_csv(WEATHER_FILE)
    if df.empty:
        return {}

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["venue_key"] = df["Venue"].apply(venue_key)
    numeric_cols = ["Temperature_C", "Humidity_Pct", "Dew_Point_C", "Rain_Probability_Pct"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.dropna(subset=["Date"])
        .groupby(["venue_key", "Date"], as_index=False)[numeric_cols]
        .mean()
    )
    return {
        (row["venue_key"], row["Date"].date()): row
        for _, row in grouped.iterrows()
    }


def lookup_weather_features(summary, venue, match_date):
    key = venue_key(venue)
    date_value = pd.to_datetime(match_date).date()
    row = summary.get((key, date_value))

    if row is None:
        same_venue = [value for (stored_key, _), value in summary.items() if stored_key == key]
        if same_venue:
            frame = pd.DataFrame(same_venue)
            row = frame[["Temperature_C", "Humidity_Pct", "Dew_Point_C", "Rain_Probability_Pct"]].mean()

    if row is None:
        temp = 30.0
        humidity = 55.0
        dew_point = 18.0
        rain = 10.0
    else:
        temp = float(row["Temperature_C"])
        humidity = float(row["Humidity_Pct"])
        dew_point = float(row["Dew_Point_C"])
        rain = float(row["Rain_Probability_Pct"])

    dew_index = np.clip((humidity / 100.0) * 0.45 + max(dew_point - 15.0, 0.0) / 15.0 * 0.55, 0.0, 1.5)
    heat_index = np.clip((temp - 20.0) / 20.0, 0.0, 1.5)
    return {
        "weather_temperature_c": temp,
        "weather_humidity_pct": humidity,
        "weather_dew_point_c": dew_point,
        "weather_rain_probability_pct": rain,
        "weather_dew_index": float(dew_index),
        "weather_heat_index": float(heat_index),
    }


def load_right_left_summary():
    if not os.path.exists(RIGHT_LEFT_FILE):
        return {}

    df = pd.read_csv(RIGHT_LEFT_FILE)
    if df.empty:
        return {}

    df["team"] = df["Current 2026 Team"].apply(normalize_team_name)
    batting_hand = df["Batting Hand"].fillna("").str.lower()
    bowling_role = df["Bowling Arm / Role"].fillna("").str.lower()
    df["left_hand_batters"] = batting_hand.str.contains("left").astype(int)
    df["right_hand_batters"] = batting_hand.str.contains("right").astype(int)
    df["left_arm_bowlers"] = bowling_role.str.contains("left-arm").astype(int)
    df["right_arm_bowlers"] = bowling_role.str.contains("right-arm").astype(int)

    grouped = df.groupby("team", as_index=False)[
        ["left_hand_batters", "right_hand_batters", "left_arm_bowlers", "right_arm_bowlers"]
    ].sum()
    return {row["team"]: row for _, row in grouped.iterrows()}


def lookup_right_left_features(summary, team1, team2):
    defaults = {
        "left_hand_batters": 0.0,
        "right_hand_batters": 0.0,
        "left_arm_bowlers": 0.0,
        "right_arm_bowlers": 0.0,
    }
    t1 = summary.get(team1, defaults)
    t2 = summary.get(team2, defaults)

    t1_left_bat = float(t1["left_hand_batters"])
    t2_left_bat = float(t2["left_hand_batters"])
    t1_right_bat = float(t1["right_hand_batters"])
    t2_right_bat = float(t2["right_hand_batters"])
    t1_left_bowl = float(t1["left_arm_bowlers"])
    t2_left_bowl = float(t2["left_arm_bowlers"])
    t1_right_bowl = float(t1["right_arm_bowlers"])
    t2_right_bowl = float(t2["right_arm_bowlers"])

    return {
        "team1_left_hand_batters": t1_left_bat,
        "team2_left_hand_batters": t2_left_bat,
        "left_hand_batter_diff": t1_left_bat - t2_left_bat,
        "right_hand_batter_diff": t1_right_bat - t2_right_bat,
        "left_arm_bowler_diff": t1_left_bowl - t2_left_bowl,
        "right_arm_bowler_diff": t1_right_bowl - t2_right_bowl,
        "batting_hand_balance_diff": (t1_left_bat - t1_right_bat) - (t2_left_bat - t2_right_bat),
    }

def load_clean_historical_matches():
    df = pd.read_csv("ipl_matches_clean.csv")
    df['team1'] = df['team1'].map(TEAM_MAPPING).fillna(df['team1'])
    df['team2'] = df['team2'].map(TEAM_MAPPING).fillna(df['team2'])
    df['winner'] = df['winner'].map(TEAM_MAPPING).fillna(df['winner'])
    df['toss_winner'] = df['toss_winner'].map(TEAM_MAPPING).fillna(df['toss_winner'])
    df['toss_decision'] = df['toss_decision'].apply(normalize_toss_decision)
    df['date'] = pd.to_datetime(df['date'])
    return df

def reconstruct_2026_matches_from_deliveries():
    df_del = pd.read_csv("ipl_2026_deliveries.csv")
    reconstructed = []
    
    # We also load recent matches to grab toss info and match metadata if possible
    df_recent = pd.read_csv("ipl_2026_recent_matches.csv")
    df_recent['Home_Team'] = df_recent['Home_Team'].map(TEAM_MAPPING).fillna(df_recent['Home_Team'])
    df_recent['Away_Team'] = df_recent['Away_Team'].map(TEAM_MAPPING).fillna(df_recent['Away_Team'])
    df_recent['Winner'] = df_recent['Winner'].map(TEAM_MAPPING).fillna(df_recent['Winner'])
    df_recent['Toss_Winner'] = df_recent['Toss_Winner'].map(TEAM_MAPPING).fillna(df_recent['Toss_Winner'])
    
    for match_id, group in df_del.groupby("match_id"):
        match_no = int(group['match_no'].iloc[0])
        # We only want to reconstruct matches 44 and above (1-43 are in matches_clean)
        if match_no < 44:
            continue
        
        date_str = group['date'].iloc[0]
        date = pd.to_datetime(date_str)
        venue = group['venue'].iloc[0]
        
        inn1 = group[group['innings'] == 1]
        inn2 = group[group['innings'] == 2]
        
        team1 = inn1['batting_team'].iloc[0] if len(inn1) > 0 else "Unknown"
        team2 = inn2['batting_team'].iloc[0] if len(inn2) > 0 else "Unknown"
        
        team1 = TEAM_MAPPING.get(team1, team1)
        team2 = TEAM_MAPPING.get(team2, team2)
        
        runs1 = inn1['runs_of_bat'].sum() + inn1['extras'].sum()
        runs2 = inn2['runs_of_bat'].sum() + inn2['extras'].sum()
        
        winner = team1 if runs1 > runs2 else team2
        if runs1 == runs2:
            winner = "Tie"
            
        # Try to find match in df_recent to enrich toss info
        recent_match = df_recent[((df_recent['Home_Team'] == team1) & (df_recent['Away_Team'] == team2)) | 
                                 ((df_recent['Home_Team'] == team2) & (df_recent['Away_Team'] == team1))]
        
        if len(recent_match) > 0:
            toss_winner = recent_match['Toss_Winner'].iloc[0]
            toss_decision = normalize_toss_decision(recent_match['Toss_Decision'].iloc[0])
            player_of_match = recent_match['Player_of_Match'].iloc[0]
            winner = recent_match['Winner'].iloc[0]
        else:
            toss_winner = team1
            toss_decision = 'field'
            player_of_match = 'Unknown'
            
        reconstructed.append({
            'match_id': int(match_id),
            'date': date,
            'season': '2026',
            'city': venue.split(',')[1].strip() if ',' in venue else venue,
            'venue': venue,
            'team1': team1,
            'team2': team2,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'winner': winner,
            'player_of_match': player_of_match,
            'match_type': 'T20',
            'overs': 20,
            'balls_per_over': 6
        })
    return pd.DataFrame(reconstructed)

def get_recent_matches_not_in_del():
    # Load recent matches to grab any specific group stage match like Match 69
    df_recent = pd.read_csv("ipl_2026_recent_matches.csv")
    df_recent['Home_Team'] = df_recent['Home_Team'].map(TEAM_MAPPING).fillna(df_recent['Home_Team'])
    df_recent['Away_Team'] = df_recent['Away_Team'].map(TEAM_MAPPING).fillna(df_recent['Away_Team'])
    df_recent['Winner'] = df_recent['Winner'].map(TEAM_MAPPING).fillna(df_recent['Winner'])
    df_recent['Toss_Winner'] = df_recent['Toss_Winner'].map(TEAM_MAPPING).fillna(df_recent['Toss_Winner'])
    
    # We know Match 69 (MI vs RR) and maybe others aren't in deliveries (since deliveries ends at 67)
    # We will manually add Match 69
    match_69 = {
        'match_id': 202669,
        'date': pd.to_datetime('2026-05-23'),
        'season': '2026',
        'city': 'Mumbai',
        'venue': 'Wankhede Stadium, Mumbai',
        'team1': 'MI',
        'team2': 'RR',
        'toss_winner': 'MI',
        'toss_decision': 'field',
        'winner': 'RR',
        'player_of_match': 'Jofra Archer',
        'match_type': 'T20',
        'overs': 20,
        'balls_per_over': 6
    }
    return pd.DataFrame([match_69])


def load_q1_actual_match():
    match = {
        'match_id': 202691,
        'date': pd.to_datetime('2026-05-26'),
        'season': '2026',
        'city': 'Dharamshala',
        'venue': 'HPCA Stadium, Dharamshala',
        'team1': 'RCB',
        'team2': 'GT',
        'toss_winner': 'GT',
        'toss_decision': 'field',
        'winner': 'RCB',
        'player_of_match': 'Rajat Patidar',
        'match_type': 'Semi Final',
        'overs': 20,
        'balls_per_over': 6
    }

    if not os.path.exists(Q1_SCORECARD_FILE):
        return match

    scorecard = pd.read_csv(Q1_SCORECARD_FILE)
    required = {'Taldea', 'Team_Total'}
    if not required.issubset(scorecard.columns):
        return match

    scorecard['team'] = scorecard['Taldea'].apply(normalize_team_name)
    scorecard['Team_Total'] = pd.to_numeric(scorecard['Team_Total'], errors='coerce')
    totals = scorecard.dropna(subset=['Team_Total']).groupby('team')['Team_Total'].max()
    if {'RCB', 'GT'}.issubset(set(totals.index)):
        match['winner'] = 'RCB' if totals['RCB'] > totals['GT'] else 'GT'

    if 'Venue' in scorecard.columns and scorecard['Venue'].notna().any():
        match['venue'] = str(scorecard['Venue'].dropna().iloc[0])
        match['city'] = match['venue'].split(',')[-1].strip()
    if 'Match_Date' in scorecard.columns and scorecard['Match_Date'].notna().any():
        match['date'] = pd.to_datetime(scorecard['Match_Date'].dropna().iloc[0])
    if 'Match_Type' in scorecard.columns and scorecard['Match_Type'].notna().any():
        match['match_type'] = str(scorecard['Match_Type'].dropna().iloc[0])

    return match


def infer_toss_from_margin(winner, margin):
    text = str(margin).lower()
    if "wicket" in text:
        return winner, "field"
    if "run" in text:
        return winner, "bat"
    return winner, "field"


def load_known_playoff_results():
    if not os.path.exists(PLAYOFF_RESULTS_FILE):
        return []

    df = pd.read_csv(PLAYOFF_RESULTS_FILE)
    required = {"Date", "Match_Type", "Team_1", "Team_2", "Venue", "Winner", "Margin"}
    if df.empty or not required.issubset(df.columns):
        return []

    matches = []
    for idx, row in df.iterrows():
        if row[["Team_1", "Team_2", "Winner", "Venue", "Match_Type"]].isna().any():
            continue
        team1 = normalize_team_name(row["Team_1"])
        team2 = normalize_team_name(row["Team_2"])
        winner = normalize_team_name(row["Winner"])
        venue = str(row["Venue"]).strip()
        match_type = str(row["Match_Type"]).strip()
        if not team1 or not team2 or not winner or team1 == "None" or team2 == "None" or winner == "None":
            continue
        if venue.lower() == "none" or "rest" in match_type.lower():
            continue

        toss_winner, toss_decision = infer_toss_from_margin(winner, row["Margin"])
        date = pd.to_datetime(row["Date"])
        matches.append(
            {
                "match_id": 202700 + idx,
                "date": date,
                "season": "2026",
                "city": venue.split(",")[-1].strip() if "," in venue else venue,
                "venue": venue,
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "player_of_match": "Unknown",
                "match_type": match_type,
                "overs": 20,
                "balls_per_over": 6,
            }
        )
    return matches


def load_all_matches():
    df_hist = load_clean_historical_matches()
    df_del_reconstructed = reconstruct_2026_matches_from_deliveries()
    df_extra = get_recent_matches_not_in_del()
    
    q1_actual = load_q1_actual_match()

    # Known playoff matches are actual results. The final row is the one future
    # fixture used for prediction feature generation.
    playoffs = [
        q1_actual,
        *load_known_playoff_results(),
        # Grand Final
        {
            'match_id': 202694,
            'date': FINAL_MATCH_DATE,
            'prediction_as_of': PREDICTION_TARGET_DATE,
            'season': '2026',
            'city': 'Ahmedabad',
            'venue': 'Narendra Modi Stadium, Ahmedabad',
            'team1': 'RCB',
            'team2': 'GT',
            'toss_winner': 'RCB',
            'toss_decision': 'field',
            'winner': 'RCB', # target label (dummy for feature creation)
            'player_of_match': 'Unknown',
            'match_type': 'Final',
            'overs': 20,
            'balls_per_over': 6
        }
    ]
    df_playoffs = pd.DataFrame(playoffs)
    
    df_all = pd.concat([
        df_hist[df_hist['season'] != '2026'],
        df_hist[df_hist['season'] == '2026'],
        df_del_reconstructed,
        df_extra,
        df_playoffs
    ], ignore_index=True)
    
    # Drop duplicates by match_id (keep original if there are overlapping ids)
    df_all = df_all.drop_duplicates(subset=['season', 'team1', 'team2', 'date'], keep='first')
    df_all = df_all.sort_values("date").reset_index(drop=True)
    return df_all

# Calculate features relative to a match date
def calculate_match_features(df_all):
    feature_rows = []
    weather_summary = load_weather_summary()
    right_left_summary = load_right_left_summary()
    
    # Pre-calculate a fast lookup of match winners
    # We will iterate through matches and compute features from the past
    for idx, row in df_all.iterrows():
        match_date = row['date']
        team1 = row['team1']
        team2 = row['team2']
        venue = row['venue']
        season = row['season']
        prediction_as_of = pd.to_datetime(row.get('prediction_as_of', match_date), errors='coerce')
        if pd.isna(prediction_as_of):
            prediction_as_of = match_date
        
        # Historical slice: matches known before the prediction date.
        # For future fixtures this avoids leaking later playoff results into the prediction row.
        df_past = df_all[df_all['date'] < prediction_as_of]
        
        # Helper to compute win rate
        def get_win_rate(team, df_slice):
            team_matches = df_slice[(df_slice['team1'] == team) | (df_slice['team2'] == team)]
            if len(team_matches) == 0:
                return 0.5
            wins = len(team_matches[team_matches['winner'] == team])
            return wins / len(team_matches)
        
        # Helper for recent form (last 5 matches)
        def get_recent_form(team, df_slice):
            team_matches = df_slice[(df_slice['team1'] == team) | (df_slice['team2'] == team)].tail(5)
            if len(team_matches) == 0:
                return 0.5
            wins = len(team_matches[team_matches['winner'] == team])
            return wins / len(team_matches)
        
        # Helper for win rate in finals
        def get_finals_win_rate(team, df_slice):
            finals = df_slice[df_slice['match_type'].str.lower().str.contains("final", na=False)]
            team_finals = finals[(finals['team1'] == team) | (finals['team2'] == team)]
            if len(team_finals) == 0:
                return 0.0
            wins = len(team_finals[team_finals['winner'] == team])
            return wins / len(team_finals)
        
        # Helper for win rate at venue
        def get_venue_win_rate(team, v, df_slice):
            # Try matching venue name containing venue string
            v_matches = df_slice[df_slice['venue'].str.lower().str.contains(v.split(',')[0].lower(), na=False)]
            team_v_matches = v_matches[(v_matches['team1'] == team) | (v_matches['team2'] == team)]
            if len(team_v_matches) == 0:
                return get_win_rate(team, df_slice) # fallback
            wins = len(team_v_matches[team_v_matches['winner'] == team])
            return wins / len(team_v_matches)
        
        # Head to head win rate
        h2h = df_past[((df_past['team1'] == team1) & (df_past['team2'] == team2)) | 
                      ((df_past['team1'] == team2) & (df_past['team2'] == team1))]
        if len(h2h) == 0:
            h2h_win_rate = 0.5
        else:
            h2h_win_rate = len(h2h[h2h['winner'] == team1]) / len(h2h)
            
        # Win rate chasing vs batting first
        def get_toss_bat_first_win_rate(team, df_slice):
            # team batted first if:
            # (toss_winner == team and toss_decision == bat) OR (toss_winner != team and toss_decision == field)
            bat_first_matches = df_slice[
                ((df_slice['toss_winner'] == team) & (df_slice['toss_decision'] == 'bat')) |
                ((df_slice['toss_winner'] != team) & (df_slice['toss_decision'] == 'field') & ((df_slice['team1'] == team) | (df_slice['team2'] == team)))
            ]
            if len(bat_first_matches) == 0:
                return 0.5
            wins = len(bat_first_matches[bat_first_matches['winner'] == team])
            return wins / len(bat_first_matches)
            
        def get_chasing_win_rate(team, df_slice):
            # team chased if:
            # (toss_winner == team and toss_decision == field) OR (toss_winner != team and toss_decision == bat)
            chase_matches = df_slice[
                ((df_slice['toss_winner'] == team) & (df_slice['toss_decision'] == 'field')) |
                ((df_slice['toss_winner'] != team) & (df_slice['toss_decision'] == 'bat') & ((df_slice['team1'] == team) | (df_slice['team2'] == team)))
            ]
            if len(chase_matches) == 0:
                return 0.5
            wins = len(chase_matches[chase_matches['winner'] == team])
            return wins / len(chase_matches)

        # Computations
        t1_overall = get_win_rate(team1, df_past)
        t2_overall = get_win_rate(team2, df_past)
        
        t1_recent = get_recent_form(team1, df_past)
        t2_recent = get_recent_form(team2, df_past)
        
        t1_finals = get_finals_win_rate(team1, df_past)
        t2_finals = get_finals_win_rate(team2, df_past)
        
        t1_venue = get_venue_win_rate(team1, venue, df_past)
        t2_venue = get_venue_win_rate(team2, venue, df_past)
        
        t1_chasing = get_chasing_win_rate(team1, df_past)
        t2_chasing = get_chasing_win_rate(team2, df_past)
        
        t1_bat_first = get_toss_bat_first_win_rate(team1, df_past)
        t2_bat_first = get_toss_bat_first_win_rate(team2, df_past)
        
        # Toss and ground effect calculations for the final venue
        # Narendra Modi Stadium stats (or the specific match venue)
        is_ahmedabad = 'ahmedabad' in venue.lower() or 'narendra modi' in venue.lower()
        if is_ahmedabad:
            toss_corr = 0.641  # from toss & ground effects file
            bat_first_venue_win_rate = 0.359
            avg_1st_score = 192.0
            dew_factor = 1.0  # high
            pitch_type = 0.5  # balanced/turn
            boundary_size = 1.2  # large
        else:
            toss_corr = 0.55
            bat_first_venue_win_rate = 0.45
            avg_1st_score = 170.0
            dew_factor = 0.5
            pitch_type = 0.5
            boundary_size = 1.0

        decision_win_rate = toss_corr if normalize_toss_decision(row['toss_decision']) == 'field' else bat_first_venue_win_rate
        if row['toss_winner'] == team1:
            team1_toss_advantage = decision_win_rate - 0.5
        elif row['toss_winner'] == team2:
            team1_toss_advantage = 0.5 - decision_win_rate
        else:
            team1_toss_advantage = 0.0
            
        # Team Momentum & Experience features
        # net run rate final from standings
        # GT: 0.695, SRH: 0.524, RCB: 0.783, RR: 0.189, etc.
        nrr_map = {'RCB': 0.783, 'GT': 0.695, 'SRH': 0.524, 'RR': 0.189, 'PBKS': 0.309, 'DC': -0.651, 'KKR': -0.147, 'CSK': -0.345, 'MI': -0.584, 'LSG': -0.74}
        t1_nrr = nrr_map.get(team1, 0.0)
        t2_nrr = nrr_map.get(team2, 0.0)
        
        # captain form rating (Shubman Gill: Excellent -> 0.9, Pat Cummins: Very Good -> 0.8, Faf: Good -> 0.7, Sanju: Average -> 0.5)
        cap_form_map = {'GT': 0.9, 'SRH': 0.8, 'RCB': 0.7, 'RR': 0.5}
        t1_cap_form = cap_form_map.get(team1, 0.6)
        t2_cap_form = cap_form_map.get(team2, 0.6)
        
        # team experience years (GT: 4, SRH: 6, RCB: 8, RR: 7)
        exp_map = {'GT': 4, 'SRH': 6, 'RCB': 8, 'RR': 7}
        t1_exp = exp_map.get(team1, 5)
        t2_exp = exp_map.get(team2, 5)
        
        # path reached after Qualifier 2: RCB and GT are confirmed finalists.
        path_map = {'RCB': 1.0, 'GT': 1.0, 'SRH': 0.0, 'RR': 0.0}
        t1_path = path_map.get(team1, 0.5)
        t2_path = path_map.get(team2, 0.5)
        
        # target label: 1 if team1 wins, 0 if team2 wins
        label = 1 if row['winner'] == team1 else 0
        
        # Add micro features from deliveries if 2025/2026 data is available
        # In a real environment, we'd compute these dynamically. Let's pre-aggregate average phase scores for teams
        # We will use static mapping representing 2026 season stats computed from the deliveries file
        # GT batting: PP 51.5, Middle 82.3, Death 55.4. Bowling: PP Wkts 1.5, Death Wkts 2.2, Economy 8.59
        # SRH batting: PP 58.2, Middle 89.1, Death 61.2. Bowling: PP Wkts 1.1, Death Wkts 1.8, Economy 9.12
        pp_bat_map = {'GT': 51.5, 'SRH': 58.2, 'RCB': 54.1, 'RR': 53.0}
        mid_bat_map = {'GT': 82.3, 'SRH': 89.1, 'RCB': 80.5, 'RR': 84.2}
        death_bat_map = {'GT': 55.4, 'SRH': 61.2, 'RCB': 50.8, 'RR': 58.9}
        dot_bat_map = {'GT': 35.0, 'SRH': 32.5, 'RCB': 36.2, 'RR': 34.1}
        
        pp_bowl_wkts_map = {'GT': 1.5, 'SRH': 1.1, 'RCB': 1.4, 'RR': 1.6}
        death_bowl_wkts_map = {'GT': 2.2, 'SRH': 1.8, 'RCB': 2.0, 'RR': 1.9}
        bowl_econ_map = {'GT': 8.59, 'SRH': 9.12, 'RCB': 8.07, 'RR': 8.77}
        weather_features = lookup_weather_features(weather_summary, venue, match_date)
        hand_features = lookup_right_left_features(right_left_summary, team1, team2)
        
        row_features = {
            'match_id': row['match_id'],
            'date': match_date,
            'season': season,
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'winner_label': label,
            
            # Historical win rate features
            'overall_win_rate_diff': t1_overall - t2_overall,
            'win_rate_last_5_diff': t1_recent - t2_recent,
            'win_rate_finals_diff': t1_finals - t2_finals,
            'win_rate_venue_diff': t1_venue - t2_venue,
            'head_to_head_win_rate': h2h_win_rate,
            'win_rate_chasing_diff': t1_chasing - t2_chasing,
            'win_rate_bat_first_diff': t1_bat_first - t2_bat_first,
            
            # Venue and ground effects
            'toss_win_match_win_corr': toss_corr,
            'bat_first_win_rate_venue': bat_first_venue_win_rate,
            'team1_toss_advantage': team1_toss_advantage,
            'avg_1st_inn_score_venue': avg_1st_score,
            'dew_factor_impact': dew_factor,
            'pitch_type_encoded': pitch_type,
            'boundary_size_factor': boundary_size,
            
            # Momentum & leadership
            'nrr_diff': t1_nrr - t2_nrr,
            'cap_form_diff': t1_cap_form - t2_cap_form,
            'team_exp_diff': t1_exp - t2_exp,
            'playoff_stage_path_diff': t1_path - t2_path,
            
            # Phase micro features
            'pp_bat_avg_diff': pp_bat_map.get(team1, 50.0) - pp_bat_map.get(team2, 50.0),
            'mid_bat_avg_diff': mid_bat_map.get(team1, 80.0) - mid_bat_map.get(team2, 80.0),
            'death_bat_avg_diff': death_bat_map.get(team1, 50.0) - death_bat_map.get(team2, 50.0),
            'dot_ball_bat_diff': dot_bat_map.get(team1, 35.0) - dot_bat_map.get(team2, 35.0),
            'pp_bowl_wkts_diff': pp_bowl_wkts_map.get(team1, 1.2) - pp_bowl_wkts_map.get(team2, 1.2),
            'death_bowl_wkts_diff': death_bowl_wkts_map.get(team1, 1.8) - death_bowl_wkts_map.get(team2, 1.8),
            'bowl_economy_diff': bowl_econ_map.get(team1, 8.5) - bowl_econ_map.get(team2, 8.5)
        }
        row_features.update(weather_features)
        row_features.update(hand_features)
        feature_rows.append(row_features)
        
    return pd.DataFrame(feature_rows)

def build_player_impact_scores():
    # Load stats files
    df_playoff_players = pd.read_csv("ipl_2026_playoff_players_full_stats.csv")
    df_top_batsmen = pd.read_csv("ipl_2026_top_batsmen.csv")
    df_top_bowlers = pd.read_csv("ipl_2026_top_bowlers.csv")
    
    player_scores = []
    
    # Calculate a score 0-100 for each player
    for idx, row in df_playoff_players.iterrows():
        name = row['Player_Name']
        team = row['Team']
        role = row['Role']
        
        # Default starting values
        batting_impact = 0.0
        bowling_impact = 0.0
        
        # 1. Batting performance impact
        if role in ['Batsman', 'All-Rounder', 'Wicketkeeper-Batsman']:
            runs = row['Batting_Runs']
            sr = row['Batting_Strike_Rate']
            sixes = row['Sixes_Hit']
            
            # Find in top batsmen list to grab averages
            top_bat = df_top_batsmen[df_top_batsmen['Player'] == name]
            if len(top_bat) > 0:
                avg = top_bat['Average'].iloc[0]
                hundreds = top_bat['Hundreds'].iloc[0]
                fifties = top_bat['Fifties'].iloc[0]
            else:
                avg = 35.0 # default
                hundreds = 0
                fifties = 2
                
            # Batting impact score calculation
            batting_impact = (runs / 10.0) * 0.4 + (sr / 3.0) * 0.3 + (avg) * 0.2 + (sixes) * 0.1
            
        # 2. Bowling performance impact
        if role in ['Bowler', 'All-Rounder']:
            wickets = row['Wickets_Taken']
            econ = row['Bowling_Economy']
            
            # Find in top bowlers list
            top_bowl = df_top_bowlers[df_top_bowlers['Player'] == name]
            if len(top_bowl) > 0:
                avg_bowl = top_bowl['Average'].iloc[0]
            else:
                avg_bowl = 25.0
                
            # Bowling impact score calculation
            # Lower economy and lower average is better
            bowling_impact = (wickets * 3.5) * 0.5 + (20.0 / (econ + 1.0)) * 25.0 * 0.3 + (50.0 / (avg_bowl + 1.0)) * 25.0 * 0.2
            
        # Overall Player Impact Score (0-100)
        if role == 'All-Rounder':
            overall_score = (batting_impact + bowling_impact) / 1.5
        elif role in ['Batsman', 'Wicketkeeper-Batsman']:
            overall_score = batting_impact
        else:
            overall_score = bowling_impact
            
        # Scale to 0-100 range and clamp
        overall_score = min(max(overall_score * 1.1, 10.0), 99.5)
        
        player_scores.append({
            'Player_Name': name,
            'Team': team,
            'Role': role,
            'Batting_Runs': row['Batting_Runs'],
            'Batting_Strike_Rate': row['Batting_Strike_Rate'],
            'Wickets_Taken': row['Wickets_Taken'],
            'Bowling_Economy': row['Bowling_Economy'],
            'Player_Impact_Score': overall_score
        })
        
    return pd.DataFrame(player_scores)

if __name__ == "__main__":
    print("Loading and preparing matches...")
    df_all_matches = load_all_matches()
    print(f"Total matches prepared: {len(df_all_matches)}")
    print(df_all_matches['season'].value_counts())
    
    print("\nCalculating match-level features...")
    df_features = calculate_match_features(df_all_matches)
    print(f"Features matrix shape: {df_features.shape}")
    df_features.to_csv("match_features.csv", index=False)
    print("Saved match_features.csv")
    
    print("\nBuilding player impact scores...")
    df_players = build_player_impact_scores()
    df_players.to_csv("player_impact_scores.csv", index=False)
    print("Saved player_impact_scores.csv")
    print(df_players.sort_values("Player_Impact_Score", ascending=False).head(10))
