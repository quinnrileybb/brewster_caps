import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import seaborn as sns

df = pd.read_csv('Brewster_Pitchers_NCAA_2025.csv')

# -------------------------
# Page & Login Configuration
# -------------------------
st.set_page_config(page_title="Brewster Pitchers", layout="wide")
# Persistent small logo in the top-right corner.

st.success("Brewster Whitecaps Pitchers")

# -------------------------
# Sidebar Dropdowns for Team/Position/Player Selection
# -------------------------
teams = ["Brewster"]
col1, col2, col3 = st.columns(3)
with col1:
    selected_team = st.selectbox("Select Team", teams)
with col2:
    position = st.selectbox("Select Position", ["Pitcher", "Batter"])
with col3:
    if position == "Pitcher":
        player_options = df[df["pitcher_cape_team"] == selected_team]["Pitcher"].unique()
    else:
        player_options = df[df["hitter_cape_team"] == selected_team]["Batter"].unique()
    else:
        player_options = df[df["pitcher_cape_team"] == selected_team]["Pitcher"].unique()
    if len(player_options) == 0:
        selected_player = st.selectbox("Select Player", ["No players available"])
    else:
        selected_player = st.selectbox("Select Player", player_options)



st.write(f"You selected Team: **{selected_team}**, Position: **{position}**, Player: **{selected_player}**")


if position == "Batter":
    # Determine and display the batter’s handedness
    sides = (
        df.loc[df["Batter"] == selected_player, "BatterSide"]
          .dropna()
          .unique()
          .tolist()
    )
    if len(sides) == 1:
        handed = sides[0]  # “Left” or “Right”
    elif len(sides) > 1:
        handed = "Switch"
    else:
        handed = "Unknown"
    st.write(f"**Batter Handedness:** {handed}")

elif position == "Pitcher":
    throws = (
        df.loc[df["Pitcher"] == selected_player, "PitcherThrows"]
          .dropna().unique().tolist()
    )
    # (almost always length‐1, but just in case…)
    if len(throws) == 1:
        pt_hand = throws[0]        # "Left" or "Right"
    elif len(throws) > 1:
        pt_hand = "Ambidextrous"
    else:
        pt_hand = "Unknown"
    st.write(f"**Pitcher Throws:** {pt_hand}")

# -------------------------
# Branch: If Batter, show In Progress; if Pitcher, show full section.
# -------------------------
if position == "Pitcher":
    # Create tabs for the Pitcher section.
    tabs = st.tabs(["Data", "Heatmaps", "Visuals", "Pitch Analyzer"])


    # =========================
    # Data Tab
    # =========================
    with tabs[0]:
        st.header("2025")
        
        pitcher_data = df[df["Pitcher"] == selected_player]
        games = pitcher_data["GameID"].nunique()
        games_started = pitcher_data.groupby("GameID")["Inning"].min().eq(1).sum()
        total_outs = pitcher_data["OutsOnPlay"].sum()
        IP = total_outs / 3
        pitcher_data["PlayResultCleaned"] = pitcher_data["PlayResult"].copy()
        pitcher_data.loc[pitcher_data["KorBB"].isin(["Strikeout", "Walk"]), "PlayResultCleaned"] = pitcher_data["KorBB"]
        pitcher_data_clean = pitcher_data[pitcher_data["PlayResultCleaned"] != "Undefined"]
        pa = pitcher_data_clean.shape[0]
        strikeouts = pitcher_data_clean[pitcher_data_clean["PlayResultCleaned"] == "Strikeout"].shape[0]
        walks = pitcher_data_clean[pitcher_data_clean["PlayResultCleaned"] == "Walk"].shape[0]
        hits = pitcher_data_clean[pitcher_data_clean["PlayResultCleaned"].isin(["Single", "HomeRun", "Double", "Triple"])].shape[0]
        K_perc = (strikeouts / pa) * 100 if pa > 0 else 0
        BB_perc = (walks / pa) * 100 if pa > 0 else 0
        WHIP = (walks + hits) / IP if IP != 0 else 0
        HR = pitcher_data[pitcher_data["PlayResultCleaned"] == "HomeRun"].shape[0]
        FIP = ((13 * HR + 3 * walks - 2 * strikeouts) / IP + 3) if IP != 0 else 0
        basic_stats = {
            "Games": [games],
            "Games Started": [games_started],
            "IP": [round(IP, 1)],
            "K%": [round(K_perc, 1)],
            "BB%": [round(BB_perc, 1)],
            "FIP": [round(FIP, 2)]
        }



        stats_df = pd.DataFrame(basic_stats)
        st.table(stats_df)

        # Create CountSituation column.
        # Create CountSituation column.
        def get_count_situation(row):
            if row['Balls'] == 0 and row['Strikes'] == 0:
                return "0-0"
            elif row['Strikes'] == 2:
                return "2Strk"
            else:
                return "Other"

        df_player = pitcher_data.copy()
        df_player['CountSituation'] = df_player.apply(get_count_situation, axis=1)

# Convert the Tilt column to numeric.
# Here we assume that you want to convert a string like "12:45" to 12.45.
        def convert_tilt(tilt_str):
            try:
                parts = tilt_str.split(":")
        # Combine parts as a decimal number.
                return float(parts[0]) + float(parts[1]) / 100.0
            except Exception as e:
                return np.nan

        df_player["Tilt_numeric"] = df_player["Tilt"].apply(convert_tilt)



        # Create extended CountSituation column.
        def get_count_situation_extended(row):
            if row['Balls'] == 0 and row['Strikes'] == 0:
                return "0-0"
            elif row['Strikes'] == 2:
                return "2Strk"
            elif row['Balls'] > row['Strikes']:
                return "Ahead"
            elif row['Strikes'] > row['Balls']:
                return "Behind"
            else:
                return "Other"

        df_player['CountSituation'] = df_player.apply(get_count_situation_extended, axis=1)

# Create a second pitch indicator: 1-0 or 0-1
        df_player['second_pitch'] = (((df_player['Balls'] == 1) & (df_player['Strikes'] == 0)) |
                             ((df_player['Balls'] == 0) & (df_player['Strikes'] == 1)))

# Compute overall totals for each condition by BatterSide.
        total_overall_RHH = df_player[df_player['BatterSide'] == "Right"].shape[0]
        total_overall_LHH = df_player[df_player['BatterSide'] == "Left"].shape[0]
        total_0_0_RHH = df_player[(df_player['BatterSide'] == "Right") & (df_player['CountSituation'] == "0-0")].shape[0]
        total_0_0_LHH = df_player[(df_player['BatterSide'] == "Left") & (df_player['CountSituation'] == "0-0")].shape[0]
        total_second_RHH = df_player[(df_player['BatterSide'] == "Right") & (df_player['second_pitch'])].shape[0]
        total_second_LHH = df_player[(df_player['BatterSide'] == "Left") & (df_player['second_pitch'])].shape[0]
        total_ahead_RHH = df_player[(df_player['BatterSide'] == "Right") & (df_player['CountSituation'] == "Ahead")].shape[0]
        total_ahead_LHH = df_player[(df_player['BatterSide'] == "Left") & (df_player['CountSituation'] == "Ahead")].shape[0]
        total_behind_RHH = df_player[(df_player['BatterSide'] == "Right") & (df_player['CountSituation'] == "Behind")].shape[0]
        total_behind_LHH = df_player[(df_player['BatterSide'] == "Left") & (df_player['CountSituation'] == "Behind")].shape[0]
        total_2Strk_RHH = df_player[(df_player['BatterSide'] == "Right") & (df_player['CountSituation'] == "2Strk")].shape[0]
        total_2Strk_LHH = df_player[(df_player['BatterSide'] == "Left") & (df_player['CountSituation'] == "2Strk")].shape[0]

        all_pitch_total = len(df_player)

# Now compute the pitch usage metrics for each pitch type.
        pitch_usage_data = []
        for pt in sorted(df_player['AutoPitchType'].unique(), key=lambda x: str(x)):
            df_pt = df_player[df_player["AutoPitchType"] == pt]
            count = len(df_pt)
            overall_usage = (count / all_pitch_total) * 100 if all_pitch_total > 0 else np.nan

    # Overall by handedness:
            count_RHH = df_pt[df_pt["BatterSide"] == "Right"].shape[0]
            count_LHH = df_pt[df_pt["BatterSide"] == "Left"].shape[0]
            RHH_overall = (count_RHH / total_overall_RHH * 100) if total_overall_RHH > 0 else np.nan
            LHH_overall = (count_LHH / total_overall_LHH * 100) if total_overall_LHH > 0 else np.nan

    # 0-0 by handedness:
            count_RHH_0_0 = df_pt[(df_pt["BatterSide"] == "Right") & (df_pt["CountSituation"] == "0-0")].shape[0]
            count_LHH_0_0 = df_pt[(df_pt["BatterSide"] == "Left") & (df_pt["CountSituation"] == "0-0")].shape[0]
            RHH_0_0 = (count_RHH_0_0 / total_0_0_RHH * 100) if total_0_0_RHH > 0 else np.nan
            LHH_0_0 = (count_LHH_0_0 / total_0_0_LHH * 100) if total_0_0_LHH > 0 else np.nan

    # Second Pitch by handedness:
            count_RHH_second = df_pt[(df_pt["BatterSide"] == "Right") & (df_pt["second_pitch"])].shape[0]
            count_LHH_second = df_pt[(df_pt["BatterSide"] == "Left") & (df_pt["second_pitch"])].shape[0]
            RHH_second = (count_RHH_second / total_second_RHH * 100) if total_second_RHH > 0 else np.nan
            LHH_second = (count_LHH_second / total_second_LHH * 100) if total_second_LHH > 0 else np.nan

    # Ahead by handedness:
            count_RHH_ahead = df_pt[(df_pt["BatterSide"] == "Right") & (df_pt["CountSituation"] == "Ahead")].shape[0]
            count_LHH_ahead = df_pt[(df_pt["BatterSide"] == "Left") & (df_pt["CountSituation"] == "Ahead")].shape[0]
            RHH_ahead = (count_RHH_ahead / total_ahead_RHH * 100) if total_ahead_RHH > 0 else np.nan
            LHH_ahead = (count_LHH_ahead / total_ahead_LHH * 100) if total_ahead_LHH > 0 else np.nan

    # Behind by handedness:
            count_RHH_behind = df_pt[(df_pt["BatterSide"] == "Right") & (df_pt["CountSituation"] == "Behind")].shape[0]
            count_LHH_behind = df_pt[(df_pt["BatterSide"] == "Left") & (df_pt["CountSituation"] == "Behind")].shape[0]
            RHH_behind = (count_RHH_behind / total_behind_RHH * 100) if total_behind_RHH > 0 else np.nan
            LHH_behind = (count_LHH_behind / total_behind_LHH * 100) if total_behind_LHH > 0 else np.nan

    # 2Strk by handedness:
            count_RHH_2strk = df_pt[(df_pt["BatterSide"] == "Right") & (df_pt["CountSituation"] == "2Strk")].shape[0]
            count_LHH_2strk = df_pt[(df_pt["BatterSide"] == "Left") & (df_pt["CountSituation"] == "2Strk")].shape[0]
            RHH_2strk = (count_RHH_2strk / total_2Strk_RHH * 100) if total_2Strk_RHH > 0 else np.nan
            LHH_2strk = (count_LHH_2strk / total_2Strk_LHH * 100) if total_2Strk_LHH > 0 else np.nan

            pitch_usage_data.append({
                "Pitch Type": pt,
                "Count": count,
                "Overall Usage": round(overall_usage, 1),
                "RHH Overall %": round(RHH_overall, 1),
                "LHH Overall %": round(LHH_overall, 1),
                "RHH 0-0%": round(RHH_0_0, 1),
                "LHH 0-0%": round(LHH_0_0, 1),
                "RHH Second Pitch%": round(RHH_second, 1),
                "LHH Second Pitch%": round(LHH_second, 1),
                "RHH Ahead%": round(RHH_ahead, 1),
                "LHH Ahead%": round(LHH_ahead, 1),
                "RHH Behind%": round(RHH_behind, 1),
                "LHH Behind%": round(LHH_behind, 1),
                "RHH 2Strk%": round(RHH_2strk, 1),
                "LHH 2Strk%": round(LHH_2strk, 1)
            })

        usage_df = pd.DataFrame(pitch_usage_data)
        usage_df = usage_df.sort_values("Count", ascending=False)


# Apply styling to the DataFrame.
        usage_df_styled = usage_df.style.format({
    "Overall Usage": "{:.1f}",
    "RHH Overall %": "{:.1f}",
    "LHH Overall %": "{:.1f}",
    "RHH 0-0%": "{:.1f}",
    "LHH 0-0%": "{:.1f}",
    "RHH Second Pitch%": "{:.1f}",
    "LHH Second Pitch%": "{:.1f}",
    "RHH Ahead%": "{:.1f}",
    "LHH Ahead%": "{:.1f}",
    "RHH Behind%": "{:.1f}",
    "LHH Behind%": "{:.1f}",
    "RHH 2Strk%": "{:.1f}",
    "LHH 2Strk%": "{:.1f}"
})

        st.subheader("Pitch Usage Table")
        st.dataframe(usage_df_styled)
        table1_df = usage_df.copy()

         

        # Third Table: Pitch Metrics
        overall_counts = df_player['AutoPitchType'].value_counts()
        total_pitches = len(df_player)
        overall_usage = (overall_counts / total_pitches * 100).round(1)
        grouped = df_player.groupby("AutoPitchType").agg({
            "RelSpeed": ["mean", "max"],
            "SpinRate": "mean",
            "Tilt_numeric": "mean",
            "InducedVertBreak": "mean",
            "HorzBreak": "mean",
            "VertApprAngle": "mean",
            "HorzApprAngle": "mean",
            "Extension": "mean",
            "RelHeight": "mean",
            "RelSide": "mean"
        })
        grouped.columns = ["MPH", "Top MPH", "RPMs", "Tilt", "IVB", "HB", "VAA", "HAA", "Extension", "RelHeight", "RelSide"]
        grouped["Overall Usage %"] = overall_usage
        grouped = grouped.reset_index().rename(columns={"AutoPitchType": "Pitch Type"})
        grouped = grouped[["Pitch Type", "Overall Usage %", "MPH", "Top MPH", "RPMs", "Tilt", "IVB", "HB", "VAA", "HAA", "Extension", "RelHeight", "RelSide"]]
        for col in ["Overall Usage %", "MPH", "Top MPH", "RPMs", "Tilt", "IVB", "HB", "VAA", "HAA", "Extension", "RelHeight", "RelSide"]:
            grouped[col] = grouped[col].round(1)
        grouped = grouped.sort_values("Overall Usage %", ascending=False)
        st.subheader("Pitch Metrics by Pitch Type")
# Use st.dataframe with a styled DataFrame to format numbers to one decimal place.
        st.dataframe(grouped.style.format({
            "Overall Usage %": "{:.1f}",
            "MPH": "{:.1f}",
            "Top MPH": "{:.1f}",
            "RPMs": "{:.1f}",
            "Tilt": "{:.1f}",
            "IVB": "{:.1f}",
            "HB": "{:.1f}",
            "VAA": "{:.1f}",
            "HAA": "{:.1f}",
            "Extension": "{:.1f}",
            "RelHeight": "{:.1f}",
            "RelSide": "{:.1f}"
        }))

        col1, col2 = st.columns(2)
        with col1:
            handedness_options = ["Combined", "Left", "Right"]
            selected_handedness = st.selectbox("Filter by Batter Handedness", handedness_options)
        with col2:
            count_options = [
                "0 Strikes", "1 Strike", "2 Strikes",
                "0 Balls",   "1 Ball",   "2 Balls",   "3 Balls"
            ]
    # default to all counts checked
            selected_counts = st.multiselect("Filter by Count", count_options, default=count_options)

# ——— Apply the handedness filter ———
        if selected_handedness == "Left":
            df_player_filtered = df_player[df_player["BatterSide"] == "Left"].copy()
        elif selected_handedness == "Right":
            df_player_filtered = df_player[df_player["BatterSide"] == "Right"].copy()
        else:
            df_player_filtered = df_player.copy()

# ——— Apply the count filter ———
# parse out which numeric strikes and balls were selected
        strike_counts = [int(s.split()[0]) for s in selected_counts if "Strike" in s]
        ball_counts  = [int(s.split()[0]) for s in selected_counts if "Ball" in s]

# keep any row that matches either a selected strikes OR a selected balls
        if strike_counts or ball_counts:
            df_player_filtered = df_player_filtered[
                df_player_filtered["Strikes"].isin(strike_counts)
                & df_player_filtered["Balls"].isin(ball_counts)
            ]
        df_player_filtered['play_by_play'] = np.where(df_player_filtered['PlayResult'] != 'Undefined',
                                                       df_player_filtered['PlayResult'],
                                                       df_player_filtered['PitchCall'])
        event_mapping = {
            'StrikeCalled': 'called_strike',
            'Double': 'double',
            'Sacrifice': 'sac_fly',
            'BallCalled': 'ball',
            'HomeRun': 'home_run',
            'StrikeSwinging': 'swinging_strike',
            'Error': 'field_error',
            'error': 'field_error',
            'Out': 'field_out',
            'BallinDirt': 'ball',
            'FoulBall': 'foul',
            'Single': 'single',
            'FieldersChoice': 'fielders_choice',
            'Triple': 'triple',
            'FoulBallNotFieldable': 'foul',
            'StolenBase': None,
            'HitByPitch': 'hit_by_pitch',
            'FoulBallFieldable': 'foul',
            'BallIntentional': 'ball',
            'CaughtStealing': None,
            'AutomaticStrike': 'called_strike'
        }
        weights = {
            'home_run': -1.374328827,
            'triple': -1.05755625,
            'double': -0.766083123,
            'single': -0.467292971,
            'ball': -0.063688329,
            'hit_by_pitch': -0.063688329,
            'blocked_ball': -0.063688329,
            'foul': 0.038050274,
            'foul_tip': 0.038050274,
            'bunt_foul': 0.038050274,
            'bunt_foul_tip': 0.038050274,
            'called_strike': 0.065092516,
            'swinging_strike': 0.118124936,
            'swinging_strike_blocked': 0.118124936,
            'force_out': 0.195568767,
            'grounded_into_double_play': 0.195568767,
            'fielders_choice_out': 0.195568767,
            'fielders_choice': 0.195568767,
            'field_out': 0.195568767,
            'double_play': 0.195568767,
            'sac_fly': 0.236889646,
            'field_error': 0.236889646,
            'sac_fly_double_play': 0.789788814,
            'triple_play': 0.789788814
        }
        def get_run_value(event):
            mapped_event = event_mapping.get(event, None)
            if mapped_event is None:
                return 0
            return weights.get(mapped_event, 0)
        df_player_filtered['run_value'] = df_player_filtered['play_by_play'].apply(get_run_value)

        # Step 2 for Results Table: Compute Metrics & Build Table using filtered data.
        # -------------------------
# Results Table
# -------------------------
# Define sets & zones
        strike_set = {"StrikeSwinging", "StrikeCalled", "AutomaticStrike", 
                      "FoulBallFieldable", "FoulBallNotFieldable", "FoulBall", "InPlay"}
        strike_zone = {"x_min": -0.83, "x_max": 0.83, "z_min": 1.5, "z_max": 3.5}
        swing_set  = {"StrikeSwinging", "InPlay", "FoulBallFieldable", "FoulBallNotFieldable", "FoulBall"}
        chase_set  = swing_set  # same as swing_set for chases
        woba_wts   = {'Out':0,'Walk':0.69,'HitByPitch':0.72,'Single':0.88,'Double':1.247,'Triple':1.578,'HomeRun':2.031}

# Build per-pitch-type rows
        results_list = []
        for pt in df_player_filtered["AutoPitchType"].unique():
            df_pt = df_player_filtered[df_player_filtered["AutoPitchType"] == pt]
            total_pt = len(df_pt)
            if total_pt == 0:
                continue

    # usage & counts
            overall_usage = total_pt / len(df_player_filtered) * 100
            strikes      = df_pt["PitchCall"].isin(strike_set).sum()
            strike_pct   = strikes / total_pt * 100
            df_zone      = df_pt[
                df_pt["PlateLocSide"].between(strike_zone["x_min"], strike_zone["x_max"]) &
                df_pt["PlateLocHeight"].between(strike_zone["z_min"], strike_zone["z_max"])
            ]
            zone_pct     = len(df_zone) / total_pt * 100

            whiffs       = df_pt["PitchCall"].eq("StrikeSwinging").sum()
            swings       = df_pt["PitchCall"].isin(swing_set).sum()
            whiff_pct    = (whiffs / swings * 100) if swings>0 else np.nan

            zone_swings  = df_zone["PitchCall"].isin(swing_set).sum()
            zone_whiffs  = df_zone["PitchCall"].eq("StrikeSwinging").sum()
            zone_whiff_pct = (zone_whiffs / zone_swings * 100) if zone_swings>0 else np.nan

            csw_count    = df_pt["PitchCall"].isin({"StrikeCalled","StrikeSwinging"}).sum()
            csw_pct      = csw_count / total_pt * 100

            df_outzone   = df_pt.drop(df_zone.index)
            chase_count  = df_outzone["PitchCall"].isin(chase_set).sum()
            chase_pct    = (chase_count / len(df_outzone) * 100) if len(df_outzone)>0 else np.nan

            df_inplay    = df_pt[df_pt["PitchCall"]=="InPlay"]
            woba_con     = df_inplay["PlayResult"].map(lambda x: woba_wts.get(x,0)).sum() / len(df_inplay) if len(df_inplay)>0 else np.nan
            hard_hit_pct = df_inplay["ExitSpeed"].ge(95).sum() / len(df_inplay) * 100 if len(df_inplay)>0 else np.nan
            
            if len(df_inplay)>0 and "Angle" in df_inplay:
                df_bb = df_inplay.assign(
                    BattedBallType=lambda d: d["Angle"].apply(
                        lambda la: "GroundBall" if la<10
                                   else "LineDrive" if la<25
                                   else "FlyBall" if la<50
                                   else "Popup"
                    )
                )
                valid = df_bb["BattedBallType"].notna().sum()
                gb = df_bb["BattedBallType"].eq("GroundBall").sum()/valid*100 if valid>0 else np.nan
                ld = df_bb["BattedBallType"].eq("LineDrive").sum()/valid*100 if valid>0 else np.nan
                fb = df_bb["BattedBallType"].eq("FlyBall").sum()/valid*100 if valid>0 else np.nan
            else:
                gb = ld = fb = np.nan
    # batted-ball types
            if "Angle" in df_inplay.columns and len(df_inplay)>0:
                # --- compute overall GB/LD/FB on all InPlay batted balls ---
                df_bb_all = (
                    df_player_filtered[df_player_filtered["PitchCall"]=="InPlay"]
                      .assign(BattedBallType=lambda d: d["Angle"].apply(
                          lambda la: "GroundBall" if la<10
                                     else "LineDrive" if la<25
                                     else "FlyBall" if la<50
                                     else "Popup"
                      ))
                )
                valid_all = df_bb_all["BattedBallType"].notna().sum()
                gb_all = df_bb_all["BattedBallType"].eq("GroundBall").sum()/valid_all*100 if valid_all>0 else np.nan
                ld_all = df_bb_all["BattedBallType"].eq("LineDrive").sum()/valid_all*100 if valid_all>0 else np.nan
                fb_all = df_bb_all["BattedBallType"].eq("FlyBall").sum()/valid_all*100 if valid_all>0 else np.nan


            rv_per_100 = df_pt["run_value"].sum()/total_pt*100 if total_pt>0 else np.nan

    # append row
            results_list.append({
                "Count":           total_pt,
                "Pitch Type":      pt,
                "Overall Usage %": round(overall_usage,1),
                "Strike%":         round(strike_pct,1),
                "Zone%":           round(zone_pct,1),
                "Whiff%":          round(whiff_pct,1) if not np.isnan(whiff_pct) else np.nan,
                "Zone-Whiff%":     round(zone_whiff_pct,1) if not np.isnan(zone_whiff_pct) else np.nan,
                "CSW%":            round(csw_pct,1),
                "Chase%":          round(chase_pct,1) if not np.isnan(chase_pct) else np.nan,
                "Wobacon":         round(woba_con,3) if not np.isnan(woba_con) else np.nan,
                "Hard Hit%":       round(hard_hit_pct,1) if not np.isnan(hard_hit_pct) else np.nan,
                "GB%":             round(gb,1) if not np.isnan(gb) else np.nan,
                "LD%":             round(ld,1) if not np.isnan(ld) else np.nan,
                "FB%":             round(fb,1) if not np.isnan(fb) else np.nan,
                "RV/100":          round(rv_per_100,1) if not np.isnan(rv_per_100) else np.nan
            })

# build DataFrame & sort
        results_df = pd.DataFrame(results_list).sort_values("Overall Usage %", ascending=False)

# compute & append overall-average row
        total_all = len(df_player_filtered)
        strikes_all = df_player_filtered["PitchCall"].isin(strike_set).sum()
        strike_perc_all = strikes_all/total_all*100 if total_all>0 else 0
        csw_all = df_player_filtered["PitchCall"].isin({"StrikeCalled","StrikeSwinging","AutomaticStrike"}).sum()
        csw_perc_all = csw_all/total_all*100 if total_all>0 else 0

        overall_avg = {
            "Count":           total_all,
            "Pitch Type":      "Overall Average",
            "Overall Usage %": 100.0,
            "Strike%":         round(strike_perc_all,1),
            "Zone%":           round(
                                  len(df_player_filtered[
                                      df_player_filtered["PlateLocSide"].between(strike_zone["x_min"],strike_zone["x_max"]) &
                                      df_player_filtered["PlateLocHeight"].between(strike_zone["z_min"],strike_zone["z_max"])
                                  ])/total_all*100,1
                              ) if total_all>0 else np.nan,
            "Whiff%":          round(
                                  df_player_filtered["PitchCall"].eq("StrikeSwinging").sum()/ 
                                  df_player_filtered["PitchCall"].isin(swing_set).sum()*100,1
                              ) if total_all>0 else np.nan,
            "Zone-Whiff%":     round(
                                  df_player_filtered[
                                      df_player_filtered["PlateLocSide"].between(strike_zone["x_min"],strike_zone["x_max"]) &
                                      df_player_filtered["PlateLocHeight"].between(strike_zone["z_min"],strike_zone["z_max"])
                                  ]["PitchCall"].eq("StrikeSwinging").sum() /
                                  df_player_filtered[
                                      df_player_filtered["PlateLocSide"].between(strike_zone["x_min"],strike_zone["x_max"]) &
                                      df_player_filtered["PlateLocHeight"].between(strike_zone["z_min"],strike_zone["z_max"])
                                  ]["PitchCall"].isin(swing_set).sum()*100,1
                              ) if total_all>0 else np.nan,
            "CSW%":            round(csw_perc_all,1),
            "Chase%":          round(
                                  df_player_filtered.drop(df_player_filtered[
                                      df_player_filtered["PlateLocSide"].between(strike_zone["x_min"],strike_zone["x_max"]) &
                                      df_player_filtered["PlateLocHeight"].between(strike_zone["z_min"],strike_zone["z_max"])
                                  ].index)["PitchCall"].isin(chase_set).sum()/
                                  (total_all - len(df_player_filtered[
                                      df_player_filtered["PlateLocSide"].between(strike_zone["x_min"],strike_zone["x_max"]) &
                                      df_player_filtered["PlateLocHeight"].between(strike_zone["z_min"],strike_zone["z_max"])
                                  ]))*100,1
                              ) if total_all>0 else np.nan,
            "Wobacon":         round(
                                  df_player_filtered[df_player_filtered["PitchCall"]=="InPlay"]["PlayResult"]
                                    .map(lambda x: woba_wts.get(x,0)).sum()/
                                  df_player_filtered[df_player_filtered["PitchCall"]=="InPlay"].shape[0],3
                              ) if total_all>0 else np.nan,
            "Hard Hit%":       round(
                                  df_player_filtered[df_player_filtered["PitchCall"]=="InPlay"]
                                    ["ExitSpeed"].ge(95).sum()/
                                  df_player_filtered[df_player_filtered["PitchCall"]=="InPlay"].shape[0]*100,1
                              ) if total_all>0 else np.nan,
            "GB%":             round(gb_all, 1),  # leave blank or compute similarly if you want
            "LD%":             round(gb_all, 1),
            "FB%":             round(gb_all, 1),
            "RV/100":          round(df_player_filtered["run_value"].sum()/total_all*100,1) if total_all>0 else np.nan
        }

        results_df = pd.concat([results_df, pd.DataFrame([overall_avg])], ignore_index=True)

# Re‐order columns so Count & Pitch Type lead
        cols = ["Count","Pitch Type"] + [c for c in results_df.columns if c not in ("Count","Pitch Type")]
        results_df = results_df[cols]

# Style & display
        fmt = {
            "Count":"{:.0f}",
            "Overall Usage %":"{:.1f}",
            "Strike%":"{:.1f}",
            "Zone%":"{:.1f}",
            "Whiff%":"{:.1f}",
            "Zone-Whiff%":"{:.1f}",
            "CSW%":"{:.1f}",
            "Chase%":"{:.1f}",
            "Wobacon":"{:.3f}",
            "Hard Hit%":"{:.1f}",
            "GB%":"{:.1f}",
            "LD%":"{:.1f}",
            "FB%":"{:.1f}",
            "RV/100":"{:.1f}"
        }
        st.subheader("Results")
        st.dataframe(results_df.style.format(fmt))


        # -------------------------
        # Heatmaps Tab
        # -------------------------

        from matplotlib.patches import Rectangle

        with tabs[1]:
            st.header("Pitch Location Heatmaps by Category")

    # — Filters (give each widget a unique key) —
            count_opts = ["0 Strikes","1 Strike","2 Strikes","0 Balls","1 Ball","2 Balls","3 Balls"]
            selected_counts = st.multiselect("Filter by Count", count_opts, default=count_opts, key="heat_counts")
            sc = [int(x.split()[0]) for x in selected_counts if "Strike" in x]
            bc = [int(x.split()[0]) for x in selected_counts if "Ball"   in x]

            throws_opts = ["Combined","Left","Right"]
            throws_sel = st.selectbox("Filter by Batter Handedness", throws_opts, key="heat_throws")

            map_types = ["All Pitches","Whiffs","Hard Hit","Softly Hit","Chases","Called Strikes"]
            map_sel = st.selectbox("Select Heatmap", map_types, key="heat_map_type")

    # — Apply filters to df_player —
            df_h = df_player.copy()
            if sc or bc:
                df_h = df_h[df_h["Strikes"].isin(sc) & df_h["Balls"].isin(bc)]
            if throws_sel != "Combined":
                df_h = df_h[df_h["BatterSide"] == throws_sel]

    # — Define the event subset for each map type —
            if map_sel == "All Pitches":
                df_event = df_h
            elif map_sel == "Whiffs":
                df_event = df_h[df_h["PitchCall"] == "StrikeSwinging"]
            elif map_sel == "Hard Hit":
                hh = (df_h["PitchCall"] == "InPlay") & (df_h["ExitSpeed"] >= 95)
                df_event = df_h[hh]
            elif map_sel == "Softly Hit":
                df_event = df_h[(df_h["PitchCall"] == "InPlay") & (df_h["ExitSpeed"] < 80)]
            elif map_sel == "Chases":
                swing_set = {"StrikeSwinging","InPlay","FoulBallFieldable","FoulBallNotFieldable","FoulBall"}
                in_zone = df_h["PlateLocSide"].between(-0.83,0.83) & df_h["PlateLocHeight"].between(1.5,3.5)
                df_event = df_h[df_h["PitchCall"].isin(swing_set) & ~in_zone]
            else:  # Called Strikes
                df_event = df_h[df_h["PitchCall"] == "StrikeCalled"]

    # — Top-5 pitch types —
            top5 = df_event["AutoPitchType"].value_counts().index.tolist()[:5]

    # — Render five small plots in their own columns —
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    if i < len(top5):
                        pt = top5[i]
                        subset = df_event[df_event["AutoPitchType"] == pt]
                        fig, ax = plt.subplots(figsize=(3, 3))
                        if len(subset) < 3:
                            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
                        else:
                            sns.kdeplot(
                                x=subset["PlateLocSide"],
                                y=subset["PlateLocHeight"],
                                fill=True,
                                levels=5,
                                thresh=0.05,
                                bw_adjust=0.5,
                                cmap="Reds",
                                ax=ax
                            )
                # draw strike zone
                        ax.add_patch(Rectangle((-0.83, 1.5), 1.66, 2.0,
                                               fill=False, edgecolor="black", linewidth=2))
                        ax.set_xlim(-2.5, 2.5)
                        ax.set_ylim(0.5, 5)
                        ax.set_xticks([]); ax.set_yticks([])
                        ax.set_title(pt, fontsize=10)
                        ax.set_xlabel("")   
                        ax.set_ylabel("") 
                        st.pyplot(fig)
                    else:
                        col.empty()


        # Visuals Tab
        # -------------------------
        with tabs[2]:
            st.header("Visuals")

    # Row 1: Movement Plot and Approach Angle Plot.
            col1, col2 = st.columns(2)
    
            with col1:
                st.subheader("Movement Plot")
                plt.figure(figsize=(6,4))
                for pitch_type in table1_df['Pitch Type']:
                    pitch_data = df_player[df_player['AutoPitchType'] == pitch_type]
                    plt.scatter(pitch_data['HorzBreak'], pitch_data['InducedVertBreak'], label=pitch_type, edgecolor='black')
                plt.axhline(0, color='black', linestyle='--')
                plt.axvline(0, color='black', linestyle='--')
                plt.xlim(-25, 25)
                plt.ylim(-25, 25)
                plt.xlabel('Horizontal Break (HB)')
                plt.ylabel('Induced Vertical Break (IVB)')
                plt.title(f'{selected_player} - 2025 Pitch Movement')
                plt.legend(fontsize='small')
                st.pyplot(plt.gcf())
                plt.clf()
    
            with col2:
                st.subheader("Approach Angle Plot")
                plt.figure(figsize=(6,4))
                for pitch_type in table1_df['Pitch Type']:
                    pitch_data = df_player[df_player['AutoPitchType'] == pitch_type]
                    plt.scatter(pitch_data['HorzApprAngle'], pitch_data['VertApprAngle'], label=pitch_type, edgecolor='black')
                plt.axhline(0, color='black', linestyle='--')
                plt.axvline(0, color='black', linestyle='--')
                plt.xlim(-10, 10)
                plt.ylim(-15, 1)
                plt.xlabel('Horizontal Approach Angle (HAA)')
                plt.ylabel('Vertical Approach Angle (VAA)')
                plt.title(f'{selected_player} - 2025 Approach Angles')
                plt.legend(fontsize='small')
                st.pyplot(plt.gcf())
                plt.clf()

    # Row 2: Release Location Plot and Release Speed by Inning Plot.
            col1, col2 = st.columns(2)
    
            with col1:
                st.subheader("Release Location Plot")
                plt.figure(figsize=(6,4))
                for pitch_type in table1_df['Pitch Type']:
                    pitch_data = df_player[df_player['AutoPitchType'] == pitch_type]
                    plt.scatter(pitch_data['RelSide'], pitch_data['RelHeight'], label=pitch_type, edgecolor='black')
                plt.axhline(0, color='black', linestyle='--')
                plt.axvline(0, color='black', linestyle='--')
                plt.xlim(-7.5, 7.5)
                plt.ylim(0, 10)
                plt.xlabel('Release Side')
                plt.ylabel('Release Height')
                plt.title(f'{selected_player} - 2025 Release Location')
                plt.legend(fontsize='small')
                st.pyplot(plt.gcf())
                plt.clf()
    
            with col2:
                st.subheader("Velo by Inning Plot")
                avg_release = df_player.groupby(['Inning', 'AutoPitchType'])['RelSpeed'].mean().reset_index()
                plt.figure(figsize=(6,4))
                for pitch_type in avg_release['AutoPitchType'].unique():
                    subset = avg_release[avg_release['AutoPitchType'] == pitch_type]
                    plt.step(subset['Inning'], subset['RelSpeed'], where='post', label=pitch_type, linewidth=2)
                plt.xlabel('Inning')
                plt.ylabel('Average Release Speed (MPH)')
                plt.title(f'{selected_player} - Avg Release Speed by Inning')
                plt.legend(fontsize='small')
                st.pyplot(plt.gcf())
                plt.clf()
    
            col1, col2 = st.columns(2)
    
            with col1:
                st.subheader("Release Speed Density Plot")
                plt.figure(figsize=(6,4))
                sns.violinplot(x="RelSpeed", y="AutoPitchType", data=df_player, palette="Reds", orient="h")
                plt.xlabel('Release Velocity (MPH)')
                plt.ylabel('Pitch Type')
                plt.title(f'{selected_player} - Release Velocity Distribution')
                st.pyplot(plt.gcf())
                plt.clf()

            df_player_sorted = df_player.sort_values(["GameID", "Inning"])  # Adjust if you have a specific pitch order column
            df_player_sorted['PitchOrder'] = df_player_sorted.groupby(['GameID', 'AutoPitchType']).cumcount() + 1
            df_player_sorted['PitchGroup'] = np.ceil(df_player_sorted['PitchOrder'] / 3).fillna(0).astype(int)
            df_player_sorted['PitchGroup'] = df_player_sorted['PitchGroup'].fillna(0).astype(int)
            avg_velo_by_group = df_player_sorted.groupby(['AutoPitchType', 'PitchGroup'])['RelSpeed'].mean().reset_index()

    
            with col2:
                st.subheader("Velo by Pitch in Outing")
                plt.figure(figsize=(6, 4))
                for pitch_type in avg_velo_by_group['AutoPitchType'].unique():
                    subset = avg_velo_by_group[avg_velo_by_group['AutoPitchType'] == pitch_type]
                    plt.plot(subset['PitchGroup'], subset['RelSpeed'], marker='o', label=pitch_type, linewidth=2)
                plt.xlabel('Pitch Number (Each point is a groupings of 3)')
                plt.ylabel('Average Release Speed (MPH)')
                plt.title(f'{selected_player} - Avg Release Speed by Pitch in Outing')
                plt.legend(fontsize='small')
                st.pyplot(plt.gcf())
                plt.clf()

            col1, col2 = st.columns(2)

            with col2:
                st.subheader("Velo By Date")
                df_player['Date'] = pd.to_datetime(df_player['Date'], errors='coerce')
                df_player_sorted_by_date = df_player.sort_values("Date")
                avg_velo_by_date = df_player_sorted_by_date.groupby(["AutoPitchType", "Date"])["RelSpeed"].mean().reset_index()
                plt.figure(figsize=(6, 4))
                for pitch_type in avg_velo_by_date["AutoPitchType"].unique():
                    subset = avg_velo_by_date[avg_velo_by_date["AutoPitchType"] == pitch_type].sort_values("Date")
                    plt.plot(subset["Date"], subset["RelSpeed"], marker='o', label=pitch_type, linewidth=2)
                plt.xlabel("Date")
                plt.ylabel("Average Release Speed (MPH)")
                plt.title(f"{selected_player} - Avg Release Speed by Date")
                plt.xticks(rotation=45)
                plt.legend(fontsize="small")
                st.pyplot(plt.gcf())
                plt.clf()

            

            with col1: 
                st.subheader("IVB By Date")
                avg_ivb_by_date = df_player_sorted_by_date.groupby(["AutoPitchType", "Date"])["InducedVertBreak"].mean().reset_index()
                plt.figure(figsize=(6, 4))
                for pitch_type in avg_ivb_by_date["AutoPitchType"].unique():
                    subset = avg_ivb_by_date[avg_ivb_by_date["AutoPitchType"] == pitch_type].sort_values("Date")
                    plt.plot(subset["Date"], subset["InducedVertBreak"], marker='o', label=pitch_type, linewidth=2)
                plt.xlabel("Date")
                plt.ylabel("Average Induced Vertical Break")
                plt.title(f"{selected_player} - IVB by Date")
                plt.xticks(rotation=45)
                plt.legend(fontsize="small")
                st.pyplot(plt.gcf())
                plt.clf()  
            
            col1, col2 = st.columns(2)

            with col1: 
                st.subheader("Spin Rate By Date")
                df_player['Date'] = pd.to_datetime(df_player['Date'], errors='coerce')
                df_player_sorted_by_date = df_player.sort_values("Date")
                avg_spin_by_date = df_player_sorted_by_date.groupby(["AutoPitchType", "Date"])["SpinRate"].mean().reset_index()
                plt.figure(figsize=(6, 4))
                for pitch_type in avg_spin_by_date["AutoPitchType"].unique():
                    subset = avg_spin_by_date[avg_spin_by_date["AutoPitchType"] == pitch_type].sort_values("Date")
                    plt.plot(subset["Date"], subset["SpinRate"], marker='o', label=pitch_type, linewidth=2)
                plt.xlabel("Date")
                plt.ylabel("Average Spin Rate")
                plt.title(f"{selected_player} - Avg Spin Rate by Date")
                plt.xticks(rotation=45)
                plt.legend(fontsize="small")
                st.pyplot(plt.gcf())
                plt.clf()

            with col2: 
                st.subheader("Average Spin Rate by Pitch in Outing")
                df_player_sorted = df_player.sort_values(["GameID", "Inning"])  # Adjust or add a pitch order column if available.
                df_player_sorted['PitchOrder'] = df_player_sorted.groupby(['GameID', 'AutoPitchType']).cumcount() + 1
                df_player_sorted['PitchGroup'] = np.ceil(df_player_sorted['PitchOrder'] / 3).fillna(0).astype(int)
                avg_spin_by_group = df_player_sorted.groupby(["AutoPitchType", "PitchGroup"])["SpinRate"].mean().reset_index()
                plt.figure(figsize=(6, 4))
                for pitch_type in avg_spin_by_group["AutoPitchType"].unique():
                    subset = avg_spin_by_group[avg_spin_by_group["AutoPitchType"] == pitch_type]
                    plt.plot(subset["PitchGroup"], subset["SpinRate"], marker='o', label=pitch_type, linewidth=2)
                plt.xlabel("Pitch in outing")
                plt.ylabel("Average Spin Rate")
                plt.title(f"{selected_player} - Avg Spin Rate by Pitch in Outing")
                plt.legend(fontsize="small")
                st.pyplot(plt.gcf())
                plt.clf()

            col1, col2 = st.columns(2)

            with col1: 

                import matplotlib.patches as mpatches
                def assign_color(row):
                    if row["PitchCall"] == "StrikeSwinging":
                        return "yellow"
                    elif row["ExitSpeed"] > 95:
                        return "red"
                    elif row["ExitSpeed"] < 80:
                        return "blue"
                    else:
                        return "white"
                df_player["PlotColor"] = df_player.apply(assign_color, axis=1)
                st.subheader("Results-Based Movement Plot")
                plt.figure(figsize=(8, 6))
                plt.scatter(df_player["HorzBreak"], df_player["InducedVertBreak"], c=df_player["PlotColor"], edgecolors="black")
                plt.axhline(0, color='black', linestyle='--')
                plt.axvline(0, color='black', linestyle='--')
                plt.xlim(-25, 25)
                plt.ylim(-25, 25)
                plt.xlabel("Horizontal Break (HB)")
                plt.ylabel("Induced Vertical Break (IVB)")
                plt.title(f'{selected_player} - 2025 Pitch Movement by Result')
                patch_swing = mpatches.Patch(color="yellow", label="Swinging Strike")
                patch_hard = mpatches.Patch(color="red", label="Hard Hit (>95 mph)")
                patch_soft = mpatches.Patch(color="blue", label="Softly Hit (<80 mph)")
                patch_other = mpatches.Patch(color="white", label="Other")
                plt.legend(handles=[patch_swing, patch_hard, patch_soft, patch_other], fontsize="small")
                st.pyplot(plt.gcf())
                plt.clf()

            with col2:

                # right after df_player is defined in the Data tab, before you do any filtering:
                df_player['play_by_play'] = np.where(
                    df_player['PlayResult'] != 'Undefined',
                    df_player['PlayResult'],
                    df_player['PitchCall']
                )
                df_player['run_value'] = df_player['play_by_play'].apply(get_run_value)

                st.subheader("RV-Based Movement Hexbin")
                fig, ax = plt.subplots(figsize=(6, 6))
                hb = ax.hexbin(
                    df_player["HorzBreak"],
                    df_player["InducedVertBreak"],
                    C=df_player["run_value"],
                    reduce_C_function=np.mean,
                    gridsize=15,
                    cmap="Reds",
                    mincnt=2
                )
                ax.axhline(0, color='black', linestyle='--')
                ax.axvline(0, color='black', linestyle='--')
                ax.set_xlim(-25, 25)
                ax.set_ylim(-25, 25)
                ax.set_xlabel("Horizontal Break (HB)")
                ax.set_ylabel("Induced Vertical Break (IVB)")
                ax.set_title(f"{selected_player} – RV-Weighted Movement")
                st.pyplot(fig)
                plt.clf()


        # #Pitch Analyzer Tab
            with tabs[3]:
                st.header("Pitch Analyzer")

                available_pitches = df_player["AutoPitchType"].dropna().unique()
                selected_pitch_type = st.selectbox("Select Pitch Type", sorted(available_pitches))

                df_selected_pitch = df_player[df_player["AutoPitchType"] == selected_pitch_type].copy()

                st.subheader("Select Feature Ranges")

    # Use columns to reduce height
                slider_cols = st.columns(4)

                def slider_range(df, col, label):
                    if df[col].dropna().empty:
                        return (0, 1)  # fallback
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    return st.slider(
                        f"{label}",
                        min_value=round(min_val - 0.5, 1),
                        max_value=round(max_val + 0.5, 1),
                        value=(round(min_val, 1), round(max_val, 1)),
                        key=label
                    )

    # Column-based sliders
                with slider_cols[0]:
                    velo_range = slider_range(df_selected_pitch, "RelSpeed", "Velocity (MPH)")
                    spin_range = slider_range(df_selected_pitch, "SpinRate", "Spin Rate")


                with slider_cols[1]:
                    ivb_range = slider_range(df_selected_pitch, "InducedVertBreak", "Induced Vertical Break")
                    hb_range = slider_range(df_selected_pitch, "HorzBreak", "Horizontal Break")

                with slider_cols[2]:
                    vaa_range = slider_range(df_selected_pitch, "VertApprAngle", "Vertical Approach Angle")
                    haa_range = slider_range(df_selected_pitch, "HorzApprAngle", "Horizontal Approach Angle")

                with slider_cols[3]:
                    rh_range = slider_range(df_selected_pitch, "RelHeight", "Release Height")
                    rs_range = slider_range(df_selected_pitch, "RelSide", "Release Side")

    # Apply filters
                df_filtered = df_selected_pitch[
                    (df_selected_pitch["RelSpeed"].between(*velo_range)) &
                    (df_selected_pitch["InducedVertBreak"].between(*ivb_range)) &
                    (df_selected_pitch["HorzBreak"].between(*hb_range)) &
                    (df_selected_pitch["SpinRate"].between(*spin_range)) &
                    (df_selected_pitch["VertApprAngle"].between(*vaa_range)) &
                    (df_selected_pitch["HorzApprAngle"].between(*haa_range)) &
                    (df_selected_pitch["RelHeight"].between(*rh_range)) &
                    (df_selected_pitch["RelSide"].between(*rs_range))
                ]

                st.markdown(f"### Filtered Results for `{selected_pitch_type}`")
                st.write(f"Total pitches after filter: {len(df_filtered)}")

                if not df_filtered.empty:
        # Pitch Metrics Table
                    grouped_filtered = pd.DataFrame([{
                        "Pitch Type": selected_pitch_type,
                        "MPH": df_filtered["RelSpeed"].mean(),
                        "Top MPH": df_filtered["RelSpeed"].max(),
                        "RPMs": df_filtered["SpinRate"].mean(),
                        "Tilt": df_filtered["Tilt_numeric"].mean(),
                        "IVB": df_filtered["InducedVertBreak"].mean(),
                        "HB": df_filtered["HorzBreak"].mean(),
                        "VAA": df_filtered["VertApprAngle"].mean(),
                        "HAA": df_filtered["HorzApprAngle"].mean(),
                        "Extension": df_filtered["Extension"].mean(),
                        "RelHeight": df_filtered["RelHeight"].mean(),
                        "RelSide": df_filtered["RelSide"].mean()
                    }])

                    st.subheader("Pitch Metrics")
                    st.dataframe(grouped_filtered.style.format({
                        "MPH": "{:.1f}",
                        "Top MPH": "{:.1f}",
                        "RPMs": "{:.0f}",
                        "Tilt": "{:.1f}",
                        "IVB": "{:.1f}",
                        "HB": "{:.1f}",
                        "VAA": "{:.1f}",
                        "HAA": "{:.1f}",
                        "Extension": "{:.1f}",
                        "RelHeight": "{:.1f}",
                        "RelSide": "{:.1f}"
                    }))

        # Pitch Results Table
                    def classify_batted_ball(la):
                        if pd.isna(la): return np.nan
                        if la < 10: return 'GroundBall'
                        elif la < 25: return 'LineDrive'
                        elif la < 50: return 'FlyBall'
                        else: return 'Popup'

                    whiffs = df_filtered[df_filtered["PitchCall"] == "StrikeSwinging"].shape[0]
                    swings = df_filtered[df_filtered["PitchCall"].isin(["StrikeSwinging", "InPlay", "FoulBallNotFieldable", "FoulBallFieldable", "FoulBall", "AutomaticStrike"])].shape[0]
                    whiff_perc = (whiffs / swings) * 100 if swings > 0 else np.nan

                    csw = df_filtered[df_filtered["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])].shape[0]
                    csw_perc = (csw / len(df_filtered)) * 100

                    strike_zone = {"x_min": -0.83, "x_max": 0.83, "z_min": 1.5, "z_max": 3.5}
                    in_zone = df_filtered[
                        (df_filtered["PlateLocSide"] >= strike_zone["x_min"]) &
                        (df_filtered["PlateLocSide"] <= strike_zone["x_max"]) &
                        (df_filtered["PlateLocHeight"] >= strike_zone["z_min"]) &
                        (df_filtered["PlateLocHeight"] <= strike_zone["z_max"])
                    ]
                    zone_perc = (len(in_zone) / len(df_filtered)) * 100
                    zone_whiffs = in_zone[in_zone["PitchCall"] == "StrikeSwinging"].shape[0]
                    zone_swings = in_zone[in_zone["PitchCall"].isin(["StrikeSwinging", "InPlay", "FoulBallNotFieldable", "FoulBallFieldable", "FoulBall", "AutomaticStrike"])].shape[0]
                    zone_whiff_perc = (zone_whiffs / zone_swings) * 100 if zone_swings > 0 else np.nan

                    df_inplay = df_filtered[df_filtered["PitchCall"] == "InPlay"]
                    woba_weights = {
                        'Out': 0,
                        'Walk': 0.69,
                        'HitByPitch': 0.72,
                        'Single': 0.88,
                        'Double': 1.247,
                        'Triple': 1.578,
                        'HomeRun': 2.031
                    }
                    wobacon = df_inplay["PlayResult"].map(lambda x: woba_weights.get(x, 0)).sum() / len(df_inplay) if len(df_inplay) > 0 else np.nan
                    hard_hit = df_inplay[df_inplay["ExitSpeed"] > 95].shape[0]
                    hard_hit_pct = (hard_hit / len(df_inplay)) * 100 if len(df_inplay) > 0 else np.nan

                    df_inplay = df_inplay.copy()
                    df_inplay["BattedBallType"] = df_inplay["Angle"].apply(classify_batted_ball)
                    total_bip = df_inplay["BattedBallType"].notna().sum()
                    gb = (df_inplay["BattedBallType"] == "GroundBall").sum() / total_bip * 100 if total_bip > 0 else np.nan
                    ld = (df_inplay["BattedBallType"] == "LineDrive").sum() / total_bip * 100 if total_bip > 0 else np.nan
                    fb = (df_inplay["BattedBallType"] == "FlyBall").sum() / total_bip * 100 if total_bip > 0 else np.nan

                    rv_per_100 = (df_filtered["run_value"].sum() / len(df_filtered)) * 100 if len(df_filtered) > 0 else np.nan

                    results_filtered = pd.DataFrame([{
                        "Pitch Type": selected_pitch_type,
                        "Strike%": round((df_filtered["PitchCall"].isin(["StrikeSwinging", "StrikeCalled", "AutomaticStrike", "FoulBallFieldable", "FoulBallNotFieldable", "FoulBall", "InPlay"]).sum() / len(df_filtered)) * 100, 1),
                        "Zone%": round(zone_perc, 1),
                        "Whiff%": round(whiff_perc, 1),
                        "Zone-Whiff%": round(zone_whiff_perc, 1),
                        "CSW%": round(csw_perc, 1),
                        "Wobacon": round(wobacon, 3),
                        "Hard Hit%": round(hard_hit_pct, 1),
                        "GB%": round(gb, 1),
                        "LD%": round(ld, 1),
                        "FB%": round(fb, 1),
                        "RV/100": round(rv_per_100, 1)
                    }])

                    st.subheader("Pitch Results")
                    st.dataframe(results_filtered.style.format({
                        "Strike%": "{:.1f}",
                        "Zone%": "{:.1f}",
                        "Whiff%": "{:.1f}",
                        "Zone-Whiff%": "{:.1f}",
                        "CSW%": "{:.1f}",
                        "Wobacon": "{:.3f}",
                        "Hard Hit%": "{:.1f}",
                        "GB%": "{:.1f}",
                        "LD%": "{:.1f}",
                        "FB%": "{:.1f}",
                        "RV/100": "{:.1f}"
                    }))
