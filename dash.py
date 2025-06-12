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
    position = st.selectbox("Select Position", ["Batter", "Pitcher"])
with col3:
    if position == "Batter":
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
if position == "Batter":
    st.header("Hitters Section")
    
    # Create tabs for the hitter section: Data, Heatmaps, Visuals, Models.
    tabs = st.tabs(["Data", "Heatmaps", "Visuals", "Pitch Level Analyzer"])
    
    with tabs[0]:
        st.subheader("2025 Hitting Data")
        # --- Filter Data for the Selected Hitter ---
        batter_data = df[df["Batter"] == selected_player].copy()
    
        # --- Create PlayResultCleaned Column ---
        # Start with PlayResult, but replace with KorBB if it indicates Strikeout or Walk.
        batter_data["PlayResultCleaned"] = batter_data["PlayResult"].copy()
        batter_data.loc[batter_data["KorBB"].isin(["Strikeout", "Walk"]), "PlayResultCleaned"] = batter_data["KorBB"]
    
        # Filter out rows with "Undefined" in PlayResultCleaned.
        batter_data_clean = batter_data[batter_data["PlayResultCleaned"] != "Undefined"]
        
        # --- Add Dropdown to Filter Hitting Data by PitcherThrows ---
        # ----- in with tabs[0] under the Batter branch -----

# Load & clean your batter_data as before…

# ——— Filters side-by-side ———

        hitting_data_clean = batter_data_clean.copy()
        col1, col2 = st.columns(2)
        with col1:
            hitting_filter = st.selectbox(
                "Filter By Pitcher Handedness",
                ["Combined", "Left", "Right"]
            )
        with col2:
            count_options = [
                "0 Strikes", "1 Strike", "2 Strikes",
                "0 Balls",   "1 Ball",   "2 Balls",   "3 Balls"
            ]
            selected_counts = st.multiselect(
                "Filter By Count",
                count_options,
                default=count_options
            )
        
# — apply handedness —
        if hitting_filter == "Left":
            hitting_data_clean = hitting_data_clean[hitting_data_clean["PitcherThrows"] == "Left"]
        elif hitting_filter == "Right":
            hitting_data_clean = hitting_data_clean[hitting_data_clean["PitcherThrows"] == "Right"]
        else:
            hitting_data_clean = hitting_data_clean.copy()

# — apply count filter —
        strike_counts = [int(s.split()[0]) for s in selected_counts if "Strike" in s]
        ball_counts   = [int(s.split()[0]) for s in selected_counts if "Ball"  in s]

        hitting_data_clean = hitting_data_clean[
            hitting_data_clean["Strikes"].isin(strike_counts)
            & hitting_data_clean["Balls"].isin(ball_counts)
        ]

# now use df_filt in place of hitting_data_clean for all stats below…
# …etc

    
        # --- Calculate Plate Appearances (PA) and At Bats (AB) ---
        PA = hitting_data_clean.shape[0]
        # AB: Exclude walks, hit-by-pitch, and sacrifices.
        AB = hitting_data_clean[~hitting_data_clean["PlayResultCleaned"].isin(["Walk", "HitByPitch", "Sacrifice"])].shape[0]
    
        # --- Count Hits and Home Runs ---
        hits = hitting_data_clean[hitting_data_clean["PlayResultCleaned"].isin(["Single", "Double", "Triple", "HomeRun"])].shape[0]
        HR = hitting_data_clean[hitting_data_clean["PlayResultCleaned"] == "HomeRun"].shape[0]
    
        # --- Calculate Batting Average (BA) ---
        BA = hits / AB if AB > 0 else 0
    
        # --- Calculate On-Base Percentage (OBP) ---
        walks = hitting_data_clean[hitting_data_clean["PlayResultCleaned"] == "Walk"].shape[0]
        HBP = hitting_data_clean[hitting_data_clean["PlayResultCleaned"] == "HitByPitch"].shape[0]
        OBP = (hits + walks + HBP) / PA if PA > 0 else 0
    
        # --- Calculate Slugging Percentage (SLG) and OPS ---
        singles = hitting_data_clean[hitting_data_clean["PlayResultCleaned"] == "Single"].shape[0]
        doubles = hitting_data_clean[hitting_data_clean["PlayResultCleaned"] == "Double"].shape[0]
        triples = hitting_data_clean[hitting_data_clean["PlayResultCleaned"] == "Triple"].shape[0]
        total_bases = singles + 2*doubles + 3*triples + 4*HR
        SLG = total_bases / AB if AB > 0 else 0
        OPS = OBP + SLG
    
        # --- Calculate Weighted On-Base Average (wOBA) ---
        woba_weights = {
            "Out": 0.00,
            "Strikeout": 0.00,
            "Walk": 0.69,
            "HitByPitch": 0.72,
            "Single": 0.88,
            "Double": 1.247,
            "Triple": 1.578,
            "HomeRun": 2.031
        }
        woba_numerator = hitting_data_clean["PlayResultCleaned"].apply(lambda x: woba_weights.get(x, 0)).sum()
        wOBA = woba_numerator / PA if PA > 0 else 0
    
        # --- Compute Strikeout and Walk Rates (percentage of PA) ---
        strikeouts = hitting_data_clean[hitting_data_clean["PlayResultCleaned"] == "Strikeout"].shape[0]
        K_rate = (strikeouts / PA) * 100 if PA > 0 else 0
        BB_rate = (walks / PA) * 100 if PA > 0 else 0
    
        # --- Compute League Average wOBA (for all batters) for wOBA+ ---
        all_batters = df[df["PlayResult"] != "Undefined"].copy()
        all_batters["PlayResultCleaned"] = all_batters["PlayResult"].copy()
        all_batters.loc[all_batters["KorBB"].isin(["Strikeout", "Walk"]), "PlayResultCleaned"] = all_batters["KorBB"]
        all_batters_clean = all_batters[all_batters["PlayResultCleaned"] != "Undefined"]
        league_PA = all_batters_clean.shape[0]
        
    
        # --- Build and Style the Hitter Stats Table ---
        hitter_stats = {
            "PA": [PA],
            "AB": [AB],
            "Hits": [hits],
            "HR": [HR],
            "BA": [round(BA, 3)],
            "OBP": [round(OBP, 3)],
            "SLG": [round(SLG, 3)],
            "OPS": [round(OPS, 3)],
            "wOBA": [round(wOBA, 3)],
            "K Rate (%)": [round(K_rate, 1)],
            "BB Rate (%)": [round(BB_rate, 1)]
        }
        hitter_stats_df = pd.DataFrame(hitter_stats)
        hitter_stats_df_styled = hitter_stats_df.style.format({
            "PA": "{:.0f}",
            "AB": "{:.0f}",
            "Hits": "{:.0f}",
            "HR": "{:.0f}",
            "BA": "{:.3f}",
            "OBP": "{:.3f}",
            "SLG": "{:.3f}",
            "OPS": "{:.3f}",
            "wOBA": "{:.3f}",
            "K Rate (%)": "{:.1f}",
            "BB Rate (%)": "{:.1f}"
        })
        st.dataframe(hitter_stats_df_styled)
        
        
        st.subheader("Plate Discipline")

# Base pitch-level df
        pitch_data = batter_data.copy()

# ——— Filters side-by-side ———
        col1, col2 = st.columns(2)
        with col1:
            discipline_option = st.selectbox(
                "Plate Discipline - Filter By Pitcher Handedness",
                ["Combined", "Left", "Right"]
            )
        with col2:
            count_options = [
                "0 Strikes", "1 Strike", "2 Strikes",
                "0 Balls",   "1 Ball",   "2 Balls",   "3 Balls"
            ]
            selected_counts = st.multiselect(
                "Plate Discipline - Filter By Count",
                count_options,
                default=count_options
            )

# ——— Apply handedness filter ———
        pd_data = pitch_data.copy()
        if discipline_option == "Left":
            pd_data = pd_data[pd_data["PitcherThrows"] == "Left"]
        elif discipline_option == "Right":
            pd_data = pd_data[pd_data["PitcherThrows"] == "Right"]

# ——— Apply count filter ———
        strike_counts = [int(s.split()[0]) for s in selected_counts if "Strike" in s]
        ball_counts   = [int(s.split()[0]) for s in selected_counts if "Ball"  in s]

        pd_data = pd_data[
            pd_data["Strikes"].isin(strike_counts) &
            pd_data["Balls"].isin(ball_counts)
        ]

    
        # Define strike zone boundaries.
        strike_zone = {"x_min": -0.83, "x_max": 0.83, "z_min": 1.5, "z_max": 3.5}
    
        # Define event sets.
        swing_set = {"StrikeSwinging", "InPlay", "FoulBallFieldable", "FoulBallNotFieldable", "FoulBall"}
        contact_set = {"InPlay", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"}
    
        # Define categories.
        categories = ["Overall", "Fastball", "Breaking Ball", "Offspeed", "90+", "High Spin"]
    
        def get_category_data(data, category):
            if category == "Fastball":
                return data[(data["AutoPitchType"].isin(["Four-Seam", "Sinker"])) | (data["RelSpeed"] > 85)]
            elif category == "90+":
                return data[data["RelSpeed"] >= 90]
            elif category == "High Spin":
                return data[data["SpinRate"] > 2400]
            elif category == "Breaking Ball":
                return data[data["AutoPitchType"].isin(["Slider", "Curveball", "Cutter"])]
            elif category == "Offspeed":
                return data[data["AutoPitchType"].isin(["Splitter", "Changeup"])]
            elif category == "Overall":
                return data
            else:
                return pd.DataFrame()
    
        pd_rows = []
        for cat in categories:
            cat_data = get_category_data(pd_data, cat)
            total_pitches = cat_data.shape[0]
            if total_pitches == 0:
                pd_rows.append({
                    "Category": cat,
                    "Count": 0,
                    "Balls in Play": 0,
                    "Swing %": np.nan,
                    "Z-Swing %": np.nan,
                    "Chase %": np.nan,
                    "Whiff %": np.nan,
                    "Z-Whiff %": np.nan
                })
                continue
    
            balls_in_play = cat_data[cat_data["PitchCall"].isin(contact_set)].shape[0]
            swings_total = cat_data[cat_data["PitchCall"].isin(swing_set)].shape[0]
            swing_pct = (swings_total / total_pitches) * 100
            bip_total = cat_data[cat_data["PitchCall"] == "InPlay"].shape[0]
    
            in_zone = cat_data[
                (cat_data["PlateLocSide"] >= strike_zone["x_min"]) &
                (cat_data["PlateLocSide"] <= strike_zone["x_max"]) &
                (cat_data["PlateLocHeight"] >= strike_zone["z_min"]) &
                (cat_data["PlateLocHeight"] <= strike_zone["z_max"])
            ]
            z_count = in_zone.shape[0]
            z_swings = in_zone[in_zone["PitchCall"].isin(swing_set)].shape[0]
            z_swing_pct = (z_swings / z_count) * 100 if z_count > 0 else np.nan
    
            out_zone = cat_data[~(
                (cat_data["PlateLocSide"] >= strike_zone["x_min"]) &
                (cat_data["PlateLocSide"] <= strike_zone["x_max"]) &
                (cat_data["PlateLocHeight"] >= strike_zone["z_min"]) &
                (cat_data["PlateLocHeight"] <= strike_zone["z_max"])
            )]
            o_count = out_zone.shape[0]
            o_swings = out_zone[out_zone["PitchCall"].isin(swing_set)].shape[0]
            chase_pct = (o_swings / o_count) * 100 if o_count > 0 else np.nan
    
            whiff_count = swings_total - cat_data[cat_data["PitchCall"].isin(contact_set)].shape[0]
            whiff_pct = (whiff_count / swings_total) * 100 if swings_total > 0 else np.nan
    
            in_zone_contact = in_zone[in_zone["PitchCall"].isin(contact_set)].shape[0]
            z_whiff_count = z_swings - in_zone_contact
            z_whiff_pct = (z_whiff_count / z_swings) * 100 if z_swings > 0 else np.nan
    
            pd_rows.append({
                "Category": cat,
                "Count": total_pitches,
                "Balls in Play": bip_total,
                "Swing %": round(swing_pct, 1),
                "Z-Swing %": round(z_swing_pct, 1) if not np.isnan(z_swing_pct) else np.nan,
                "Chase %": round(chase_pct, 1) if not np.isnan(chase_pct) else np.nan,
                "Whiff %": round(whiff_pct, 1) if not np.isnan(whiff_pct) else np.nan,
                "Z-Whiff %": round(z_whiff_pct, 1) if not np.isnan(z_whiff_pct) else np.nan
            })
    
        plate_discipline_df = pd.DataFrame(pd_rows)
    
        # Define a style function that bolds the "Overall" row.
        def bold_overall(row):
            return ['font-weight: bold' if row['Category'] == 'Overall' else '' for _ in row]
    
        plate_discipline_styled = plate_discipline_df.style.apply(bold_overall, axis=1).format({
            "Count": "{:.0f}",
            "Balls in Play": "{:.0f}",
            "Swing %": "{:.1f}",
            "Z-Swing %": "{:.1f}",
            "Chase %": "{:.1f}",
            "Whiff %": "{:.1f}",
            "Z-Whiff %": "{:.1f}"
        })
        st.dataframe(plate_discipline_styled)

        st.subheader("Batted Ball Direction")

        bb_data = batter_data.copy()
        bb_data = bb_data[bb_data["PitchCall"] == "InPlay"]

# Add a dropdown filter by PitcherThrows (Combined, Left, Right)
        bb_filter = st.selectbox("Batted Ball Data - Filter by PitcherThrows", ["Combined", "Left", "Right"])
        if bb_filter == "Left":
            bb_data = bb_data[bb_data["PitcherThrows"] == "Left"].copy()
        elif bb_filter == "Right":
            bb_data = bb_data[bb_data["PitcherThrows"] == "Right"].copy()

        def get_category_data(data, category):
            if category == "Fastball":
                return data[(data["AutoPitchType"].isin(["Four-Seam", "Sinker"])) | (data["RelSpeed"] > 85)]
            elif category == "90+":
                return data[data["RelSpeed"] >= 90]
            elif category == "High Spin":
                return data[data["SpinRate"] > 2400]
            elif category == "Breaking Ball":
                return data[data["AutoPitchType"].isin(["Slider", "Curveball", "Cutter"])]
            elif category == "Offspeed":
                return data[data["AutoPitchType"].isin(["Splitter", "Changeup"])]
            elif category == "Overall":
                return data
            else:
                return pd.DataFrame()

        categories = ["Overall", "Fastball", "Breaking Ball", "Offspeed", "90+", "High Spin"]

# Define a simple classifier function for batted ball type based on the Angle column.
        def classify_batted_ball(la):
            if pd.isna(la):
                return np.nan
            if la < 10:
                return 'GroundBall'
            elif la < 25:
                return 'LineDrive'
            elif la < 50:
                return 'FlyBall'
            else:
                return 'Popup'

# Build a list to collect category-specific metrics.
        bb_rows = []
        for cat in categories:
            cat_data = get_category_data(bb_data, cat)
            total_bb = cat_data.shape[0]

            if total_bb > 0: 
                valid_angles = cat_data["Angle"].dropna()
                if valid_angles.empty:
                    avg_la = np.nan
                    la_sweetspot_pct = np.nan
                else: 
                    avg_la = valid_angles.mean()
                    
                    sweetspot_count = valid_angles.between(8, 32).sum()
                    la_sweetspot_pct = (sweetspot_count / valid_angles.count()) * 100

            else: 
                    avg_la = np.nan
                    la_sweetspot_pct = np.nan 

            gb_pct = ld_pct = fb_pct = np.nan
            if total_bb > 0:
                cat_data = cat_data.copy()  # avoid SettingWithCopyWarning
                cat_data["BattedBallType"] = cat_data["Angle"].apply(classify_batted_ball)
        # Consider only rows that returned a valid batted ball type.
                valid_bb = cat_data[cat_data["BattedBallType"].notna()]
                total_valid = valid_bb.shape[0]
                if total_valid > 0:
                    gb_count = (valid_bb["BattedBallType"] == "GroundBall").sum()
                    ld_count = (valid_bb["BattedBallType"] == "LineDrive").sum()
                    fb_count = (valid_bb["BattedBallType"] == "FlyBall").sum()
                    gb_pct = (gb_count / total_valid) * 100
                    ld_pct = (ld_count / total_valid) * 100
                    fb_pct = (fb_count / total_valid) * 100

    # Set directional percentages as NaN (as requested)
            pull_pct = center_pct = oppo_pct = pulled_fb_pct = np.nan

            bb_rows.append({
                "Category": cat,
                "Count": total_bb,
                "Avg LA": round(avg_la, 1) if not np.isnan(avg_la) else np.nan,
                "LA Sweetspot%": round(la_sweetspot_pct, 1) if not np.isnan(la_sweetspot_pct) else np.nan,
                "GB%": round(gb_pct, 1) if not np.isnan(gb_pct) else np.nan,
                "LD%": round(ld_pct, 1) if not np.isnan(ld_pct) else np.nan,
                "FB%": round(fb_pct, 1) if not np.isnan(fb_pct) else np.nan,
                "Pull%": pull_pct,
                "Center%": center_pct,
                "Oppo%": oppo_pct,
                "Pulled FB%": pulled_fb_pct
            })

        bb_df = pd.DataFrame(bb_rows)
        bb_df_styled = bb_df.style.format({
            "Count": "{:.0f}",
            "Avg LA": "{:.1f}",
            "LA Sweetspot%": "{:.1f}",
            "GB%": "{:.1f}",
            "LD%": "{:.1f}",
            "FB%": "{:.1f}",
            "Pull%": "{:.1f}",
            "Center%": "{:.1f}",
            "Oppo%": "{:.1f}",
            "Pulled FB%": "{:.1f}"
        })
        st.dataframe(bb_df_styled)

        st.subheader("Exit Velocity")
        ev_data = batter_data.copy()
        ev_data = ev_data[ev_data["PitchCall"] == "InPlay"]


        ev_filter = st.selectbox("Exit Velocity Data - Filter by Pitcher Handedness", ["Combined", "Left", "Right"])
        if ev_filter == "Left":
            ev_data = ev_data[ev_data["PitcherThrows"] == "Left"].copy()
        elif ev_filter == "Right":
            ev_data = ev_data[ev_data["PitcherThrows"] == "Right"].copy()

# Define your categories
        categories = ["Overall", "Fastball", "Breaking Ball", "Offspeed", "90+", "High Spin"]

# The same helper function used in your earlier code to segregate data by pitch type/speed.
        def get_category_data(data, category):
            if category == "Fastball":
                return data[(data["AutoPitchType"].isin(["Four-Seam", "Sinker"])) | (data["RelSpeed"] > 85)]
            elif category == "90+":
                return data[data["RelSpeed"] >= 90]
            elif category == "High Spin":
                return data[data["SpinRate"] > 2400]
            elif category == "Breaking Ball":
                return data[data["AutoPitchType"].isin(["Slider", "Curveball", "Cutter"])]
            elif category == "Offspeed":
                return data[data["AutoPitchType"].isin(["Splitter", "Changeup"])]
            elif category == "Overall":
                return data
            else:
                return pd.DataFrame()


        woba_weights = {
            "Out": 0.00,
            "Single": 0.88,
            "Double": 1.247,
            "Triple": 1.578,
            "HomeRun": 2.031
        }


        ev_rows = []
        for cat in categories:
            cat_data = get_category_data(ev_data, cat)
            count = cat_data.shape[0]
            if count > 0 and "ExitSpeed" in cat_data.columns:
        # Average exit velocity
                avg_ev = cat_data["ExitSpeed"].mean()
        # Calculate 90th percentile exit velocity (make sure to drop missing values)
                ev_nonnull = cat_data["ExitSpeed"].dropna()
                p90_ev = np.percentile(ev_nonnull, 90) if len(ev_nonnull) > 0 else np.nan
        # Maximum exit velocity
                max_ev = cat_data["ExitSpeed"].max()
        # Hard Hit%: count events with EV >= 95 (you can adjust the threshold as needed)
                hard_hits = cat_data[cat_data["ExitSpeed"] >= 95].shape[0]
                hard_hit_pct = (hard_hits / count) * 100
        # wOBAcon: average the weights applied to each PlayResultCleaned outcome
                woba_num = cat_data["PlayResultCleaned"].apply(lambda x: woba_weights.get(x, 0)).sum()
                woba_con = woba_num / count
            else:
                avg_ev, p90_ev, max_ev, hard_hit_pct, woba_con = np.nan, np.nan, np.nan, np.nan, np.nan

            ev_rows.append({
                "Category": cat,
                "Count": count,
                "Avg EV": avg_ev,
                "90th Percentile EV": p90_ev,
                "Max EV": max_ev,
                "Hard Hit%": hard_hit_pct,
                "wOBAcon": woba_con
            })

# Create a DataFrame to display the table
        ev_df = pd.DataFrame(ev_rows)
        ev_df_styled = ev_df.style.format({
            "Count": "{:.0f}",
            "Avg EV": "{:.1f}",
            "90th Percentile EV": "{:.1f}",
            "Max EV": "{:.1f}",
            "Hard Hit%": "{:.1f}",
            "wOBAcon": "{:.3f}"
        })
        st.dataframe(ev_df_styled)

    from matplotlib.patches import Rectangle
    
    with tabs[1]:
        st.header("Heatmaps")
        st.markdown("#### Pitch Heatmaps by Category")

    # base df
        df_player = batter_data.copy().dropna(subset=["PlateLocSide","PlateLocHeight"])

    # ——— Filters ———
        col1, col2, col3 = st.columns(3)
        with col1:
        # Count filter
            count_options = [
                "0 Strikes","1 Strike","2 Strikes",
                "0 Balls","1 Ball","2 Balls","3 Balls"
            ]
            selected_counts = st.multiselect("Filter by Count", count_options, default=count_options)
        with col2:
        # Handedness filter
            handed_opts = ["Overall","RHP","LHP"]
            handed_sel = st.selectbox("Filter by Pitcher Handedness", handed_opts)
        with col3:
        # Heatmap type filter
            map_opts = ["Swings","Whiffs","Hard Hit","Softly Hit","Chases","Called Strikes"]
            map_sel = st.selectbox("Select Heatmap", map_opts)

    # apply count filter
        strike_counts = [int(x.split()[0]) for x in selected_counts if "Strike" in x]
        ball_counts   = [int(x.split()[0]) for x in selected_counts if "Ball"   in x]
        df_player = df_player[
            df_player["Strikes"].isin(strike_counts) |
            df_player["Balls"].isin(ball_counts)
        ]

    # apply handedness
        if handed_sel == "RHP":
            df_player = df_player[df_player["PitcherThrows"]=="Right"]
        elif handed_sel == "LHP":
            df_player = df_player[df_player["PitcherThrows"]=="Left"]

    # define row filter for the selected heatmap
        swing_set = {"StrikeSwinging","InPlay","FoulBallFieldable","FoulBallNotFieldable","FoulBall"}
        row_filters = {
            "Swings":       lambda df: df[df["PitchCall"].isin(swing_set)],
            "Whiffs":       lambda df: df[df["PitchCall"]=="StrikeSwinging"],
            "Hard Hit":     lambda df: df[(df["PitchCall"]=="InPlay") & (df["ExitSpeed"]>95)],
            "Softly Hit":   lambda df: df[(df["PitchCall"]=="InPlay") & (df["ExitSpeed"]<80)],
            "Chases":       lambda df: df[
                                 df["PitchCall"].isin(swing_set) &
                                 ~((df["PlateLocSide"].between(-0.83,0.83)) &
                                   (df["PlateLocHeight"].between(1.5,3.5)))
                             ],
            "Called Strikes": lambda df: df[df["PitchCall"]=="StrikeCalled"]
        }
        df_event = row_filters[map_sel](df_player)

    # column filters (pitch categories)
        col_filters = {
            "Overall":      lambda df: df,
            "Fastball":     lambda df: df[(df["AutoPitchType"].isin(["Four-Seam","Sinker"])) | (df["RelSpeed"]>85)],
            "Breaking Ball":lambda df: df[df["AutoPitchType"].isin(["Slider","Curveball","Cutter"])],
            "Offspeed":     lambda df: df[df["AutoPitchType"].isin(["Splitter","Changeup"])]
        }
        col_names = list(col_filters.keys())

    # 1×4 grid
        fig, axs = plt.subplots(1, len(col_names), figsize=(len(col_names)*3, 3), constrained_layout=True)
        for i, cat in enumerate(col_names):
            ax = axs[i]
            df_cell = col_filters[cat](df_event)[["PlateLocSide","PlateLocHeight"]].dropna().astype(float)
            if len(df_cell) < 3:
                ax.text(0.5,0.5,"No Data",ha="center",va="center",transform=ax.transAxes)
            else:
                sns.kdeplot(
                    data=df_cell, x="PlateLocSide", y="PlateLocHeight",
                    ax=ax, fill=True, cmap="Reds", bw_adjust=0.5, levels=5, thresh=0.05
                )
        # draw strike zone
            rect = Rectangle(
                (-0.83,1.5), 1.66, 2.0,
                fill=False, edgecolor="black", linewidth=2
            )
            ax.add_patch(rect)
            ax.set(xlim=(-2.5,2.5), ylim=(0.5,5), xticks=[], yticks=[], xlabel="", ylabel="")
            ax.set_title(cat, fontsize=10)

        st.pyplot(fig)


    with tabs[2]:
        st.header("Visuals")
        
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Launch Angle Density")
        # Create a figure for Launch Angle Density
            fig_la, ax_la = plt.subplots(figsize=(6, 4))
        # Plot the density for launch angle (assumed to be in the 'Angle' column)
            sns.kdeplot(
                data=batter_data, 
                x="Angle", 
                ax=ax_la, 
                color="red", 
                linewidth=1,
                fill=False
            )
            ax_la.set_xlabel("Launch Angle")
            ax_la.set_ylabel("Density")
            ax_la.set_xlim(-50, 100)
            st.pyplot(fig_la)
            
        
        with col2:
            st.subheader("Exit Velocity Density")
        # Create a figure for Exit Velocity Density
            fig_ev, ax_ev = plt.subplots(figsize=(6, 4))
        # Plot the density for exit velocity (assumed to be in the 'EV' column)
            sns.kdeplot(
                data=batter_data, 
                x="ExitSpeed", 
                ax=ax_ev, 
                color="red", 
                linewidth=1,
                fill=False
            )
            ax_ev.set_xlabel("Exit Velocity")
            ax_ev.set_ylabel("Density")
            st.pyplot(fig_ev)

        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1: 
            st.subheader("Exit Velocity vs Launch Angle by Outcome")
    # pick off-ball events
            vis_df = batter_data.copy()
            vis_df = vis_df[vis_df["PlayResultCleaned"]
                     .isin(["Out","Single","Double","Triple","HomeRun"])]
            vis_df = vis_df.dropna(subset=["ExitSpeed","Angle"])

    # map doubles+triples → “ExtraBase”
            vis_df["OutcomeGroup"] = vis_df["PlayResultCleaned"].map(
                lambda x: "ExtraBase" if x in ["Double","Triple"] else x
            )

            color_map = {
                "Out":       "blue",
                "Single":    "yellow",
                "ExtraBase": "orange",
                "HomeRun":   "green"
            }

            plt.figure(figsize=(6,4))
            for outcome, color in color_map.items():
                sub = vis_df[vis_df["OutcomeGroup"] == outcome]
                plt.scatter(
                    sub["Angle"],
                    sub["ExitSpeed"],
                    label=outcome,
                    color=color,
                    alpha=0.6,
                    s=20
                )

            plt.xlim(-100, 100) 
            plt.ylim(0, 120)
            plt.xlabel("Launch Angle")
            plt.ylabel("Exit Velocity (mph)")
            plt.title("EV vs LA by Batted-Ball Outcome")
            plt.legend(title="Outcome", fontsize="small", frameon=False)
            st.pyplot(plt.gcf())
            plt.clf()

        with row2_col2:            
            st.subheader("Pitch Movement & Outcome")
            mv = batter_data.copy().dropna(subset=["HorzBreak","InducedVertBreak"])

            def classify(row):
                if row["PitchCall"] == "StrikeSwinging":
                    return "Whiff"
                if row["PitchCall"] == "InPlay" and row.get("ExitSpeed", 0) < 80:
                    return "SoftHit"
                if row["PlayResultCleaned"] in ["Double","Triple","HomeRun"] or row.get("ExitSpeed",0) >= 95:
                    return "Hard Hit/XBH"
                return None

            mv["Outcome"] = mv.apply(classify, axis=1)
            mv = mv.dropna(subset=["Outcome"])
            colormap = {"SoftHit":"blue","Whiff":"red","Hard Hit/XBH":"green"}

            fig, ax = plt.subplots(figsize=(6,4))
            for outcome, color in colormap.items():
                sub = mv[mv["Outcome"] == outcome]
                ax.scatter(sub["HorzBreak"], sub["InducedVertBreak"],
                           c=color, s=20, alpha=0.7, label=outcome)
            ax.axhline(0, linestyle="--", color="black")
            ax.axvline(0, linestyle="--", color="black")
            ax.set_ylim(-25, 25)
            ax.set_xlim(-25, 25)
            ax.set(xlabel="Horizontal Break", ylabel="Induced Vertical Break")
            ax.legend(title="Outcome", fontsize="small", frameon=False)
            st.pyplot(fig)
            plt.clf()
            



        with tabs[3]:
            st.header("Pitch Level Analyzer")

    # Base df
            df_pl = batter_data.copy()

    # --- Three dropdown filters: Pitch Category, Pitcher Throws, Count ---
            cat_col, hand_col, count_col = st.columns(3)
            with cat_col:
                category_options = ["Overall", "Fastball", "Breaking Ball", "Offspeed"]
                selected_category = st.selectbox("Filter by Pitch Category", category_options)
            with hand_col:
                throws_options = ["Combined", "Left", "Right"]
                selected_throws = st.selectbox("Filter by Pitcher Throws", throws_options)
            with count_col:
                count_options = ["0 Strikes", "1 Strike", "2 Strikes", "0 Balls", "1 Ball", "2 Balls", "3 Balls"]
                selected_counts = st.multiselect("Filter by Count", count_options, default=count_options, key="hitter_counts")

    # apply category filter
            if selected_category == "Fastball":
                df_pl = df_pl[(df_pl["AutoPitchType"].isin(["Four-Seam","Sinker"])) | (df_pl["RelSpeed"] > 85)]
            elif selected_category == "Breaking Ball":
                df_pl = df_pl[df_pl["AutoPitchType"].isin(["Slider","Curveball","Cutter"])]
            elif selected_category == "Offspeed":
                df_pl = df_pl[df_pl["AutoPitchType"].isin(["Splitter","Changeup"])]
    # else "Overall" leaves df_pl unchanged

    # apply throws filter
            if selected_throws == "Left":
                df_pl = df_pl[df_pl["PitcherThrows"] == "Left"]
            elif selected_throws == "Right":
                df_pl = df_pl[df_pl["PitcherThrows"] == "Right"]

    # apply count filter (AND logic)
            sc = [int(x.split()[0]) for x in selected_counts if "Strike" in x]
            bc = [int(x.split()[0]) for x in selected_counts if "Ball"   in x]
            if sc or bc:
                df_pl = df_pl[df_pl["Strikes"].isin(sc) & df_pl["Balls"].isin(bc)]

    # --- Six pitcher-level sliders ---
            def slider_range(df, col, label):
                vals = df[col].dropna()
                if vals.empty:
                    return (0, 1)
                lo, hi = float(vals.min()), float(vals.max())
                return st.slider(label, round(lo - 0.5, 1), round(hi + 0.5, 1), (round(lo, 1), round(hi, 1)))

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                velo_range = slider_range(df_pl, "RelSpeed", "Velocity (MPH)")
                spin_range = slider_range(df_pl, "SpinRate", "Spin Rate")
            with s2:
                ivb_range  = slider_range(df_pl, "InducedVertBreak", "Induced Vertical Break")
                hb_range   = slider_range(df_pl, "HorzBreak", "Horizontal Break")
            with s3:
                vaa_range  = slider_range(df_pl, "VertApprAngle", "Vertical Approach Angle")
                haa_range  = slider_range(df_pl, "HorzApprAngle", "Horizontal Approach Angle")
            with s4:
                rh_range   = slider_range(df_pl, "RelHeight", "Release Height")
                rs_range   = slider_range(df_pl, "RelSide", "Release Side")

    # apply slider filters
            df_pl = df_pl[
                df_pl["RelSpeed"].between(*velo_range) &
                df_pl["SpinRate"].between(*spin_range) &
                df_pl["InducedVertBreak"].between(*ivb_range) &
                df_pl["HorzBreak"].between(*hb_range) &
                df_pl["VertApprAngle"].between(*vaa_range) &
                df_pl["HorzApprAngle"].between(*haa_range) &
                df_pl["RelHeight"].between(*rh_range) &
                df_pl["RelSide"].between(*rs_range)
            ]

    # --- Plate Discipline Table ---
                # --- Basic Hitting Stats ---

    # Clean and filter
            stats_df = df_pl.copy()
            stats_df["PlayResultCleaned"] = stats_df["PlayResult"]
            stats_df.loc[stats_df["KorBB"].isin(["Strikeout","Walk"]), "PlayResultCleaned"] = stats_df["KorBB"]
            stats_df = stats_df[stats_df["PlayResultCleaned"] != "Undefined"]

    # Compute rate stats
            PA      = len(stats_df)
            AB      = stats_df[~stats_df["PlayResultCleaned"].isin(["Walk","HitByPitch","Sacrifice"])].shape[0]
            hits    = stats_df["PlayResultCleaned"].isin(["Single","Double","Triple","HomeRun"]).sum()
            HRs     = stats_df["PlayResultCleaned"].eq("HomeRun").sum()
            BA      = hits/AB if AB>0 else 0
            walks   = stats_df["PlayResultCleaned"].eq("Walk").sum()
            HBP     = stats_df["PlayResultCleaned"].eq("HitByPitch").sum()
            OBP     = (hits+walks+HBP)/PA if PA>0 else 0
            singles = stats_df["PlayResultCleaned"].eq("Single").sum()
            doubles = stats_df["PlayResultCleaned"].eq("Double").sum()
            triples = stats_df["PlayResultCleaned"].eq("Triple").sum()
            total_bases = singles + 2*doubles + 3*triples + 4*HRs
            SLG     = total_bases/AB if AB>0 else 0
            OPS     = OBP + SLG

            woba_wts = {
                "Out":0.00, "Strikeout":0.00, "Walk":0.69, "HitByPitch":0.72,
                "Single":0.88, "Double":1.247, "Triple":1.578, "HomeRun":2.031
            }
            woba_num = stats_df["PlayResultCleaned"].map(lambda x: woba_wts.get(x,0)).sum()
            wOBA     = woba_num/PA if PA>0 else 0

            K_rate   = stats_df["PlayResultCleaned"].eq("Strikeout").sum()/PA*100 if PA>0 else 0
            BB_rate  = walks/PA*100 if PA>0 else 0

    # Build and display table
            basic_stats = pd.DataFrame({
                "PA": [PA],
                "AB": [AB],
                "Hits": [hits],
                "HR": [HRs],
                "BA": [round(BA,3)],
                "OBP": [round(OBP,3)],
                "SLG": [round(SLG,3)],
                "OPS": [round(OPS,3)],
                "wOBA": [round(wOBA,3)],
                "K Rate (%)": [round(K_rate,1)],
                "BB Rate (%)": [round(BB_rate,1)]
            })

            st.dataframe(
                basic_stats.style.format({
                    "PA": "{:.0f}",
                    "AB": "{:.0f}",
                    "Hits": "{:.0f}",
                    "HR": "{:.0f}",
                    "BA": "{:.3f}",
                    "OBP": "{:.3f}",
                    "SLG": "{:.3f}",
                    "OPS": "{:.3f}",
                    "wOBA": "{:.3f}",
                    "K Rate (%)": "{:.1f}",
                    "BB Rate (%)": "{:.1f}"
                })
            )

    # (insert existing Plate Discipline code, using pd_data)

            pd_data = df_pl.copy()
            strike_zone = {"x_min": -0.83, "x_max": 0.83, "z_min": 1.5, "z_max": 3.5}
            swing_set   = {"StrikeSwinging", "InPlay", "FoulBallFieldable", "FoulBallNotFieldable", "FoulBall"}
            contact_set = {"InPlay", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"}

            total_pitches = len(pd_data)
            swings_total  = pd_data["PitchCall"].isin(swing_set).sum()
            swing_pct     = swings_total / total_pitches * 100 if total_pitches>0 else np.nan

            bip_total     = pd_data["PitchCall"].eq("InPlay").sum()

            in_zone       = pd_data[
                pd_data["PlateLocSide"].between(strike_zone["x_min"], strike_zone["x_max"]) &
                pd_data["PlateLocHeight"].between(strike_zone["z_min"], strike_zone["z_max"])
            ]
            z_count       = len(in_zone)
            z_swings      = in_zone["PitchCall"].isin(swing_set).sum()
            z_swing_pct   = z_swings / z_count * 100 if z_count>0 else np.nan

            out_zone      = pd_data.drop(in_zone.index)
            o_swings      = out_zone["PitchCall"].isin(swing_set).sum()
            chase_pct     = o_swings / len(out_zone) * 100 if len(out_zone)>0 else np.nan

            total_contacts = pd_data["PitchCall"].isin(contact_set).sum()
            whiff_count    = swings_total - total_contacts
            whiff_pct      = whiff_count / swings_total * 100 if swings_total>0 else np.nan

            z_whiff_pct    = (z_swings - in_zone["PitchCall"].isin(contact_set).sum()) / z_swings * 100 \
                                if z_swings>0 else np.nan

            pd_row = {
                "Count":         total_pitches,
                "Balls in Play": bip_total,
                "Swing %":       round(swing_pct,1),
                "Z-Swing %":     round(z_swing_pct,1) if not np.isnan(z_swing_pct) else np.nan,
                "Chase %":       round(chase_pct,1)   if not np.isnan(chase_pct)   else np.nan,
                "Whiff %":       round(whiff_pct,1)   if not np.isnan(whiff_pct)   else np.nan,
                "Z-Whiff %":     round(z_whiff_pct,1) if not np.isnan(z_whiff_pct) else np.nan
            }

            plate_discipline_df = pd.DataFrame([pd_row])
            plate_discipline_styled = plate_discipline_df.style.format({
                "Count": "{:.0f}",
                "Balls in Play": "{:.0f}",
                "Swing %": "{:.1f}",
                "Z-Swing %": "{:.1f}",
                "Chase %": "{:.1f}",
                "Whiff %": "{:.1f}",
                "Z-Whiff %": "{:.1f}"
            })
            st.dataframe(plate_discipline_styled)

    # --- Batted Ball Direction Table ---
            bb_data = df_pl[df_pl["PitchCall"] == "InPlay"].copy()

    # Count
            total_bb = len(bb_data)

            if total_bb > 0:
        # Average launch angle
                avg_la = bb_data["Angle"].mean()

        # Sweetspot % (8°–32°)
                sweet_count = bb_data["Angle"].between(8, 32).sum()
                sweet_pct   = sweet_count / total_bb * 100

        # Classify batted-ball types
                def classify_la(la):
                    if la < 10:   return "GroundBall"
                    if la < 25:   return "LineDrive"
                    if la < 50:   return "FlyBall"
                    return "Popup"

                bb_data["BattedBallType"] = bb_data["Angle"].apply(classify_la)
                valid = bb_data["BattedBallType"].notna().sum()
                gb_pct = bb_data["BattedBallType"].eq("GroundBall").sum() / valid * 100 if valid>0 else np.nan
                ld_pct = bb_data["BattedBallType"].eq("LineDrive").sum() / valid * 100 if valid>0 else np.nan
                fb_pct = bb_data["BattedBallType"].eq("FlyBall").sum() / valid * 100 if valid>0 else np.nan
            else:
                avg_la = sweet_pct = gb_pct = ld_pct = fb_pct = np.nan

            bb_row = {
                "Count":         total_bb,
                "Avg LA":        round(avg_la, 1) if not np.isnan(avg_la) else np.nan,
                "LA Sweetspot%": round(sweet_pct, 1) if not np.isnan(sweet_pct) else np.nan,
                "GB%":           round(gb_pct, 1) if not np.isnan(gb_pct) else np.nan,
                "LD%":           round(ld_pct, 1) if not np.isnan(ld_pct) else np.nan,
                "FB%":           round(fb_pct, 1) if not np.isnan(fb_pct) else np.nan
            }

            bb_df = pd.DataFrame([bb_row])
            st.dataframe(
                bb_df.style.format({
                    "Count": "{:.0f}",
                    "Avg LA": "{:.1f}",
                    "LA Sweetspot%": "{:.1f}",
                    "GB%": "{:.1f}",
                    "LD%": "{:.1f}",
                    "FB%": "{:.1f}"
                }))

    # --- Exit Velocity Table ---
            ev_data = df_pl[df_pl["PitchCall"]=="InPlay"].copy()
            count = len(ev_data)
            if count>0 and "ExitSpeed" in ev_data.columns:
                avg_ev = ev_data["ExitSpeed"].mean()
                p90_ev = np.percentile(ev_data["ExitSpeed"].dropna(),90)
                max_ev = ev_data["ExitSpeed"].max()
                hard_hit_pct = ev_data["ExitSpeed"].ge(95).sum()/count*100
                woba_con = ev_data["PlayResultCleaned"].apply(
                    lambda x: woba_wts.get(x,0)
                ).sum()/count
            else:
                avg_ev=p90_ev=max_ev=hard_hit_pct=woba_con=np.nan

            ev_row = {
                "Count": count,
                "Avg EV": avg_ev,
                "90th Percentile EV": p90_ev,
                "Max EV": max_ev,
                "Hard Hit%": hard_hit_pct,
                "wOBAcon": woba_con
            }
            st.dataframe(pd.DataFrame([ev_row]).style.format({
                "Count":"{:.0f}",
                "Avg EV":"{:.1f}",
                "90th Percentile EV":"{:.1f}",
                "Max EV":"{:.1f}",
                "Hard Hit%":"{:.1f}",
                "wOBAcon":"{:.3f}"
            }))
        

    
       


    
else:
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
