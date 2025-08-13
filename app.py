# app.py
# Powerleague Stats - Final Version with Responsive FotMob-Style Pitch

import streamlit as st
from supabase import create_client
import pandas as pd
import uuid
import base64
import datetime
import io

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Powerleague Stats", layout="wide", initial_sidebar_state="collapsed")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
AVATAR_BUCKET = st.secrets["AVATAR_BUCKET"]

sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
sb_write = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ==============================
# UTILS
# ==============================
@st.cache_data(ttl=60)
def load_table(name):
    return pd.DataFrame(sb.table(name).select("*").execute().data)

def initials(name):
    return "".join([p[0] for p in name.split()][:2]).upper()

def grass_texture():
    # base64-encoded subtle grass SVG
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <rect width="100%" height="100%" fill="#3e9e4d"/>
      <path d="M0 10h1000v10H0z" fill="#46a851" opacity="0.2"/>
      <path d="M0 30h1000v10H0z" fill="#46a851" opacity="0.2"/>
      <path d="M0 50h1000v10H0z" fill="#46a851" opacity="0.2"/>
    </svg>
    """
    return base64.b64encode(svg.encode()).decode()

def render_pitch(players, formation, motm_name):
    formation_lines = [int(x) for x in formation.split("-")]
    total_lines = len(formation_lines)
    pitch_html = f"""
    <div style="position: relative; width: 100%; padding-top: 150%; 
        background: url('data:image/svg+xml;base64,{grass_texture()}'); 
        background-size: cover; border-radius: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
    """
    pitch_width = 100
    pitch_height = 100
    y_gap = pitch_height / (total_lines + 1)

    idx = 0
    for line_idx, num_players in enumerate(formation_lines):
        y = y_gap * (line_idx + 1)
        x_gap = pitch_width / (num_players + 1)
        for slot in range(num_players):
            x = x_gap * (slot + 1)
            if idx < len(players):
                p = players[idx]
                name = p["player_name"]
                goals = p["goals"]
                assists = p["assists"]
                is_motm = (name == motm_name)
                photo_url = p.get("photo_url", "")
                if not photo_url:
                    avatar_html = f"""
                    <div style='width:50px;height:50px;border-radius:50%;
                        background:#ccc;display:flex;align-items:center;justify-content:center;
                        font-weight:bold;font-size:16px;color:#333;'>{initials(name)}</div>
                    """
                else:
                    avatar_html = f"<img src='{photo_url}' style='width:50px;height:50px;border-radius:50%;object-fit:cover;'>"

                chips_html = ""
                if goals > 0:
                    chips_html += f"<div style='background:#fff;padding:2px 5px;border-radius:10px;font-size:12px;'>⚽ {goals}</div>"
                if assists > 0:
                    chips_html += f"<div style='background:#fff;padding:2px 5px;border-radius:10px;font-size:12px;'>🅰 {assists}</div>"

                motm_star = ""
                if is_motm:
                    motm_star = "<div style='position:absolute;top:-8px;right:-8px;background:gold;color:#000;border-radius:50%;padding:2px 4px;font-size:12px;'>★</div>"

                pitch_html += f"""
                <div style='position:absolute;left:{x}%;top:{y}%;transform:translate(-50%,-50%);
                    display:flex;flex-direction:column;align-items:center;'>
                    <div style='position:relative;'>{avatar_html}{motm_star}</div>
                    <div style='font-size:12px;font-weight:bold;color:white;text-shadow:1px 1px 2px black;'>{name}</div>
                    <div style='display:flex;gap:4px;margin-top:2px;'>{chips_html}</div>
                </div>
                """
            idx += 1

    pitch_html += "</div>"
    st.markdown(pitch_html, unsafe_allow_html=True)

# ==============================
# MAIN PAGES
# ==============================
def page_matches():
    matches = load_table("matches")
    players = load_table("players")
    lineups = load_table("lineups")

    m = st.selectbox("Select Match", matches.sort_values(["season","gw"])[["season","gw"]].astype(str).agg("GW".join, axis=1))
    if m:
        # Placeholder: pull correct match data
        match_row = matches.iloc[0]  # Replace with actual selection logic
        st.markdown(f"### {match_row['team_a']} {match_row['score_a']} – {match_row['score_b']} {match_row['team_b']}")
        st.markdown(f"**MOTM:** {match_row['motm_name']}")

        # Render pitches
        team_a_players = lineups[lineups["team"] == "Non-bibs"].to_dict("records")
        team_b_players = lineups[lineups["team"] == "Bibs"].to_dict("records")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Non-bibs")
            render_pitch(team_a_players, match_row.get("formation_a", "1-2-1"), match_row["motm_name"])
        with col2:
            st.subheader("Bibs")
            render_pitch(team_b_players, match_row.get("formation_b", "1-2-1"), match_row["motm_name"])

# ==============================
# ROUTER
# ==============================
pages = {
    "Matches": page_matches,
    # Add other pages here
}

page = st.sidebar.radio("Go to", list(pages.keys()))
pages[page]()
