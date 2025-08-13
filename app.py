# app.py — minimal, solid fix for formation=None and proper lineup rendering
# - Safe formation parsing with fallbacks (5s -> 1-2-1, 7s -> 2-1-2-1)
# - GK row separated and spaced nicely
# - Uses ONLY players from the selected match (no cross-match bleed)
# - Merges player photos into lineup records (photo_url or initials fallback)
# - Chips for goals/assists; MOTM star overlay
# - No emoji icon to keep it clean

import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
import base64

# ------------------ App config ------------------
st.set_page_config(page_title="Powerleague Stats", layout="wide", page_icon=None)

# Secrets
def _required_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        st.error(f"Missing secret: {key}")
        st.stop()

SUPABASE_URL = _required_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = _required_secret("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = _required_secret("SUPABASE_SERVICE_KEY")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")

# Clients
sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
sb_write: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ------------------ Caching helpers ------------------
@st.cache_data(ttl=60)
def fetch_players() -> pd.DataFrame:
    res = sb.table("players").select("*").order("name").execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        df = pd.DataFrame(columns=["id","name","photo_url","notes"])
    return df

@st.cache_data(ttl=60)
def fetch_matches() -> pd.DataFrame:
    res = sb.table("matches").select("*").order("season").order("gw").execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        df = pd.DataFrame(columns=[
            "id","season","gw","side_count","team_a","team_b","score_a","score_b",
            "date","motm_name","is_draw","formation_a","formation_b","notes"
        ])
    return df

@st.cache_data(ttl=60)
def fetch_lineups_for_match(match_id: str) -> pd.DataFrame:
    res = sb.table("lineups").select("*").eq("match_id", match_id).order("team").order("line").order("slot").execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        df = pd.DataFrame(columns=["id","season","gw","match_id","team","player_id","player_name","is_gk","goals","assists","line","slot","position"])
    return df

def _grass_svg_base64() -> str:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="1200" viewBox="0 0 900 1200">
      <defs>
        <linearGradient id="g" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#109e3e"/>
          <stop offset="100%" stop-color="#0b7f30"/>
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(#g)"/>
      <!-- subtle mowing stripes -->
      <g opacity="0.15">
        <rect x="0" y="0" width="900" height="80" fill="#fff"/>
        <rect x="0" y="160" width="900" height="80" fill="#fff"/>
        <rect x="0" y="320" width="900" height="80" fill="#fff"/>
        <rect x="0" y="480" width="900" height="80" fill="#fff"/>
        <rect x="0" y="640" width="900" height="80" fill="#fff"/>
        <rect x="0" y="800" width="900" height="80" fill="#fff"/>
        <rect x="0" y="960" width="900" height="80" fill="#fff"/>
        <rect x="0" y="1120" width="900" height="80" fill="#fff"/>
      </g>
      <!-- pitch lines -->
      <g stroke="#ffffff" stroke-width="6" opacity="0.8" fill="none">
        <line x1="0" y1="600" x2="900" y2="600"/>
        <circle cx="450" cy="600" r="90"/>
        <rect x="120" y="40" width="660" height="180" />
        <rect x="300" y="40" width="300" height="80" />
        <rect x="120" y="980" width="660" height="180" />
        <rect x="300" y="1080" width="300" height="80" />
      </g>
    </svg>
    """
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8")

PITCH_CSS = f"""
<style>
.pitch {{
  position: relative;
  width: 100%;
  aspect-ratio: 3 / 4;
  background: url('data:image/svg+xml;base64,{_grass_svg_base64()}');
  background-size: cover;
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.22);
  overflow: hidden;
}}
.player {{
  position: absolute;
  transform: translate(-50%, -50%);
  text-align: center;
  color: #fff;
  width: clamp(58px, 16vw, 76px);
}}
.ava {{
  width: 100%;
  aspect-ratio: 1/1;
  border-radius: 50%;
  border: 3px solid rgba(255,255,255,0.95);
  background: rgba(255,255,255,0.18);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: clamp(16px, 4.2vw, 22px);
  box-shadow: 0 3px 10px rgba(0,0,0,0.25);
  overflow: hidden;
}}
.name {{ font-weight: 800; font-size: clamp(11px, 3vw, 13px); margin-top: 5px; text-shadow: 0 1px 2px rgba(0,0,0,0.45); }}
.chips {{ display: inline-flex; gap: 6px; margin-top: 3px; }}
.chip {{
  padding: 2px 8px; border-radius: 999px; font-size: clamp(11px, 3.1vw, 13px);
  border: 1px solid rgba(255,255,255,0.25); background: rgba(0,0,0,0.35);
}}
.star {{
  position: absolute; right: -6px; top: -6px; background: #ffb400; color: #111;
  font-weight: 900; border-radius: 999px; padding: 2px 6px; font-size: 11px; border: 2px solid #fff;
  box-shadow: 0 2px 6px rgba(0,0,0,0.35);
}}
.team-title {{ font-weight: 800; margin: 6px 0 8px; }}
</style>
"""
st.markdown(PITCH_CSS, unsafe_allow_html=True)

# ------------------ Formation + pitch rendering ------------------
def _default_formation(side_count: int | None) -> str:
    if side_count == 7:
        return "2-1-2-1"  # your requested default for 7s
    return "1-2-1"        # default for 5s (and other)

def _parse_formation(formation: str | None, side_count: int | None, outfield_n: int) -> list[int]:
    """
    Return a list like [1,2,1]. Safe against None/garbage. If the formation sum
    doesn't match outfield_n, we still use it & fill remaining players on an extra row.
    """
    if not isinstance(formation, str) or "-" not in formation:
        formation = _default_formation(side_count or 5)
    parts = []
    for part in formation.split("-"):
        try:
            val = int(str(part).strip())
            if val < 1: val = 1
            parts.append(val)
        except Exception:
            continue
    if not parts:
        parts = [outfield_n] if outfield_n > 0 else [1]
    return parts

def _initials(full_name: str) -> str:
    bits = [b for b in str(full_name).strip().split() if b]
    if not bits: return "?"
    return (bits[0][0] + (bits[1][0] if len(bits) > 1 else "")).upper()

def _enrich_team_lineup(team_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    # Merge in photos by player_id
    P = players_df[["id", "photo_url", "name"]].rename(columns={"id": "pid"})
    T = team_df.merge(P, left_on="player_id", right_on="pid", how="left")
    # Clean numeric
    for col in ["goals","assists","slot","line"]:
        if col in T.columns:
            T[col] = pd.to_numeric(T[col], errors="coerce").fillna(0).astype(int)
    T["is_gk"] = T["is_gk"].astype(bool) if "is_gk" in T.columns else False
    return T

def _player_html(name: str, photo_url: str | None, is_gk: bool, goals: int, assists: int, is_motm: bool, key: str) -> str:
    if isinstance(photo_url, str) and photo_url.strip():
        ava = f"<img src='{photo_url}' class='ava'/>"
    else:
        ava = f"<div class='ava'>{'🧤' if is_gk else _initials(name)}</div>"
    star = f"<div class='star'>★</div>" if is_motm else ""
    chips = []
    if goals and int(goals) > 0: chips.append(f"<div class='chip'>⚽ x{int(goals)}</div>")
    if assists and int(assists) > 0: chips.append(f"<div class='chip'>🅰️ x{int(assists)}</div>")
    chips_html = f"<div class='chips'>{''.join(chips)}</div>" if chips else "<div class='chips'></div>"
    return f"""
      <div style="position:relative;">{ava}{star}</div>
      <div class='name'>{name}</div>
      {chips_html}
    """

def render_pitch(team_df: pd.DataFrame, team_label: str, formation: str | None, motm_name: str | None, side_count: int | None, container_key: str):
    # Split GK and outfield
    gk_df = team_df[team_df["is_gk"] == True]
    out_df = team_df[team_df["is_gk"] == False].sort_values(["slot","player_name"])

    # Formation lines for outfielders
    form_lines = _parse_formation(formation, side_count, outfield_n=len(out_df))

    # Y positions (percent). GK ~12%, first outfield at ~32%, then spread to ~90%.
    rows = []
    if not gk_df.empty:
        rows.append([gk_df.iloc[0]])          # GK row
    # Build rows per formation group
    idx = 0
    for cnt in form_lines:
        rows.append([r for _, r in out_df.iloc[idx: idx+cnt].iterrows()])
        idx += cnt
    if idx < len(out_df):
        rows.append([r for _, r in out_df.iloc[idx:].iterrows()])

    total_rows = max(1, len(rows))
    ys = []
    for i in range(total_rows):
        if i == 0:
            ys.append(12)  # GK far from first line
        else:
            rem = max(1, total_rows - 1)
            ys.append(32 + (i - 1) * (58 / rem))  # 32%..90%

    # Build pitch HTML
    st.markdown(f"<div class='team-title'>#### {team_label}</div>", unsafe_allow_html=True)
    html = ["<div class='pitch'>"]
    for row_idx, players in enumerate(rows):
        if not players: continue
        y = ys[row_idx]
        n = len(players)
        xs = [50] if n == 1 else list(np.linspace(15, 85, n))
        for x, r in zip(xs, players):
            name = str(r.get("player_name", ""))
            photo = r.get("photo_url", None)
            is_gk = bool(r.get("is_gk", False))
            goals = int(r.get("goals", 0))
            assists = int(r.get("assists", 0))
            is_motm = (isinstance(motm_name, str) and name.lower() == motm_name.strip().lower())
            card = _player_html(name, photo, is_gk, goals, assists, is_motm, key=f"{container_key}_{name}")
            html.append(f"<div class='player' style='left:{x}%; top:{y}%;'>{card}</div>")
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)

# ------------------ Matches page (fixed) ------------------
def page_matches():
    st.header("Matches")
    matches = fetch_matches()
    if matches.empty:
        st.info("No matches yet.")
        return

    # Build human labels and map back to index
    labels = []
    for _, r in matches.iterrows():
        lab = f"S{r.get('season')} · GW{int(r.get('gw',0))} — {r.get('team_a','Non-bibs')} {int(r.get('score_a',0))}–{int(r.get('score_b',0))} {r.get('team_b','Bibs')}"
        labels.append(lab)
    choice = st.selectbox("Select match", labels, index=max(0, len(labels)-1), key="match_select_safe")
    m = matches.iloc[labels.index(choice)]

    # Load lineups for this match only and enrich with photos
    L = fetch_lineups_for_match(m["id"])
    players = fetch_players()
    if L.empty:
        st.info("No lineups for this match yet.")
        return
    # Enrich each team with photo_url
    nb = _enrich_team_lineup(L[L["team"] == "Non-bibs"].copy(), players)
    bb = _enrich_team_lineup(L[L["team"] == "Bibs"].copy(), players)

    # MOTM & formation safe fallbacks
    motm = m.get("motm_name") or ""  # display-only; case-insensitive star match
    side_count = int(m.get("side_count") or 5)
    fa = m.get("formation_a") or _default_formation(side_count)
    fb = m.get("formation_b") or _default_formation(side_count)

    # Banner
    d_txt = f" · {m['date']}" if isinstance(m.get("date"), str) and m["date"] else ""
    st.subheader(f"S{m['season']} · GW{int(m['gw'])}")
    st.caption(f"{m['team_a']} vs {m['team_b']} · {int(m['score_a'])}–{int(m['score_b'])}{d_txt} · ⭐ MOTM: {motm or '—'}")

    c1, c2 = st.columns(2)
    with c1:
        render_pitch(nb, "Non-bibs", fa, motm, side_count, container_key=f"nb_{m['id']}")
    with c2:
        render_pitch(bb, "Bibs", fb, motm, side_count, container_key=f"bb_{m['id']}")

# ------------------ Router (keep simple for now) ------------------
PAGES = {"Matches": page_matches}
page = st.sidebar.radio("Go to", list(PAGES.keys()), key="nav")
PAGES[page]()
