import io
import json
import math
from datetime import date, datetime
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from PIL import Image
import pillow_heif

from supabase import create_client, Client

# ============================
# App Setup
# ============================

st.set_page_config(
    page_title="Powerleague Stats",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Mobile-first tweaks & FotMob-style look
MOBILE_CSS = '''
<style>
/* Base */
:root {
  --pitch-green: #1e7f3a;
  --pitch-lines: rgba(255,255,255,0.2);
  --chip-bg: rgba(255,255,255,0.06);
  --chip-border: rgba(255,255,255,0.18);
}
html, body, [class^="css"]  {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
/* Hide Streamlit footer & menu on mobile for more space */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
/* Buttons & inputs a bit larger for touch */
button, .stButton>button, .stDownloadButton>button {
  border-radius: 9999px;
  padding: 0.6rem 1rem;
}
/* Pitch */
.pitch {
  background: radial-gradient(circle at 50% 20%, #2aa14d 0%, #1f8440 60%, #166332 100%);
  border-radius: 18px;
  padding: 10px;
  position: relative;
  box-shadow: 0 8px 24px rgba(0,0,0,0.2);
  border: 1px solid var(--pitch-lines);
  min-height: 460px;
}
.pitch .grid {
  display: grid;
  grid-template-rows: auto;
  gap: 18px;
}
.pitch .line {
  display: grid;
  gap: 10px;
}
.pitch .slot {
  display: flex;
  align-items: center;
  justify-content: center;
}
.chip {
  backdrop-filter: blur(6px);
  background: var(--chip-bg);
  border: 1px solid var(--chip-border);
  color: #fff;
  padding: 6px 10px;
  border-radius: 14px;
  font-size: 13px;
  line-height: 1.2;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 100%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.chip img {
  width: 20px;
  height: 20px;
  border-radius: 9999px;
  object-fit: cover;
  border: 1px solid rgba(255,255,255,0.25);
}
.chip .stats {
  opacity: 0.9;
  font-size: 12px;
}
.banner {
  background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
  color: #fff;
  padding: 10px 14px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border: 1px solid rgba(255,255,255,0.15);
}
.badge {
  background: #22c55e;
  color: #052e14;
  padding: 4px 10px;
  border-radius: 9999px;
  font-weight: 600;
  font-size: 12px;
}
.small {
  font-size: 12px;
  opacity: 0.85;
}
.hr { height: 1px; background: rgba(0,0,0,0.08); margin: 10px 0; }
@media (max-width: 640px) {
  .pitch { min-height: 420px; }
}
</style>
'''
st.markdown(MOBILE_CSS, unsafe_allow_html=True)

# ============================
# Secrets
# ============================
REQ_KEYS = ["SUPABASE_URL","SUPABASE_ANON_KEY","ADMIN_PASSWORD","AVATAR_BUCKET"]
for k in REQ_KEYS:
    if k not in st.secrets:
        st.error(f"Missing secret: {k}. Please set it in Streamlit secrets.")
        st.stop()

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", None)
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
AVATAR_BUCKET = st.secrets.get("AVATAR_BUCKET", "avatars")

# ============================
# Clients
# ============================
sb_public: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def get_service_client() -> Optional[Client]:
    if st.session_state.get("is_admin") and SUPABASE_SERVICE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return None

# Ensure avatars bucket exists (best-effort)
def ensure_bucket():
    try:
        sb = get_service_client()
        if sb is None:
            return
        # supabase-py 2.x
        buckets = sb.storage.list_buckets()
        names = {b.name if hasattr(b,"name") else b.get("name") for b in buckets}
        if AVATAR_BUCKET not in names:
            sb.storage.create_bucket(AVATAR_BUCKET, public=True)
    except Exception:
        pass
ensure_bucket()

# ============================
# HEIC registration
# ============================
try:
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ============================
# Helpers
# ============================

@st.cache_data(ttl=30)
def fetch_players() -> pd.DataFrame:
    res = sb_public.table("players").select("*").execute()
    data = res.data or []
    return pd.DataFrame(data)

@st.cache_data(ttl=30)
def fetch_matches() -> pd.DataFrame:
    res = sb_public.table("matches").select("*").order("season").order("gw").execute()
    data = res.data or []
    return pd.DataFrame(data)

@st.cache_data(ttl=30)
def fetch_lineups() -> pd.DataFrame:
    res = sb_public.table("lineups").select("*").execute()
    data = res.data or []
    return pd.DataFrame(data)

@st.cache_data(ttl=30)
def fetch_awards() -> pd.DataFrame:
    res = sb_public.table("awards").select("*").execute()
    data = res.data or []
    return pd.DataFrame(data)

def clear_caches():
    fetch_players.clear()
    fetch_matches.clear()
    fetch_lineups.clear()
    fetch_awards.clear()

def formation_to_lines(form: str) -> List[int]:
    if not form or form.strip() == "":
        return [1,2,1]  # default 5s
    parts = [int(x) for x in form.strip().split("-") if x.strip().isdigit()]
    return [1] + parts  # prepend GK line

def slot_positions(form: str) -> List[Tuple[int,int]]:
    # Returns list of (line, slots_in_line)
    lines = formation_to_lines(form)
    return [(i, lines[i]) for i in range(len(lines))]

def chip_html(name: str, ga: Tuple[int,int], photo_url: Optional[str]) -> str:
    goals, assists = ga
    img = f'<img src="{photo_url}" alt="" />' if photo_url else ""
    stats = []
    if goals: stats.append(f"⚽ {goals}")
    if assists: stats.append(f"🅰️ {assists}")
    stat_html = f'<span class="stats">{" · ".join(stats)}</span>' if stats else ""
    return f'<span class="chip">{img}<span>{name}</span>{stat_html}</span>'

def public_image_url(path: str) -> str:
    try:
        # supabase-py 2.x returns object with .public_url
        url = sb_public.storage.from_(AVATAR_BUCKET).get_public_url(path)
        if isinstance(url, str):
            return url
        # sometimes dict
        if hasattr(url, "get"):
            u = url.get("publicUrl") or url.get("public_url")
            if u: return u
    except Exception:
        pass
    # fallback
    return f"{SUPABASE_URL}/storage/v1/object/public/{AVATAR_BUCKET}/{path}"

def upload_avatar(image_file, preferred_name: str) -> Optional[str]:
    sb = get_service_client()
    if sb is None:
        st.error("Admin required to upload avatars.")
        return None
    # Convert to PNG (handles HEIC as PIL now can open after pillow_heif opener)
    img = Image.open(image_file)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    safe_name = preferred_name.lower().replace(" ", "_")
    key = f"{safe_name}.png"
    try:
        sb.storage.from_(AVATAR_BUCKET).upload(key, buf.getvalue(), {"content-type":"image/png","upsert":"true"})
    except Exception:
        # try upsert by removing existing then upload
        try:
            sb.storage.from_(AVATAR_BUCKET).remove([key])
        except Exception:
            pass
        sb.storage.from_(AVATAR_BUCKET).upload(key, buf.getvalue(), {"content-type":"image/png"})
    return public_image_url(key)

def upsert_players(df: pd.DataFrame) -> None:
    sb = get_service_client()
    if sb is None:
        st.error("Admin required.")
        return
    # Only relevant columns
    cols = ["name","photo_url","notes"]
    payload = df[cols].fillna("").to_dict(orient="records")
    # Upsert on name
    sb.table("players").upsert(payload, on_conflict="name").execute()
    clear_caches()


def upsert_matches(df: pd.DataFrame) -> None:
    sb = get_service_client()
    if sb is None:
        st.error("Admin required.")
        return
    df = df.copy()

    # default formations based on side_count if blank
    def _default_form(row, which):
        val = str(row.get(which, "") or "").strip()
        if val:
            return val
        sc = int(row.get("side_count") or 5)
        return "1-2-1" if sc == 5 else "2-1-2-1"
    if "formation_a" in df.columns and "formation_b" in df.columns:
        df["formation_a"] = df.apply(lambda r: _default_form(r, "formation_a"), axis=1)
        df["formation_b"] = df.apply(lambda r: _default_form(r, "formation_b"), axis=1)

    # Normalize booleans and dates; replace NaN with None for JSON
    if "is_draw" in df.columns:
        def _is_draw(row):
            val = row.get("is_draw")
            if pd.isna(val) or str(val).strip() == "":
                try:
                    return int(row.get("score_a") or 0) == int(row.get("score_b") or 0)
                except Exception:
                    return False
            # strings like 'true'/'false'
            if isinstance(val, str):
                return val.strip().lower() in ["true", "t", "1", "yes", "y"]
            return bool(val)
        df["is_draw"] = df.apply(_is_draw, axis=1)

    # Normalize date to 'YYYY-MM-DD' or None
    if "date" in df.columns:
        def _norm_date(x):
            if pd.isna(x) or str(x).strip() == "":
                return None
            try:
                return str(pd.to_datetime(x).date())
            except Exception:
                return None
        df["date"] = df["date"].apply(_norm_date)

    # Ensure numeric ints where appropriate
    for col in ["season","gw","side_count","score_a","score_b"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: int(v) if (pd.notna(v) and str(v).strip()!="") else None)

    cols = ["season","gw","side_count","team_a","team_b","score_a","score_b","date","motm_name","is_draw","formation_a","formation_b","notes"]
    # Replace NaN/NaT with None to avoid json encoder ValueError
    df = df.reindex(columns=cols).where(pd.notnull(df), None)
    payload = df.to_dict(orient="records")

    sb.table("matches").upsert(payload, on_conflict="season,gw").execute()

    # Ensure MOTM awards exist
    awards = []
    for r in payload:
        if r.get("motm_name"):
            awards.append({
                "season": r.get("season"),
                "month": pd.to_datetime(r.get("date")).month if r.get("date") else None,
                "type": "MOTM",
                "gw": r.get("gw"),
                "player_name": r.get("motm_name")
            })
    if awards:
        sb.table("awards").insert(awards, count="none").execute()
    clear_caches()



def insert_lineups(df: pd.DataFrame) -> None:
    sb = get_service_client()
    if sb is None:
        st.error("Admin required.")
        return

    players = fetch_players()
    name_to_id = {str(n).strip().lower(): pid for n,pid in zip(players["name"].fillna(""), players["id"].fillna(""))}

    def safe_int(v, default=None):
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in ("nan","none","null"):
            return default
        try:
            return int(float(s))
        except Exception:
            return default

    rows = []
    affected_match_ids = set()

    # Collect rows with resolved match_id
    for _, r in df.iterrows():
        mid = r.get('match_id')
        if (pd.isna(mid) or (isinstance(mid, str) and mid.strip()=='')) and pd.notna(r.get('season')) and pd.notna(r.get('gw')):
            try:
                mres = sb.table('matches').select('id').eq('season', int(r.get('season'))).eq('gw', int(r.get('gw'))).single().execute()
                mid = mres.data['id']
            except Exception:
                mid = None

        team_val = (r.get("team") or "").strip()
        if team_val.lower() == "non-bibs": team_val = "Non-bibs"
        if team_val.lower() == "bibs": team_val = "Bibs"

        is_gk_val = r.get("is_gk")
        if isinstance(is_gk_val, str):
            is_gk = is_gk_val.strip().lower() in ("1","true","t","yes","y")
        else:
            try:
                is_gk = bool(int(is_gk_val))
            except Exception:
                is_gk = bool(is_gk_val)

        row_out = {
            "season": safe_int(r.get("season")),
            "gw": safe_int(r.get("gw")),
            "match_id": mid,
            "team": team_val,
            "player_name": r.get("player_name"),
            "player_id": r.get("player_id") or name_to_id.get(str(r.get("player_name") or "").strip().lower()),
            "is_gk": is_gk,
            "goals": safe_int(r.get("goals"), 0),
            "assists": safe_int(r.get("assists"), 0),
            "line": safe_int(r.get("line")),
            "slot": safe_int(r.get("slot")),
            "position": r.get("position") or ""
        }
        if row_out["match_id"] and row_out["team"] in ("Non-bibs","Bibs"):
            rows.append(row_out)
            affected_match_ids.add(row_out["match_id"])

    if not rows:
        st.warning("No valid lineup rows to import (check season/gw or team names).")
        return

    # Bulk delete all lineups for affected matches in one API call
    try:
        sb.table("lineups").delete().in_("match_id", list(affected_match_ids)).execute()
    except Exception as e:
        st.warning(f"Warning: bulk delete failed, attempting per-match deletes. ({e})")
        for mid in affected_match_ids:
            try:
                sb.table("lineups").delete().eq("match_id", mid).execute()
            except Exception:
                pass

    # Chunked inserts to avoid payload limits and rate limiting
    CHUNK = 500
    p = st.progress(0, text="Importing lineups...")
    total = len(rows)
    for i in range(0, total, CHUNK):
        chunk = rows[i:i+CHUNK]
        sb.table("lineups").insert(chunk).execute()
        p.progress(min(1.0, (i+len(chunk))/total), text=f"Imported {i+len(chunk)}/{total} rows")
    p.empty()

    clear_caches()

# ============================
# Stats computation
# ============================

def build_fact_tables(players: pd.DataFrame, matches: pd.DataFrame, lineups: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if matches.empty:
        return lineups.copy(), matches.copy()
    m = matches.copy()
    l = lineups.copy()
    # Normalize teams: map team text to side A/B by joining on match_id + team text
    # We'll assume team_a == 'Non-bibs' and team_b == 'Bibs' unless different labels are stored.
    # Use team names from matches rows.
    mi = m[["id","team_a","team_b","score_a","score_b","season","gw","date","is_draw","motm_name"]].rename(columns={"id":"match_id"})
    l = l.merge(mi, on="match_id", how="left")
    # Determine side for lineup row
    l["side"] = np.where(l["team"] == l["team_a"], "A", np.where(l["team"] == l["team_b"], "B", None))
    # Result for row
    l["team_goals"] = np.where(l["side"]=="A", l["score_a"], np.where(l["side"]=="B", l["score_b"], None))
    l["opp_goals"] = np.where(l["side"]=="A", l["score_b"], np.where(l["side"]=="B", l["score_a"], None))
    l["result"] = np.where(l["is_draw"], "D", np.where(l["team_goals"]>l["opp_goals"], "W", "L"))
    l["ga"] = l["goals"] + l["assists"]
    # Attach player names (keep player_name prefer)
    players = players.rename(columns={"id":"player_id"})
    l = l.merge(players[["player_id","name","photo_url"]], on="player_id", how="left")
    l["name"] = l["player_name"].combine_first(l["name"])
    l["photo"] = l["photo_url"]
    return l, m

def player_aggregate(l: pd.DataFrame, season: Optional[int]=None, min_games: int=0, last_gw: Optional[int]=None) -> pd.DataFrame:
    df = l.copy()
    if season:
        df = df[df["season"] == season]
    if last_gw:
        df = df[df["gw"] >= (df["gw"].max() - last_gw + 1)]
    if df.empty:
        return pd.DataFrame(columns=["name","gp","w","d","l","win_pct","goals","assists","ga","ga_pg","team_contrib_pct","photo"])
    gp = df.groupby("name").size().rename("gp")
    w = df[df["result"]=="W"].groupby("name").size().reindex(gp.index, fill_value=0).rename("w")
    d = df[df["result"]=="D"].groupby("name").size().reindex(gp.index, fill_value=0).rename("d")
    lcnt = df[df["result"]=="L"].groupby("name").size().reindex(gp.index, fill_value=0).rename("l")
    goals = df.groupby("name")["goals"].sum().reindex(gp.index, fill_value=0)
    assists = df.groupby("name")["assists"].sum().reindex(gp.index, fill_value=0)
    ga = goals + assists
    ga_pg = (ga / gp).round(2)
    # team contribution: sum player ga divided by sum team goals for games they played
    team_goals = df.groupby(["name"])["team_goals"].sum().reindex(gp.index, fill_value=0).replace(0, np.nan)
    contrib = ((ga / team_goals) * 100).round(1).fillna(0)
    photo = df.groupby("name")["photo"].last().reindex(gp.index)
    out = pd.DataFrame({
        "name": gp.index,
        "gp": gp.values,
        "w": w.values,
        "d": d.values,
        "l": lcnt.values,
        "win_pct": ((w.values / np.maximum(gp.values,1))*100).round(1),
        "goals": goals.values,
        "assists": assists.values,
        "ga": ga.values,
        "ga_pg": ga_pg.values,
        "team_contrib_pct": contrib.values,
        "photo": photo.values
    }).sort_values(["ga","goals","assists"], ascending=False)
    if min_games > 0:
        out = out[out["gp"] >= min_games]
    return out

def best_worst_duos(l: pd.DataFrame, top_n: int = 10, season: Optional[int]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = l.copy()
    if season:
        df = df[df["season"] == season]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Pairs who played on same team in the same match
    pairs = []
    for (mid, team), g in df.groupby(["match_id","team"]):
        names = list(g["name"].dropna().unique())
        for a,b in combinations(sorted(names), 2):
            res = g[["result"]].iloc[0]["result"]
            pairs.append({"pair": f"{a} × {b}", "a": a, "b": b, "result": res})
    p = pd.DataFrame(pairs)
    if p.empty:
        return pd.DataFrame(), pd.DataFrame()
    gp = p.groupby("pair").size().rename("gp")
    w = p[p["result"]=="W"].groupby("pair").size().reindex(gp.index, fill_value=0).rename("w")
    lcnt = p[p["result"]=="L"].groupby("pair").size().reindex(gp.index, fill_value=0).rename("l")
    d = p[p["result"]=="D"].groupby("pair").size().reindex(gp.index, fill_value=0).rename("d")
    win_pct = ((w / gp) * 100).round(1)
    out = pd.DataFrame({"pair": gp.index, "gp": gp.values, "w": w.values, "d": d.values, "l": lcnt.values, "win_pct": win_pct.values})
    best = out[out["gp"]>=3].sort_values(["win_pct","gp"], ascending=[False, False]).head(top_n)
    worst = out[out["gp"]>=3].sort_values(["win_pct","gp"], ascending=[True, False]).head(top_n)
    return best, worst

def nemesis_table(l: pd.DataFrame, top_n: int = 10, season: Optional[int]=None) -> pd.DataFrame:
    df = l.copy()
    if season:
        df = df[df["season"] == season]
    # Player vs opposing player win% when both appear
    rows = []
    # Build match-wise rosters
    for mid, g in df.groupby("match_id"):
        if g.empty: continue
        a_team = g[g["side"]=="A"]["name"].dropna().unique().tolist()
        b_team = g[g["side"]=="B"]["name"].dropna().unique().tolist()
        # Determine result for side A/B
        resA = g.iloc[0]["result"] if g.iloc[0]["side"]=="A" else ("W" if g.iloc[0]["result"]=="L" else "L" if g.iloc[0]["result"]=="W" else "D")
        # For each cross pair
        for a in a_team:
            for b in b_team:
                rows.append({"a": a, "b": b, "res": resA})
    p = pd.DataFrame(rows)
    if p.empty: return p
    gp = p.groupby(["a","b"]).size().rename("gp")
    w = p[p["res"]=="W"].groupby(["a","b"]).size().reindex(gp.index, fill_value=0).rename("w")
    lcnt = p[p["res"]=="L"].groupby(["a","b"]).size().reindex(gp.index, fill_value=0).rename("l")
    win_pct = ((w / gp)*100).round(1)
    out = pd.DataFrame({"a":[x[0] for x in gp.index], "b":[x[1] for x in gp.index],
                        "gp": gp.values, "w": w.values, "l": lcnt.values, "win_pct": win_pct.values})
    # Nemesis: for each a, find b with lowest win_pct (min 3 games)
    out = out[out["gp"]>=3]
    nemesis = out.sort_values(["a","win_pct","gp"], ascending=[True, True, False]).groupby("a").head(1)
    return nemesis.sort_values("win_pct").head(top_n)

# ============================
# UI Components
# ============================

def header():
    left, right = st.columns([1,1], gap="small")
    with left:
        st.title("⚽ Powerleague Stats")
    with right:
        # Admin login
        if "is_admin" not in st.session_state:
            st.session_state["is_admin"] = False
        if st.session_state["is_admin"]:
            st.success("Admin mode", icon="🔐")
            if st.button("Logout", key="logout_btn"):
                st.session_state["is_admin"] = False
                st.rerun()
        else:
            with st.popover("Admin login"):
                pwd = st.text_input("Password", type="password", key="admin_pwd")
                if st.button("Login", use_container_width=True):
                    if pwd == ADMIN_PASSWORD:
                        st.session_state["is_admin"] = True
                        st.success("Welcome, admin.")
                        st.rerun()
                    else:
                        st.error("Invalid password.")

def draw_pitch(formation: str, lineup_rows: pd.DataFrame, show_photos: bool = True):
    # Build a grid where each line has N slots centered
    lines = formation_to_lines(formation)
    max_slots = max(lines)
    html_lines = []
    for i, slots in enumerate(lines):
        # Determine names in this line
        g = lineup_rows[lineup_rows["line"]==i].sort_values("slot")
        # Generate centered grid columns
        grid_tpl = " ".join(["1fr"]*max_slots)
        items_html = []
        # Fill with empty & placed slots
        placed = {int(r["slot"]): r for _,r in g.iterrows() if pd.notna(r["slot"])}
        for s in range(slots):
            # center them: compute slot index among max_slots positions
            offset = (max_slots - slots)//2
            global_slot = s + offset
            r = placed.get(global_slot)
            if r is not None:
                photo = r.get("photo") if show_photos else None
                items_html.append(f'<div class="slot">{chip_html(r.get("name") or "", (int(r.get("goals") or 0), int(r.get("assists") or 0)), photo)}</div>')
            else:
                items_html.append(f'<div class="slot"></div>')
        html_line = f'<div class="line" style="grid-template-columns:{grid_tpl}">{"".join(items_html)}</div>'
        html_lines.append(html_line)
    html = f'<div class="pitch"><div class="grid">{"".join(html_lines)}</div></div>'
    st.markdown(html, unsafe_allow_html=True)

def match_banner(m):
    left, right = st.columns([3,2], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="banner"><div><div><strong>Season {m["season"]} · GW {m["gw"]}</strong></div>'
            f'<div class="small">{m.get("date") or ""}</div></div>'
            f'<div><strong>{m["team_a"]} {m["score_a"]} – {m["score_b"]} {m["team_b"]}</strong></div></div>',
            unsafe_allow_html=True
        )
    with right:
        mm = m.get("motm_name")
        if mm:
            st.markdown(f'<div class="banner"><span>Man of the Match</span><span class="badge">🏅 {mm}</span></div>', unsafe_allow_html=True)

# ============================
# Pages
# ============================

def page_matches():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact, _ = build_fact_tables(players, matches, lineups)

    st.subheader("Matches")
    view_tab, add_tab, edit_tab = st.tabs(["📋 All Matches","➕ Add Match","✏️ Edit Match"])

    with view_tab:
        if matches.empty:
            st.info("No matches yet.")
        for _, m in matches.sort_values(["season","gw"]).iterrows():
            match_banner(m)
            # Split lineups by team & match
            g = lfact[lfact["match_id"]==m["id"]]
            a_rows = g[g["team"]==m["team_a"]]
            b_rows = g[g["team"]==m["team_b"]]
            show_photos = st.toggle("Show photos", value=True, key=f"photos_{m['id']}")
            a_col, b_col = st.columns(2, gap="large")
            with a_col:
                st.caption(m["team_a"])
                draw_pitch(m.get("formation_a") or "1-2-1", a_rows, show_photos)
            with b_col:
                st.caption(m["team_b"])
                draw_pitch(m.get("formation_b") or "1-2-1", b_rows, show_photos)
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    with add_tab:
        if not st.session_state.get("is_admin"):
            st.warning("Admin only.")
        else:
            with st.form("add_match_form", clear_on_submit=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    season = st.number_input("Season", value=1, step=1, min_value=1, key="am_season")
                    gw = st.number_input("Gameweek", value=1, step=1, min_value=1, key="am_gw")
                    date_val = st.date_input("Date", value=date.today(), key="am_date")
                with c2:
                    side_count = st.selectbox("Side Count", [5,7], index=0, key="am_side")
                    team_a = st.text_input("Team A", value="Non-bibs", key="am_team_a")
                    team_b = st.text_input("Team B", value="Bibs", key="am_team_b")
                with c3:
                    default_form = "1-2-1" if side_count == 5 else "2-1-2-1"
                    formation_a = st.text_input("Formation A", value=default_form, key="am_form_a")
                    formation_b = st.text_input("Formation B", value=default_form, key="am_form_b")

                st.markdown("#### Select Players")
                names = players["name"].sort_values().tolist()
                a_sel = st.multiselect("Team A players", names, key="am_players_a")
                b_sel = st.multiselect("Team B players", names, key="am_players_b")

                st.markdown("#### Assign Positions & Stats")
                def lineup_editor(team_label: str, selected_names: List[str], formation: str, prefix: str):
                    lines = formation_to_lines(formation)
                    slots = max(lines)
                    data = []
                    # GK
                    if selected_names:
                        default_gk = selected_names[0]
                    else:
                        default_gk = None
                    gk = st.selectbox(f"{team_label} GK", [None] + selected_names, index=0 if default_gk is None else ([None]+selected_names).index(default_gk), key=f"{prefix}_gk")
                    if gk:
                        selected_names_no_gk = [n for n in selected_names if n != gk]
                    else:
                        selected_names_no_gk = selected_names[:]
                    # Build lines
                    for line_idx in range(1, len(lines)):  # skip GK line 0
                        st.caption(f"{team_label} Line {line_idx} ({lines[line_idx]} players)")
                        for s in range(slots):
                            # only allow center-aligned slots that are within the number of players
                            offset = (slots - lines[line_idx])//2
                            if s < offset or s >= offset + lines[line_idx]:
                                continue
                            key = f"{prefix}_l{line_idx}_s{s}"
                            choice = st.selectbox(f"Slot {s-offset+1}", [None] + selected_names_no_gk, key=key, index=0)
                            if choice:
                                goals = st.number_input(f"⚽ {choice}", min_value=0, step=1, value=0, key=f"{key}_g")
                                assists = st.number_input(f"🅰️ {choice}", min_value=0, step=1, value=0, key=f"{key}_a")
                                data.append({"name": choice, "line": line_idx, "slot": s, "goals": goals, "assists": assists, "is_gk": False})
                    # GK stats
                    if gk:
                        g_goals = st.number_input(f"⚽ {gk} (GK)", min_value=0, step=1, value=0, key=f"{prefix}_gk_g")
                        g_assists = st.number_input(f"🅰️ {gk} (GK)", min_value=0, step=1, value=0, key=f"{prefix}_gk_a")
                        data.append({"name": gk, "line": 0, "slot": (slots//2), "goals": g_goals, "assists": g_assists, "is_gk": True})
                    return data

                a_data = lineup_editor(team_a, a_sel, formation_a, "A")
                b_data = lineup_editor(team_b, b_sel, formation_b, "B")

                c4, c5, c6 = st.columns(3)
                with c4:
                    score_a = st.number_input(f"{team_a} score", min_value=0, value=0, step=1, key="am_score_a")
                with c5:
                    score_b = st.number_input(f"{team_b} score", min_value=0, value=0, step=1, key="am_score_b")
                with c6:
                    motm = st.selectbox("Man of the Match", [None] + (a_sel + b_sel), index=0, key="am_motm")
                is_draw = (score_a == score_b)

                submitted = st.form_submit_button("Save Match", use_container_width=True)
                if submitted:
                    sb = get_service_client()
                    if sb is None:
                        st.error("Admin required.")
                    else:
                        # upsert match
                        match_payload = {
                            "season": int(season), "gw": int(gw), "side_count": int(side_count),
                            "team_a": team_a, "team_b": team_b, "score_a": int(score_a), "score_b": int(score_b),
                            "date": str(date_val), "motm_name": motm, "is_draw": bool(is_draw),
                            "formation_a": formation_a, "formation_b": formation_b
                        }
                        res = sb.table("matches").upsert(match_payload, on_conflict="season,gw").execute()
                        # fetch inserted/updated id
                        # Query by season/gw
                        mrow = sb.table("matches").select("*").eq("season", season).eq("gw", gw).single().execute().data
                        mid = mrow["id"]
                        # Build lineups rows
                        players_df = fetch_players()
                        name_to_id = {n.lower(): pid for n, pid in zip(players_df["name"].fillna(""), players_df["id"].fillna(""))}
                        def pack(team_label, data_rows):
                            out = []
                            for r in data_rows:
                                out.append({
                                    "season": int(season),
                                    "gw": int(gw),
                                    "match_id": mid,
                                    "team": team_label,
                                    "player_name": r["name"],
                                    "player_id": name_to_id.get(r["name"].lower()),
                                    "is_gk": bool(r["is_gk"]),
                                    "goals": int(r["goals"]),
                                    "assists": int(r["assists"]),
                                    "line": int(r["line"]),
                                    "slot": int(r["slot"]),
                                    "position": ""
                                })
                            return out
                        rows_a = pack(team_a, a_data)
                        rows_b = pack(team_b, b_data)
                        # delete-then-insert
                        sb.table("lineups").delete().eq("match_id", mid).eq("team", team_a).execute()
                        sb.table("lineups").delete().eq("match_id", mid).eq("team", team_b).execute()
                        if rows_a: sb.table("lineups").insert(rows_a).execute()
                        if rows_b: sb.table("lineups").insert(rows_b).execute()
                        # award MOTM
                        if motm:
                            sb.table("awards").insert({
                                "season": int(season),
                                "month": int(date_val.month),
                                "type": "MOTM",
                                "gw": int(gw),
                                "player_name": motm
                            }).execute()
                        clear_caches()
                        st.success("Match saved.")
                        st.rerun()

    with edit_tab:
        if not st.session_state.get("is_admin"):
            st.warning("Admin only.")
        else:
            if matches.empty:
                st.info("No matches to edit.")
            else:
                options = matches.apply(lambda r: f"S{r['season']} GW{r['gw']} · {r['team_a']} {r['score_a']}–{r['score_b']} {r['team_b']}", axis=1).tolist()
                idx = st.selectbox("Select match", list(range(len(options))), format_func=lambda i: options[i], key="em_sel")
                m = matches.iloc[idx].to_dict()
                st.write("Adjust formations, GK, goals & assists then save.")
                # Show current lineups
                g = lfact[lfact["match_id"]==m["id"]]
                for team_label, form_key in [(m["team_a"], "formation_a"), (m["team_b"], "formation_b")]:
                    st.markdown(f"#### {team_label}")
                    formation = st.text_input(f"Formation ({team_label})", value=m.get(form_key) or "1-2-1", key=f"em_form_{team_label}")
                    rows = g[g["team"]==team_label].copy()
                    # Inline editors
                    for i, r in rows.sort_values(["line","slot"]).iterrows():
                        c1, c2, c3, c4 = st.columns([2,1,1,1])
                        c1.write(f"{'🧤 ' if r['is_gk'] else ''}{r['name']}")
                        gval = c2.number_input("⚽", min_value=0, step=1, value=int(r.get("goals") or 0), key=f"em_g_{r['id']}")
                        aval = c3.number_input("🅰️", min_value=0, step=1, value=int(r.get("assists") or 0), key=f"em_a_{r['id']}")
                        gk = c4.checkbox("GK", value=bool(r.get("is_gk")), key=f"em_k_{r['id']}")
                        rows.loc[i,"goals"] = gval
                        rows.loc[i,"assists"] = aval
                        rows.loc[i,"is_gk"] = gk
                    if st.button(f"Save {team_label}", key=f"em_save_{team_label}"):
                        sb = get_service_client()
                        if sb is None:
                            st.error("Admin required.")
                        else:
                            # update formation
                            sb.table("matches").update({form_key: formation}).eq("id", m["id"]).execute()
                            # write lineups
                            out = rows[["id","goals","assists","is_gk"]].to_dict(orient="records")
                            for r in out:
                                rid = r.pop("id")
                                sb.table("lineups").update(r).eq("id", rid).execute()
                            clear_caches()
                            st.success("Saved.")
                            st.rerun()

def page_players():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()

    st.subheader("Players")
    names = players["name"].sort_values().tolist()
    selected = st.selectbox("Select player", [None]+names, index=0, key="pp_sel")
    if not selected:
        st.info("Choose a player to view profile.")
        return
    p = players[players["name"]==selected].iloc[0].to_dict()
    lfact, _ = build_fact_tables(players, matches, lineups)
    my_rows = lfact[lfact["name"]==selected]
    agg = player_aggregate(lfact)
    me = agg[agg["name"]==selected].iloc[0] if not agg[agg["name"]==selected].empty else None

    col1, col2 = st.columns([1,2], gap="large")
    with col1:
        if p.get("photo_url"):
            st.image(p.get("photo_url"), width=140)
        else:
            st.image("https://placehold.co/200x200?text=No+Photo", width=140)
        st.markdown(f"### {p['name']}")
        if p.get("notes"):
            st.caption(p["notes"])
        if st.session_state.get("is_admin"):
            img = st.file_uploader("Update photo (HEIC/JPG/PNG)", type=["heic","HEIC","jpg","jpeg","png"], key="pp_upload")
            if img is not None and st.button("Upload photo"):
                url = upload_avatar(img, p["name"])
                if url:
                    sb = get_service_client()
                    sb.table("players").update({"photo_url": url}).eq("id", p["id"]).execute()
                    clear_caches()
                    st.success("Photo updated.")
                    st.rerun()

    with col2:
        if me is not None:
            st.markdown(f"**Career** — GP: {int(me['gp'])} · W-D-L: {int(me['w'])}-{int(me['d'])}-{int(me['l'])} · Win%: {me['win_pct']}%")
            st.markdown(f"**Goals**: {int(me['goals'])} · **Assists**: {int(me['assists'])} · **G+A**: {int(me['ga'])} · **G+A/PG**: {me['ga_pg']} · **Team Contribution**: {me['team_contrib_pct']}%")

        # Recent games
        st.markdown("#### Recent Games")
        if my_rows.empty:
            st.info("No games yet.")
        else:
            for _, r in my_rows.sort_values(["season","gw"], ascending=[False, False]).head(10).iterrows():
                s = f"S{r['season']} GW{r['gw']} · {r['team_a']} {r['score_a']}–{r['score_b']} {r['team_b']} — {r['team']}"
                s2 = f"⚽ {int(r['goals'])} · 🅰️ {int(r['assists'])} · Result: {r['result']}"
                st.write(s)
                st.caption(s2)

        # Awards
        st.markdown("#### Awards")
        aw = fetch_awards()
        mine = aw[(aw["player_name"]==selected) | (aw["player_id"]==p["id"])]
        if mine.empty:
            st.caption("No awards yet.")
        else:
            for _, a in mine.sort_values(["season","month","gw"]).iterrows():
                st.write(f"🏅 {a['type']} — Season {a['season']} · {('Month '+str(a['month'])) if pd.notna(a['month']) else ''} · GW {a.get('gw') or ''}")

def page_stats():
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    lfact, _ = build_fact_tables(players, matches, lineups)

    st.subheader("Stats & Leaderboards")
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Season", [None] + sorted(matches["season"].dropna().unique().tolist()), index=0, key="st_season")
    min_games = c2.number_input("Min games", min_value=0, value=0, step=1, key="st_min")
    last_x = c3.selectbox("Last X GWs", [None,3,5,10], index=0, key="st_last")
    show_photos = st.toggle("Show photos", value=True, key="st_photos")

    agg = player_aggregate(lfact, season=season, min_games=min_games, last_gw=last_x)

    def leaderboard(df: pd.DataFrame, value_col: str, title: str, n: int = 10):
        if df.empty:
            st.info(f"No data for {title}.")
            return
        st.markdown(f"### {title}")
        rows = df.sort_values([value_col, "ga","goals"], ascending=False).head(n).to_dict(orient="records")
        for r in rows:
            col1, col2 = st.columns([4,1])
            with col1:
                img = r.get("photo") if show_photos else None
                st.markdown(chip_html(r['name'], (int(r.get("goals",0)), int(r.get("assists",0))), img), unsafe_allow_html=True)
            with col2:
                st.metric(value_col.replace("_"," ").title(), r[value_col])

    leaderboard(agg, "goals", "Top Scorers")
    leaderboard(agg, "assists", "Top Assisters")
    leaderboard(agg, "ga", "G + A")
    leaderboard(agg.sort_values("team_contrib_pct", ascending=False), "team_contrib_pct", "Team Contribution %")

    best, worst = best_worst_duos(lfact, season=season)
    st.markdown("### Best Duos (min 3 games)")
    if best.empty:
        st.caption("Not enough data.")
    else:
        for _, r in best.iterrows():
            st.write(f"{r['pair']} — {r['w']}-{r['d']}-{r['l']} · {r['win_pct']}%")

    st.markdown("### Worst Duos (min 3 games)")
    if worst.empty:
        st.caption("Not enough data.")
    else:
        for _, r in worst.iterrows():
            st.write(f"{r['pair']} — {r['w']}-{r['d']}-{r['l']} · {r['win_pct']}%")

    st.markdown("### Nemesis (min 3 H2H games)")
    nem = nemesis_table(lfact, season=season)
    if nem.empty:
        st.caption("Not enough data.")
    else:
        for _, r in nem.iterrows():
            st.write(f"{r['a']} vs {r['b']} — {r['w']}W/{r['l']}L · {r['win_pct']}%")

def page_awards():
    aw = fetch_awards()
    st.subheader("Awards")
    if st.session_state.get("is_admin"):
        with st.form("add_award_form"):
            season = st.number_input("Season", value=1, step=1, min_value=1, key="aw_season")
            month = st.number_input("Month (1-12, for POTM)", value=1, step=1, min_value=1, max_value=12, key="aw_month")
            atype = st.selectbox("Type", ["MOTM","POTM"], key="aw_type")
            gw = st.number_input("Gameweek (for MOTM)", value=1, step=1, min_value=1, key="aw_gw")
            player_name = st.text_input("Player name", key="aw_name")
            notes = st.text_input("Notes", key="aw_notes")
            if st.form_submit_button("Add Award"):
                sb = get_service_client()
                if sb is None:
                    st.error("Admin required.")
                else:
                    sb.table("awards").insert({
                        "season": int(season),
                        "month": int(month) if atype=="POTM" else None,
                        "type": atype,
                        "gw": int(gw) if atype=="MOTM" else None,
                        "player_name": player_name,
                        "notes": notes
                    }).execute()
                    clear_caches()
                    st.success("Award saved.")
                    st.rerun()
    st.markdown("#### Player of the Month")
    potm = aw[aw["type"]=="POTM"]
    if potm.empty:
        st.caption("No POTM yet.")
    else:
        for _, r in potm.sort_values(["season","month"]).iterrows():
            st.write(f"🏆 Season {r['season']} · Month {int(r['month'])}: {r['player_name']}")
    st.markdown("#### Man of the Match (History)")
    motm = aw[aw["type"]=="MOTM"]
    if motm.empty:
        st.caption("No MOTM yet.")
    else:
        for _, r in motm.sort_values(["season","gw"]).iterrows():
            extra = f" · Notes: {r['notes']}" if r.get("notes") else ""
            st.write(f"🎖️ S{r['season']} GW{r['gw']}: {r['player_name']}{extra}")

def page_import_export():
    st.subheader("Import / Export")
    if not st.session_state.get("is_admin"):
        st.warning("Admin only.")
        return
    st.caption("Upload CSVs in this order: players → matches → lineups")
    c1, c2, c3 = st.columns(3)
    up_players = c1.file_uploader("players.csv", type=["csv"])
    up_matches = c2.file_uploader("matches.csv", type=["csv"])
    up_lineups = c3.file_uploader("lineups.csv", type=["csv"])
    if up_players and st.button("Import players"):
        df = pd.read_csv(up_players)
        upsert_players(df)
        st.success("Players imported.")
    if up_matches and st.button("Import matches"):
        df = pd.read_csv(up_matches)
        upsert_matches(df)
        st.success("Matches imported.")
    if up_lineups and st.button("Import lineups"):
        df = pd.read_csv(up_lineups)
        insert_lineups(df)
        st.success("Lineups imported.")
    st.divider()
    # Exports — headers order
    players = fetch_players()
    matches = fetch_matches()
    lineups = fetch_lineups()
    def to_csv_download(df: pd.DataFrame, cols: List[str], label: str, key: str):
        out = io.StringIO()
        df = df[cols] if not df.empty else pd.DataFrame(columns=cols)
        df.to_csv(out, index=False)
        st.download_button(label, out.getvalue().encode("utf-8"), file_name=key, mime="text/csv")
    c1, c2, c3 = st.columns(3)
    to_csv_download(players, ["name","photo_url","notes"], "Export players.csv", "players.csv")
    to_csv_download(matches, ["season","gw","side_count","team_a","team_b","score_a","score_b","date","motm_name","is_draw","formation_a","formation_b","notes"], "Export matches.csv", "matches.csv")
    to_csv_download(lineups, ["season","gw","match_id","team","player_name","player_id","is_gk","goals","assists","line","slot","position"], "Export lineups.csv", "lineups.csv")
    st.button("Force refresh", on_click=clear_caches)



# ============================
# Router
# ============================
header()
st.markdown("")
nav_fn = getattr(st, "navigation", None)
Page = getattr(st, "Page", None)

if callable(nav_fn) and Page is not None:
    # Build sections using st.Page objects
    main_pages = [
        Page(page_matches, title="Matches", icon="📋"),
        Page(page_players, title="Players", icon="👤"),
        Page(page_stats, title="Stats", icon="📊"),
        Page(page_awards, title="Awards", icon="🏆"),
    ]
    pages_dict = {"Main": main_pages}
    if st.session_state.get("is_admin"):
        pages_dict["Admin"] = [Page(page_import_export, title="Import/Export", icon="⤴️")]
    nav = nav_fn(pages_dict)
    nav.run()
else:
    # Fallback for older Streamlit versions
    pages = {
        "Matches": page_matches,
        "Players": page_players,
        "Stats": page_stats,
        "Awards": page_awards,
    }
    if st.session_state.get("is_admin"):
        pages["Import/Export"] = page_import_export
    choice = st.sidebar.radio("Navigate", list(pages.keys()), index=0)
    pages[choice]()
