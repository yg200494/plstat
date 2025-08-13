# app.py — Powerleague Stats (Streamlit + Supabase)
# Mobile-first, admin-only writes, CSV import/export, drag-order lineups, avatars (HEIC→PNG).
# Fixes:
# - Duplicate key crash in Matches (unique keys per tab)
# - FotMob-style green pitch rendering
# - Drag-and-drop ordering (add/edit)
# - Full Player Profile page with nemesis + best teammate
# - NaN-safe CSV import; avatar join hot-fix

import io
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pillow_heif import read_heif
from streamlit_sortables import sort_items
from supabase import create_client, Client

# -----------------------
# Config & Clients
# -----------------------
st.set_page_config(page_title="Powerleague Stats", page_icon="⚽", layout="centered")

SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")
AVATAR_BUCKET = st.secrets.get("AVATAR_BUCKET", "avatars")

if not (SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_SERVICE_KEY):
    st.error("Missing Supabase credentials in secrets. Please set SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_clients() -> Tuple[Client, Client]:
    # anon for reads, service for writes
    anon = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return anon, service

sb_read, sb_write = get_clients()

# -----------------------
# Auth (single admin password)
# -----------------------
def is_admin() -> bool:
    return st.session_state.get("is_admin", False)

def admin_gate():
    if is_admin():
        c1, c2 = st.columns([1,3])
        with c1:
            st.success("Admin mode")
        with c2:
            if st.button("Sign out", key="btn_signout"):
                st.session_state["is_admin"] = False
                st.rerun()
        return True
    pw = st.text_input("Admin password", type="password", key="admin_pw")
    if st.button("Enter admin", key="btn_admin_enter"):
        if pw and pw == ADMIN_PASSWORD:
            st.session_state["is_admin"] = True
            st.success("Admin mode enabled")
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

# -----------------------
# Helpers & Caching
# -----------------------
def _nan_to_none(v):
    """Normalize pandas NaN/empty strings to None for JSON encoding."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return None
    return v

def _to_int(v, default=0):
    v = _nan_to_none(v)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)

def _to_bool(v, default=False):
    v = _nan_to_none(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true","1","t","yes","y")
    return bool(default)

@st.cache_data(ttl=60)
def fetch_players_df() -> pd.DataFrame:
    res = sb_read.table("players").select("*").order("name").execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        df = pd.DataFrame(columns=["id","name","photo_url","notes"])
    return df

@st.cache_data(ttl=60)
def fetch_matches_df() -> pd.DataFrame:
    res = sb_read.table("matches").select("*").order("season").order("gw").execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        df = pd.DataFrame(columns=[
            "id","season","gw","side_count","team_a","team_b","score_a","score_b",
            "date","motm_name","is_draw","formation_a","formation_b","notes"])
    return df

@st.cache_data(ttl=60)
def fetch_lineups_by_match(match_id: str) -> pd.DataFrame:
    res = sb_read.table("lineups").select("*").eq("match_id", match_id)\
        .order("team").order("line").order("slot").execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        df = pd.DataFrame(columns=[
            "id","season","gw","match_id","team","player_id","player_name","is_gk",
            "goals","assists","line","slot","position"])
    return df

@st.cache_data(ttl=60)
def fetch_awards(season: Optional[int]=None) -> pd.DataFrame:
    q = sb_read.table("awards").select("*")
    if season:
        q = q.eq("season", season)
    res = q.order("season").order("month").order("gw").execute()
    return pd.DataFrame(res.data or [])

def clear_cache():
    fetch_players_df.clear()
    fetch_matches_df.clear()
    fetch_lineups_by_match.clear()
    fetch_awards.clear()

# name normalization for imports
ALIAS = {
    "Ani": "Anirudh Gautam",
    "Abdullah Y13": "Mohammad Abdullah",
}

# -----------------------
# Storage: image upload / HEIC convert
# -----------------------
def _image_bytes_from_upload(file) -> Tuple[bytes, str]:
    name = file.name.lower()
    if name.endswith(".heic") or name.endswith(".heif"):
        heif = read_heif(file.read())
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    else:
        img = Image.open(file)
        buf = io.BytesIO()
        img.save(buf, format="PNG")  # normalize to PNG
        return buf.getvalue(), "image/png"

def upload_avatar(player_id: str, file) -> Optional[str]:
    content, mime = _image_bytes_from_upload(file)
    path = f"{player_id}.png"
    # delete any existing
    try:
        sb_write.storage.from_(AVATAR_BUCKET).remove([path])
    except Exception:
        pass
    res = sb_write.storage.from_(AVATAR_BUCKET).upload(
        path, content, {"content-type": mime, "upsert": True}
    )
    if hasattr(res, "error") and res.error:
        st.error(f"Upload error: {res.error}")
        return None
    public_url = sb_write.storage.from_(AVATAR_BUCKET).get_public_url(path)
    return public_url

# -----------------------
# CSV Import / Export (Admin)
# -----------------------
def upsert_players(df: pd.DataFrame):
    """players.csv: name,photo_url,notes (NaN-safe). Upsert by name."""
    rows = []
    for _, r in df.iterrows():
        name = _nan_to_none(r.get("name"))
        if not name:
            continue
        rows.append({
            "name": str(name).strip(),
            "photo_url": _nan_to_none(r.get("photo_url")),
            "notes": _nan_to_none(r.get("notes")),
        })
    if not rows:
        return
    sb_write.table("players").upsert(rows, on_conflict="name").execute()
    clear_cache()

def upsert_matches(df: pd.DataFrame):
    """matches.csv NaN-safe; upsert by (season,gw)."""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "season": _to_int(r.get("season")),
            "gw": _to_int(r.get("gw")),
            "side_count": _to_int(r.get("side_count"), default=5),
            "team_a": (_nan_to_none(r.get("team_a")) or "Non-bibs"),
            "team_b": (_nan_to_none(r.get("team_b")) or "Bibs"),
            "score_a": _to_int(r.get("score_a")),
            "score_b": _to_int(r.get("score_b")),
            "date": _nan_to_none(r.get("date")),
            "motm_name": _nan_to_none(r.get("motm_name")),
            "is_draw": _to_bool(r.get("is_draw"), default=False),
            "formation_a": _nan_to_none(r.get("formation_a")),
            "formation_b": _nan_to_none(r.get("formation_b")),
            "notes": _nan_to_none(r.get("notes")),
        })
    if rows:
        sb_write.table("matches").upsert(rows, on_conflict="season,gw").execute()
        clear_cache()

def insert_lineups(df: pd.DataFrame):
    """
    lineups.csv: delete-then-insert per (season,gw,team).
    columns: season,gw,team,player_name,is_gk,goals,assists,line,slot,position
    """
    players_df = fetch_players_df()
    matches_df = fetch_matches_df()
    name_to_id = dict(zip(players_df["name"], players_df["id"]))

    df = df.copy()
    df["player_name"] = df["player_name"].astype(str).map(lambda x: ALIAS.get(x, x))

    for (season, gw, team), sub in df.groupby(["season","gw","team"]):
        season_i = _to_int(season); gw_i = _to_int(gw)
        match_row = matches_df[(matches_df["season"]==season_i) & (matches_df["gw"]==gw_i)]
        if match_row.empty:
            st.error(f"Missing match for season {season_i} GW {gw_i} (cannot import lineups).")
            continue
        match_id = match_row.iloc[0]["id"]
        sb_write.table("lineups").delete().eq("match_id", match_id).eq("team", str(team)).execute()
        rows = []
        for _, r in sub.iterrows():
            pname = str(_nan_to_none(r.get("player_name")) or "")
            if not pname:
                continue
            pid = name_to_id.get(pname)
            if not pid:
                st.warning(f"Player '{pname}' not found in players table. Skipping.")
                continue
            rows.append({
                "season": season_i,
                "gw": gw_i,
                "match_id": match_id,
                "team": str(team),
                "player_id": pid,
                "player_name": pname,
                "is_gk": _to_bool(r.get("is_gk"), default=False),
                "goals": _to_int(r.get("goals")),
                "assists": _to_int(r.get("assists")),
                "line": _to_int(r.get("line"), default=1),
                "slot": _to_int(r.get("slot"), default=1),
                "position": _nan_to_none(r.get("position"))
            })
        if rows:
            sb_write.table("lineups").insert(rows).execute()
    clear_cache()

def export_table_csv(table: str, filename: str):
    res = sb_read.table(table).select("*").execute()
    df = pd.DataFrame(res.data or [])
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name=filename, mime="text/csv", key=f"dl_{table}")

# -----------------------
# UI Components
# -----------------------
def kpi(label: str, value: str):
    st.metric(label, value)

def fotmob_pitch(team_df: pd.DataFrame, photos_on: bool, motm_name: Optional[str], team_label: str):
    """
    Render a simple FotMob-style green pitch:
    - GK row (single)
    - Outfield rows grouped by 'line' then 'slot'
    """
    # Style wrapper
    st.markdown(
        """
        <style>
        .pitch {
            background: #0b6e0b;
            border-radius: 16px;
            padding: 10px 10px 4px 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
            color: white;
        }
        .pitch .row { margin: 6px 0; }
        .chip { font-size: 12px; opacity: 0.95; }
        .card {
            background: rgba(255,255,255,0.10);
            border-radius: 12px;
            padding: 6px 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"#### {team_label}")

    def chip(g,a):
        chips=[]
        if int(g)>0: chips.append(f"⚽x{int(g)}")
        if int(a)>0: chips.append(f"🅰️x{int(a)}")
        return "  ".join(chips)

    # Merge player photos (hot-fix join by player_id)
    P = fetch_players_df()[["id","photo_url"]]
    df = team_df.merge(P, left_on="player_id", right_on="id", how="left")
    if "id_y" in df.columns:
        df = df.drop(columns=["id_y"])

    gk = df[df["is_gk"]==True].copy()
    out = df[df["is_gk"]==False].copy()

    # GK single row
    if not gk.empty:
        st.markdown('<div class="pitch">', unsafe_allow_html=True)
        for _, r in gk.iterrows():
            star = " ⭐" if (motm_name and str(r["player_name"])==motm_name) else ""
            c = st.columns([1,5], vertical_alignment="center")
            with c[0]:
                if photos_on and r.get("photo_url"):
                    st.image(r["photo_url"], width=48)
                else:
                    st.markdown("🧤")
            with c[1]:
                st.markdown(f"<div class='card'><b>{r['player_name']}{star}</b><div class='chip'>{chip(r['goals'], r['assists'])}</div></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Outfield grouped by 'line'
    if not out.empty:
        for ln, sub in out.groupby("line"):
            sub = sub.sort_values(["slot","player_name"])
            st.markdown('<div class="pitch">', unsafe_allow_html=True)
            cols = st.columns(len(sub)) if len(sub) > 0 else st.columns(1)
            for col, (_, r) in zip(cols, sub.iterrows()):
                with col:
                    star = " ⭐" if (motm_name and str(r["player_name"])==motm_name) else ""
                    if photos_on and r.get("photo_url"):
                        st.image(r["photo_url"], width=48)
                    else:
                        st.markdown("👟")
                    st.markdown(f"<div class='card'><b>{r['player_name']}{star}</b><div class='chip'>{chip(r['goals'], r['assists'])}</div></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def match_selector(key: str):
    matches = fetch_matches_df()
    if matches.empty:
        st.info("No matches yet.")
        return None
    options = matches.apply(
        lambda r: f"S{r['season']} GW{int(r['gw'])} — {r['team_a']} {r['score_a']}–{r['score_b']} {r['team_b']}",
        axis=1
    ).tolist()
    choice = st.selectbox("Select match", options, key=key)
    idx = options.index(choice)
    return matches.iloc[idx]

# -----------------------
# Stats Computation
# -----------------------
def compute_player_stats(season: Optional[int]=None, last_gw: Optional[int]=None, min_games: int=0):
    matches = fetch_matches_df()
    if season:
        matches = matches[matches["season"]==season]
    if last_gw:
        max_gw = matches["gw"].max() if not matches.empty else 0
        matches = matches[matches["gw"]>max_gw-last_gw]
    mids = matches["id"].tolist()
    ln = []
    for mid in mids:
        df = fetch_lineups_by_match(mid)
        if not df.empty:
            m = matches[matches["id"]==mid].iloc[0]
            df = df.copy()
            df["score_a"] = m["score_a"]
            df["score_b"] = m["score_b"]
            df["team_a"] = m["team_a"]
            df["team_b"] = m["team_b"]
            df["motm_name"] = m.get("motm_name")
            df["is_draw"] = bool(m.get("is_draw"))
            df["gw"] = int(m["gw"])
            ln.append(df)
    if not ln:
        return pd.DataFrame()
    L = pd.concat(ln, ignore_index=True)

    def result(row):
        if row["team"] == row["team_a"]:
            a, b = int(row["score_a"]), int(row["score_b"])
        else:
            a, b = int(row["score_b"]), int(row["score_a"])
        if bool(row["is_draw"]) or a==b:
            return "D"
        return "W" if a>b else "L"

    L["result"] = L.apply(result, axis=1)
    team_goals = L.groupby(["season","gw","team"], as_index=False)["goals"].sum().rename(columns={"goals":"team_goals"})
    L = L.merge(team_goals, on=["season","gw","team"], how="left")

    agg = L.groupby(["player_id","player_name"], as_index=False).agg(
        GP=("match_id","nunique"),
        W=("result", lambda s: (s=="W").sum()),
        D=("result", lambda s: (s=="D").sum()),
        L_=("result", lambda s: (s=="L").sum()),
        Goals=("goals","sum"),
        Assists=("assists","sum"),
        TeamGoals=("team_goals","sum")
    )
    agg["GA"] = agg["Goals"] + agg["Assists"]
    agg["Win%"] = (agg["W"] / agg["GP"]).round(3)
    agg["G/GM"] = (agg["Goals"] / agg["GP"]).replace([np.inf, np.nan], 0).round(3)
    agg["A/GM"] = (agg["Assists"] / agg["GP"]).replace([np.inf, np.nan], 0).round(3)
    agg["(G+A)/GM"] = (agg["GA"] / agg["GP"]).replace([np.inf, np.nan], 0).round(3)
    agg["Team Contribution%"] = ((agg["GA"] / agg["TeamGoals"]).replace([np.inf, np.nan], 0) * 100).round(1)
    agg = agg[agg["GP"] >= int(min_games)]
    agg = agg.rename(columns={"L_":"L"})
    return agg.sort_values(["GA","Goals"], ascending=False)

def compute_duos(min_games_together=3):
    """Pairs on SAME team; Win%."""
    matches = fetch_matches_df()
    L_all: List[tuple] = []
    for _, m in matches.iterrows():
        lid = m["id"]
        df = fetch_lineups_by_match(lid)
        if df.empty:
            continue
        for team in ["Non-bibs","Bibs"]:
            t = df[df["team"]==team]
            ids = t["player_id"].tolist()
            names = t["player_name"].tolist()
            if len(ids) < 2:
                continue
            a, b = int(m["score_a"]), int(m["score_b"])
            team_score = a if team==m["team_a"] else b
            opp_score = b if team==m["team_a"] else a
            res = "D" if (bool(m["is_draw"]) or team_score==opp_score) else ("W" if team_score>opp_score else "L")
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    L_all.append((ids[i], names[i], ids[j], names[j], res))
    if not L_all:
        return pd.DataFrame()
    D = pd.DataFrame(L_all, columns=["p1","n1","p2","n2","res"])
    grp = D.groupby(["p1","n1","p2","n2"], as_index=False).agg(
        GP=("res","count"),
        W=("res", lambda s:(s=="W").sum()),
        D=("res", lambda s:(s=="D").sum()),
        L=("res", lambda s:(s=="L").sum())
    )
    grp = grp[grp["GP"]>=int(min_games_together)].copy()
    grp["Win%"] = (grp["W"]/grp["GP"]).round(3)
    return grp.sort_values("Win%", descending=False).sort_values("Win%", ascending=False)

def compute_duo_ga_per_game():
    """Combined G+A per game for pairs on SAME team."""
    matches = fetch_matches_df()
    rows = []
    for _, m in matches.iterrows():
        lid = m["id"]
        L = fetch_lineups_by_match(lid)
        if L.empty:
            continue
        for team in ["Non-bibs","Bibs"]:
            T = L[L["team"]==team]
            if T.empty: continue
            for i in range(len(T)):
                for j in range(i+1, len(T)):
                    r1 = T.iloc[i]; r2 = T.iloc[j]
                    rows.append({
                        "p1": r1["player_id"], "n1": r1["player_name"],
                        "p2": r2["player_id"], "n2": r2["player_name"],
                        "ga": int(r1["goals"])+int(r1["assists"]) + int(r2["goals"])+int(r2["assists"]),
                        "match_id": lid
                    })
    if not rows:
        return pd.DataFrame()
    D = pd.DataFrame(rows)
    g = D.groupby(["p1","n1","p2","n2"], as_index=False).agg(
        GP=("match_id","nunique"),
        GA=("ga","sum")
    )
    g["GA/GM"] = (g["GA"]/g["GP"]).round(3)
    return g.sort_values("GA/GM", ascending=False)

def compute_nemesis_for_player(pid: str) -> pd.DataFrame:
    """
    Opposite-side opponent that gives this player the worst Win% (min 2 meetings).
    """
    matches = fetch_matches_df()
    rows = []
    for _, m in matches.iterrows():
        lid = m["id"]
        L = fetch_lineups_by_match(lid)
        if L.empty: continue
        if pid not in L["player_id"].values:
            continue
        # find player's team
        my_row = L[L["player_id"]==pid].iloc[0]
        my_team = my_row["team"]
        opp_team = "Bibs" if my_team=="Non-bibs" else "Non-bibs"
        # result from my team perspective
        a, b = int(m["score_a"]), int(m["score_b"])
        my_score = a if my_team==m["team_a"] else b
        opp_score = b if my_team==m["team_a"] else a
        res = "D" if (bool(m.get("is_draw")) or my_score==opp_score) else ("W" if my_score>opp_score else "L")
        # accumulate vs each opponent on the other team
        for _, opp in L[L["team"]==opp_team].iterrows():
            rows.append((opp["player_id"], opp["player_name"], res))
    if not rows:
        return pd.DataFrame()
    D = pd.DataFrame(rows, columns=["opp_id","opp_name","res"])
    G = D.groupby(["opp_id","opp_name"], as_index=False).agg(
        GP=("res","count"),
        W=("res", lambda s:(s=="W").sum()),
        D=("res", lambda s:(s=="D").sum()),
        L=("res", lambda s:(s=="L").sum())
    )
    G = G[G["GP"]>=2].copy()
    if G.empty:
        return G
    G["Win%"] = (G["W"]/G["GP"]).round(3)
    return G.sort_values(["Win%","GP"], ascending=[True, False])

# -----------------------
# Pages
# -----------------------
def page_overview():
    st.header("Powerleague Overview")
    matches = fetch_matches_df()
    players = fetch_players_df()
    if matches.empty:
        st.info("No matches yet. Import CSVs (Admin → Import).")
        return
    total_matches = len(matches)
    total_goals = (matches["score_a"].astype(int) + matches["score_b"].astype(int)).sum()
    unique_players = len(players)
    k1, k2, k3 = st.columns(3)
    with k1: kpi("Matches", str(total_matches))
    with k2: kpi("Total Goals", str(int(total_goals)))
    with k3: kpi("Players", str(unique_players))

    st.subheader("Latest Match")
    latest = matches.iloc[-1]
    st.write(f"**S{latest['season']} GW{int(latest['gw'])}** — {latest['team_a']} {latest['score_a']}–{latest['score_b']} {latest['team_b']}")
    st.caption(f"MOTM: {latest.get('motm_name') or '—'}  ·  {'Draw' if latest.get('is_draw') else 'Result'}")

    st.divider()
    st.subheader("Leaders (All time)")
    agg = compute_player_stats(min_games=0)
    if not agg.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            top_scorers = agg.sort_values(["Goals","GA"], ascending=False).head(5)[["player_name","Goals","GP"]]
            st.dataframe(top_scorers, hide_index=True, use_container_width=True)
        with c2:
            top_assisters = agg.sort_values(["Assists","GA"], ascending=False).head(5)[["player_name","Assists","GP"]]
            st.dataframe(top_assisters, hide_index=True, use_container_width=True)
        with c3:
            top_ga = agg.sort_values(["GA","Goals"], ascending=False).head(5)[["player_name","GA","GP"]]
            st.dataframe(top_ga, hide_index=True, use_container_width=True)

def page_matches():
    st.header("Matches")
    tab1, tab2, tab3 = st.tabs(["Summary", "Add/Edit", "Export"])

    # --- Summary ---
    with tab1:
        m = match_selector("match_select_summary")
        if m is not None:
            st.subheader(f"S{m['season']} · GW{int(m['gw'])}")
            st.caption(f"{m['team_a']} vs {m['team_b']} · Score: {m['score_a']}–{m['score_b']}")
            st.caption(f"MOTM: {m.get('motm_name') or '—'}  ·  {'Draw' if m.get('is_draw') else 'Result'}")
            photos_on = st.toggle("Show photos", value=True, key="pitch_photos_on")
            L = fetch_lineups_by_match(m["id"])
            if L.empty:
                st.info("No lineups yet.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    nb = L[L["team"]=="Non-bibs"]
                    fotmob_pitch(nb, photos_on, m.get("motm_name"), "Non-bibs")
                with c2:
                    bb = L[L["team"]=="Bibs"]
                    fotmob_pitch(bb, photos_on, m.get("motm_name"), "Bibs")

    # --- Add/Edit ---
    with tab2:
        if not admin_gate():
            st.info("Admin only.")
            return

        st.subheader("Add Match")
        season = st.number_input("Season", min_value=2024, max_value=2100, value=2025, step=1, key="add_season")
        gw = st.number_input("Gameweek", min_value=1, max_value=200, value=1, step=1, key="add_gw")
        date = st.date_input("Date", value=None, key="add_date")
        side_count = st.selectbox("Side count", [5,6,7], index=0, key="add_side")
        formations = {
            5: ["1-2-1","2-1-1","1-3-0","2-0-2"],
            6: ["1-2-2-0","1-2-1-1","1-1-2-1"],
            7: ["1-2-3-1","1-3-2-1","1-2-2-1","1-1-3-1"]
        }
        fa = st.selectbox("Formation (Non-bibs)", formations[side_count], key="form_a")
        fb = st.selectbox("Formation (Bibs)", formations[side_count], key="form_b")

        players = fetch_players_df()
        options = players["name"].tolist()
        nb = st.multiselect("Select players (Non-bibs)", options, max_selections=side_count, key="sel_nb")
        bb = st.multiselect("Select players (Bibs)", options, max_selections=side_count, key="sel_bb")
        st.caption("Drag players to order (GK first).")

        col1, col2 = st.columns(2)
        with col1:
            nb_order = sort_items(nb, direction="horizontal", key="sort_nb")
        with col2:
            bb_order = sort_items(bb, direction="horizontal", key="sort_bb")

        if st.button("Create match & lineups", key="btn_create_match"):
            mrow = {
                "season": int(season), "gw": int(gw), "side_count": int(side_count),
                "team_a": "Non-bibs", "team_b": "Bibs",
                "score_a": 0, "score_b": 0,
                "date": str(date) if date else None,
                "motm_name": None, "is_draw": False,
                "formation_a": fa, "formation_b": fb, "notes": None
            }
            sb_write.table("matches").upsert([mrow], on_conflict="season,gw").execute()
            clear_cache()
            mdf = fetch_matches_df()
            row = mdf[(mdf["season"]==int(season)) & (mdf["gw"]==int(gw))].iloc[0]
            match_id = row["id"]
            # build lineups
            pid_map = dict(zip(players["name"], players["id"]))
            def build_rows(team, ordered):
                out=[]
                for i, name in enumerate(ordered):
                    out.append({
                        "season": int(season), "gw": int(gw), "match_id": match_id, "team": team,
                        "player_id": pid_map.get(name), "player_name": name,
                        "is_gk": i==0, "goals": 0, "assists": 0,
                        "line": 0 if i==0 else 1, "slot": 1 if i==0 else i, "position": None
                    })
                return out
            sb_write.table("lineups").delete().eq("match_id", match_id).execute()
            rows = build_rows("Non-bibs", nb_order) + build_rows("Bibs", bb_order)
            if rows:
                sb_write.table("lineups").insert(rows).execute()
            clear_cache()
            st.success("Match created.")
            st.rerun()

        st.divider()
        st.subheader("Edit Match")
        m = match_selector("match_select_edit")
        if m is not None:
            st.caption("Edit formations, GK, goals/assists, score, MOTM. Drag to change on-field order.")
            la = fetch_lineups_by_match(m["id"]).sort_values(["team","line","slot"])
            # Drag reorder per team (keeps GK first; outfield order changes slot)
            for team in ["Non-bibs","Bibs"]:
                st.markdown(f"### {team}")
                tdf = la[la["team"]==team].copy()
                # Build two lists: GK [single] and Outfield (names)
                if tdf.empty:
                    st.info("No players for this team yet.")
                    continue
                gk_name = tdf[tdf["is_gk"]==True]["player_name"].tolist()
                if not gk_name and not tdf.empty:
                    # force first as GK if none marked
                    gk_name = [tdf.iloc[0]["player_name"]]
                out_names = [n for n in tdf["player_name"].tolist() if n not in gk_name]
                st.caption("Outfield order (left→right):")
                new_order = sort_items(out_names, direction="horizontal", key=f"edit_sort_{team}_{m['id']}")
                # Inline GA + slot edits
                for _, r in tdf.iterrows():
                    c = st.columns([3,1,1,1])
                    with c[0]:
                        st.checkbox("GK", value=bool(r["is_gk"]), key=f"gk_{r['id']}")
                        if st.button("👤 Profile", key=f"profile_{r['id']}"):
                            st.session_state["profile_player_id"] = r["player_id"]
                            st.session_state["nav_radio"] = "Player Profile"
                            st.rerun()
                        st.markdown(f"**{r['player_name']}**")
                    with c[1]:
                        st.number_input("G", min_value=0, value=int(r["goals"]), key=f"g_{r['id']}")
                    with c[2]:
                        st.number_input("A", min_value=0, value=int(r["assists"]), key=f"a_{r['id']}")
                    with c[3]:
                        # Slot will be recomputed from drag order for outfielders on save
                        st.number_input("Slot", min_value=1, value=int(r["slot"]), key=f"s_{r['id']}")
                # Save button applies both drag order and field edits
                if st.button(f"Save {team}", key=f"btn_save_{team}_{m['id']}"):
                    # persist GA, GK and slot updates
                    tdf_now = fetch_lineups_by_match(m["id"])
                    # compute target slots from new_order (GK slot=1, outfield slots follow)
                    slot_map: Dict[str,int] = {}
                    slot_map[gk_name[0]] = 1 if gk_name else 1
                    for i, name in enumerate(new_order, start=2):
                        slot_map[name] = i
                    for _, r in tdf_now[tdf_now["team"]==team].iterrows():
                        pname = r["player_name"]
                        new_slot = int(slot_map.get(pname, r["slot"]))
                        sb_write.table("lineups").update({
                            "is_gk": bool(st.session_state.get(f"gk_{r['id']}")),
                            "goals": int(st.session_state.get(f"g_{r['id']}")),
                            "assists": int(st.session_state.get(f"a_{r['id']}")),
                            "slot": new_slot,
                            "line": 0 if bool(st.session_state.get(f"gk_{r['id']}")) else 1,
                        }).eq("id", r["id"]).execute()
                    clear_cache()
                    st.success(f"{team} saved.")
            # match meta (one save)
            m_motm = st.text_input("MOTM name", value=m.get("motm_name") or "", key="edit_motm")
            m_form_a = st.text_input("Formation A", value=m.get("formation_a") or "", key="edit_fa")
            m_form_b = st.text_input("Formation B", value=m.get("formation_b") or "", key="edit_fb")
            score_a = st.number_input("Score A", min_value=0, value=int(m["score_a"]), key="edit_sa")
            score_b = st.number_input("Score B", min_value=0, value=int(m["score_b"]), key="edit_sb")
            is_draw = st.checkbox("Draw", value=bool(m.get("is_draw")), key="edit_draw")
            if st.button("Save match meta", key="btn_save_matchmeta"):
                sb_write.table("matches").update({
                    "motm_name": (m_motm or None),
                    "formation_a": (m_form_a or None),
                    "formation_b": (m_form_b or None),
                    "score_a": int(score_a),
                    "score_b": int(score_b),
                    "is_draw": bool(is_draw)
                }).eq("id", m["id"]).execute()
                clear_cache()
                st.success("Match meta saved.")
                st.rerun()

    # --- Export ---
    with tab3:
        st.subheader("Export CSV")
        export_table_csv("players", "players_export.csv")
        export_table_csv("matches", "matches_export.csv")
        export_table_csv("lineups", "lineups_export.csv")
        export_table_csv("awards", "awards_export.csv")

def page_players():
    st.header("Players")
    players = fetch_players_df()
    if players.empty:
        st.info("No players yet.")
    else:
        # Gallery + admin edit controls
        for _, r in players.iterrows():
            with st.container():
                row = st.columns([1,3,3,2])
                with row[0]:
                    if r.get("photo_url"):
                        st.image(r["photo_url"], width=72)
                    else:
                        st.markdown("🧑‍🎤")
                with row[1]:
                    st.markdown(f"**{r['name']}**")
                    st.caption(r.get("notes") or "")
                    if st.button("Open profile", key=f"open_prof_{r['id']}"):
                        st.session_state["profile_player_id"] = r["id"]
                        st.session_state["nav_radio"] = "Player Profile"
                        st.rerun()
                if is_admin():
                    with row[2]:
                        with st.expander("Edit player", expanded=False):
                            new_name = st.text_input("Name", value=r["name"], key=f"pname_{r['id']}")
                            new_notes = st.text_area("Notes", value=r.get("notes") or "", key=f"pnotes_{r['id']}")
                            if st.button("Save", key=f"psave_{r['id']}"):
                                sb_write.table("players").update({
                                    "name": new_name.strip(),
                                    "notes": (new_notes or None)
                                }).eq("id", r["id"]).execute()
                                clear_cache()
                                st.success("Updated.")
                                st.rerun()
                            if st.button("Delete player", key=f"pdel_{r['id']}"):
                                sb_write.table("players").delete().eq("id", r["id"]).execute()
                                clear_cache()
                                st.warning("Deleted.")
                                st.rerun()
                    with row[3]:
                        up = st.file_uploader(f"Avatar ({r['name']})", type=["png","jpg","jpeg","heic","heif"], key=f"up_{r['id']}")
                        if up:
                            url = upload_avatar(r["id"], up)
                            if url:
                                sb_write.table("players").update({"photo_url": url}).eq("id", r["id"]).execute()
                                clear_cache()
                                st.success("Avatar updated.")
                                st.rerun()
    if is_admin():
        st.divider()
        st.subheader("Add new player")
        new_p = st.text_input("Player name", key="new_player_name")
        new_notes = st.text_input("Notes (optional)", key="new_player_notes")
        if st.button("Add player", key="btn_add_player") and new_p.strip():
            try:
                sb_write.table("players").insert({
                    "name": new_p.strip(),
                    "notes": (new_notes or None),
                    "photo_url": None
                }).execute()
                clear_cache()
                st.success("Player added.")
                st.rerun()
            except Exception:
                st.error("Could not add player (duplicate name?)")

def page_player_profile():
    st.header("Player Profile")
    players = fetch_players_df()
    if players.empty:
        st.info("No players yet.")
        return
    # Selector + deep link via session_state
    default_idx = 0
    if "profile_player_id" in st.session_state:
        pid = st.session_state["profile_player_id"]
        if pid in players["id"].values:
            default_idx = players.index[players["id"]==pid][0]
    p_choice = st.selectbox("Select player", players["name"].tolist(), index=default_idx, key="pp_select")
    P = players[players["name"]==p_choice].iloc[0]
    pid = P["id"]

    # Fetch all appearances
    matches = fetch_matches_df()
    rows = []
    for _, m in matches.iterrows():
        L = fetch_lineups_by_match(m["id"])
        if L.empty: continue
        me = L[L["player_id"]==pid]
        if me.empty: continue
        r = me.iloc[0]
        # result
        a, b = int(m["score_a"]), int(m["score_b"])
        my_team = r["team"]
        my_score = a if my_team==m["team_a"] else b
        opp_score = b if my_team==m["team_a"] else a
        res = "D" if (bool(m.get("is_draw")) or my_score==opp_score) else ("W" if my_score>opp_score else "L")
        rows.append({
            "season": int(m["season"]),
            "gw": int(m["gw"]),
            "match_id": m["id"],
            "team": my_team,
            "goals": int(r["goals"]),
            "assists": int(r["assists"]),
            "res": res,
            "score_for": my_score,
            "score_against": opp_score
        })
    if not rows:
        st.info("No appearances yet.")
        return
    A = pd.DataFrame(rows).sort_values(["season","gw"])

    # KPIs
    GP = len(A)
    W = (A["res"]=="W").sum()
    D = (A["res"]=="D").sum()
    L = (A["res"]=="L").sum()
    G = int(A["goals"].sum())
    A_ = int(A["assists"].sum())
    GA = G + A_
    winp = round(W/GP, 3) if GP else 0.0
    gpg = round(G/GP, 3) if GP else 0.0
    apg = round(A_/GP, 3) if GP else 0.0
    gapg = round(GA/GP, 3) if GP else 0.0

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("GP", GP)
    with c2: st.metric("W‑D‑L", f"{W}-{D}-{L}")
    with c3: st.metric("Win%", f"{winp:.3f}")

    c4,c5,c6 = st.columns(3)
    with c4: st.metric("Goals", G)
    with c5: st.metric("Assists", A_)
    with c6: st.metric("G+A / GM", f"{gapg:.3f}")

    # Streaks
    def streaks(seq: List[str], target: str) -> Dict[str,int]:
        longest = curr = 0
        for s in seq:
            if s == target:
                curr += 1
                longest = max(longest, curr)
            else:
                curr = 0
        # current streak (from end)
        curr2 = 0
        for s in reversed(seq):
            if s == target:
                curr2 += 1
            else:
                break
        return {"longest": longest, "current": curr2}

    ws = streaks(A["res"].tolist(), "W")
    ls = streaks(A["res"].tolist(), "L")
    st.caption(f"Streaks — Longest W: {ws['longest']} · Current W: {ws['current']} · Longest L: {ls['longest']} · Current L: {ls['current']}")

    # Best teammate by Win% and by GA/GM
    duos = compute_duos(min_games_together=2)
    duo_ga = compute_duo_ga_per_game()
    best_win = None
    if not duos.empty:
        d1 = duos[(duos["p1"]==pid) | (duos["p2"]==pid)].copy()
        if not d1.empty:
            d1["mate_id"] = np.where(d1["p1"]==pid, d1["p2"], d1["p1"])
            d1["mate_name"] = np.where(d1["p1"]==pid, d1["n2"], d1["n1"])
            best_win = d1.sort_values("Win%", ascending=False).head(1)
    best_ga = None
    if not duo_ga.empty:
        g1 = duo_ga[(duo_ga["p1"]==pid) | (duo_ga["p2"]==pid)].copy()
        if not g1.empty:
            g1["mate_id"] = np.where(g1["p1"]==pid, g1["p2"], g1["p1"])
            g1["mate_name"] = np.where(g1["p1"]==pid, g1["n2"], g1["n1"])
            best_ga = g1.sort_values("GA/GM", ascending=False).head(1)

    # Nemesis
    nem = compute_nemesis_for_player(pid)

    st.subheader("Teammate & Nemesis")
    c7, c8, c9 = st.columns(3)
    with c7:
        st.markdown("**Best teammate (Win%)**")
        if best_win is None or best_win.empty:
            st.caption("—")
        else:
            r = best_win.iloc[0]
            st.caption(f"{r['mate_name']} · {r['Win%']:.3f} ({int(r['W'])}-{int(r['D'])}-{int(r['L'])}, {int(r['GP'])} GP)")
    with c8:
        st.markdown("**Best teammate (G+A per game)**")
        if best_ga is None or best_ga.empty:
            st.caption("—")
        else:
            r = best_ga.iloc[0]
            st.caption(f"{r['mate_name']} · {r['GA/GM']:.3f} ({int(r['GA'])} G+A in {int(r['GP'])} GP)")
    with c9:
        st.markdown("**Nemesis (worst Win%)**")
        if nem is None or nem.empty:
            st.caption("—")
        else:
            r = nem.iloc[0]
            st.caption(f"{r['opp_name']} · {r['Win%']:.3f} ({int(r['W'])}-{int(r['D'])}-{int(r['L'])}, {int(r['GP'])} GP)")

    st.subheader("Recent games")
    show = A.sort_values(["season","gw"], ascending=False).head(10)[["season","gw","team","res","score_for","score_against","goals","assists"]]
    st.dataframe(show.rename(columns={
        "res":"Result","score_for":"For","score_against":"Against","goals":"G","assists":"A"
    }), hide_index=True, use_container_width=True)

def page_stats():
    st.header("Stats")
    st.caption("Filters")
    season = st.selectbox("Season", ["All","2025"], index=1, key="stats_season")
    min_games = st.slider("Min games", 0, 30, 0, key="stats_min_g")
    last_x = st.number_input("Last X GWs (optional, 0=all)", min_value=0, value=0, step=1, key="stats_lastx")
    s_val = None if season=="All" else int(season)
    agg = compute_player_stats(season=s_val, last_gw=(None if last_x==0 else last_x), min_games=min_games)
    if agg.empty:
        st.info("No data.")
        return
    mode = st.selectbox("Leaderboard", ["Top scorers","Top assisters","Top G+A","Team Contribution%","MOTM","Best duos (win%)","Worst duos (win%)"], key="stats_mode")
    st.toggle("Show photos", value=False, key="stats_photos")

    if mode in ["Top scorers","Top assisters","Top G+A","Team Contribution%"]:
        if mode == "Top scorers":
            df = agg.sort_values(["Goals","GA"], ascending=False)[["player_name","GP","W","D","L","Goals","G/GM","Win%","Team Contribution%"]]
        elif mode == "Top assisters":
            df = agg.sort_values(["Assists","GA"], ascending=False)[["player_name","GP","W","D","L","Assists","A/GM","Win%","Team Contribution%"]]
        elif mode == "Top G+A":
            df = agg.sort_values(["GA","Goals"], ascending=False)[["player_name","GP","W","D","L","Goals","Assists","GA","(G+A)/GM","Win%"]]
        else:
            df = agg.sort_values("Team Contribution%", ascending=False)[["player_name","GP","GA","Team Contribution%","Win%"]]
        st.dataframe(df, hide_index=True, use_container_width=True)
    elif mode == "MOTM":
        m = fetch_matches_df()
        mm = m[m["motm_name"].notnull()]
        df = mm.groupby("motm_name", as_index=False).agg(MOTM=("motm_name","count"))
        st.dataframe(df.sort_values("MOTM", ascending=False), hide_index=True, use_container_width=True)
    else:
        duos = compute_duos(min_games_together=max(2, min_games))
        if duos.empty:
            st.info("Not enough duo data.")
        else:
            if mode.startswith("Best"):
                df = duos.sort_values("Win%", ascending=False).head(50)
            else:
                df = duos.sort_values("Win%", ascending=True).head(50)
            df = df.rename(columns={"n1":"Player 1","n2":"Player 2"})
            st.dataframe(df[["Player 1","Player 2","GP","W","D","L","Win%"]], hide_index=True, use_container_width=True)

def page_awards():
    st.header("Awards")
    season = st.selectbox("Season", [2025], key="aw_season")
    a = fetch_awards(season=season)
    potm = a[a["type"]=="POTM"].sort_values(["month"])
    motm = a[a["type"]=="MOTM"].sort_values(["gw"])
    st.subheader("Player of the Month")
    if potm.empty:
        st.caption("No POTM yet.")
    else:
        st.dataframe(potm[["month","player_name","notes"]], hide_index=True, use_container_width=True)
    st.subheader("Man of the Match (history)")
    if motm.empty:
        st.caption("No MOTM entries yet.")
    else:
        st.dataframe(motm[["gw","player_name","notes"]], hide_index=True, use_container_width=True)

    if admin_gate():
        st.subheader("Add Award")
        typ = st.selectbox("Type", ["MOTM","POTM"], key="aw_type")
        month = None
        gw = None
        if typ == "POTM":
            month = st.selectbox("Month (1-12)", list(range(1,13)), key="aw_month")
        else:
            gw = st.number_input("Gameweek", min_value=1, max_value=500, value=1, step=1, key="aw_gw")
        p_df = fetch_players_df()
        pname = st.selectbox("Player", p_df["name"].tolist(), key="aw_player")
        pid = p_df[p_df["name"]==pname]["id"].iloc[0]
        notes = st.text_input("Notes (optional)", key="aw_notes")
        if st.button("Save award", key="btn_save_aw"):
            row = {
                "season": int(season),
                "month": int(month) if month else None,
                "type": typ,
                "gw": int(gw) if gw else None,
                "player_id": pid,
                "player_name": pname,
                "notes": notes or None
            }
            sb_write.table("awards").insert(row).execute()
            clear_cache()
            st.success("Award saved.")
            st.rerun()

def page_import():
    st.header("Import / Export CSV")
    if not admin_gate():
        st.info("Admin only.")
        return
    st.subheader("Import")
    st.caption("Import in this order: players.csv → matches.csv → lineups.csv")
    up1 = st.file_uploader("players.csv", type=["csv"], key="csv_p")
    up2 = st.file_uploader("matches.csv", type=["csv"], key="csv_m")
    up3 = st.file_uploader("lineups.csv", type=["csv"], key="csv_l")

    if st.button("Import players", key="btn_imp_p") and up1:
        df = pd.read_csv(up1)
        upsert_players(df)
        st.success("Players imported.")

    if st.button("Import matches", key="btn_imp_m") and up2:
        df = pd.read_csv(up2)
        upsert_matches(df)
        st.success("Matches imported.")

    if st.button("Import lineups", key="btn_imp_l") and up3:
        df = pd.read_csv(up3)
        insert_lineups(df)
        st.success("Lineups imported.")

    st.divider()
    st.subheader("Export")
    export_table_csv("players", "players.csv")
    export_table_csv("matches", "matches.csv")
    export_table_csv("lineups", "lineups.csv")
    export_table_csv("awards", "awards.csv")

# -----------------------
# App Nav
# -----------------------
PAGES = {
    "Overview": page_overview,
    "Matches": page_matches,
    "Players": page_players,
    "Player Profile": page_player_profile,
    "Stats": page_stats,
    "Awards": page_awards,
    "Import (Admin)": page_import,
}

with st.sidebar:
    st.title("⚽ Powerleague")
    page = st.radio("Navigate", list(PAGES.keys()), key="nav_radio")

PAGES[page]()
