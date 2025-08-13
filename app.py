
import io
import os
import uuid
import itertools
from datetime import datetime, date
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import streamlit as st
from dateutil.relativedelta import relativedelta
from PIL import Image
from pillow_heif import read_heif
from supabase import create_client, Client

# ----------------------------------
# Config & Secrets
# ----------------------------------
st.set_page_config(page_title="Powerleague Stats", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")

REQUIRED_SECRETS = ["SUPABASE_URL","SUPABASE_ANON_KEY","SUPABASE_SERVICE_KEY","ADMIN_PASSWORD","AVATAR_BUCKET"]
for key in REQUIRED_SECRETS:
    if key not in st.secrets:
        st.error(f"Missing secret: {key}. Add it to .streamlit/secrets.toml", icon="⚠️")
        st.stop()

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE = st.secrets["SUPABASE_SERVICE_KEY"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
AVATAR_BUCKET = st.secrets["AVATAR_BUCKET"]

# ----------------------------------
# Clients
# ----------------------------------
@st.cache_resource(show_spinner=False)
def get_clients() -> Dict[str, Client]:
    return {
        "anon": create_client(SUPABASE_URL, SUPABASE_ANON),
        "svc": create_client(SUPABASE_URL, SUPABASE_SERVICE)
    }

cli = get_clients()
sb: Client = cli["anon"]
svc: Client = cli["svc"]

# ----------------------------------
# Styles (FotMob feel, mobile-first)
# ----------------------------------
st.markdown(
    '<style>'
    ':root { --green:#1d6b3a; --pitch:#0b7a41; --chip:#eef2f6; --ink:#0f172a; }'
    '.block-container { padding-top:.6rem; padding-bottom:3.5rem; max-width: 970px; }'
    'header { visibility:hidden; height:0 }'
    '.banner { background:linear-gradient(135deg, var(--green), #0a9b50); color:#fff; padding:12px 14px; border-radius:16px; }'
    '.chip { display:inline-flex; align-items:center; gap:.35rem; padding:.22rem .5rem; border-radius:999px; background:var(--chip); color:var(--ink); font-size:.85rem; }'
    '.pitch { background: radial-gradient(ellipse at center, #0e8f4a 0%, #0b7a41 100%); border-radius:18px; padding:10px; color:white; }'
    '.grid { display:grid; gap:6px; }'
    '.player { background:rgba(255,255,255,.15); backdrop-filter: blur(2px); padding:4px 6px; border-radius:12px; text-align:center; font-weight:600; }'
    '.subtitle { opacity:.9; font-size:.9rem; }'
    '.muted { opacity:.7; }'
    '.avatar { width:28px; height:28px; border-radius:999px; object-fit:cover; margin-right:6px; border:2px solid rgba(255,255,255,.4); }'
    '@media (max-width: 600px){ .block-container { padding: .5rem .8rem; } }'
    '</style>',
    unsafe_allow_html=True
)

# ----------------------------------
# Auth helpers
# ----------------------------------
def is_admin() -> bool:
    return st.session_state.get("is_admin", False)

def login_area():
    with st.expander("🔐 Admin login", expanded=False):
        pwd = st.text_input("Password", type="password", key="__admin_pwd")
        if st.button("Log in", use_container_width=True):
            if pwd == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.success("Admin mode enabled")
                st.rerun()
            else:
                st.error("Wrong password")

# ----------------------------------
# Data access
# ----------------------------------
@st.cache_data(ttl=45, show_spinner=False)
def fetch_table(name: str) -> pd.DataFrame:
    res = sb.table(name).select("*").execute().data or []
    return pd.DataFrame(res)

def refresh_cache():
    fetch_table.clear()

# Upserts/replaces
def upsert_players(df: pd.DataFrame):
    cols = ["id","name","photo_url","notes"]
    payload = df[cols].where(pd.notnull(df), None).to_dict(orient="records")
    svc.table("players").upsert(payload, on_conflict="name").execute()

def upsert_matches(df: pd.DataFrame):
    # Default formations when blanks based on side_count
    def default_form(x):
        sc = int(x.get("side_count") or 5)
        if not x.get("formation_a"): x["formation_a"] = "1-2-1" if sc==5 else "2-1-2-1"
        if not x.get("formation_b"): x["formation_b"] = "1-2-1" if sc==5 else "2-1-2-1"
        if x.get("is_draw") in [1,"1","true","True",True]: x["is_draw"] = True
        elif x.get("is_draw") in [0,"0","false","False",False,None,""]: x["is_draw"] = False
        return x
    df = df.apply(default_form, axis=1)
    cols = ["id","season","gw","side_count","team_a","team_b","score_a","score_b","date","motm_name","is_draw","formation_a","formation_b","notes"]
    payload = df[cols].where(pd.notnull(df), None).to_dict(orient="records")
    svc.table("matches").upsert(payload, on_conflict="season,gw").execute()

def replace_lineups(df: pd.DataFrame):
    required = ["id","season","gw","match_id","team","player_id","player_name","is_gk","goals","assists","line","slot","position"]
    df = df[required]
    groups = df.groupby(["match_id","team"])
    for (mid, team), g in groups:
        svc.table("lineups").delete().eq("match_id", mid).eq("team", team).execute()
        payload = g.where(pd.notnull(g), None).to_dict(orient="records")
        if payload:
            svc.table("lineups").insert(payload).execute()

# ----------------------------------
# Images
# ----------------------------------
def heic_to_png_bytes(raw: bytes) -> bytes:
    heif = read_heif(raw)
    img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

def upload_avatar(file) -> str:
    img_bytes = file.read()
    name = f"{uuid.uuid4()}.png"
    if file.type in ["image/heic","image/heif"] or file.name.lower().endswith(".heic"):
        img_bytes = heic_to_png_bytes(img_bytes)
    if file.type in ["image/jpeg","image/jpg","image/png"] and (not file.name.lower().endswith(".png")):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            b = io.BytesIO()
            img.save(b, format="PNG")
            img_bytes = b.getvalue()
        except Exception:
            pass
    svc.storage.from_(AVATAR_BUCKET).upload(name, img_bytes, {"content-type": "image/png", "x-upsert": "true"})
    return sb.storage.from_(AVATAR_BUCKET).get_public_url(name)

# ----------------------------------
# Computations
# ----------------------------------
def result_for(team: str, row_m: pd.Series) -> str:
    a, b = int(row_m.get("score_a") or 0), int(row_m.get("score_b") or 0)
    if row_m.get("is_draw"): return "D"
    if team == "Non-bibs":
        return "W" if a>b else ("L" if a<b else "D")
    else:
        return "W" if b>a else ("L" if b<a else "D")

def join_lineups_matches(players: pd.DataFrame, lineups: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    df = lineups.merge(matches, left_on="match_id", right_on="id", how="left", suffixes=("_l","_m"))
    df.rename(columns={"id_l":"lineup_row_id","id_m":"match_row_id"}, inplace=True)
    # team goals for this player's team
    df["team_goals"] = np.where(df["team"]=="Non-bibs", df["score_a"], df["score_b"])
    df["opp_goals"] = np.where(df["team"]=="Non-bibs", df["score_b"], df["score_a"])
    df["result"] = df.apply(lambda r: result_for(r["team"], r), axis=1)
    return df

def per_player_aggregates(players, lineups, matches, season_filter=None, last_x=None):
    df = join_lineups_matches(players, lineups, matches)
    df = df.sort_values(["season_m","gw"])
    if season_filter and season_filter!="All":
        df = df[df["season_m"]==season_filter]
    if last_x and last_x>0:
        tail_ids = matches.sort_values(["season","gw"]).tail(last_x)["id"].tolist()
        df = df[df["match_id"].isin(tail_ids)]

    g = df.groupby("player_name").agg(
        games=("match_id","nunique"),
        goals=("goals","sum"),
        assists=("assists","sum"),
        team_goals=("team_goals","sum")
    )
    if g.empty: return g
    g["ga"] = g["goals"] + g["assists"]
    # Win/Draw/Loss
    res = df.pivot_table(index="player_name", values="result", aggfunc=lambda s: "".join(s))
    w = df[df["result"]=="W"].groupby("player_name").size().rename("wins")
    d = df[df["result"]=="D"].groupby("player_name").size().rename("draws")
    l = df[df["result"]=="L"].groupby("player_name").size().rename("losses")
    out = g.join(w, how="left").join(d, how="left").join(l, how="left").fillna(0)
    out[["wins","draws","losses"]] = out[["wins","draws","losses"]].astype(int)
    out["winp"] = np.where(out["games"]>0, (out["wins"]/out["games"]*100).round(1), 0)
    # Per-game
    out["gpg"] = (out["goals"]/out["games"]).replace([np.inf,np.nan],0).round(2)
    out["apg"] = (out["assists"]/out["games"]).replace([np.inf,np.nan],0).round(2)
    out["gapg"] = (out["ga"]/out["games"]).replace([np.inf,np.nan],0).round(2)
    # Contribution %
    out["contrib_pct"] = np.where(out["team_goals"]>0, (out["ga"]/out["team_goals"]*100).round(1), 0.0)
    # MOTM
    motm = matches.groupby("motm_name").size().rename("motm_count")
    out = out.join(motm, how="left").fillna({"motm_count":0})
    out["motm_count"] = out["motm_count"].astype(int)
    return out.sort_values(["ga","goals","assists"], ascending=False)

def streaks_for_player(name: str, lineups, matches) -> Tuple[int,int,str]:
    df = join_lineups_matches(None, lineups[lineups["player_name"]==name], matches).sort_values(["season_m","gw"])
    seq = df["result"].tolist()
    # current streak
    cur = 0; cur_type = ""
    for r in reversed(seq):
        if cur_type=="":
            cur_type = r; cur = 1
        elif r==cur_type:
            cur += 1
        else:
            break
    # best win streak
    best = 0; run=0
    for r in seq:
        if r=="W":
            run += 1; best = max(best, run)
        else:
            run = 0
    return best, cur, cur_type or "-"

def duo_stats(lineups, matches, min_games=2):
    df = join_lineups_matches(None, lineups, matches)
    rows = []
    for mid, grp in df.groupby("match_id"):
        # same-team duos
        for team, tg in grp.groupby("team"):
            players = tg["player_name"].tolist()
            res = tg["result"].iloc[0]
            goals = tg["goals"].sum(); assists = tg["assists"].sum()
            for a, b in itertools.combinations(sorted(players), 2):
                rows.append({"a":a,"b":b,"team":team,"result":res,
                             "ga": (tg[tg["player_name"].isin([a,b])]["goals"].sum() + tg[tg["player_name"].isin([a,b])]["assists"].sum())})
    if not rows: return pd.DataFrame()
    d = pd.DataFrame(rows)
    agg = d.groupby(["a","b"]).agg(
        games=("result","count"),
        wins=("result", lambda s: (s=="W").sum()),
        draws=("result", lambda s: (s=="D").sum()),
        losses=("result", lambda s: (s=="L").sum()),
        ga=("ga","sum")
    ).reset_index()
    agg = agg[agg["games"]>=min_games].copy()
    if agg.empty: return agg
    agg["winp"] = (agg["wins"]/agg["games"]*100).round(1)
    agg["ga_per_game"] = (agg["ga"]/agg["games"]).round(2)
    agg = agg.sort_values(["winp","ga_per_game","games"], ascending=[False,False,False])
    return agg

def nemesis_stats(lineups, matches, min_meetings=2):
    df = join_lineups_matches(None, lineups, matches)
    rows = []
    for mid, grp in df.groupby("match_id"):
        nb = grp[grp["team"]=="Non-bibs"]["player_name"].tolist()
        bb = grp[grp["team"]=="Bibs"]["player_name"].tolist()
        a, b = int(grp["score_a"].iloc[0] or 0), int(grp["score_b"].iloc[0] or 0)
        res_nb = "W" if a>b else ("L" if a<b else "D")
        res_bb = "W" if b>a else ("L" if b<a else "D")
        for x in nb:
            for y in bb:
                rows.append({"me":x,"opp":y,"res":res_nb})
        for x in bb:
            for y in nb:
                rows.append({"me":x,"opp":y,"res":res_bb})
    if not rows: return pd.DataFrame()
    d = pd.DataFrame(rows)
    agg = d.groupby(["me","opp"]).agg(
        games=("res","count"),
        losses=("res", lambda s: (s=="L").sum()),
        wins=("res", lambda s: (s=="W").sum()),
        draws=("res", lambda s: (s=="D").sum())
    ).reset_index()
    agg["loss_delta"] = agg["losses"] - agg["wins"]
    agg = agg[agg["games"]>=min_meetings]
    return agg.sort_values(["loss_delta","games"], ascending=[False,False])

# ----------------------------------
# Pitch
# ----------------------------------
def fotmob_pitch(formation: str, players: List[Dict[str, Any]], title: str):
    try:
        layers = [int(x) for x in str(formation).split("-")]
    except Exception:
        layers = [1,2,1]
    st.markdown("<div class='pitch'><div class='subtitle'>" + title + " • " + str(formation) + "</div>", unsafe_allow_html=True)
    gks = [p for p in players if p.get("is_gk")]
    def row_html(names):
        return "<div class='grid' style='grid-template-columns: repeat(" + str(max(1,len(names))) + ", 1fr);'>" + "".join(
            "<div class='player'><span>" + (n or "") + "</span></div>" for n in names
        ) + "</div>"
    st.markdown(row_html([p.get('name','GK') for p in gks]) or row_html(["GK"]), unsafe_allow_html=True)
    idx = 0
    outfield = [p for p in players if not p.get("is_gk")]
    for count in layers:
        names = [(outfield[i+idx]["name"] if (i+idx) < len(outfield) else "") for i in range(count)]
        idx += count
        st.markdown(row_html(names), unsafe_allow_html=True)
    chips = []
    for p in players:
        g = int(p.get("goals",0) or 0); a=int(p.get("assists",0) or 0)
        tags = []
        if g: tags.append("⚽ " + str(g))
        if a: tags.append("🅰️ " + str(a))
        if tags:
            chips.append("<span class='chip'>" + p.get('name','') + " • " + "  ".join(tags) + "</span>")
    if chips:
        st.markdown("<div style='margin-top:6px;display:flex;flex-wrap:wrap;gap:6px;'>" + "".join(chips) + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------
# PAGES
# ----------------------------------
def page_matches():
    st.title("🏟️ Matches")
    dfm = fetch_table("matches").sort_values(["season","gw"], ascending=[False, False])
    if dfm.empty:
        st.info("No matches yet.")
        return
    all_lineups = fetch_table("lineups")
    for _, m in dfm.iterrows():
        score = f"{int(m.score_a) if pd.notna(m.score_a) else '-'}—{int(m.score_b) if pd.notna(m.score_b) else '-'}"
        banner = (
            "<div class='banner'>"
            f"<div style='font-size:1rem;font-weight:700'>Season {m.season} • GW {m.gw} • {m.get('date','')}</div>"
            f"<div style='font-size:1.25rem; margin-top:2px;'>{m.team_a} {score} {m.team_b}</div>"
            f"<div class='subtitle'>MOTM: <b>{m.get('motm_name','—')}</b></div>"
            "</div>"
        )
        st.markdown(banner, unsafe_allow_html=True)
        la = all_lineups[(all_lineups["match_id"]==m["id"]) & (all_lineups["team"]=="Non-bibs")]
        lb = all_lineups[(all_lineups["match_id"]==m["id"]) & (all_lineups["team"]=="Bibs")]
        def pack(df):
            return [dict(name=r.get("player_name",""), goals=int(r.get("goals",0) or 0), assists=int(r.get("assists",0) or 0),
                         is_gk=bool(r.get("is_gk", False))) for _, r in df.iterrows()]
        c1,c2 = st.columns(2)
        with c1: fotmob_pitch(m.get("formation_a") or "1-2-1", pack(la), m.get("team_a","Non-bibs"))
        with c2: fotmob_pitch(m.get("formation_b") or "1-2-1", pack(lb), m.get("team_b","Bibs"))
        st.divider()

def add_match_wizard():
    st.title("➕ Add Match")
    if not is_admin():
        st.warning("Admin only.")
        login_area()
        return
    presets = {5:"1-2-1", 6:"2-2-1", 7:"2-1-2-1"}
    with st.form("add_match"):
        season = st.number_input("Season", min_value=2000, max_value=2100, value=date.today().year, step=1)
        gw = st.number_input("Gameweek", min_value=1, value=1, step=1)
        d = st.date_input("Date", value=date.today())
        side = st.selectbox("Side count", [5,6,7], index=0)
        team_a = st.text_input("Team A name", "Non-bibs")
        team_b = st.text_input("Team B name", "Bibs")
        form_a = st.text_input("Formation A", presets.get(side, "1-2-1"))
        form_b = st.text_input("Formation B", presets.get(side, "1-2-1"))
        score_a = st.number_input("Score A", 0, 99, 0)
        score_b = st.number_input("Score B", 0, 99, 0)
        is_draw = st.checkbox("Record as draw", value=(score_a==score_b))
        motm = st.text_input("MOTM name", "")
        notes = st.text_input("Notes")
        submitted = st.form_submit_button("Create match", use_container_width=True)
    if submitted:
        payload = {"season": int(season), "gw": int(gw), "side_count": int(side), "team_a": team_a, "team_b": team_b,
                   "score_a": int(score_a), "score_b": int(score_b), "date": str(d), "motm_name": motm,
                   "is_draw": bool(is_draw), "formation_a": form_a or presets.get(side,"1-2-1"),
                   "formation_b": form_b or presets.get(side,"1-2-1"), "notes": notes}
        res = svc.table("matches").upsert(payload, on_conflict="season,gw").execute()
        mid = res.data[0]["id"]
        st.success("Match saved.")
        refresh_cache()
        st.experimental_set_query_params(match_id=mid)
        st.rerun()

def edit_match():
    st.title("✏️ Edit Match & Lineups")
    if not is_admin():
        st.warning("Admin only.")
        login_area()
        return
    dfm = fetch_table("matches").sort_values(["season","gw"])
    if dfm.empty:
        st.info("No matches to edit."); return
    options = {f"S{r.season} GW{r.gw} ({r.team_a} vs {r.team_b})": r.id for _, r in dfm.iterrows()}
    sel = st.selectbox("Choose match", list(options.keys()))
    mid = options[sel]
    m = dfm[dfm["id"]==mid].iloc[0].to_dict()
    lups = fetch_table("lineups")
    la = lups[(lups["match_id"]==mid) & (lups["team"]=="Non-bibs")].copy()
    lb = lups[(lups["match_id"]==mid) & (lups["team"]=="Bibs")].copy()

    with st.form("edit_match_form"):
        c1,c2 = st.columns(2)
        with c1: m["formation_a"] = st.text_input("Formation A", m.get("formation_a","1-2-1"))
        with c2: m["formation_b"] = st.text_input("Formation B", m.get("formation_b","1-2-1"))
        c3,c4,c5 = st.columns(3)
        with c3: m["motm_name"] = st.text_input("MOTM", m.get("motm_name",""))
        with c4: m["score_a"] = st.number_input("Score A", 0, 99, int(m.get("score_a") or 0))
        with c5: m["score_b"] = st.number_input("Score B", 0, 99, int(m.get("score_b") or 0))
        m["is_draw"] = st.checkbox("Record as draw", value=bool(m.get("is_draw", False)))
        m["notes"] = st.text_input("Notes", m.get("notes") or "")
        st.subheader("Non-bibs")
        for i, row in la.iterrows():
            col = st.columns([3,1,1,1])
            la.at[i,"player_name"] = col[0].text_input(f"Name {i}", row["player_name"] or "", key=f"nba_{i}")
            la.at[i,"is_gk"] = col[1].checkbox("GK", bool(row.get("is_gk", False)), key=f"nbgk_{i}")
            la.at[i,"goals"] = col[2].number_input("G", 0, 50, int(row.get("goals") or 0), key=f"nbg_{i}")
            la.at[i,"assists"] = col[3].number_input("A", 0, 50, int(row.get("assists") or 0), key=f"nbaa_{i}")
        st.subheader("Bibs")
        for i, row in lb.iterrows():
            col = st.columns([3,1,1,1])
            lb.at[i,"player_name"] = col[0].text_input(f"Name {i}", row["player_name"] or "", key=f"bba_{i}")
            lb.at[i,"is_gk"] = col[1].checkbox("GK", bool(row.get("is_gk", False)), key=f"bbgk_{i}")
            lb.at[i,"goals"] = col[2].number_input("G", 0, 50, int(row.get("goals") or 0), key=f"bbg_{i}")
            lb.at[i,"assists"] = col[3].number_input("A", 0, 50, int(row.get("assists") or 0), key=f"bbaa_{i}")
        submitted = st.form_submit_button("Save changes", use_container_width=True)
    if submitted:
        svc.table("matches").update({
            "formation_a": m["formation_a"], "formation_b": m["formation_b"],
            "motm_name": m["motm_name"], "score_a": int(m["score_a"]), "score_b": int(m["score_b"]),
            "is_draw": bool(m["is_draw"]), "notes": m["notes"]
        }).eq("id", mid).execute()
        svc.table("lineups").delete().eq("match_id", mid).eq("team","Non-bibs").execute()
        svc.table("lineups").delete().eq("match_id", mid).eq("team","Bibs").execute()
        if not la.empty: svc.table("lineups").insert(la.to_dict(orient="records")).execute()
        if not lb.empty: svc.table("lineups").insert(lb.to_dict(orient="records")).execute()
        st.success("Updated."); refresh_cache(); st.rerun()

def page_player():
    st.title("👤 Player")
    players = fetch_table("players").sort_values("name")
    if players.empty: st.info("Add players to view profiles."); return
    name = st.selectbox("Choose player", players["name"].tolist())
    p = players[players["name"]==name].iloc[0]
    l = fetch_table("lineups")
    m = fetch_table("matches")
    if l.empty or m.empty: st.info("No data yet."); return
    agg = per_player_aggregates(players, l, m)
    row = agg.loc[name] if name in agg.index else None
    if row is None: st.info("No appearances yet."); return

    st.markdown(
        "<div class='banner'>"
        "<div style='display:flex; align-items:center; gap:.75rem;'>"
        f"<img class='avatar' src='{p.get('photo_url') or ''}' onerror=\"this.style.display='none'\"/>"
        "<div>"
        f"<div style='font-size:1.25rem;font-weight:800'>{p['name']}</div>"
        f"<div class='subtitle'>{p.get('notes') or ''}</div>"
        "</div></div></div>", unsafe_allow_html=True
    )

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("GP", int(row["games"]))
    c2.metric("W-D-L", f"{int(row['wins'])}-{int(row['draws'])}-{int(row['losses'])}")
    c3.metric("Win %", f"{row['winp']}%")
    c4.metric("Goals", int(row["goals"]))
    c5.metric("Assists", int(row["assists"]))
    c6.metric("G+A", int(row["ga"]))
    c7,c8,c9 = st.columns(3)
    c7.metric("G/GP", row["gpg"]); c8.metric("A/GP", row["apg"]); c9.metric("G+A/GP", row["gapg"])
    c10,c11 = st.columns(2)
    c10.metric("Team Contrib %", f"{row['contrib_pct']}%")
    c11.metric("MOTM", int(row["motm_count"]))

    best_win, cur_len, cur_type = streaks_for_player(name, l, m)
    st.caption(f"Best win streak: **{best_win}** • Current streak: **{cur_len} {cur_type}**")

    # Recent games table
    plays = l[l["player_name"]==name]
    mids = plays["match_id"].unique().tolist()
    mm = m[m["id"].isin(mids)].sort_values("gw", ascending=False)[["season","gw","team_a","team_b","score_a","score_b","date","motm_name","is_draw"]]
    st.subheader("Recent games")
    st.dataframe(mm.head(15), hide_index=True, use_container_width=True)

def page_stats():
    st.title("📊 Stats")
    m = fetch_table("matches")
    l = fetch_table("lineups")
    p = fetch_table("players")
    if m.empty or l.empty: st.info("Not enough data yet."); return

    c1,c2,c3,c4 = st.columns(4)
    seasons = ["All"] + sorted(m["season"].dropna().unique().tolist())
    season = c1.selectbox("Season", seasons, index=0)
    min_gp = int(c2.number_input("Min games", 1, 50, 1))
    last_x = int(c3.number_input("Last X GWs (0=all)", 0, 100, 0))
    show_photos = c4.toggle("Show photos", value=False)

    agg = per_player_aggregates(p, l, m, season_filter=season, last_x=last_x)
    if agg.empty: st.info("No data for filters."); return
    agg_f = agg[agg["games"]>=min_gp]

    def render_table(df, cols, title):
        st.subheader(title)
        dd = df.reset_index()[["player_name"] + cols].rename(columns={"player_name":"Player"})
        st.dataframe(dd, hide_index=True, use_container_width=True)

    render_table(agg_f.sort_values("goals", ascending=False), ["games","goals","gpg"], "Top Scorers")
    render_table(agg_f.sort_values("assists", ascending=False), ["games","assists","apg"], "Top Assisters")
    gad = agg_f.assign(GA=agg_f["ga"]).sort_values("GA", ascending=False)
    render_table(gad, ["games","GA","gapg"], "G + A")
    render_table(agg_f.sort_values("contrib_pct", ascending=False), ["games","ga","team_goals","contrib_pct"], "Team Contribution %")
    render_table(agg_f.sort_values("motm_count", ascending=False), ["games","motm_count"], "MOTM Leaderboard")
    # Duos & Nemesis
    st.subheader("Best Duos (same team)")
    duo = duo_stats(l, m, min_games=2)
    if duo.empty: st.info("No duo data yet.")
    else:
        st.dataframe(duo.head(20), hide_index=True, use_container_width=True)
    st.subheader("Nemesis (who you lose to)")
    nem = nemesis_stats(l, m, min_meetings=2)
    if nem.empty: st.info("No nemesis data yet.")
    else:
        st.dataframe(nem.head(20), hide_index=True, use_container_width=True)

def page_awards():
    st.title("🏆 Awards")
    a = fetch_table("awards").sort_values(["season","month","gw"]).reset_index(drop=True)
    st.subheader("All Awards")
    st.dataframe(a, hide_index=True, use_container_width=True)
    st.divider()
    if is_admin():
        st.subheader("Add Award")
        with st.form("add_award"):
            season = st.number_input("Season", 2000, 2100, date.today().year)
            month = st.number_input("Month", 0, 12, 0)
            typ = st.selectbox("Type", ["MOTM","POTM"])
            gw = st.number_input("Gameweek (optional)", 0, 60, 0)
            player = st.text_input("Player name")
            notes = st.text_input("Notes")
            if st.form_submit_button("Save", use_container_width=True):
                svc.table("awards").insert({
                    "season": int(season), "month": int(month), "type": typ, "gw": int(gw or 0),
                    "player_name": player, "notes": notes
                }).execute()
                st.success("Saved"); refresh_cache(); st.rerun()
    else:
        login_area()

def page_import_export():
    st.title("⬆️⬇️ CSV Import / Export")
    if not is_admin():
        st.warning("Admin only."); login_area(); return
    st.info("Import order: players → matches → lineups")

    up_players = st.file_uploader("Upload players.csv", type=["csv"], accept_multiple_files=False, key="upl_p")
    if up_players and st.button("Import players"):
        df = pd.read_csv(up_players)
        upsert_players(df); st.success("Players upserted."); refresh_cache()

    up_matches = st.file_uploader("Upload matches.csv", type=["csv"], accept_multiple_files=False, key="upl_m")
    if up_matches and st.button("Import matches"):
        df = pd.read_csv(up_matches, parse_dates=["date"], dayfirst=True, keep_default_na=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        upsert_matches(df); st.success("Matches upserted."); refresh_cache()

    up_lineups = st.file_uploader("Upload lineups.csv", type=["csv"], accept_multiple_files=False, key="upl_l")
    if up_lineups and st.button("Import lineups"):
        df = pd.read_csv(up_lineups)
        if "is_gk" in df.columns:
            df["is_gk"] = df["is_gk"].astype(int).astype(bool)
        replace_lineups(df); st.success("Lineups replaced per match/team."); refresh_cache()

    st.divider()
    st.subheader("Export")
    for name in ["players","matches","lineups","awards"]:
        df = fetch_table(name)
        st.download_button(label=f"Download {name}.csv",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{name}.csv",
                           mime="text/csv",
                           use_container_width=True)

    st.divider()
    st.subheader("Avatar upload")
    f = st.file_uploader("Upload player photo (HEIC/JPG/PNG)", type=["heic","heif","jpg","jpeg","png"])
    if f and st.button("Upload to avatars bucket"):
        url = upload_avatar(f)
        st.success(f"Uploaded. Public URL: {url}")

def page_admin():
    st.title("🛠️ Admin Editor")
    if not is_admin():
        st.warning("Admin only."); login_area(); return
    st.caption("Quick-edit any table below. Use with care.")

    # Players editor
    st.subheader("Players")
    p = fetch_table("players")
    p_edit = st.data_editor(p, num_rows="dynamic", use_container_width=True, key="ed_players")
    if st.button("Save players", type="primary"):
        if not p_edit.empty: upsert_players(p_edit)
        st.success("Players saved."); refresh_cache()

    st.subheader("Matches")
    m = fetch_table("matches")
    m_edit = st.data_editor(m, num_rows="dynamic", use_container_width=True, key="ed_matches")
    if st.button("Save matches"):
        if not m_edit.empty: upsert_matches(m_edit)
        st.success("Matches saved."); refresh_cache()

    st.subheader("Lineups")
    l = fetch_table("lineups")
    l_edit = st.data_editor(l, num_rows="dynamic", use_container_width=True, key="ed_lineups")
    if st.button("Save lineups"):
        if not l_edit.empty:
            # delete+insert by match+team to keep invariant
            replace_lineups(l_edit)
        st.success("Lineups saved."); refresh_cache()

    st.subheader("Awards")
    a = fetch_table("awards")
    a_edit = st.data_editor(a, num_rows="dynamic", use_container_width=True, key="ed_awards")
    if st.button("Save awards"):
        if not a_edit.empty:
            # crude upsert by id if exists else insert
            rows = a_edit.where(pd.notnull(a_edit), None).to_dict(orient="records")
            for r in rows:
                rid = r.get("id")
                if rid:
                    svc.table("awards").update(r).eq("id", rid).execute()
                else:
                    svc.table("awards").insert(r).execute()
        st.success("Awards saved."); refresh_cache()

# ----------------------------------
# Navigation
# ----------------------------------
tabs = {
    "🏟️ Matches": page_matches,
    "➕ Add Match": add_match_wizard,
    "✏️ Edit Match": edit_match,
    "👤 Player": page_player,
    "📊 Stats": page_stats,
    "🏆 Awards": page_awards,
    "⬆️⬇️ Import / Export": page_import_export,
    "🛠️ Admin Editor": page_admin,
}
choice = st.sidebar.radio("Navigate", list(tabs.keys()), index=0, key="nav")
if is_admin():
    st.sidebar.success("Admin mode")
else:
    st.sidebar.info("Read-only")
tabs[choice]()
