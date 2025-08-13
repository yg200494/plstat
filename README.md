# Powerleague Stats — Streamlit + Supabase

Mobile-first, production-ready app for weekly Powerleague stats. Includes admin tools, CSV import/export, HEIC → PNG avatar conversion, formations, pitches, leaderboards, duo/nemesis logic, and awards.

---

## 0) What you get

- `app.py` — complete Streamlit app (no TODOs). **Hot-fix applied** for avatar joins.
- `schema.sql` — Postgres schema, RLS, storage bucket + policies.
- `players.csv`, `matches.csv`, `lineups.csv` — ready to import.
- `requirements.txt` — Python dependencies.
- `secrets.example.toml` — template for Streamlit secrets.

---

## 1) Create Supabase project (one-time)

1. Go to **https://supabase.com** → New project.
2. Open **SQL Editor** → paste **`schema.sql`** → **Run**.  
   - This creates tables (`players`, `matches`, `lineups`, `awards`), indexes, RLS, and a **public `avatars`** storage bucket.
3. Project Settings → **API**: copy
   - **Project URL** → `SUPABASE_URL`
   - **anon public** key → `SUPABASE_ANON_KEY`
   - **service_role** key → `SUPABASE_SERVICE_KEY` (server-side only; do not expose).

> RLS model: **public read-only**, writes via **service role** (the app uses the service key for writes server-side).

---

## 2) Streamlit secrets

Create `.streamlit/secrets.toml` (locally) or paste into Streamlit Cloud → App → Settings → **Secrets**.

```toml
# .streamlit/secrets.toml
SUPABASE_URL = "https://<project-ref>.supabase.co"
SUPABASE_ANON_KEY = "<your-anon-key>"
SUPABASE_SERVICE_KEY = "<your-service-role-key>"
ADMIN_PASSWORD = "<choose-a-strong-admin-password>"
AVATAR_BUCKET = "avatars"
```

> Single admin password controls all write actions inside the app.

---

## 3) Run the app

### Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud
- Create a new app from this repo.
- Paste **secrets** (same TOML as above) into App → Settings → Secrets.
- Deploy.

---

## 4) Load the data

In the running app:
1. Sidebar → **Import (Admin)** → enter your `ADMIN_PASSWORD`.
2. Import in this order:
   - **players.csv**
   - **matches.csv**
   - **lineups.csv**
3. Navigate to **Matches / Overview / Stats** to explore.

---

## 5) Day-to-day admin

- **Add new match:** `Matches → Add/Edit` → choose side count + formations → pick players for both teams (searchable) → order them (GK first) → **Create match & lineups**, then edit goals/assists/MOTM/score.
- **Edit existing match:** `Matches → Add/Edit` (same page) — inline edit GK, goals/assists, formations, MOTM, final score.
- **Avatars:** `Players` page — upload HEIC/JPG/PNG; HEIC auto-converts to PNG and is stored under `avatars/<player_id>.png` (public read).
- **Awards:** `Awards` page — add MOTM (by GW) or POTM (by month). They appear on player profile and stats views.
- **Exports:** `Import (Admin)` → Export area, or `Matches → Export` tab.

---

## 6) Notes and defaults

- **GW2** is 7‑a‑side; all others default to 5‑a‑side.
- **GW21** is marked as a draw regardless of score.
- Formation presets stored as strings (`formation_a`/`formation_b`).
- Pitch is mobile-friendly; MOTM shown with ⭐; chips show ⚽ and 🅰️ counts.
- Caching (`st.cache_data`/`st.cache_resource`) reduces Supabase reads; use **Refresh** via actions like Save/Import (they call `clear_cache()` and `st.rerun()`).
- Widget keys are namespaced to avoid duplicate element IDs.

---

## 7) Troubleshooting

- **No images showing:** ensure `avatars` bucket exists (created by `schema.sql`) and secrets are set. Upload at least one avatar to test.
- **Imports fail for lineups:** check that `players.csv` and `matches.csv` were imported first; importer uses `season+gw` to find `match_id`, and player name must match (aliases `Ani→Anirudh Gautam`, `Abdullah Y13→Mohammad Abdullah` handled).
- **Writes blocked:** confirm you're in Admin mode and the app has the **service_role** key in secrets. RLS otherwise enforces read-only.
- **Streamlit Cloud**: if deployment stalls, click **Rerun** after adding secrets.

---

## 8) File inventory

```
app.py
schema.sql
players.csv
matches.csv
lineups.csv
requirements.txt
secrets.example.toml   # copy to .streamlit/secrets.toml
README.md
```

Happy shipping! ⚽
