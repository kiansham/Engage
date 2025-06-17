from __future__ import annotations

import duckdb, pandas as pd, re, tempfile, zipfile
from pathlib import Path
from functools import lru_cache

# ────────────────────────── paths & constants
DB_PATH      = Path("engagement.duckdb")
EXCEL_SOURCE = Path("Engagement_Tracker_2025_Filled.xlsx")
DATE_COLS    = [
    "start_date", "target_date", "last_interaction_date",
    "outcome_date", "next_action_date",
]
ESG_COLS = ["e", "s", "g"]
_SPLIT   = re.compile(r"[,;/]| and | & ", flags=re.I)

# ────────────────────────── loader
@lru_cache(maxsize=1)
def load_db(xlsx: str | Path = EXCEL_SOURCE,
            db_path: str | Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection.  If DB doesn't exist, seed it from Excel.
    """
    db_path = Path(db_path)
    if db_path.exists():
        return duckdb.connect(db_path)

    con  = duckdb.connect(db_path)
    raw  = pd.read_excel(xlsx, skiprows=1)

    for c in DATE_COLS:
        raw[c] = pd.to_datetime(raw.get(c), errors="coerce").dt.date
    for c in ESG_COLS:
        raw[c] = raw[c].astype(str).str.lower().isin({"y", "yes", "true", "1"})

    # engagement table
    base_cols = [
        "company_name", "isin", "aqr_id", "gics_sector",
        "country", "region", "program", "start_date", "e", "s", "g",
    ]
    engagement = (
        raw[base_cols]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(engagement_id=lambda d: d.index + 1)
    )
    id_map = engagement.set_index(base_cols)["engagement_id"].to_dict()
    raw["engagement_id"] = raw[base_cols].apply(tuple, axis=1).map(id_map)

    # interaction table
    inter_cols = [
        "engagement_id", "interaction_type", "interaction_summary",
        "last_interaction_date", "outcome_status", "outcome_date",
        "next_action_date", "escalation_level", "milestone",
        "milestone_status", "target_date",
    ]
    interaction = raw[inter_cols].reset_index(drop=True).assign(
        interaction_id=lambda d: d.index + 1
    )

    # theme / objective link tables
    def _explode(col: str) -> pd.DataFrame:
        if col not in raw:
            return pd.DataFrame(columns=["engagement_id", col, f"{col}_id"])
        out = (
            raw[["engagement_id", col]]
            .dropna()
            .assign(**{col: lambda d: d[col].str.replace("Multi", "", regex=False)})
            .assign(**{col: lambda d: d[col].str.split(_SPLIT)})
            .explode(col)
            .assign(**{col: lambda d: d[col].str.strip()})
            .query(f"{col} != ''")
            .drop_duplicates()
            .reset_index(drop=True)
        )
        out[f"{col}_id"] = out.groupby(col, sort=True).ngroup() + 1
        return out

    theme_link     = _explode("theme")
    objective_link = _explode("objective")

    lookup = (
        pd.read_excel(xlsx, sheet_name=1, header=None)
        .pipe(lambda d: d.rename(columns=dict(zip(range(len(d.columns)), raw.columns))))
        .melt(var_name="field", value_name="value")
        .dropna()
        .reset_index(drop=True)
    )

    # write tables
    for name, df in {
        "engagement": engagement,
        "interaction": interaction,
        "theme_link": theme_link,
        "objective_link": objective_link,
        "lookup": lookup,
    }.items():
        con.register(name, df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}")

    con.execute(
        """
        CREATE VIEW engagement_latest AS
        SELECT i.*
        FROM interaction i
        JOIN (
            SELECT engagement_id, MAX(last_interaction_date) mx
            FROM interaction GROUP BY engagement_id
        ) t USING (engagement_id)
        WHERE i.last_interaction_date = t.mx;
        """
    )
    return con

# ────────────────────────── backup & lookup helpers
def backup_bytes() -> bytes:
    """
    Export all tables to Parquet and zip them—safe for download while DB is open.
    """
    con = load_db()
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "export"
        exp_dir.mkdir()
        con.execute(f"EXPORT DATABASE '{exp_dir}' (FORMAT parquet)")
        zip_path = Path(tmpdir) / "engagement_backup.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in exp_dir.rglob("*"):
                z.write(f, f.relative_to(exp_dir))
        return zip_path.read_bytes()


def get_lookup_values(field: str) -> list[str]:
    con = load_db()
    return [r[0] for r in con.execute(
        "SELECT value FROM lookup WHERE field=? ORDER BY value", [field]
    ).fetchall()]


def add_lookup_value(field: str, value: str) -> None:
    con = load_db()
    exists = con.execute(
        "SELECT 1 FROM lookup WHERE field=? AND value=? LIMIT 1", [field, value]
    ).fetchone()
    if not exists:
        con.execute("INSERT INTO lookup VALUES (?, ?)", [field, value])
        load_db.cache_clear()
