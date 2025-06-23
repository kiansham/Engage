from __future__ import annotations

import duckdb, pandas as pd, re, tempfile, zipfile, streamlit as st
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from config import Config

# DATA PROCESSING UTILITIES

def safe_get_column_data(df: pd.DataFrame, column: str, default_type=pd.Series) -> pd.Series:
    """Safely get column data with fallback."""
    return df.get(column, default_type(dtype=bool if 'complete' in column else object))

def calculate_metrics(data: pd.DataFrame) -> Dict[str, any]:
    """Calculate key engagement metrics."""
    if data.empty:
        return {"total": 0, "completed": 0, "completion_rate": 0, "on_time": 0, "late": 0, "effectiveness": 0, "active": 0}
    
    total = len(data)
    completed = safe_get_column_data(data, "is_complete").sum()
    completion_rate = (completed / total * 100) if total > 0 else 0
    on_time = safe_get_column_data(data, "on_time").sum()
    late = safe_get_column_data(data, "late").sum()
    effectiveness = (on_time / (on_time + late) * 100) if (on_time + late) > 0 else 0
    active = len(data[~safe_get_column_data(data, "is_complete")])
    
    return {
        "total": total, "completed": completed, "completion_rate": completion_rate,
        "on_time": on_time, "late": late, "effectiveness": effectiveness, "active": active
    }

def apply_data_filters(df: pd.DataFrame, filters: tuple) -> pd.DataFrame:
    """Apply all filters to the dataframe."""
    if df.empty:
        return df
        
    progs, sector, region, country, mile, status, esg, show_urgent, show_overdue = filters
    
    filter_map = {
        "program": progs, "gics_sector": sector, "region": region,
        "country": country, "milestone": mile, "milestone_status": status
    }
    
    for col, values in filter_map.items():
        if values and col in df.columns:
            df = df[df[col].isin(values)]
    
    if esg and all(col in df.columns for col in esg):
        df = df[df[esg].any(axis=1)]
    
    if show_urgent and "urgent" in df.columns:
        df = df[df["urgent"] == True]
    if show_overdue and "overdue" in df.columns:
        df = df[df["overdue"] == True]
    
    return df

def get_filtered_options(field: str, exclude_terms: List[str] = None) -> List[str]:
    """Get filtered lookup options excluding header terms."""
    exclude_terms = exclude_terms or [field.lower(), field.replace("_", " ").lower()]
    options = [opt for opt in get_lookup_values(field) if opt.lower() not in exclude_terms]
    return options if options else ["Technology", "Healthcare", "Finance"]

def prepare_engagement_data(form_data: Dict) -> Dict:
    """Prepare and validate engagement data."""
    return {
        **form_data,
        "created_by": "System"
    }

def prepare_interaction_data(form_data: Dict, engagement_id: int, additional_data: Dict) -> Dict:
    """Prepare interaction data for database insertion."""
    return {
        "engagement_id": engagement_id,
        **form_data,
        **additional_data
    }

def format_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Format date columns in dataframe for display."""
    df_copy = df.copy()
    for col in date_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col]).dt.date
    return df_copy

def get_esg_focus_areas(company_data: pd.Series) -> List[str]:
    """Get ESG focus areas for a company."""
    return [
        label for flag, label in [("e", "Environmental"), ("s", "Social"), ("g", "Governance")]
        if company_data.get(flag)
    ]

def calculate_task_days(task_date, today: datetime.date) -> Tuple[bool, str]:
    """Calculate days remaining for a task."""
    try:
        if pd.notna(task_date):
            if hasattr(task_date, 'date'):
                task_date = task_date.date()
            else:
                task_date = pd.to_datetime(task_date).date()
            
            days_left = (task_date - today).days
            if days_left < 0:
                return False, f"Overdue by {abs(days_left)} days"
            else:
                return True, f"{days_left} days left"
        else:
            return None, "No due date set"
    except:
        return None, "Date error"

# CHART CONFIGURATION UTILITIES

def get_chart_colors() -> Dict[str, str]:
    """Get standardized chart colors."""
    return {
        "on_time": Config.COLORS["success"],
        "late": Config.COLORS["danger"],
        "active": Config.COLORS["warning"],
        "primary": Config.COLORS["primary"]
    }

def prepare_chart_data(data: pd.DataFrame, field: str, limit: int = None) -> pd.Series:
    """Prepare data for chart visualization."""
    if field not in data.columns:
        return pd.Series(dtype=object)
    
    chart_data = data[field].value_counts()
    if limit:
        chart_data = chart_data.head(limit)
    return chart_data

# VALIDATION FUNCTIONS

def validate_engagement_data(data: Dict) -> List[str]:
    """Validate engagement creation data."""
    errors = []
    
    if not data.get("company_name", "").strip():
        errors.append("Company name is required")
    
    if data.get("isin") and not re.match(r"^[A-Z]{2}[A-Z0-9]{10}$", data["isin"]):
        errors.append("Invalid ISIN format (should be 12 characters: 2 letters + 10 alphanumeric)")
    
    if not any([data.get("e"), data.get("s"), data.get("g")]):
        errors.append("At least one ESG flag (E, S, or G) must be selected")
    
    if data.get("start_date") and data.get("target_date"):
        if data["start_date"] > data["target_date"]:
            errors.append("Target date must be after start date")
    
    return errors

def validate_interaction_data(data: Dict) -> List[str]:
    """Validate interaction logging data."""
    errors = []
    
    if not data.get("interaction_summary", "").strip():
        errors.append("Interaction summary is required")
    
    if not data.get("interaction_type"):
        errors.append("Interaction type is required")
    
    if data.get("next_action_date") and data.get("last_interaction_date"):
        if data["next_action_date"] <= data["last_interaction_date"]:
            errors.append("Next action date should be after interaction date")
    
    return errors

def validate_isin(isin: str) -> bool:
    """Validate ISIN format."""
    if not isin:
        return False
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]{10}$", isin.upper()))

# DATABASE FUNCTIONS

@lru_cache(maxsize=1)
def load_db(xlsx: str | Path = Config.EXCEL_SOURCE,
            db_path: str | Path = Config.DB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection. If DB doesn't exist, seed it from Excel.
    Enhanced with user management and assignments.
    """
    db_path = Path(db_path)
    
    # First try to connect and check if tables exist
    if db_path.exists():
        try:
            con = duckdb.connect(db_path)
            # Verify that required tables exist
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            required_tables = ['engagement', 'interaction']
            
            if all(table in tables for table in required_tables):
                # Check if engagement_summary view exists
                try:
                    con.execute("SELECT COUNT(*) FROM engagement_summary LIMIT 1")
                    return con
                except:
                    st.info("Recreating database views...")
                    _create_views(con)
                    return con
            else:
                st.warning("Database missing required tables. Will recreate after closing connection...")
                con.close()  # Close connection before attempting to recreate
                return _recreate_database(xlsx, db_path)
                
        except Exception as e:
            st.warning(f"Database connection error: {e}. Creating new database...")
            try:
                con.close()
            except:
                pass
            return _recreate_database(xlsx, db_path)
    
    # Create new database
    return _recreate_database(xlsx, db_path)

def _recreate_database(xlsx: str | Path, db_path: Path) -> duckdb.DuckDBPyConnection:
    """
    Recreate the database from scratch.
    """
    # If file exists, try to rename it instead of deleting
    if db_path.exists():
        try:
            backup_path = db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
            db_path.rename(backup_path)
            st.info(f"Backed up old database to {backup_path.name}")
        except Exception as e:
            st.error(f"Could not backup old database: {e}")
            st.error("Please close all applications using the database and refresh the page.")
            # Create with a different name to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_path = db_path.with_name(f"engagement_{timestamp}.duckdb")
            st.info(f"Creating new database: {db_path.name}")

    con = duckdb.connect(db_path)
    
    try:
        if not Path(xlsx).exists():
            st.warning(f"Excel file not found: {xlsx}. Creating sample data.")
            # Create a minimal database structure anyway
            _create_enhanced_schema(con)
            return con
            
        raw = pd.read_excel(xlsx, skiprows=1)
        st.success(f"Successfully loaded data from {xlsx}")
    except Exception as e:
        st.warning(f"Could not load Excel file: {e}. Creating sample data.")
        # Create a minimal database structure anyway
        _create_enhanced_schema(con)
        return con

    # Data preprocessing
    for c in Config.DATE_COLS:
        if c in raw.columns:
            raw[c] = pd.to_datetime(raw.get(c), errors="coerce").dt.date
    
    esg_cols = ["e", "s", "g"]
    for c in esg_cols:
        if c in raw.columns:
            raw[c] = raw[c].astype(str).str.lower().isin({"y", "yes", "true", "1"})

    # Create enhanced schema
    _create_enhanced_schema(con)

    # Engagement table - include target_date in base engagement
    base_cols = [
        "company_name", "isin", "aqr_id", "gics_sector",
        "country", "region", "program", "start_date", "target_date", "e", "s", "g",
    ]
    
    # Filter only columns that exist in the raw data
    available_base_cols = [col for col in base_cols if col in raw.columns]
    missing_cols = [col for col in base_cols if col not in raw.columns]
    
    if missing_cols:
        st.warning(f"Missing columns in Excel file: {missing_cols}. Using available data.")
    
    engagement = (
        raw[available_base_cols]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(engagement_id=lambda d: d.index + 1)
        .assign(created_date=datetime.now().date())
        .assign(created_by="System")
    )
    
    # Fill missing columns with defaults
    for col in missing_cols:
        if col == "target_date":
            engagement[col] = (datetime.now() + timedelta(days=90)).date()
        elif col in ["e", "s", "g"]:
            engagement[col] = False
        else:
            engagement[col] = ""
    id_map = engagement.set_index([col for col in base_cols if col in engagement.columns])["engagement_id"].to_dict()
    raw["engagement_id"] = raw[[col for col in base_cols if col in raw.columns]].apply(tuple, axis=1).map(id_map)

    # Interaction table - remove target_date since it's now in engagement
    inter_cols = [
        "engagement_id", "interaction_type", "interaction_summary",
        "last_interaction_date", "outcome_status", "outcome_date",
        "next_action_date", "escalation_level", "milestone",
        "milestone_status",
    ]
    
    # Filter only columns that exist in the raw data
    available_inter_cols = [col for col in inter_cols if col in raw.columns]
    
    if available_inter_cols:
        interaction = (
            raw[available_inter_cols]
            .dropna(subset=["engagement_id"])
            .reset_index(drop=True)
            .assign(interaction_id=lambda d: d.index + 1)
            .assign(logged_date=datetime.now().date())
            .assign(logged_by="System")
        )
        
        # Fill missing columns with defaults
        for col in inter_cols:
            if col not in interaction.columns:
                if col == "milestone_status":
                    interaction[col] = "In Progress"
                elif col == "escalation_level":
                    interaction[col] = "Standard"
                elif col == "interaction_type":
                    interaction[col] = "Email"
                elif col == "outcome_status":
                    interaction[col] = "Neutral"
                elif col == "milestone":
                    interaction[col] = "Initial Contact"
                else:
                    interaction[col] = None
    else:
        # Create empty interaction table if no interaction data
        interaction = pd.DataFrame(columns=inter_cols + ["interaction_id", "logged_date", "logged_by"])

    # Theme / objective link tables
    theme_link = _explode_column(raw, "theme")
    objective_link = _explode_column(raw, "objective")

    # Lookup table
    lookup = _create_lookup_table(raw, xlsx)

    # Write all tables
    tables = {
        "engagement": engagement,
        "interaction": interaction,
        "theme_link": theme_link,
        "objective_link": objective_link,
        "lookup": lookup,
    }

    for name, df in tables.items():
        if not df.empty:
            con.register(name, df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}")
        else:
            # Create empty table with proper structure
            if name == "interaction":
                con.execute(f"""
                    CREATE TABLE {name} (
                        interaction_id INTEGER,
                        engagement_id INTEGER,
                        interaction_type VARCHAR,
                        interaction_summary VARCHAR,
                        last_interaction_date DATE,
                        outcome_status VARCHAR,
                        outcome_date DATE,
                        next_action_date DATE,
                        escalation_level VARCHAR,
                        milestone VARCHAR,
                        milestone_status VARCHAR,
                        logged_date DATE,
                        logged_by VARCHAR
                    )
                """)
            elif name in ["theme_link", "objective_link"]:
                field_name = name.split("_")[0]
                con.execute(f"""
                    CREATE TABLE {name} (
                        engagement_id INTEGER,
                        {field_name} VARCHAR,
                        {field_name}_id INTEGER
                    )
                """)

    # Create views
    _create_views(con)
    
    return con

# Update the backup function to handle connection issues
def backup_bytes() -> bytes:
    """Export all tables to Parquet and zip them—safe for download while DB is open."""
    try:
        con = load_db()
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "export"
            exp_dir.mkdir()
            
            # Export each table individually to avoid issues
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            
            for table in tables:
                try:
                    df = con.execute(f"SELECT * FROM {table}").df()
                    df.to_parquet(exp_dir / f"{table}.parquet")
                except Exception as e:
                    st.warning(f"Could not export table {table}: {e}")
            
            zip_path = Path(tmpdir) / "engagement_backup.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for f in exp_dir.rglob("*.parquet"):
                    z.write(f, f.name)
            
            return zip_path.read_bytes()
    except Exception as e:
        st.error(f"Backup failed: {e}")
        return b""

def reset_database() -> Tuple[bool, str]:
    """
    Reset the database by clearing the cache and recreating.
    Safe way to handle database issues.
    """
    try:
        # Clear the cache first
        load_db.cache_clear()
        
        # Get a new connection (will recreate if needed)
        con = load_db()
        
        # Test the connection
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        
        return True, f"Database reset successfully. Found tables: {', '.join(tables)}"
        
    except Exception as e:
        return False, f"Failed to reset database: {str(e)}"

def get_database_info() -> Dict:
    """Get information about the current database."""
    try:
        con = load_db()
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        
        info = {"tables": tables, "counts": {}}
        
        for table in tables:
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                info["counts"][table] = count
            except:
                info["counts"][table] = "Error"
        
        return info
    except Exception as e:
        return {"error": str(e)}
    """
    Reset the database by clearing the cache and recreating.
    Safe way to handle database issues.
    """
    try:
        # Clear the cache first
        load_db.cache_clear()
        
        # Get a new connection (will recreate if needed)
        con = load_db()
        
        # Test the connection
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        
        return True, f"Database reset successfully. Found tables: {', '.join(tables)}"
        
    except Exception as e:
        return False, f"Failed to reset database: {str(e)}"

def get_database_info() -> Dict:
    """Get information about the current database."""
    try:
        con = load_db()
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        
        info = {"tables": tables, "counts": {}}
        
        for table in tables:
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                info["counts"][table] = count
            except:
                info["counts"][table] = "Error"
        
        return info
    except Exception as e:
        return {"error": str(e)}


def get_database_info() -> Dict:
    """Get information about the current database."""
    try:
        con = load_db()
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        
        info = {"tables": tables, "counts": {}}
        
        for table in tables:
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                info["counts"][table] = count
            except:
                info["counts"][table] = "Error"
        
        return info
    except Exception as e:
        return {"error": str(e)}

# Update the backup function to handle connection issues
def backup_bytes() -> bytes:
    """Export all tables to Parquet and zip them—safe for download while DB is open."""
    try:
        con = load_db()
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "export"
            exp_dir.mkdir()
            
            # Export each table individually to avoid issues
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            
            for table in tables:
                try:
                    df = con.execute(f"SELECT * FROM {table}").df()
                    df.to_parquet(exp_dir / f"{table}.parquet")
                except Exception as e:
                    st.warning(f"Could not export table {table}: {e}")
            
            zip_path = Path(tmpdir) / "engagement_backup.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for f in exp_dir.rglob("*.parquet"):
                    z.write(f, f.name)
            
            return zip_path.read_bytes()
    except Exception as e:
        st.error(f"Backup failed: {e}")
        return b""

def _create_enhanced_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create enhanced database schema."""
    
    # Impact tracking
    con.execute("""
        CREATE TABLE IF NOT EXISTS impact_milestones (
            milestone_id INTEGER PRIMARY KEY,
            engagement_id INTEGER NOT NULL,
            milestone_type VARCHAR, -- Policy Change, Disclosure Improvement, etc.
            description TEXT,
            target_date DATE,
            achieved_date DATE,
            status VARCHAR DEFAULT 'Planned',
            impact_score INTEGER, -- 1-5 scale
            evidence_url VARCHAR,
            notes TEXT
        )
    """)
    
    # Engagement history for audit trail
    con.execute("""
        CREATE TABLE IF NOT EXISTS engagement_history (
            history_id INTEGER PRIMARY KEY,
            engagement_id INTEGER NOT NULL,
            field_changed VARCHAR,
            old_value VARCHAR,
            new_value VARCHAR,
            changed_by VARCHAR,
            changed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

def _explode_column(raw: pd.DataFrame, col: str) -> pd.DataFrame:
    """Enhanced column explosion with better handling."""
    if col not in raw or raw[col].isna().all():
        return pd.DataFrame(columns=["engagement_id", col, f"{col}_id"])
    
    split_pattern = re.compile(r"[,;/]| and | & ", flags=re.I)
    
    out = (
        raw[["engagement_id", col]]
        .dropna()
        .assign(**{col: lambda d: d[col].astype(str).str.replace("Multi", "", regex=False)})
        .assign(**{col: lambda d: d[col].str.split(split_pattern)})
        .explode(col)
        .assign(**{col: lambda d: d[col].str.strip()})
        .query(f"{col} != '' and {col} != 'nan'")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    
    if not out.empty:
        out[f"{col}_id"] = out.groupby(col, sort=True).ngroup() + 1
    else:
        out[f"{col}_id"] = []
    
    return out

def _create_lookup_table(raw: pd.DataFrame, xlsx: str | Path) -> pd.DataFrame:
    """Create enhanced lookup table with proper filtering."""
    try:
        lookup_raw = pd.read_excel(xlsx, sheet_name=1, header=None)
        
        # Get column names from the original data
        column_names = raw.columns.tolist()
        
        # Map sheet columns to data columns (assuming same order)
        lookup_raw.columns = column_names[:len(lookup_raw.columns)]
        
        # Melt the data
        lookup = lookup_raw.melt(var_name="field", value_name="value").dropna()
        
        # Filter out invalid values
        invalid_values = set([
            "program", "gics_sector", "country", "region", "objective", "theme",
            "Program", "GICS Sector", "Country", "Region", "Objective", "Theme",
            "", "nan", "NaN"
        ])
        
        # Also filter out any column names that appear as values
        invalid_values.update(column_names)
        
        lookup = lookup[~lookup["value"].astype(str).str.strip().isin(invalid_values)]
        lookup = lookup[lookup["value"].astype(str).str.strip() != ""]
        
        # Remove duplicates
        lookup = lookup.drop_duplicates().reset_index(drop=True)
        
        return lookup
        
    except Exception as e:
        # Create default lookup if sheet doesn't exist or has issues
        lookup = pd.DataFrame([
            {"field": "program", "value": "Climate Action"},
            {"field": "program", "value": "Governance"},
            {"field": "program", "value": "Social Impact"},
            {"field": "gics_sector", "value": "Technology"},
            {"field": "gics_sector", "value": "Healthcare"},
            {"field": "gics_sector", "value": "Finance"},
            {"field": "region", "value": "North America"},
            {"field": "region", "value": "Europe"},
            {"field": "region", "value": "Asia Pacific"},
            {"field": "country", "value": "United States"},
            {"field": "country", "value": "United Kingdom"},
            {"field": "country", "value": "Germany"},
            {"field": "objective", "value": "Policy Change"},
            {"field": "objective", "value": "Disclosure Improvement"},
            {"field": "objective", "value": "Practice Change"},
        ])
        
        return lookup

def _create_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create database views."""
    
    # Latest interaction view
    con.execute("""
        CREATE VIEW engagement_latest AS
        SELECT i.*
        FROM interaction i
        JOIN (
            SELECT engagement_id, MAX(last_interaction_date) mx
            FROM interaction 
            WHERE last_interaction_date IS NOT NULL
            GROUP BY engagement_id
        ) t USING (engagement_id)
        WHERE i.last_interaction_date = t.mx
    """)
    
    # Engagement summary view - remove user assignment references
    con.execute("""
        CREATE VIEW engagement_summary AS
        SELECT 
            e.*,
            COALESCE(i.interaction_id, 0) as interaction_id,
            COALESCE(i.interaction_type, 'No Interactions') as interaction_type,
            COALESCE(i.interaction_summary, 'No interactions recorded') as interaction_summary,
            COALESCE(i.milestone, 'Initial Contact') as milestone,
            COALESCE(i.milestone_status, 'Not Started') as milestone_status,
            i.last_interaction_date,
            i.outcome_status,
            i.outcome_date,
            i.next_action_date,
            COALESCE(i.escalation_level, 'Standard') as escalation_level
        FROM engagement e
        LEFT JOIN engagement_latest i USING (engagement_id)
    """)

# DATA RETRIEVAL FUNCTIONS

@st.cache_data(ttl=Config.CACHE_TTL)
def get_latest_view() -> pd.DataFrame:
    """Get the latest engagement data with enhanced error handling."""
    try:
        con = load_db()
        df = con.execute("""
            SELECT * FROM engagement_summary
        """).df()
        
        if df.empty:
            return df
        
        # Process dates with robust error handling
        for c in Config.DATE_COLS:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        
        now = datetime.now()
        
        # Handle target_date calculations
        if "target_date" in df.columns:
            df["days_to_target"] = (df["target_date"] - now).dt.days
        else:
            df["days_to_target"] = None
        
        # Handle next_action_date calculations  
        if "next_action_date" in df.columns:
            df["days_to_next_action"] = (df["next_action_date"] - now).dt.days
        else:
            df["days_to_next_action"] = None
        
        # Status calculations
        df["is_complete"] = df["milestone_status"] == "Complete"
        
        # Only calculate on_time/late if we have target_date
        if "target_date" in df.columns and "days_to_target" in df.columns:
            df["on_time"] = df["is_complete"] & (df["days_to_target"] >= 0)
            df["late"] = df["is_complete"] & (df["days_to_target"] < 0)
        else:
            df["on_time"] = False
            df["late"] = False
        
        df["open"] = ~df["is_complete"]
        
        # Handle overdue calculations
        if "next_action_date" in df.columns:
            df["overdue"] = (df["next_action_date"] < now) & (~df["is_complete"])
        else:
            df["overdue"] = False
        
        # Alert flags
        if "days_to_next_action" in df.columns:
            df["urgent"] = df["days_to_next_action"] <= Config.ALERT_DAYS["urgent"]
            df["warning"] = df["days_to_next_action"] <= Config.ALERT_DAYS["warning"]
        else:
            df["urgent"] = False
            df["warning"] = False
        
        return df
        
    except Exception as e:
        st.error(f"Error loading engagement data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=Config.CACHE_TTL)
def get_upcoming_tasks(days: int = 7) -> pd.DataFrame:
    """Get tasks due within specified days."""
    try:
        con = load_db()
        return con.execute("""
            SELECT 
                e.company_name,
                e.next_action_date,
                e.milestone,
                e.escalation_level,
                e.engagement_id
            FROM engagement_summary e
            WHERE e.next_action_date <= CURRENT_DATE + INTERVAL %d DAYS
            AND e.milestone_status != 'Complete'
            ORDER BY e.next_action_date
        """ % days).df()
    except Exception as e:
        st.error(f"Error loading upcoming tasks: {e}")
        return pd.DataFrame()

# DATA MODIFICATION FUNCTIONS

def create_engagement(data: Dict) -> Tuple[bool, str]:
    """Create a new engagement with validation."""
    errors = validate_engagement_data(data)
    if errors:
        return False, "; ".join(errors)
    
    try:
        con = load_db()
        
        # Check for duplicate
        existing = con.execute("""
            SELECT engagement_id FROM engagement 
            WHERE company_name = ? AND program = ?
        """, [data["company_name"], data["program"]]).fetchone()
        
        if existing:
            return False, "Engagement already exists for this company and program"
        
        # Get next ID
        next_id = con.execute("SELECT COALESCE(MAX(engagement_id), 0) + 1 FROM engagement").fetchone()[0]
        
        # Insert engagement
        con.execute("""
            INSERT INTO engagement (
                engagement_id, company_name, isin, gics_sector, country, region,
                program, start_date, target_date, e, s, g, created_date, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id, data["company_name"], data.get("isin", ""),
            data["gics_sector"], data["country"], data["region"],
            data["program"], data["start_date"], data.get("target_date"), 
            data["e"], data["s"], data["g"],
            datetime.now().date(), data.get("created_by", "System")
        ])
        
        # Clear cache
        get_latest_view.clear()
        
        return True, f"Engagement created successfully (ID: {next_id})"
        
    except Exception as e:
        return False, f"Database error: {str(e)}"

def log_interaction(data: Dict) -> Tuple[bool, str]:
    """Log a new interaction with validation."""
    errors = validate_interaction_data(data)
    if errors:
        return False, "; ".join(errors)
    
    try:
        con = load_db()
        
        # Get next interaction ID
        next_id = con.execute("SELECT COALESCE(MAX(interaction_id), 0) + 1 FROM interaction").fetchone()[0]
        
        # Insert interaction
        con.execute("""
            INSERT INTO interaction (
                interaction_id, engagement_id, interaction_type, interaction_summary,
                last_interaction_date, outcome_status, next_action_date,
                milestone, milestone_status, escalation_level, logged_date, logged_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id, data["engagement_id"], data["interaction_type"],
            data["interaction_summary"], data.get("last_interaction_date", datetime.now().date()),
            data["outcome_status"], data.get("next_action_date"),
            data.get("milestone"), data.get("milestone_status", "In Progress"),
            data.get("escalation_level", "Standard"), datetime.now().date(),
            data.get("logged_by", "System")
        ])
        
        # Clear cache
        get_latest_view.clear()
        
        return True, f"Interaction logged successfully (ID: {next_id})"
        
    except Exception as e:
        return False, f"Database error: {str(e)}"

def update_milestone_status(engagement_id: int, status: str, user: str = "System") -> Tuple[bool, str]:
    """Update milestone status for an engagement."""
    try:
        con = load_db()
        
        # Get current status for history
        current = con.execute("""
            SELECT milestone_status FROM interaction 
            WHERE engagement_id = ? 
            ORDER BY interaction_id DESC LIMIT 1
        """, [engagement_id]).fetchone()
        
        if current:
            old_status = current[0]
            
            # Update status
            con.execute("""
                UPDATE interaction 
                SET milestone_status = ?, outcome_date = CASE WHEN ? = 'Complete' THEN CURRENT_DATE ELSE NULL END
                WHERE engagement_id = ? AND interaction_id = (
                    SELECT MAX(interaction_id) FROM interaction WHERE engagement_id = ?
                )
            """, [status, status, engagement_id, engagement_id])
            
            # Log history
            con.execute("""
                INSERT INTO engagement_history (engagement_id, field_changed, old_value, new_value, changed_by)
                VALUES (?, 'milestone_status', ?, ?, ?)
            """, [engagement_id, old_status, status, "System"])
            
            get_latest_view.clear()
            return True, "Status updated successfully"
        else:
            return False, "No interactions found for this engagement"
            
    except Exception as e:
        return False, f"Database error: {str(e)}"

# BACKUP & LOOKUP FUNCTIONS

def backup_bytes() -> bytes:
    """Export all tables to Parquet and zip them—safe for download while DB is open."""
    try:
        con = load_db()
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "export"
            exp_dir.mkdir()
            con.execute(f"EXPORT DATABASE '{exp_dir}' (FORMAT parquet)")
            zip_path = Path(tmpdir) / "engagement_backup.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for f in exp_dir.rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(exp_dir))
            return zip_path.read_bytes()
    except Exception as e:
        st.error(f"Backup failed: {e}")
        return b""

def get_lookup_values(field: str) -> List[str]:
    """Get lookup values for a field."""
    try:
        con = load_db()
        return [r[0] for r in con.execute(
            "SELECT DISTINCT value FROM lookup WHERE field=? ORDER BY value", [field]
        ).fetchall()]
    except Exception as e:
        st.error(f"Error loading lookup values: {e}")
        return []

def add_lookup_value(field: str, value: str) -> Tuple[bool, str]:
    """Add a new lookup value."""
    try:
        con = load_db()
        exists = con.execute(
            "SELECT 1 FROM lookup WHERE field=? AND value=? LIMIT 1", [field, value]
        ).fetchone()
        
        if exists:
            return False, "Value already exists"
        
        con.execute("INSERT INTO lookup VALUES (?, ?)", [field, value])
        load_db.cache_clear()
        return True, "Value added successfully"
    except Exception as e:
        return False, f"Database error: {str(e)}"

# ────────────────────────── ANALYTICS FUNCTIONS

@st.cache_data(ttl=Config.CACHE_TTL)
def get_engagement_analytics() -> Dict:
    """Get comprehensive engagement analytics."""
    try:
        con = load_db()
        
        # Response time analysis
        response_times = con.execute("""
            SELECT 
                gics_sector,
                AVG(DATEDIFF('day', start_date, last_interaction_date)) as avg_response_days,
                COUNT(*) as engagement_count
            FROM engagement_summary
            WHERE last_interaction_date IS NOT NULL
            GROUP BY gics_sector
            ORDER BY avg_response_days
        """).df()
        
        # Success rates by sector
        success_rates = con.execute("""
            SELECT 
                gics_sector,
                COUNT(*) as total,
                SUM(CASE WHEN milestone_status = 'Complete' THEN 1 ELSE 0 END) as completed,
                ROUND(100.0 * SUM(CASE WHEN milestone_status = 'Complete' THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
            FROM engagement_summary
            GROUP BY gics_sector
            ORDER BY success_rate DESC
        """).df()
        
        # Monthly trends
        monthly_trends = con.execute("""
            SELECT 
                DATE_TRUNC('month', start_date) as month,
                COUNT(*) as new_engagements,
                SUM(CASE WHEN milestone_status = 'Complete' THEN 1 ELSE 0 END) as completed
            FROM engagement_summary
            WHERE start_date >= CURRENT_DATE - INTERVAL 12 MONTHS
            GROUP BY DATE_TRUNC('month', start_date)
            ORDER BY month
        """).df()
        
        return {
            "response_times": response_times,
            "success_rates": success_rates,
            "monthly_trends": monthly_trends
        }
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        return {"response_times": pd.DataFrame(), "success_rates": pd.DataFrame(), "monthly_trends": pd.DataFrame()}

# ────────────────────────── IMPACT TRACKING

def add_impact_milestone(engagement_id: int, data: Dict) -> Tuple[bool, str]:
    """Add an impact milestone for tracking."""
    try:
        con = load_db()
        
        next_id = con.execute("SELECT COALESCE(MAX(milestone_id), 0) + 1 FROM impact_milestones").fetchone()[0]
        
        con.execute("""
            INSERT INTO impact_milestones (
                milestone_id, engagement_id, milestone_type, description,
                target_date, status, impact_score, evidence_url, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id, engagement_id, data["milestone_type"], data["description"],
            data.get("target_date"), data.get("status", "Planned"),
            data.get("impact_score"), data.get("evidence_url"), data.get("notes")
        ])
        
        return True, "Impact milestone added successfully"
        
    except Exception as e:
        return False, f"Database error: {str(e)}"

@st.cache_data(ttl=Config.CACHE_TTL)
def get_impact_summary() -> pd.DataFrame:
    """Get impact milestone summary."""
    try:
        con = load_db()
        return con.execute("""
            SELECT 
                im.milestone_type,
                COUNT(*) as total_milestones,
                SUM(CASE WHEN achieved_date IS NOT NULL THEN 1 ELSE 0 END) as achieved,
                AVG(impact_score) as avg_impact_score,
                e.company_name,
                e.gics_sector
            FROM impact_milestones im
            JOIN engagement e USING (engagement_id)
            GROUP BY im.milestone_type, e.company_name, e.gics_sector
            ORDER BY achieved DESC, avg_impact_score DESC
        """).df()
    except Exception as e:
        st.error(f"Error loading impact summary: {e}")
        return pd.DataFrame()