from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from typing import Dict, List, Optional

from config import Config
from utils import (
    load_db, get_latest_view, get_upcoming_tasks,
    create_engagement, log_interaction, update_milestone_status,
    backup_bytes, get_lookup_values, add_lookup_value,
    get_engagement_analytics, get_impact_summary, add_impact_milestone,
    reset_database, get_database_info
)

st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .alert-urgent { background-color: #ffe6e6; border-left: 4px solid #e74c3c; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .alert-warning { background-color: #fff3cd; border-left: 4px solid #f39c12; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS

def create_metric_row(metrics: List[tuple]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

def create_chart(data: pd.Series, title: str, xlab: str, chart_type: str = "bar") -> go.Figure:
    if chart_type == "bar":
        fig = px.bar(x=data.index, y=data.values, title=title, labels={"x": xlab, "y": "Count"}, 
                    color=data.index, color_discrete_sequence=Config.CB_SAFE_PALETTE)
        fig.update_layout(showlegend=False, height=400, yaxis=dict(tickformat='d'))
    elif chart_type == "pie":
        fig = px.pie(values=data.values, names=data.index, title=title, color_discrete_sequence=Config.CB_SAFE_PALETTE)
    return fig

def create_status_chart(data: pd.DataFrame) -> go.Figure:
    if data.empty:
        return go.Figure()
    
    on_time = data.get("on_time", pd.Series(dtype=bool)).sum()
    late = data.get("late", pd.Series(dtype=bool)).sum()
    active = len(data[~data.get("is_complete", pd.Series(dtype=bool))])
    
    fig = go.Figure()
    colors = [Config.COLORS["success"], Config.COLORS["danger"], Config.COLORS["warning"]]
    
    for i, (label, value) in enumerate([("On‑time", on_time), ("Late", late), ("Open", active)]):
        fig.add_bar(y=["Engagements"], x=[value], name=label, orientation="h", marker_color=colors[i])
    
    fig.update_layout(barmode="stack", height=140, title="Status Breakdown", 
                     margin=dict(l=80, r=10, t=40, b=10), showlegend=True,
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                     xaxis=dict(tickformat='d'))
    return fig

def create_info_display(items: List[tuple], use_html: bool = False) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        if use_html:
            col.markdown(f"**{label}**<br><span style='font-size: 1.1em;'>{value}</span>", unsafe_allow_html=True)
        else:
            col.metric(label, value)

def get_filtered_lookup(field: str, exclude_terms: List[str] = None) -> List[str]:
    exclude_terms = exclude_terms or [field.lower(), field.replace("_", " ").lower()]
    options = [opt for opt in get_lookup_values(field) if opt.lower() not in exclude_terms]
    return options if options else ["Technology", "Healthcare", "Finance"]

def create_form_section(title: str, inputs: List[tuple]) -> Dict:
    st.markdown(f"### {title}")
    cols = st.columns(len(inputs))
    results = {}
    
    for col, (key, input_type, label, options) in zip(cols, inputs):
        with col:
            if input_type == "text":
                results[key] = st.text_input(label)
            elif input_type == "select":
                results[key] = st.selectbox(label, options)
            elif input_type == "date":
                results[key] = st.date_input(label, value=options if options else datetime.now().date())
            elif input_type == "checkbox":
                results[key] = st.checkbox(label)
    return results

def handle_task_date_display(task_date, today) -> None:
    try:
        if pd.notna(task_date):
            if hasattr(task_date, 'date'):
                task_date = task_date.date()
            else:
                task_date = pd.to_datetime(task_date).date()
            
            days_left = (task_date - today).days
            if days_left < 0:
                st.error(f"Overdue by {abs(days_left)} days")
            else:
                st.warning(f"{days_left} days left")
        else:
            st.info("No due date set")
    except:
        st.caption("Date error")

def create_alert_section(urgent_tasks: pd.DataFrame, overdue_tasks: pd.DataFrame) -> None:
    if not urgent_tasks.empty or not overdue_tasks.empty:
        st.markdown("### Alerts")
        col1, col2 = st.columns(2)
        
        if not urgent_tasks.empty:
            col1.markdown(f"""<div class="alert-urgent"><strong>⚠️ {len(urgent_tasks)} Urgent Tasks</strong><br>
                         Due within {Config.ALERT_DAYS['urgent']} days</div>""", unsafe_allow_html=True)
        
        if not overdue_tasks.empty:
            col2.markdown(f"""<div class="alert-warning"><strong>📅 {len(overdue_tasks)} Overdue Tasks</strong><br>
                         Past due date</div>""", unsafe_allow_html=True)

# FILTER FUNCTIONS

def sidebar_filters(df: pd.DataFrame) -> tuple:
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        
        with st.expander("Alerts", expanded=False):
            show_urgent = st.checkbox("Show urgent only")
            show_overdue = st.checkbox("Show overdue only")
        
        with st.expander("Engagement Type", expanded=True):
            progs = st.multiselect("Engagement Program", sorted(df["program"].unique()) if not df.empty else [])
            
        with st.expander("Geo & Sector", expanded=True):
            region = st.multiselect("Region", sorted(df["region"].unique()) if not df.empty else [])
            country = st.multiselect("Country", sorted(df["country"].unique()) if not df.empty else [])
            sector = st.multiselect("GICS Sector", sorted(df["gics_sector"].unique()) if not df.empty else [])
            
        with st.expander("Engagement Status", expanded=True):
            mile = st.multiselect("Milestone", sorted(df["milestone"].unique()) if not df.empty else [])
            status = st.multiselect("Status", Config.MILESTONE_STATUSES)
            
        with st.expander("ESG Focus", expanded=True):
            col_e, col_s, col_g = st.columns(3)
            env = col_e.checkbox("E", value=True)
            soc = col_s.checkbox("S", value=True)
            gov = col_g.checkbox("G", value=True)

    esg = [c for c, b in zip(["e", "s", "g"], [env, soc, gov]) if b] or ["e", "s", "g"]
    return progs, sector, region, country, mile, status, esg, show_urgent, show_overdue

def apply_filters(df: pd.DataFrame, filters: tuple) -> pd.DataFrame:
    if df.empty:
        return df
        
    progs, sector, region, country, mile, status, esg, show_urgent, show_overdue = filters
    
    filter_map = {"program": progs, "gics_sector": sector, "region": region, 
                 "country": country, "milestone": mile, "milestone_status": status}
    
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

# PAGE FUNCTIONS

def dashboard():
    df = get_latest_view()
    if df.empty:
        st.warning("No engagement data available. Please check your data source.")
        return
    
    filters = sidebar_filters(df)
    data = apply_filters(df, filters)
    st.session_state["DATA"] = data
    
    if data.empty:
        st.warning("No data matches the current filters.")
        return
    
    # Alerts
    urgent_tasks = data[data.get("urgent", False) == True] if "urgent" in data.columns else pd.DataFrame()
    overdue_tasks = data[data.get("overdue", False) == True] if "overdue" in data.columns else pd.DataFrame()
    create_alert_section(urgent_tasks, overdue_tasks)
    
    # Key metrics
    st.markdown("### 📈 Key Metrics")
    total = len(data)
    completed = data.get("is_complete", pd.Series(dtype=bool)).sum()
    completion_rate = (completed / total * 100) if total > 0 else 0
    on_time = data.get("on_time", pd.Series(dtype=bool)).sum()
    late = data.get("late", pd.Series(dtype=bool)).sum()
    effectiveness = (on_time / (on_time + late) * 100) if (on_time + late) > 0 else 0
    active = len(data[~data.get("is_complete", pd.Series(dtype=bool))])
    
    create_metric_row([
        ("Total Engagements", total), ("Completion Rate", f"{completion_rate:.1f}%"),
        ("On‑time Effectiveness", f"{effectiveness:.1f}%"), ("Active Engagements", active), ("Overdue", late)
    ])
    
    # Charts
    st.plotly_chart(create_status_chart(data), use_container_width=True)
    
    col1, col2 = st.columns(2)
    for col, (field, title) in zip([col1, col2], [("gics_sector", "Sector"), ("region", "Region")]):
        if field in data.columns:
            col.plotly_chart(create_chart(data[field].value_counts(), f"Engagements by {title}", title), use_container_width=True)
    
    if "milestone" in data.columns:
        st.markdown("### 🎯 Milestone Progress")
        st.write('AQR Categorises Engagements by Milestone Stage. The Milestone Stage is the current stage of the engagement according to the receptiveness of the company to the engagement.')
        st.plotly_chart(create_chart(data["milestone"].value_counts(), "Engagements by Milestone Stage", "Milestone"), use_container_width=True)

def engagement_operations():    
    tab1, tab2, tab3 = st.tabs(["➕ Create Engagement", "📝 Log Interaction", "🔧 Data Management"])
    
    with tab1:
        with st.form("new_engagement", clear_on_submit=True):
            # Company Information
            company_info = create_form_section('Log New Engagement Target', [
                ("company_name", "text", "Company Name *", None),
                ("gics_sector", "select", "GICS Sector *", get_filtered_lookup("gics_sector")),
                ("region", "select", "Region *", get_filtered_lookup("region"))
            ])
            
            # Additional fields
            col1, col2 = st.columns(2)
            with col1:
                isin = st.text_input("ISIN", help="12-character international identifier")
                country = st.selectbox("Country *", get_filtered_lookup("country"))
            with col2:
                program = st.selectbox("Program *", get_filtered_lookup("program"))
            
            # ESG Focus
            st.markdown("### ESG Focus Areas")
            col_e, col_s, col_g = st.columns(3)
            esg_flags = {}
            for col, (flag, label, themes) in zip([col_e, col_s, col_g], 
                                                 [("e", "Environmental", "E"), ("s", "Social", "S"), ("g", "Governance", "G")]):
                with col:
                    esg_flags[flag] = st.checkbox(label, help=f"{label} issues")
                    if esg_flags[flag]:
                        st.multiselect(f"{label} Themes", Config.ESG_THEMES[themes])
            
            # Objectives and Timeline
            objectives = st.multiselect("Select Objectives", get_filtered_lookup("objective"), help="### Objectives")
            
            timeline_data = create_form_section("Timeline", [
                ("start_date", "date", "Start Date *", datetime.now().date()),
                ("target_date", "date", "Target Date", datetime.now().date() + timedelta(days=90))
            ])
            
            notes = st.text_area("Additional Notes", height=100)
            
            if st.form_submit_button("Create Engagement", type="primary"):
                if not company_info["company_name"] or not any(esg_flags.values()):
                    st.error("Company name and at least one ESG focus area are required")
                else:
                    engagement_data = {**company_info, "isin": isin, "country": country, "program": program,
                                     **timeline_data, **esg_flags, "created_by": "System"}
                    
                    success, message = create_engagement(engagement_data)
                    st.success(message) if success else st.error(message)
                    if success:
                        st.balloons()
    
    with tab2:
        st.markdown("### Log Interaction")
        df = get_latest_view()
        if df.empty:
            st.warning("No engagements available. Create an engagement first.")
            return
        
        selected_company = st.selectbox("Select Company *", [""] + sorted(df["company_name"].unique()))
        if not selected_company:
            st.info("Please select a company to log an interaction.")
            return
        
        engagement_data = df[df["company_name"] == selected_company].iloc[0]
        
        # Current status display
        with st.expander("Current Engagement Status", expanded=True):
            create_info_display([
                ("Current Milestone", engagement_data.get("milestone", "N/A")),
                ("Status", engagement_data.get("milestone_status", "N/A")),
                ("Escalation Level", engagement_data.get("escalation_level", "Standard"))
            ], use_html=True)
        
        with st.form("log_interaction"):
            # Interaction details
            interaction_data = create_form_section("Interaction Details", [
                ("interaction_type", "select", "Interaction Type *", Config.INTERACTION_TYPES),
                ("outcome_status", "select", "Outcome Status *", Config.OUTCOME_STATUSES)
            ])
            
            col1, col2 = st.columns(2)
            with col1:
                interaction_date = st.date_input("Interaction Date *", value=datetime.now().date())
            with col2:
                escalation_level = st.selectbox("Escalation Level", Config.ESCALATION_LEVELS)
            
            interaction_summary = st.text_area("Interaction Summary *", height=150)
            
            # Milestone and next steps
            milestone_data = create_form_section("Milestone Update", [
                ("milestone", "select", "Current Milestone", ["Initial Contact", "Commitment Sought", "Action Plan", "Verification", "Complete"]),
                ("milestone_status", "select", "Milestone Status", Config.MILESTONE_STATUSES)
            ])
            
            next_action_date = st.date_input("Next Action Date", value=datetime.now().date() + timedelta(days=14))
            next_action_notes = st.text_area("Next Action Required", height=100)
            
            if st.form_submit_button("Log Interaction", type="primary"):
                if not interaction_summary.strip():
                    st.error("Interaction summary is required")
                else:
                    full_data = {
                        "engagement_id": engagement_data["engagement_id"],
                        "last_interaction_date": interaction_date,
                        "next_action_date": next_action_date,
                        "escalation_level": escalation_level,
                        "interaction_summary": interaction_summary,
                        **interaction_data, **milestone_data
                    }
                    
                    success, message = log_interaction(full_data)
                    st.success(message) if success else st.error(message)
                    if success:
                        st.rerun()
    
    with tab3:
        st.markdown("### Data Management")
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["💾 Backup & Export", "📝 Lookup Management", "🔧 Database Tools"])
        
        with sub_tab1:
            st.markdown("### Database Backup")
            st.info("Download a complete backup of your engagement database.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Backup", type="primary"):
                    with st.spinner("Creating backup..."):
                        backup_data = backup_bytes()
                        if backup_data:
                            st.download_button("⬇️ Download Backup", backup_data,
                                             f"engagement_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", "application/zip")
                            st.success("Backup created successfully!")
                        else:
                            st.error("Backup failed. Check the error messages above.")
            
            with col2:
                st.markdown("**Backup includes:**\n- All engagement data\n- Interaction history\n- Impact milestones")
        
        with sub_tab2:
            st.markdown("### Lookup Value Management")
            try:
                con = load_db()
                fields = [r[0] for r in con.execute("SELECT DISTINCT field FROM lookup ORDER BY field").fetchall()]
            except:
                fields = []
            
            if fields:
                selected_field = st.selectbox("Select Field", fields)
                if selected_field:
                    current_values = get_lookup_values(selected_field)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Current Values:**")
                        for value in current_values:
                            st.write(f"• {value}")
                    
                    with col2:
                        st.markdown("**Add New Value:**")
                        new_value = st.text_input("Value", key=f"new_{selected_field}")
                        if st.button("Add Value") and new_value:
                            success, message = add_lookup_value(selected_field, new_value)
                            st.success(message) if success else st.error(message)
                            if success:
                                st.rerun()
            else:
                st.info("No lookup fields available.")
        
        with sub_tab3:
            st.markdown("### Database Diagnostics & Tools")
            
            # Database info
            st.markdown("#### Database Information")
            db_info = get_database_info()
            
            if "error" not in db_info:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Tables:**")
                    for table in db_info["tables"]:
                        st.write(f"• {table}")
                with col2:
                    st.markdown("**Record Counts:**")
                    for table, count in db_info["counts"].items():
                        st.write(f"• {table}: {count}")
            else:
                st.error(f"Database error: {db_info['error']}")
            
            st.markdown("---")
            
            # Database tools
            st.markdown("#### Database Tools")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Reset Database**")
                st.caption("Clear cache and recreate database if corrupted")
                if st.button("🔄 Reset Database", type="secondary"):
                    with st.spinner("Resetting database..."):
                        success, message = reset_database()
                        if success:
                            st.success(message)
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(message)
            
            with col2:
                st.markdown("**Clear Cache**")
                st.caption("Clear application cache to force data reload")
                if st.button("🧹 Clear Cache"):
                    st.cache_data.clear()
                    load_db.cache_clear()
                    st.success("Cache cleared successfully!")
                    st.rerun()
            
            # File information
            st.markdown("#### File Information")
            file_status = "✅ Excel file found" if Path(Config.EXCEL_SOURCE).exists() else "⚠️ Excel file not found"
            st.success(f"{file_status}: {Config.EXCEL_SOURCE}") if "✅" in file_status else st.warning(f"{file_status}: {Config.EXCEL_SOURCE}")
            
            if not Path(Config.EXCEL_SOURCE).exists():
                st.info("Application will use sample data.")
            
            if Path(Config.DB_PATH).exists():
                db_size = Path(Config.DB_PATH).stat().st_size / 1024 / 1024
                st.info(f"📊 Database file: {Config.DB_PATH} ({db_size:.2f} MB)")
            else:
                st.info("📊 Database will be created on first use.")

def task_management():
    task_counts = [get_upcoming_tasks(days) for days in [Config.ALERT_DAYS["urgent"], Config.ALERT_DAYS["warning"], Config.ALERT_DAYS["upcoming"]]]
    
    create_metric_row([
        ("Urgent (≤3 days)", len(task_counts[0])),
        ("Warning (≤7 days)", len(task_counts[1])),
        ("Upcoming (≤14 days)", len(task_counts[2]))
    ])
    
    tab1, tab2, tab3 = st.tabs(["🚨 Urgent", "⚠️ This Week", "📅 Upcoming"])
    
    for tab, tasks, label in zip([tab1, tab2, tab3], task_counts, ["Urgent", "This Week", "Upcoming"]):
        with tab:
            if not tasks.empty:
                if tab == tab1:  # Urgent tab with detailed view
                    st.markdown("### Tasks Due Within 3 Days")
                    today = datetime.now().date()
                    for _, task in tasks.iterrows():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.markdown(f"**{task['company_name']}**")
                            st.caption(f"Milestone: {task.get('milestone', 'N/A')}")
                        with col2:
                            handle_task_date_display(task['next_action_date'], today)
                        with col3:
                            if st.button("Mark Complete", key=f"urgent_{task['engagement_id']}"):
                                success, msg = update_milestone_status(task['engagement_id'], "Complete")
                                if success:
                                    st.success(msg)
                                    st.rerun()
                        st.divider()
                else:  # Table view for other tabs
                    display_tasks = tasks.copy()
                    if 'next_action_date' in display_tasks.columns:
                        display_tasks['next_action_date'] = pd.to_datetime(display_tasks['next_action_date']).dt.date
                    st.dataframe(display_tasks[["company_name", "next_action_date", "milestone"]], use_container_width=True)
            else:
                st.info(f"No {label.lower()} tasks!" + (" 🎉" if tab == tab1 else ""))

def enhanced_analysis():
    analytics_data = get_engagement_analytics()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Engagement Analysis","🌍 Geographic Analysis", "⏱️ Monthly Trends", "📈 Engagement Effectiveness", "📊 Benchmarking"])

    with tab1:
        st.markdown("### Engagement Analysis")
        df = st.session_state.get("DATA", get_latest_view())
        if not df.empty:
            esg_data = pd.Series({
                "Environmental": df.get("e", pd.Series(dtype=bool)).sum(),
                "Social": df.get("s", pd.Series(dtype=bool)).sum(),
                "Governance": df.get("g", pd.Series(dtype=bool)).sum()
            })
            st.plotly_chart(create_chart(esg_data, "ESG Focus Distribution", "Flag", "pie"), use_container_width=True)
        
        impact_data = get_impact_summary()
        if not impact_data.empty:
            st.markdown("### Impact Milestones")
            st.dataframe(impact_data, use_container_width=True)


    with tab2:
        st.markdown("### Geographic Analysis")
        df = st.session_state.get("DATA", get_latest_view())
        if df.empty:
            st.warning("No data available for geographic analysis.")
            return
        
        # Global distribution
        if "country" in df.columns:
            country_data = df.groupby("country").size().reset_index(name="engagement_count")
            fig = px.choropleth(country_data, locations="country", locationmode="country names",
                              color="engagement_count", color_continuous_scale="Blues",
                              title="Global Engagement Distribution", labels={"engagement_count": "Number of Engagements"})
            fig.update_layout(height=500, geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional breakdown
        col1, col2 = st.columns(2)
        for col, (field, title) in zip([col1, col2], [("region", "Region"), ("country", "Top 10 Countries")]):
            if field in df.columns:
                data = df[field].value_counts()
                if field == "country":
                    data = data.head(10)
                col.plotly_chart(create_chart(data, f"Engagements by {title}", title.split()[-1]), use_container_width=True)

    with tab3:
        st.markdown("### Monthly Trends")
        if not analytics_data["monthly_trends"].empty:
            fig = go.Figure()
            for name, color in [("New Engagements", Config.COLORS["primary"]), ("Completed", Config.COLORS["success"])]:
                fig.add_trace(go.Scatter(
                    x=analytics_data["monthly_trends"]["month"],
                    y=analytics_data["monthly_trends"][name.lower().replace(" ", "_")],
                    mode='lines+markers', name=name, line=dict(color=color)
                ))
            fig.update_layout(title="Monthly Engagement Trends", xaxis_title="Month", yaxis_title="Count", 
                            height=400, yaxis=dict(tickformat='d'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Engagement Effectiveness Analysis")
        col1, col2 = st.columns(2)
        
        chart_configs = [
            (analytics_data["success_rates"], "gics_sector", "success_rate", "Success Rate by Sector", "Success Rate (%)", "Sector"),
            (analytics_data["response_times"], "gics_sector", "avg_response_days", "Average Response Time by Sector", "Days", "Sector")
        ]
        
        for col, (data, x, y, title, ylabel, xlabel) in zip([col1, col2], chart_configs):
            if not data.empty:
                fig = px.bar(data, x=x, y=y, title=title, labels={y: ylabel, x: xlabel})
                fig.update_layout(height=400, yaxis=dict(tickformat='d'))
                col.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### Benchmarking & Performance")
        df = st.session_state.get("DATA", get_latest_view())
        if not df.empty and "program" in df.columns:
            program_performance = df.groupby("program").agg({
                "is_complete": ["count", "sum"], "days_to_target": "mean"
            }).round(2)
            program_performance.columns = ["Total", "Completed", "Avg Days to Target"]
            program_performance["Success Rate %"] = (program_performance["Completed"] / program_performance["Total"] * 100).round(1)
            st.dataframe(program_performance, use_container_width=True)
    


def company_deep_dive():
    df = get_latest_view()
    if df.empty:
        st.warning("No engagement data available.")
        return
    
    selected_company = st.selectbox("Select Company", sorted(df["company_name"].unique()))
    if not selected_company:
        return
    
    company_data = df[df["company_name"] == selected_company].iloc[0]
    st.markdown(f"## {selected_company}")
    
    create_info_display([
        ("Sector", company_data.get("gics_sector", "N/A")),
        ("Region", company_data.get("region", "N/A")),
        ("Status", company_data.get("milestone_status", "N/A")),
        ("Escalation", company_data.get("escalation_level", "Standard"))
    ], use_html=True)
    
    # ESG focus
    st.markdown("### ESG Focus Areas")
    esg_focus = [label for flag, label in [("e", "Environmental"), ("s", "Social"), ("g", "Governance")] if company_data.get(flag)]
    st.write(", ".join(esg_focus) if esg_focus else "Not specified")
    
    # Interaction timeline
    st.markdown("### Interaction Timeline")
    try:
        con = load_db()
        interactions = con.execute("""
            SELECT i.last_interaction_date, e.target_date, i.interaction_type, i.milestone, i.outcome_status, i.interaction_summary
            FROM interaction i JOIN engagement e USING (engagement_id)
            WHERE e.company_name = ? ORDER BY i.last_interaction_date DESC
        """, [selected_company]).df()
        
        if not interactions.empty:
            # Timeline visualization
            fig = go.Figure()
            for i, row in interactions.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["last_interaction_date"]], y=[i], mode='markers+text',
                    marker=dict(size=15, color=Config.CB_SAFE_PALETTE[i % len(Config.CB_SAFE_PALETTE)]),
                    text=row["interaction_type"], textposition="middle right",
                    name=row["interaction_type"], showlegend=False
                ))
            
            fig.update_layout(title="Interaction Timeline", xaxis_title="Date", yaxis_title="Interactions",
                            height=400, yaxis=dict(showticklabels=False))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent interactions
            st.markdown("### Recent Interactions")
            for _, interaction in interactions.head(5).iterrows():
                with st.expander(f"{interaction['interaction_type']} - {interaction['last_interaction_date']}", expanded=False):
                    col1, col2 = st.columns(2)
                    col1.write(f"**Milestone:** {interaction.get('milestone', 'N/A')}")
                    col1.write(f"**Outcome:** {interaction.get('outcome_status', 'N/A')}")
                    col2.write(f"**Target Date:** {interaction.get('target_date', 'N/A')}")
                    st.write("**Summary:**")
                    st.write(interaction.get("interaction_summary", "No summary available"))
        else:
            st.info("No interactions recorded for this company.")
    except Exception as e:
        st.error(f"Error loading interaction data: {e}")

def impact_tracking():
    tab1, tab2 = st.tabs(["📊 Impact Overview", "➕ Add Impact Milestone"])
    
    with tab1:
        impact_data = get_impact_summary()
        if not impact_data.empty:
            st.markdown("### Impact Milestones Summary")
            st.dataframe(impact_data, use_container_width=True)
            if "milestone_type" in impact_data.columns:
                st.plotly_chart(create_chart(impact_data["milestone_type"].value_counts(), "Impact Milestones by Type", "Milestone Type"), use_container_width=True)
        else:
            st.info("No impact milestones recorded yet.")
    
    with tab2:
        st.markdown("### Add New Impact Milestone")
        df = get_latest_view()
        if df.empty:
            st.warning("No engagements available.")
            return
        
        selected_company = st.selectbox("Select Company", [""] + sorted(df["company_name"].unique()))
        if selected_company:
            engagement_id = df[df["company_name"] == selected_company].iloc[0]["engagement_id"]
            
            with st.form("add_impact_milestone"):
                milestone_data = create_form_section("", [
                    ("milestone_type", "select", "Impact Type", ["Policy Change", "Disclosure Improvement", "Practice Change", "Target Setting", "Reporting Enhancement", "Stakeholder Engagement"]),
                    ("impact_score", "text", "Impact Score (1-5)", None)  # Will handle as slider below
                ])
                
                col1, col2 = st.columns(2)
                with col1:
                    target_date = st.date_input("Target Date")
                    impact_score = st.slider("Impact Score (1-5)", 1, 5, 3)
                with col2:
                    status = st.selectbox("Status", ["Planned", "In Progress", "Achieved", "Delayed"])
                
                description = st.text_area("Description", height=100)
                evidence_url = st.text_input("Evidence URL (optional)")
                notes = st.text_area("Notes", height=80)
                
                if st.form_submit_button("Add Impact Milestone"):
                    full_milestone_data = {
                        **milestone_data, "target_date": target_date, "status": status,
                        "impact_score": impact_score, "description": description,
                        "evidence_url": evidence_url or None, "notes": notes or None
                    }
                    
                    success, message = add_impact_milestone(engagement_id, full_milestone_data)
                    st.success(message) if success else st.error(message)
                    if success:
                        st.rerun()

# NAVIGATION AND MAIN APP

def enhanced_sidebar():
    with st.sidebar:
        st.markdown("### ESG Engagement Platform")
        df = get_latest_view()
        if not df.empty:
            st.markdown("#### Engagement Stats")
            total = len(df)
            active = len(df[~df.get("is_complete", pd.Series(dtype=bool))])
            urgent = len(df[df.get("urgent", False) == True]) if "urgent" in df.columns else 0
            
            col1, col2 = st.columns(2)
            col1.metric("YTD Total", total)
            col2.metric("Active", active)
            
            if urgent > 0:
                st.error(f"🚨 {urgent} urgent tasks")
        st.markdown("---")

PAGES = {
    "Dashboard": dashboard, "Analytics": enhanced_analysis, "Company Profiles": company_deep_dive,
    "Engagement Operations": engagement_operations, "Task Management": task_management,
    "Impact Tracking": impact_tracking,
}

ICON_MAP = {
    "Dashboard": "speedometer2", "Analytics": "graph-up-arrow", "Company Profiles": "building",
    "Engagement Operations": "folder-plus", "Task Management": "list-check",
    "Impact Tracking": "target",
}

def navigation():
    enhanced_sidebar()
    current_page = st.query_params.get("page", "Dashboard")
    if current_page not in PAGES:
        current_page = "Dashboard"
    
    with st.sidebar:
        selected_page = option_menu("Navigation", list(PAGES.keys()),
                                  icons=[ICON_MAP.get(k, "circle") for k in PAGES.keys()],
                                  default_index=list(PAGES.keys()).index(current_page),
                                  styles={"container": {"padding": "0!important"}, "icon": {"font-size": "14px"},
                                         "nav-link": {"font-size": "14px", "--hover-color": "#e8f4fd", "border-radius": "8px", "margin": "2px 0"},
                                         "nav-link-selected": {"background-color": Config.COLORS["primary"], "color": "white"}})
    
    st.query_params["page"] = selected_page
    
    with st.sidebar:
        st.markdown("---")
        st.caption("ESG Engagement Platform")
        st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    
    return selected_page

def main():
    try:
        with st.spinner("Initializing database..."):
            con = load_db()
        
        try:
            con.execute("SELECT COUNT(*) FROM engagement").fetchone()
        except Exception as e:
            st.warning(f"Database connection issue: {e}")
            st.info("Go to Engagement Operations > Data Management > Database Tools to reset the database.")
        
        current_page = navigation()
        PAGES[current_page]()
        
    except Exception as e:
        st.error("## Application Error")
        st.error(f"**Error:** {str(e)}")
        
        st.markdown("### 🔧 Quick Fixes")
        col1, col2, col3 = st.columns(3)
        
        buttons = [("🔄 Refresh Page", st.rerun), ("🧹 Clear Cache", lambda: [st.cache_data.clear(), load_db.cache_clear(), st.rerun()]),
                  ("🔧 Reset Database", lambda: st.success("Database reset") if reset_database()[0] else st.error("Reset failed"))]
        
        for col, (label, action) in zip([col1, col2, col3], buttons):
            if col.button(label):
                try:
                    action() if label != "🔧 Reset Database" else (
                        (lambda success, msg: st.success(msg) if success else st.error(msg))(*reset_database()) or st.rerun()
                    )
                except Exception as reset_error:
                    st.error(f"Action failed: {reset_error}")

if __name__ == "__main__":
    main()