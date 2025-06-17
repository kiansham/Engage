from __future__ import annotations
import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from datetime import datetime, timedelta
from io import BytesIO
from streamlit_option_menu import option_menu
from utils import load_db, backup_bytes, get_lookup_values, add_lookup_value

# ────────────────────────── CONFIG
st.set_page_config("ESG Engagement Platform", "📊", layout="wide")
ENG_DB = load_db()

CB_SAFE = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]
DATE_COLS = [
    "start_date", "target_date", "last_interaction_date",
    "outcome_date", "next_action_date",
]

# ────────────────────────── DATA
@st.cache_data(ttl=600)
def latest_view() -> pd.DataFrame:
    df = ENG_DB.execute(
        """
        SELECT e.*, i.interaction_id, i.interaction_type, i.interaction_summary,
               i.milestone, i.milestone_status, i.target_date, i.last_interaction_date,
               i.outcome_status, i.outcome_date, i.next_action_date, i.escalation_level
        FROM engagement e
        JOIN engagement_latest i USING (engagement_id)
        """
    ).df()
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    now = datetime.now()
    df["days_to_target"] = (df["target_date"] - now).dt.days
    df["is_complete"] = df["milestone_status"] == "Complete"
    df["on_time"] = df["is_complete"] & (df["days_to_target"] >= 0)
    df["late"]    = df["is_complete"] & (df["days_to_target"] < 0)
    df["open"]    = ~df["is_complete"]
    return df

# ────────────────────────── FILTER HELPERS

def sidebar_filters(df):
    with st.sidebar.expander("Filters", False):
        rng = st.date_input(
            "Start-date range",
            (df["start_date"].min(), df["start_date"].max()),
            df["start_date"].min(), df["start_date"].max(),
        )
        progs   = st.multiselect("Program",  df["program"].unique())
        sector  = st.multiselect("Sector",   df["gics_sector"].unique())
        region  = st.multiselect("Region",   df["region"].unique())
        country = st.multiselect("Country",  df["country"].unique())
        mile    = st.multiselect("Milestone",df["milestone"].unique())
        st.markdown("ESG flags")
        env,soc,gov = st.columns(3)
        env = env.checkbox("E", True); soc = soc.checkbox("S", True); gov = gov.checkbox("G", True)
    esg = [c for c,b in zip(["e","s","g"],[env,soc,gov]) if b] or ["e","s","g"]
    return rng, progs, sector, region, country, mile, esg

def apply_filters(df, f):
    rng, progs, sector, region, country, mile, esg = f
    sel = df[(df["start_date"].dt.date >= rng[0]) & (df["start_date"].dt.date <= rng[1])]
    if progs:   sel = sel[sel["program"].isin(progs)]
    if sector:  sel = sel[sel["gics_sector"].isin(sector)]
    if region:  sel = sel[sel["region"].isin(region)]
    if country: sel = sel[sel["country"].isin(country)]
    if mile:    sel = sel[sel["milestone"].isin(mile)]
    return sel[sel[esg].any(axis=1)]

@st.cache_data(ttl=300)
def bar(series, title, xlab):
    fig = px.bar(
        x=series.index, y=series.values, title=title,
        labels={"x":xlab,"y":"Engagements"},
        color=series.index, color_discrete_sequence=CB_SAFE,
    ); fig.update_layout(showlegend=False); return fig

# ────────────────────────── DASHBOARD

def dashboard():
    df,filt = latest_view(), None
    filt = sidebar_filters(df); data = apply_filters(df,filt)
    st.session_state["DATA"] = data
    st.markdown("#### ESG Engagement Dashboard")
    if data.empty: st.warning("No data"); return

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total", len(data))
    c2.metric("Completion", f"{data['is_complete'].mean()*100:.1f}%")
    on_time,late = data["on_time"].sum(), data["late"].sum()
    eff = on_time/(on_time+late)*100 if on_time+late else 0
    c3.metric("On‑time effectiveness", f"{eff:.1f}%")
    active = len(data[~data["is_complete"]]); c4.metric("Active", active)
    c5.metric("Overdue", late)

    # stacked bar
    fig = go.Figure()
    for i,(lbl,val) in enumerate({"On‑time":on_time,"Late":late,"Open":active}.items()):
        fig.add_bar(y=["Engagements"], x=[val], name=lbl, orientation="h", marker_color=CB_SAFE[i])
    fig.update_layout(barmode="stack", height=140, title="Status breakdown",
                      margin=dict(l=30,r=10,t=40,b=10)); st.plotly_chart(fig,use_container_width=True)

    col1,col2 = st.columns(2)
    col1.plotly_chart(bar(data["gics_sector"].value_counts(),"By sector","Sector"), use_container_width=True)
    col2.plotly_chart(bar(data["region"].value_counts(),"By region","Region"), use_container_width=True)

    st.markdown("**Milestone**"); st.caption("Milestone = stage (Commitment, Action, Verification, etc.)")
    st.plotly_chart(bar(data["milestone"].value_counts(),"By milestone","Milestone"), use_container_width=True)

# ────────────────────────── ANALYSIS

def analysis():
    df = st.session_state.get("DATA", latest_view())
    if df.empty: st.info("Run Dashboard first"); return
    t1,t2 = st.tabs(["Topic","Timeline"])
    with t1:
        st.plotly_chart(bar(pd.Series({"E":df["e"].sum(),"S":df["s"].sum(),"G":df["g"].sum()}),"ESG flags","Flag"),use_container_width=True)
        st.plotly_chart(bar(df["program"].value_counts(),"Programs","Program"), use_container_width=True)
        if "theme" in df:
            st.plotly_chart(bar(df["theme"].value_counts().head(10),"Top themes","Theme"), use_container_width=True)
    with t2:
        df["days_to_target"] = (df["target_date"]-datetime.now()).dt.days
        st.plotly_chart(px.histogram(df,x="days_to_target",nbins=30,title="Days to target"),use_container_width=True)

# ────────────────────────── COMPANY DEEP DIVE

def company_page():
    df = latest_view(); comp = st.selectbox("Company", sorted(df["company_name"].unique()))
    if not comp: return
    row = df[df["company_name"]==comp].iloc[0]
    c1,c2,c3 = st.columns(3)
    c1.metric("Sector",row["gics_sector"]); c2.metric("Region",row["region"]); c3.metric("Status",row["milestone_status"])
    intr = ENG_DB.execute("""SELECT last_interaction_date AS interaction_date,target_date,interaction_type FROM interaction JOIN engagement USING(engagement_id) WHERE company_name=?""",[comp]).df()
    if not intr.empty and intr["interaction_date"].notna().any():
        fig = px.timeline(intr,x_start="interaction_date",x_end="target_date",y=[""]*len(intr),color="interaction_type",color_discrete_sequence=CB_SAFE)
        fig.update_yaxes(showticklabels=False); st.plotly_chart(fig,use_container_width=True)
    st.markdown("---"); st.write(row["interaction_summary"])

# ────────────────────────── GEOGRAPHIC

def geo():
    df = latest_view(); c = df.groupby("country").size().reset_index(name="count")
    fig = px.choropleth(c,locations="country",locationmode="country names",color="count",color_continuous_scale="Blues",title="Global distribution")
    fig.update_layout(geo=dict(showframe=False,showcoastlines=True)); st.plotly_chart(fig,use_container_width=True)

# ────────────────────────── DATA MGMT (lookup + backup only for brevity)

def data_mgmt():
    st.header("Data Management")
    st.download_button("⬇️ Backup", backup_bytes(),"engagement_backup.zip","application/zip")
    with st.expander("Edit lookup lists"):
        field = st.selectbox("Field", sorted({r[0] for r in ENG_DB.execute("SELECT DISTINCT field FROM lookup").fetchall()}))
        st.write("Current:",", ".join(get_lookup_values(field)))
        val = st.text_input("New value");
        if st.button("Add") and val:
            add_lookup_value(field,val); st.success("Added")

# ────────────────────────── ROUTER
PAGES = {
    "Dashboard": dashboard,
    "Analysis": analysis,
    "Company Deep Dive": company_page,
    "Geographic View": geo,
    "Data Management": data_mgmt,
}
ICON_MAP = {
    "Dashboard":"speedometer2","Analysis":"graph-up","Company Deep Dive":"building","Geographic View":"globe","Data Management":"pencil-square",
}

def nav():
    qp = st.query_params.get("page", ["Dashboard"])[0]
    if qp not in PAGES: qp="Dashboard"
    with st.sidebar:
        page = option_menu("Navigation", list(PAGES.keys()), icons=[ICON_MAP[k] for k in PAGES], default_index=list(PAGES).index(qp),
                           styles={"container":{"padding":"0!important"},"icon":{"font-size":"14px"},"nav-link":{"font-size":"14px","--hover-color":"#eee"},"nav-link-selected":{"background-color":"#3498db","color":"white"}})
    st.query_params["page"] = page; return page

def main():
    page = nav(); PAGES[page]()
    st.sidebar.markdown("---"); st.sidebar.caption("ESG Engagement Platform • "+datetime.now().strftime("%d %b %Y"))

if __name__ == "__main__":
    main()
