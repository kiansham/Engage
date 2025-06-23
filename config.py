from pathlib import Path
from datetime import timedelta

class Config:
    # Database and file paths
    DB_PATH = Path("engagement.duckdb")
    EXCEL_SOURCE = Path("Engagement_Tracker_2025_Filled.xlsx")
    
    # Application settings
    APP_TITLE = "ESG Engagement Platform"
    APP_ICON = "📊"
    BACKUP_SCHEDULE = "daily"
    MAX_FILE_SIZE = 50  # MB
    CACHE_TTL = 600  # seconds
    
    # Database connection settings
    DB_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    # Date settings
    DATE_COLS = [
        "start_date", "target_date", "last_interaction_date",
        "outcome_date", "next_action_date",
    ]
    
    # Alert thresholds
    ALERT_DAYS = {
        "urgent": 3,
        "warning": 7,
        "upcoming": 14
    }
    
    # ESG Categories and themes
    ESG_THEMES = {
        "E": [
            "Climate Change", "Water",
            "Forests", "Other"
            ],
        "S": [
            "Supply Chain", "Modern Slavery"
        ],
        "G": [ "Executive Compensation", "Ethics",
              "Transparency", "Shareholder Rights"
        ]
    }
    
    # Interaction types
    INTERACTION_TYPES = [
        "Email", "Call", "Meeting", "Letter",
        "AGM", "Conference Call"
    ]
    
    # Outcome statuses
    OUTCOME_STATUSES = [
        "Achieved", "Partial", "Neutral", "No Progress",
        "Backwards Step", "Withdrawal", "Escalation Required", "N/A"
    ]
    
    # Milestone statuses
    MILESTONE_STATUSES = [
        "Initiated", "No Response", "Negative Response", 
        "Further Meetings", "Issue Acknowledged, Request Not",
        "Request Acknowledged", "Aspiration", "Commitment", 
        "Partial Disclosure", "Full Disclosure", "Verified", "Success", 
        "Cancelled"
    ]
    
    # Escalation levels
    ESCALATION_LEVELS = [
        "None Required", "Reminder", "Follow Up", "Collaborative Push", 
        "Senior Appeal (from AQR)", "Senior Appeal (to Company)", 
        "Escalate - Vote Against Committee", "Escalate - Vote Against Chair",
        "Escalate - Vote Against Board", "Escalate - Divest"
    ]
    
    # Color palette (colorblind safe)
    COLORS = {
        "primary": "#3498db",
        "success": "#2ecc71",
        "warning": "#f39c12",
        "danger": "#e74c3c",
        "info": "#17a2b8",
        "light": "#f8f9fa",
        "dark": "#343a40"
    }
    
    CB_SAFE_PALETTE = [
        "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
        "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
    ]

# Page configuration
PAGES_CONFIG = {
    "Dashboard": {"icon": "speedometer2", "function": "dashboard"},
    "Engagement Management": {"icon": "plus-circle", "function": "engagement_management"},
    "Task Management": {"icon": "list-check", "function": "task_management"},
    "Analytics": {"icon": "graph-up-arrow", "function": "enhanced_analysis"},
    "Company Deep Dive": {"icon": "building", "function": "company_deep_dive"},
    "Impact Tracking": {"icon": "target", "function": "impact_tracking"},
    "Data Management": {"icon": "gear", "function": "data_management"},
}

# CSS Styles
CSS_STYLES = """
<style>
    .alert-urgent { 
        background-color: #ffe6e6; 
        border-left: 4px solid #e74c3c; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 1rem 0; 
    }
    .alert-warning { 
        background-color: #fff3cd; 
        border-left: 4px solid #f39c12; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 1rem 0; 
    }
</style>
"""

# Navigation styles
NAV_STYLES = {
    "container": {"padding": "0!important"},
    "icon": {"font-size": "14px"},
    "nav-link": {
        "font-size": "12px",
        "--hover-color": "#e8f4fd",
        "border-radius": "8px",
        "margin": "2px 0"
    },
    "nav-link-selected": {
        "background-color": Config.COLORS["primary"],
        "color": "white"
    }
}

# Form field configurations
FORM_CONFIGS = {
    "company_info": [
        ("company_name", "text", "Company Name *", None),
        ("gics_sector", "select", "GICS Sector *", "gics_sector"),
        ("region", "select", "Region *", "region")
    ],
    "interaction_details": [
        ("interaction_type", "select", "Interaction Type *", Config.INTERACTION_TYPES),
        ("outcome_status", "select", "Outcome Status *", Config.OUTCOME_STATUSES)
    ],
    "milestone_update": [
        ("milestone", "select", "Current Milestone", ["Initial Contact", "Commitment Sought", "Action Plan", "Verification", "Complete"]),
        ("milestone_status", "select", "Milestone Status", Config.MILESTONE_STATUSES)
    ],
    "timeline": [
        ("start_date", "date", "Start Date *", "now"),
        ("target_date", "date", "Target Date", "90_days")
    ]
}

# Chart configurations
CHART_CONFIGS = {
    "bar": {
        "height": 400,
        "showlegend": False,
        "yaxis": {"tickformat": "d"},
        "margin": {"l": 50, "r": 50, "t": 60, "b": 50}
    },
    "status": {
        "height": 140,
        "barmode": "stack",
        "margin": {"l": 80, "r": 10, "t": 40, "b": 10},
        "showlegend": True,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
        "xaxis": {"tickformat": "d"}
    },
    "geographic": {
        "height": 500,
        "geo": {
            "showframe": False,
            "showcoastlines": True,
            "projection_type": "equirectangular"
        }
    }
}