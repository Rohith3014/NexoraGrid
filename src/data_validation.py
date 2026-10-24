"""
EcoGrid AI - Data Validation Module
Validates data quality and produces audit reports.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from .utils import get_logger

logger = get_logger("data_validation")


def validate_dataset(data: dict, output_dir: Path = None) -> dict:
    """Run comprehensive data validation and produce audit report."""
    audit = {
        "validation_timestamp": pd.Timestamp.now().isoformat(),
        "datasets": {},
        "issues": [],
        "summary": {}
    }
    
    for name, df in data.items():
        if df is None or name.startswith("_"):
            continue
        
        ds_audit = {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "null_counts": {c: int(df[c].isnull().sum()) for c in df.columns},
            "null_pct": {c: round(df[c].isnull().mean() * 100, 2) for c in df.columns},
            "duplicate_rows": int(df.duplicated().sum()),
        }
        
        # Check numeric ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                neg_count = int((vals < 0).sum())
                if neg_count > 0 and col not in ["ingestion_lag_min"]:
                    audit["issues"].append({
                        "dataset": name,
                        "column": col,
                        "issue": "negative_values",
                        "count": neg_count,
                        "severity": "warning"
                    })
        
        # Check for datetime columns
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        for col in datetime_cols:
            ds_audit[f"{col}_range"] = {
                "min": str(df[col].min()),
                "max": str(df[col].max())
            }
        
        audit["datasets"][name] = ds_audit
    
    # Specific validations
    if "buildings" in data and data["buildings"] is not None:
        bids = data["buildings"]["building_id"].tolist()
        audit["summary"]["buildings_count"] = len(bids)
        audit["summary"]["building_ids"] = bids
    
    if "daily_summary" in data and data["daily_summary"] is not None:
        ds = data["daily_summary"]
        audit["summary"]["daily_date_range"] = {
            "start": str(ds["date"].min()),
            "end": str(ds["date"].max()),
            "days": int((ds["date"].max() - ds["date"].min()).days + 1)
        }
    
    if "hourly_streams" in data and data["hourly_streams"] is not None:
        hs = data["hourly_streams"]
        audit["summary"]["hourly_records"] = len(hs)
        audit["summary"]["hourly_timestamp_range"] = {
            "start": str(hs["timestamp"].min()),
            "end": str(hs["timestamp"].max())
        }
    
    if "waste_train" in data and data["waste_train"] is not None:
        wt = data["waste_train"]
        audit["summary"]["waste_train_count"] = len(wt)
        audit["summary"]["waste_classes"] = wt["category"].value_counts().to_dict()
        audit["summary"]["waste_feature_nulls"] = {
            c: int(wt[c].isnull().sum()) for c in 
            ["weight_g", "moisture_pct", "metal_signal", "transparency", "acoustic_hardness"]
            if c in wt.columns
        }
    
    if "waste_test" in data and data["waste_test"] is not None:
        audit["summary"]["waste_test_count"] = len(data["waste_test"])
    
    audit["summary"]["total_issues"] = len(audit["issues"])
    audit["summary"]["is_demo"] = data.get("_is_demo", False)
    
    # Save reports
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        with open(output_dir / "dataset_audit.json", "w") as f:
            json.dump(audit, f, indent=2, default=str)
        
        # Markdown report
        md = generate_audit_markdown(audit)
        with open(output_dir / "dataset_audit.md", "w") as f:
            f.write(md)
        
        logger.info(f"Audit reports saved to {output_dir}")
    
    return audit


def generate_audit_markdown(audit: dict) -> str:
    """Generate a human-readable markdown audit report."""
    lines = [
        "# EcoGrid AI - Dataset Audit Report",
        f"\n**Generated:** {audit['validation_timestamp']}",
        f"**Demo Mode:** {'Yes' if audit['summary'].get('is_demo') else 'No'}",
        "\n## Summary\n",
    ]
    
    summary = audit["summary"]
    if "buildings_count" in summary:
        lines.append(f"- **Buildings:** {summary['buildings_count']} ({', '.join(summary.get('building_ids', []))})")
    if "daily_date_range" in summary:
        dr = summary["daily_date_range"]
        lines.append(f"- **Daily data range:** {dr['start']} to {dr['end']} ({dr['days']} days)")
    if "hourly_records" in summary:
        lines.append(f"- **Hourly telemetry records:** {summary['hourly_records']:,}")
    if "waste_train_count" in summary:
        lines.append(f"- **Waste training events:** {summary['waste_train_count']:,}")
    if "waste_test_count" in summary:
        lines.append(f"- **Waste test events:** {summary['waste_test_count']:,}")
    if "waste_classes" in summary:
        lines.append(f"- **Waste classes:** {summary['waste_classes']}")
    
    lines.append(f"\n## Data Quality Issues: {summary.get('total_issues', 0)}\n")
    
    for issue in audit.get("issues", []):
        lines.append(f"- [{issue['severity'].upper()}] `{issue['dataset']}.{issue['column']}`: {issue['issue']} ({issue['count']} occurrences)")
    
    lines.append("\n## Dataset Details\n")
    for name, ds in audit["datasets"].items():
        lines.append(f"### {name}")
        lines.append(f"- Rows: {ds['rows']:,}")
        lines.append(f"- Columns: {len(ds['columns'])}")
        lines.append(f"- Duplicates: {ds['duplicate_rows']}")
        null_cols = {k: v for k, v in ds["null_counts"].items() if v > 0}
        if null_cols:
            lines.append(f"- Null values: {null_cols}")
        lines.append("")
    
    return "\n".join(lines)


def check_data_freshness(data: dict) -> dict:
    """Check freshness of data across datasets."""
    freshness = {}
    
    for name, df in data.items():
        if df is None or name.startswith("_"):
            continue
        ts_cols = [c for c in df.columns if "timestamp" in c.lower() or c in ["date", "logged_at"]]
        for col in ts_cols:
            try:
                latest = pd.to_datetime(df[col]).max()
                freshness[f"{name}.{col}"] = {
                    "latest": str(latest),
                    "age_hours": round((pd.Timestamp.now() - latest).total_seconds() / 3600, 1)
                }
            except Exception:
                continue
    
    return freshness
