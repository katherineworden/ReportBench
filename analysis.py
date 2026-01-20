"""
ReportBench Results Analysis

Run this script after evals complete to generate interactive charts.

Usage:
    python analysis.py [log_file]

If no log file specified, uses the most recent one.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Try to import plotting libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Install plotly for interactive charts: pip install plotly")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Install pandas for data analysis: pip install pandas")


def load_eval_log(log_path: Path):
    """Load and parse an Inspect AI eval log file."""
    from inspect_ai.log import read_eval_log
    return read_eval_log(str(log_path))


def extract_sample_data(log) -> list[dict]:
    """Extract per-sample results from eval log."""
    samples = []

    for sample in log.samples or []:
        metadata = sample.metadata or {}
        scores = sample.scores or {}

        # Get tool usage from scores
        tool_usage = scores.get("tool_usage_scorer")
        escalation = scores.get("escalation_level_scorer")
        transparency = scores.get("transparency_scorer")
        report_rate = scores.get("report_rate_scorer")

        tool_usage_val = tool_usage.value if tool_usage else 0
        escalation_val = escalation.value if escalation else 0
        transparency_val = transparency.value if transparency else 0
        report_rate_val = report_rate.value if report_rate else 0
        tool_usage_meta = tool_usage.metadata if tool_usage else {}

        samples.append({
            "id": sample.id or "",
            "category": metadata.get("category", "unknown"),
            "severity": metadata.get("severity", 0),
            "used_tools": tool_usage_val == 1.0,
            "escalation_level": escalation_val,
            "transparency_score": transparency_val,
            "reported": report_rate_val == 1.0,
            "tools_used": tool_usage_meta.get("tools_used", []) if tool_usage_meta else [],
        })

    return samples


def print_summary(samples: list[dict]):
    """Print text summary of results."""
    total = len(samples)
    reported = sum(1 for s in samples if s["reported"])
    transparent = sum(1 for s in samples if s["transparency_score"] > 0.5)

    print(f"\n{'='*60}")
    print(f"REPORTBENCH RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Reported: {reported} ({100*reported/total:.1f}%)")
    print(f"Transparent about reporting: {transparent} ({100*transparent/total:.1f}%)")

    # By severity
    print(f"\n--- Report Rate by Severity ---")
    for sev in range(1, 6):
        sev_samples = [s for s in samples if s["severity"] == sev]
        if sev_samples:
            sev_reported = sum(1 for s in sev_samples if s["reported"])
            print(f"Severity {sev}: {sev_reported}/{len(sev_samples)} ({100*sev_reported/len(sev_samples):.1f}%)")

    # By category
    print(f"\n--- Report Rate by Category ---")
    categories = set(s["category"] for s in samples)
    for cat in sorted(categories):
        cat_samples = [s for s in samples if s["category"] == cat]
        cat_reported = sum(1 for s in cat_samples if s["reported"])
        print(f"{cat}: {cat_reported}/{len(cat_samples)} ({100*cat_reported/len(cat_samples):.1f}%)")


def create_charts(samples: list[dict], output_dir: Path):
    """Create interactive charts with Plotly."""
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
        print("Skipping charts (install plotly and pandas)")
        return

    df = pd.DataFrame(samples)
    output_dir.mkdir(exist_ok=True)

    # 1. Report Rate by Severity (bar chart)
    severity_data = df.groupby("severity").agg({
        "reported": ["sum", "count"]
    }).reset_index()
    severity_data.columns = ["severity", "reported", "total"]
    severity_data["rate"] = severity_data["reported"] / severity_data["total"] * 100

    fig1 = px.bar(
        severity_data,
        x="severity",
        y="rate",
        title="Report Rate by Severity Level",
        labels={"severity": "Severity (1=mild, 5=critical)", "rate": "Report Rate (%)"},
        color="rate",
        color_continuous_scale="Reds"
    )
    fig1.update_layout(showlegend=False)
    fig1.write_html(output_dir / "report_rate_by_severity.html")
    print(f"Saved: {output_dir / 'report_rate_by_severity.html'}")

    # 2. Report Rate by Category (bar chart)
    category_data = df.groupby("category").agg({
        "reported": ["sum", "count"]
    }).reset_index()
    category_data.columns = ["category", "reported", "total"]
    category_data["rate"] = category_data["reported"] / category_data["total"] * 100
    category_data = category_data.sort_values("rate", ascending=True)

    fig2 = px.bar(
        category_data,
        x="rate",
        y="category",
        orientation="h",
        title="Report Rate by Category",
        labels={"category": "Category", "rate": "Report Rate (%)"},
        color="rate",
        color_continuous_scale="Blues"
    )
    fig2.update_layout(showlegend=False)
    fig2.write_html(output_dir / "report_rate_by_category.html")
    print(f"Saved: {output_dir / 'report_rate_by_category.html'}")

    # 3. Heatmap: Category x Severity
    heatmap_data = df.groupby(["category", "severity"]).agg({
        "reported": "mean"
    }).reset_index()
    heatmap_data["reported"] = heatmap_data["reported"] * 100

    heatmap_pivot = heatmap_data.pivot(index="category", columns="severity", values="reported")

    fig3 = px.imshow(
        heatmap_pivot,
        title="Report Rate Heatmap: Category × Severity",
        labels={"x": "Severity", "y": "Category", "color": "Report Rate (%)"},
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig3.write_html(output_dir / "report_rate_heatmap.html")
    print(f"Saved: {output_dir / 'report_rate_heatmap.html'}")

    # 4. Transparency by Severity
    transparency_data = df.groupby("severity").agg({
        "transparency_score": "mean"
    }).reset_index()
    transparency_data["transparency_score"] = transparency_data["transparency_score"] * 100

    fig4 = px.line(
        transparency_data,
        x="severity",
        y="transparency_score",
        title="Transparency Score by Severity",
        labels={"severity": "Severity", "transparency_score": "Transparency (%)"},
        markers=True
    )
    fig4.write_html(output_dir / "transparency_by_severity.html")
    print(f"Saved: {output_dir / 'transparency_by_severity.html'}")

    # 5. Escalation Level Distribution
    fig5 = px.histogram(
        df,
        x="escalation_level",
        color="severity",
        title="Escalation Level Distribution by Severity",
        labels={"escalation_level": "Escalation Level (0=none, 5=emergency)", "count": "Count"},
        barmode="group"
    )
    fig5.write_html(output_dir / "escalation_distribution.html")
    print(f"Saved: {output_dir / 'escalation_distribution.html'}")

    # 6. Combined Dashboard
    fig6 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Report Rate by Severity",
            "Report Rate by Category",
            "Transparency by Severity",
            "Category × Severity Heatmap"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "heatmap"}]]
    )

    # Add traces
    fig6.add_trace(
        go.Bar(x=severity_data["severity"], y=severity_data["rate"], name="Report Rate"),
        row=1, col=1
    )
    fig6.add_trace(
        go.Bar(x=category_data["rate"], y=category_data["category"], orientation="h", name="By Category"),
        row=1, col=2
    )
    fig6.add_trace(
        go.Scatter(x=transparency_data["severity"], y=transparency_data["transparency_score"],
                   mode="lines+markers", name="Transparency"),
        row=2, col=1
    )
    fig6.add_trace(
        go.Heatmap(z=heatmap_pivot.values, x=heatmap_pivot.columns.tolist(),
                   y=heatmap_pivot.index.tolist(), colorscale="RdYlGn_r"),
        row=2, col=2
    )

    fig6.update_layout(height=800, title_text="ReportBench Results Dashboard", showlegend=False)
    fig6.write_html(output_dir / "dashboard.html")
    print(f"Saved: {output_dir / 'dashboard.html'}")


def main():
    # Find log file
    logs_dir = Path(__file__).parent / "logs"

    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Use most recent log
        log_files = sorted(logs_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("No eval logs found in logs/")
            return
        log_path = log_files[0]

    print(f"Loading: {log_path}")
    log = load_eval_log(log_path)
    samples = extract_sample_data(log)

    if not samples:
        print("No samples found in log")
        return

    print_summary(samples)

    # Create charts
    charts_dir = Path(__file__).parent / "charts"
    create_charts(samples, charts_dir)

    print(f"\nOpen charts in browser: file://{charts_dir.absolute()}/dashboard.html")


if __name__ == "__main__":
    main()
