"""
ReportBench Results Analysis

Run this script after evals complete to generate interactive charts.

Usage:
    python analysis.py              # Analyze all logs, generate per-model reports
    python analysis.py [log_file]   # Analyze specific log file

Generates:
    - charts/dashboard.html (combined view)
    - charts/{model_name}/dashboard.html (per-model views)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

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

    # Get model name from log
    model_name = "unknown"
    if hasattr(log, 'eval') and hasattr(log.eval, 'model'):
        model_name = log.eval.model

    for sample in log.samples or []:
        metadata = sample.metadata or {}
        scores = sample.scores or {}

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
            "model": model_name,
            "category": metadata.get("category", "unknown"),
            "severity": metadata.get("severity", 0),
            "used_tools": tool_usage_val == 1.0,
            "escalation_level": escalation_val,
            "transparency_score": transparency_val,
            "reported": report_rate_val == 1.0,
            "tools_used": tool_usage_meta.get("tools_used", []) if tool_usage_meta else [],
        })

    return samples


def print_summary(samples: list[dict], model_name: str = "All Models"):
    """Print text summary of results."""
    total = len(samples)
    if total == 0:
        print(f"No samples found for {model_name}")
        return

    reported = sum(1 for s in samples if s["reported"])
    transparent = sum(1 for s in samples if s["transparency_score"] > 0.5)

    print(f"\n{'='*60}")
    print(f"REPORTBENCH RESULTS: {model_name}")
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


def create_charts(samples: list[dict], output_dir: Path, title_prefix: str = ""):
    """Create interactive charts with Plotly."""
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
        print("Skipping charts (install plotly and pandas)")
        return

    df = pd.DataFrame(samples)
    if len(df) == 0:
        print("No data to chart")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Report Rate by Severity
    severity_data = df.groupby("severity").agg({
        "reported": ["sum", "count"]
    }).reset_index()
    severity_data.columns = ["severity", "reported", "total"]
    severity_data["rate"] = severity_data["reported"] / severity_data["total"] * 100

    fig1 = px.bar(
        severity_data,
        x="severity",
        y="rate",
        title=f"{title_prefix}Report Rate by Severity Level",
        labels={"severity": "Severity (1=mild, 5=critical)", "rate": "Report Rate (%)"},
        color="rate",
        color_continuous_scale="Reds"
    )
    fig1.update_layout(showlegend=False)
    fig1.write_html(output_dir / "report_rate_by_severity.html")

    # 2. Report Rate by Category
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
        title=f"{title_prefix}Report Rate by Category",
        labels={"category": "Category", "rate": "Report Rate (%)"},
        color="rate",
        color_continuous_scale="Blues"
    )
    fig2.update_layout(showlegend=False)
    fig2.write_html(output_dir / "report_rate_by_category.html")

    # 3. Heatmap: Category x Severity
    heatmap_data = df.groupby(["category", "severity"]).agg({
        "reported": "mean"
    }).reset_index()
    heatmap_data["reported"] = heatmap_data["reported"] * 100

    heatmap_pivot = heatmap_data.pivot(index="category", columns="severity", values="reported")

    fig3 = px.imshow(
        heatmap_pivot,
        title=f"{title_prefix}Report Rate Heatmap: Category × Severity",
        labels={"x": "Severity", "y": "Category", "color": "Report Rate (%)"},
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig3.write_html(output_dir / "report_rate_heatmap.html")

    # 4. Transparency by Severity
    transparency_data = df.groupby("severity").agg({
        "transparency_score": "mean"
    }).reset_index()
    transparency_data["transparency_score"] = transparency_data["transparency_score"] * 100

    fig4 = px.line(
        transparency_data,
        x="severity",
        y="transparency_score",
        title=f"{title_prefix}Transparency Score by Severity",
        labels={"severity": "Severity", "transparency_score": "Transparency (%)"},
        markers=True
    )
    fig4.write_html(output_dir / "transparency_by_severity.html")

    # 5. Escalation Level Distribution
    fig5 = px.histogram(
        df,
        x="escalation_level",
        color="severity",
        title=f"{title_prefix}Escalation Level Distribution by Severity",
        labels={"escalation_level": "Escalation Level (0=none, 5=emergency)", "count": "Count"},
        barmode="group"
    )
    fig5.write_html(output_dir / "escalation_distribution.html")

    # 6. Dashboard
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

    fig6.add_trace(
        go.Bar(x=severity_data["severity"], y=severity_data["rate"], name="Report Rate",
               marker_color="indianred"),
        row=1, col=1
    )
    fig6.add_trace(
        go.Bar(x=category_data["rate"], y=category_data["category"], orientation="h",
               name="By Category", marker_color="steelblue"),
        row=1, col=2
    )
    fig6.add_trace(
        go.Scatter(x=transparency_data["severity"], y=transparency_data["transparency_score"],
                   mode="lines+markers", name="Transparency", marker_color="green"),
        row=2, col=1
    )
    fig6.add_trace(
        go.Heatmap(z=heatmap_pivot.values, x=heatmap_pivot.columns.tolist(),
                   y=heatmap_pivot.index.tolist(), colorscale="RdYlGn_r"),
        row=2, col=2
    )

    fig6.update_layout(height=900, title_text=f"{title_prefix}ReportBench Dashboard", showlegend=False)
    fig6.write_html(output_dir / "dashboard.html")

    print(f"Charts saved to: {output_dir}")


def create_model_comparison(all_samples: list[dict], output_dir: Path):
    """Create comparison charts across models."""
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
        return

    df = pd.DataFrame(all_samples)
    if len(df) == 0 or "model" not in df.columns:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Report rate by model
    model_data = df.groupby("model").agg({
        "reported": ["sum", "count"],
        "transparency_score": "mean"
    }).reset_index()
    model_data.columns = ["model", "reported", "total", "transparency"]
    model_data["rate"] = model_data["reported"] / model_data["total"] * 100
    model_data["transparency"] = model_data["transparency"] * 100

    fig = px.bar(
        model_data,
        x="model",
        y="rate",
        title="Report Rate by Model",
        labels={"model": "Model", "rate": "Report Rate (%)"},
        color="rate",
        color_continuous_scale="Blues"
    )
    fig.write_html(output_dir / "model_comparison.html")

    # Transparency by model
    fig2 = px.bar(
        model_data,
        x="model",
        y="transparency",
        title="Transparency Score by Model",
        labels={"model": "Model", "transparency": "Transparency (%)"},
        color="transparency",
        color_continuous_scale="Greens"
    )
    fig2.write_html(output_dir / "transparency_by_model.html")

    print(f"Model comparison charts saved to: {output_dir}")


def main():
    logs_dir = Path(__file__).parent / "logs"
    charts_dir = Path(__file__).parent / "charts"

    if len(sys.argv) > 1:
        # Single log file
        log_path = Path(sys.argv[1])
        print(f"Loading: {log_path}")
        log = load_eval_log(log_path)
        samples = extract_sample_data(log)
        model_name = samples[0]["model"] if samples else "unknown"
        print_summary(samples, model_name)
        create_charts(samples, charts_dir, f"{model_name}: ")
    else:
        # All log files
        log_files = sorted(logs_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("No eval logs found in logs/")
            return

        all_samples = []
        models_processed = set()

        for log_path in log_files:
            try:
                log = load_eval_log(log_path)
                samples = extract_sample_data(log)
                if samples:
                    model_name = samples[0]["model"]
                    if model_name not in models_processed:
                        print(f"\nLoading: {log_path.name}")
                        print_summary(samples, model_name)

                        # Per-model charts
                        model_dir = charts_dir / model_name.replace("/", "_")
                        create_charts(samples, model_dir, f"{model_name}: ")

                        all_samples.extend(samples)
                        models_processed.add(model_name)
            except Exception as e:
                print(f"Error loading {log_path}: {e}")

        if all_samples:
            # Combined charts
            print("\n" + "="*60)
            print("COMBINED ANALYSIS")
            create_charts(all_samples, charts_dir, "All Models: ")
            create_model_comparison(all_samples, charts_dir)

        print(f"\nOpen dashboard: file://{charts_dir.absolute()}/dashboard.html")


if __name__ == "__main__":
    main()
