"""
succes/reporter.py
------------------
SUCCES HTML Report Generator

Called automatically at the end of every solver run, or manually:

    from succes.reporter import generate_report
    generate_report(results_dict, out_path="my_run.html")

    # CLI:
    python -m succes.reporter results_europe_core.json
    python -m succes.reporter results.json -o report.html

Produces a standalone self-contained HTML with Plotly charts:
  - Executive summary cards + window breakdown table
  - Per-country: stacked generation vs demand + electricity price + commitment Gantt
  - Cross-border flow chart
  - GA convergence curves
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
import numpy as np


# ── Colour palettes ────────────────────────────────────────────────────────────

FUEL_COLOURS = {
    "nuclear": "#7B2FBE",
    "coal":    "#5C4033",
    "gas":     "#E87722",
    "oil":     "#8B0000",
    "hydro":   "#1E90FF",
    "biomass": "#3CB371",
    "battery": "#2ECC71",
    "storage": "#2ECC71",
    "pumped":  "#00CED1",
    "wind":    "#87CEEB",
    "solar":   "#FFD700",
    "other":   "#AAA",
}

REGION_COLOURS = {
    "DE": "#009933", "FR": "#0055A4", "BE": "#FFD700",
    "AT": "#C8102E", "NL": "#FF6600", "CH": "#FF0000",
    "CZ": "#D7241E", "PL": "#DC143C", "DK": "#C60C30",
}

COUNTRY_NAMES = {
    "DE": "Germany", "FR": "France", "BE": "Belgium",
    "AT": "Austria", "NL": "Netherlands", "CH": "Switzerland",
    "CZ": "Czech Republic", "PL": "Poland", "DK": "Denmark",
}


def _fuel_colour(unit_name: str) -> str:
    name_lower = unit_name.lower()
    for key, colour in FUEL_COLOURS.items():
        if key in name_lower:
            return colour
    return FUEL_COLOURS["other"]


def _fuel_type_label(unit_name: str) -> str:
    """Map a unit name to a canonical fuel-type label for aggregation."""
    n = unit_name.lower()
    if "nuclear"  in n:                        return "Nuclear"
    if "lignite"  in n:                        return "Lignite"
    if "biomass"  in n:                        return "Biomass"
    if "coal"     in n:                        return "Hard Coal"
    if "ocgt"     in n or "peaker" in n:       return "OCGT"
    if "np_"      in n or "newpeak" in n:      return "OCGT"   # NewPeak = fast gas
    if "ccgt"     in n or "gas"    in n:       return "CCGT"
    if "pumped"   in n or "ph_"    in n:       return "Pumped Hydro"
    if "hydro"    in n or "ror"    in n or "reservoir" in n or "rof" in n: return "Hydro"
    if "battery"  in n or "storage" in n:      return "Battery"
    if "wind"     in n:                        return "Wind"
    if "solar"    in n or "pv"     in n:       return "Solar"
    return "Other"


# Canonical display order for fuel-type stacks (baseload → peakers)
FUEL_STACK_ORDER = [
    "Nuclear", "Lignite", "Hard Coal", "Hydro", "Pumped Hydro",
    "CCGT", "OCGT", "Wind", "Solar", "Battery", "Other",
]

FUEL_TYPE_COLOURS = {
    "Nuclear":      "#7B2FBE",
    "Lignite":      "#5C4033",
    "Hard Coal":    "#8B7355",
    "Hydro":        "#1E90FF",
    "Pumped Hydro": "#00CED1",
    "CCGT":         "#E87722",
    "OCGT":         "#FF4500",
    "Wind":         "#87CEEB",
    "Solar":        "#FFD700",
    "Battery":      "#2ECC71",
    "Other":        "#AAA",
}


def _aggregate_by_fuel(generation: dict, T: int = None) -> dict:
    """
    Aggregate per-unit generation timeseries into fuel-type totals.
    Input:  {unit_name: [MW × T]}
    Output: {fuel_type_label: [MW × T]}  — ordered by FUEL_STACK_ORDER
    T: if provided, truncate/align all arrays to exactly T elements.
       Guards against overlap-window arrays being longer than committed horizon.
    """
    from collections import defaultdict
    import numpy as np
    buckets: dict = defaultdict(lambda: None)
    for unit_name, values in generation.items():
        if not values:
            continue
        label = _fuel_type_label(unit_name)
        arr   = np.array(values, dtype=float)
        if T is not None:
            arr = arr[:T]
        if buckets[label] is None:
            buckets[label] = arr.copy()
        else:
            n = min(len(buckets[label]), len(arr))
            buckets[label] = buckets[label][:n] + arr[:n]
    # Return in canonical order, skipping empty buckets
    return {
        label: buckets[label].tolist()
        for label in FUEL_STACK_ORDER
        if label in buckets and buckets[label] is not None
    }


# ── Data loading & reshaping ───────────────────────────────────────────────────

def load_results(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def _get_regions(data: dict) -> list[str]:
    if "regions" in data and isinstance(data["regions"], list):
        return sorted(data["regions"])
    if "region_data" in data:
        return sorted(data["region_data"].keys())
    # Fallback: derive from region key in to_dict output
    return ["coupled"]


def _hours(data: dict) -> list[int]:
    if "total_hours" in data:
        return list(range(data["total_hours"]))
    if "hours" in data:
        return data["hours"]
    # Infer from windows
    n = sum(w.get("hour_end", 0) - w.get("hour_start", 0)
            for w in data.get("windows", []))
    return list(range(n))


# ── Plotly trace builder helpers ───────────────────────────────────────────────

def _stacked_area_traces(generation: dict, hours: list) -> list[dict]:
    """
    Build Plotly stacked area traces, aggregated by fuel type.
    Individual units are summed into their fuel-type bucket so the chart
    remains readable regardless of how many plants are in the fleet.
    """
    aggregated = _aggregate_by_fuel(generation, T=len(hours))
    traces = []
    for fuel_label, values in aggregated.items():
        if not values or max(abs(v) for v in values) < 0.1:
            continue
        colour = FUEL_TYPE_COLOURS.get(fuel_label, FUEL_COLOURS["other"])
        traces.append({
            "type":       "scatter",
            "mode":       "lines",
            "name":       fuel_label,
            "x":          hours,
            "y":          values,
            "fill":       "tonexty" if traces else "tozeroy",
            "fillcolor":  colour,
            "line":       {"color": colour, "width": 0.5},
            "stackgroup": "one",
            "hovertemplate": f"{fuel_label}: %{{y:.0f}} MW<extra></extra>",
        })
    return traces


def _demand_trace(demand: list, hours: list, label: str = "Demand") -> dict:
    return {
        "type":  "scatter",
        "mode":  "lines",
        "name":  label,
        "x":     hours,
        "y":     demand,
        "line":  {"color": "#333", "width": 2.5, "dash": "dot"},
        "hovertemplate": f"{label}: %{{y:.0f}} MW<extra></extra>",
    }


def _price_trace(prices: list, hours: list, colour: str = "#E63946") -> dict:
    return {
        "type":  "scatter",
        "mode":  "lines",
        "name":  "Price (€/MWh)",
        "x":     hours,
        "y":     prices,
        "line":  {"color": colour, "width": 2},
        "fill":  "tozeroy",
        "fillcolor": colour.replace(")", ", 0.15)").replace("rgb", "rgba"),
        "hovertemplate": "€%{y:.1f}/MWh<extra></extra>",
    }


def _commitment_gantt_traces(commitment: dict, unit_order: list, hours: list) -> list:
    """Convert binary commitment schedule to a heatmap trace."""
    if not commitment or not unit_order:
        return []
    matrix = []
    for unit in unit_order:
        row = commitment.get(unit, [0] * len(hours))
        matrix.append(row)
    return [{
        "type":       "heatmap",
        "z":          matrix,
        "x":          hours,
        "y":          unit_order,
        "colorscale": [[0, "#EEE"], [1, "#1A5276"]],
        "showscale":  False,
        "hovertemplate": "%{y}: %{z:.0f}<extra>h%{x}</extra>",
        "zmin": 0, "zmax": 1,
    }]


def _convergence_traces(windows: list) -> list:
    traces = []
    for w in windows:
        conv = w.get("convergence", [])
        if not conv:
            continue
        label = f"Window {w['window_idx']} (h{w['hour_start']}–{w['hour_end']})"
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": label,
            "x":    list(range(len(conv))),
            "y":    conv,
            "line": {"width": 1.5},
        })
    return traces


# ── Synthetic price/demand fallback (when Julia prices not in results) ─────────

def _synthesize_prices(generation: dict, demand: list, hours: list) -> list[float]:
    """
    Heuristic price estimation when MCP prices are not stored in results.
    Uses a simple merit-order approximation based on generation stack residual.
    """
    prices = []
    total_hours = len(hours)
    for t in range(total_hours):
        gen  = sum(v[t] for v in generation.values() if v and t < len(v))
        dem  = demand[t] if t < len(demand) else gen
        gap  = dem - gen
        if gap > 500:
            p = 200.0 + gap * 0.5      # scarcity zone
        elif gap > 0:
            p = 60.0 + gap * 0.02      # peaking zone
        elif gap > -1000:
            p = max(0.0, 40.0 + gap * 0.01)
        else:
            p = max(-20.0, gap * 0.005)  # curtailment zone
        prices.append(round(p, 1))
    return prices


# ── Layout builder ─────────────────────────────────────────────────────────────

def _layout(title: str, xaxis: dict, yaxis: dict, **kwargs) -> dict:
    base = {
        "title":      {"text": title, "font": {"size": 14}},
        "margin":     {"l": 55, "r": 20, "t": 40, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "font":       {"family": "Inter, sans-serif", "size": 11},
        "legend":     {"orientation": "h", "y": -0.18, "font": {"size": 10}},
        "hovermode":  "x unified",
        "xaxis":      {"gridcolor": "#EEE", "title": xaxis.get("title", "Hour"), **xaxis},
        "yaxis":      {"gridcolor": "#EEE", **yaxis},
    }
    base.update(kwargs)
    return base


# ── Per-country section builder ────────────────────────────────────────────────

def _country_section(region: str, region_data: dict, hours: list) -> str:
    gen         = region_data.get("generation", {})
    demand      = region_data.get("demand", [])
    prices      = region_data.get("prices", [])
    storage_net = region_data.get("storage_net", {})
    imports_h   = region_data.get("imports", [])
    exports_h   = region_data.get("exports", [])
    country  = COUNTRY_NAMES.get(region, region)
    colour   = REGION_COLOURS.get(region, "#555")
    T        = len(hours)

    # Use heuristic prices only if truly absent
    if not prices or (len(prices) == len(hours) and all(abs(p) < 0.01 for p in prices)):
        prices = _synthesize_prices(gen, demand, hours)
    avg_p = sum(prices) / max(len(prices), 1)

    # ── Generation + demand + storage + trade chart ───────────────────────────
    gen_traces = _stacked_area_traces(gen, hours)

    # Storage discharge (positive = supplying grid) — green
    # Storage charge (negative = consuming from grid) — blue
    for st_name, st_vals in storage_net.items():
        if not st_vals or max(abs(v) for v in st_vals) < 0.5:
            continue
        dc = [max(v, 0.0) for v in st_vals]
        ch = [min(v, 0.0) for v in st_vals]
        if max(dc) > 0.5:
            gen_traces.append({
                "type": "scatter", "mode": "lines",
                "name": f"{st_name} (discharge)",
                "x": hours, "y": dc, "fill": "tozeroy",
                "fillcolor": "rgba(39,174,96,0.35)",
                "line": {"color": "#27AE60", "width": 1.2},
                "hovertemplate": f"{st_name} DC: %{{y:.0f}} MW<extra></extra>",
            })
        if min(ch) < -0.5:
            gen_traces.append({
                "type": "scatter", "mode": "lines",
                "name": f"{st_name} (charge)",
                "x": hours, "y": ch, "fill": "tozeroy",
                "fillcolor": "rgba(41,128,185,0.35)",
                "line": {"color": "#2980B9", "width": 1.2},
                "hovertemplate": f"{st_name} CH: %{{y:.0f}} MW<extra></extra>",
            })

    # Imports: positive contribution from neighbours — dashed green
    if imports_h and max(imports_h) > 1:
        gen_traces.append({
            "type": "scatter", "mode": "lines", "name": "Imports",
            "x": hours, "y": imports_h, "fill": "tozeroy",
            "fillcolor": "rgba(39,174,96,0.15)",
            "line": {"color": "#27AE60", "width": 1.8, "dash": "dashdot"},
            "hovertemplate": "Import: %{y:.0f} MW<extra></extra>",
        })

    # Exports: negative (removes from local supply) — dashed red
    if exports_h and max(exports_h) > 1:
        gen_traces.append({
            "type": "scatter", "mode": "lines", "name": "Exports",
            "x": hours, "y": [-v for v in exports_h], "fill": "tozeroy",
            "fillcolor": "rgba(231,76,60,0.15)",
            "line": {"color": "#E74C3C", "width": 1.8, "dash": "dashdot"},
            "hovertemplate": "Export: %{y:.0f} MW<extra></extra>",
        })

    # Demand line on top
    gen_traces.append(_demand_trace(demand, hours))

    gen_layout = _layout(
        f"{country} ({region}) — Generation, Storage & Trade",
        {"title": "Hour"}, {"title": "MW"},
        height=300,
    )
    gen_id = f"gen_{region}"

    # ── Price chart ────────────────────────────────────────────────────────────
    price_traces = [_price_trace(prices, hours, colour)]
    price_traces.append({
        "type": "scatter", "mode": "lines",
        "name": f"Avg €{avg_p:.1f}",
        "x": hours, "y": [avg_p] * T,
        "line": {"color": "#666", "width": 1, "dash": "dash"},
    })
    price_layout = _layout(
        f"{country} — Electricity Price (MCP)",
        {"title": "Hour"}, {"title": "€/MWh"},
        height=220,
    )
    price_id = f"price_{region}"

    return f"""
<section class="country-section">
  <div class="country-header" style="border-left:5px solid {colour}">
    <h2>{country} <span class="region-tag">{region}</span></h2>
    <div class="country-stats">
      <span>Peak demand: <b>{max(demand, default=0)/1e3:.1f} GW</b></span>
      <span>Avg price: <b>€{avg_p:.1f}/MWh</b></span>
      <span>Peak price: <b>€{max(prices, default=0):.0f}/MWh</b></span>
    </div>
  </div>
  <div class="chart-row">
    <div id="{gen_id}" class="chart chart-wide"></div>
    <div id="{price_id}" class="chart chart-narrow"></div>
  </div>
</section>

<script>
Plotly.newPlot('{gen_id}',
  {json.dumps(gen_traces)},
  {json.dumps(gen_layout)},
  {{responsive:true, displayModeBar:false}}
);
Plotly.newPlot('{price_id}',
  {json.dumps(price_traces)},
  {json.dumps(price_layout)},
  {{responsive:true, displayModeBar:false}}
);
</script>
"""


# ── Cross-border flow chart ────────────────────────────────────────────────────

def _flow_section(data: dict, hours: list) -> str:
    """
    Cross-border flow visualisation.
    Panel 1: stacked bar chart of |flow| per hour (top 12 links).
    Panel 2: summary table — mean/max/net per link.
    """
    link_names = list(data.get("network_links", []))
    windows    = data.get("windows", [])
    T_total    = len(hours)

    if not windows:
        return ""

    n_links = max((len(w.get("mean_flows", [])) for w in windows), default=0)
    if n_links == 0:
        return ""

    while len(link_names) < n_links:
        link_names.append(f"Link{len(link_names)}")

    all_flows = np.zeros((n_links, T_total))
    h = 0
    for w in windows:
        mf = w.get("mean_flows", [])
        if not mf:
            h += w.get("hour_end", h+24) - w.get("hour_start", h)
            continue
        T_w = len(mf[0]) if mf else 0
        for li in range(min(len(mf), n_links)):
            vals = np.array(mf[li][:T_w])
            all_flows[li, h:h+len(vals)] = vals
        h += T_w

    mean_abs = np.abs(all_flows).mean(axis=1)
    max_abs  = np.abs(all_flows).max(axis=1)
    net_dir  = all_flows.mean(axis=1)

    # Top-12 active links
    active = [int(i) for i in np.argsort(mean_abs)[::-1][:12]
              if mean_abs[i] > 1.0]
    if not active:
        return ""

    COLOURS = ["#3498DB","#E74C3C","#2ECC71","#F39C12","#9B59B6",
               "#1ABC9C","#E67E22","#34495E","#D35400","#27AE60","#8E44AD","#2980B9"]

    # Stacked bar traces
    bar_traces = []
    for rank, li in enumerate(active):
        name = link_names[li]
        bar_traces.append({
            "type": "bar", "name": name,
            "x": list(hours),
            "y": [abs(float(v)) for v in all_flows[li]],
            "marker": {"color": COLOURS[rank % len(COLOURS)], "opacity": 0.85},
            "hovertemplate": f"{name}: %{{y:.0f}} MW<extra></extra>",
        })

    bar_layout = _layout(
        "Cross-Border Flow Magnitude — Top Links (mean over scenarios)",
        {"title": "Hour"}, {"title": "|MW|"}, height=260,
    )
    bar_layout["barmode"] = "stack"

    # Summary table
    CWE_AC = {"DE-FR","DE-NL","DE-BE","DE-AT","DE-CH","DE-CZ","DE-DK","DE-PL",
               "FR-BE","FR-CH","BE-NL","AT-CH","AT-CZ","CZ-PL","NL-DK"}
    rows = ""
    for li in np.argsort(mean_abs)[::-1]:
        if mean_abs[li] < 0.5:
            continue
        name   = link_names[li]
        is_ac  = name in CWE_AC
        marker = "" if is_ac else " <span style='color:#888'>(HVDC)</span>"
        nd     = float(net_dir[li])
        net_str = f"+{nd:.0f}" if nd >= 0 else f"{nd:.0f}"
        rows += (
            f"<tr><td><strong>{name}</strong>{marker}</td>"
            f"<td style='text-align:right'>{mean_abs[li]:.0f}</td>"
            f"<td style='text-align:right'>{max_abs[li]:.0f}</td>"
            f"<td style='text-align:right'>{net_str}</td></tr>"
        )

    bar_traces_json  = json.dumps(bar_traces)
    bar_layout_json  = json.dumps(bar_layout)
    n_active         = len(active)

    return f"""
<section class="country-section">
  <div class="country-header" style="border-left:5px solid #2980B9">
    <h2>Cross-Border Flows
      <span class="region-tag">PTDF+ATC &middot; top {n_active} links</span>
    </h2>
  </div>
  <div id="flows_bar" class="chart chart-full"></div>
  <script>
  Plotly.newPlot('flows_bar', {bar_traces_json}, {bar_layout_json},
    {{responsive:true, displayModeBar:false}});
  </script>
  <div style="overflow-x:auto;margin:8px 0 0 0">
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
    <thead><tr style="background:#2C3E50;color:#fff">
      <th style="padding:5px 10px;text-align:left">Link</th>
      <th style="padding:5px 10px;text-align:right">Mean |MW|</th>
      <th style="padding:5px 10px;text-align:right">Max |MW|</th>
      <th style="padding:5px 10px;text-align:right">Net dir. MW</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
</section>
"""

# ── Convergence section ────────────────────────────────────────────────────────

def _convergence_section(data: dict) -> str:
    windows = data.get("windows", [])
    traces  = _convergence_traces(windows)
    if not traces:
        return ""

    layout = _layout(
        "GA Convergence (objective value per epoch)",
        {"title": "Epoch"},
        {"title": "Objective (€)"},
        height=260,
    )

    return f"""
<section class="country-section">
  <div class="country-header" style="border-left:5px solid #888">
    <h2>🧬 GA Convergence</h2>
  </div>
  <div id="conv_chart" class="chart chart-full"></div>
</section>
<script>
Plotly.newPlot('conv_chart',
  {json.dumps(traces)},
  {json.dumps(layout)},
  {{responsive:true, displayModeBar:false}}
);
</script>
"""


# ── Summary cards ──────────────────────────────────────────────────────────────

def _summary_html(data: dict, regions: list) -> str:
    total_cost   = data.get("total_cost",        data.get("mean_cost", 0.0))
    cvar         = data.get("mean_cvar",          0.0)
    fuel_cost    = data.get("total_fuel_cost",    0.0)
    startup_cost = data.get("total_startup_cost", 0.0)
    penalty_cost = data.get("total_penalty_cost", 0.0)
    n_windows    = data.get("n_windows",          len(data.get("windows", [])))
    total_hours  = data.get("total_hours",        72)

    # Penalty breakdown totals
    pen_sc  = data.get("total_pen_scarcity",    0.0)
    pen_cu  = data.get("total_pen_curtailment", 0.0)
    pen_mr  = data.get("total_pen_must_run",    0.0)
    pen_in  = data.get("total_pen_inertia",     0.0)
    pen_ra  = data.get("total_pen_ramp",        0.0)

    cards = [
        ("💰", "Total Cost",         f"€{total_cost/1e6:.2f}M",   "#3498DB"),
        ("📊", "CVaR (5%)",          f"€{cvar/1e6:.2f}M",         "#E74C3C"),
        ("🔥", "Fuel Cost",          f"€{fuel_cost/1e6:.2f}M",    "#E67E22"),
        ("🔛", "Startup Cost",       f"€{startup_cost/1e3:.0f}k", "#9B59B6"),
        ("⚠️", "Penalty Cost",       f"€{penalty_cost/1e3:.0f}k", "#E74C3C"),
        ("🗓️", "Windows / Hours",    f"{n_windows} × {total_hours//max(n_windows,1)}h", "#27AE60"),
        ("🌍", "Regions",            str(len(regions)),             "#2980B9"),
    ]

    card_html = "\n".join(
        f'<div class="summary-card" style="border-top:4px solid {c}">'
        f'<div class="card-icon">{icon}</div>'
        f'<div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>'
        f'</div>'
        for icon, label, value, c in cards
    )

    # Penalty breakdown mini-table
    pen_rows = ""
    if penalty_cost > 0:
        pen_items = [
            ("Scarcity (unserved demand)",  pen_sc,  "#E74C3C"),
            ("Curtailment (surplus)",        pen_cu,  "#E67E22"),
            ("Must-run violation",           pen_mr,  "#9B59B6"),
            ("Inertia constraint",           pen_in,  "#1ABC9C"),
            ("Ramp rate violation",          pen_ra,  "#3498DB"),
        ]
        for label, val, color in pen_items:
            pct = val / penalty_cost * 100 if penalty_cost > 0 else 0
            bar = f'<div style="height:8px;width:{pct:.0f}%;background:{color};border-radius:4px;min-width:2px"></div>'
            pen_rows += (
                f'<tr>'
                f'<td>{label}</td>'
                f'<td style="text-align:right">€{val/1e3:.0f}k</td>'
                f'<td style="text-align:right">{pct:.1f}%</td>'
                f'<td style="width:120px;padding-left:8px">{bar}</td>'
                f'</tr>'
            )

    pen_section = f"""
  <h3 style="margin-top:2rem">Penalty Breakdown  <span style="color:#888;font-size:0.85em">(total: €{penalty_cost/1e3:.0f}k)</span></h3>
  <div class="table-wrapper">
    <table>
      <thead><tr><th>Type</th><th style="text-align:right">Cost</th><th style="text-align:right">Share</th><th></th></tr></thead>
      <tbody>{pen_rows}</tbody>
    </table>
  </div>""" if pen_rows else ""

    # Window table with penalty breakdown
    windows = data.get("windows", [])

    def _pbd(w):
        """Format penalty_breakdown dict into a compact cell."""
        bd = w.get("penalty_breakdown", {})
        if not bd or sum(bd.values()) < 1:
            return "—"
        parts = []
        labels = [("sc","scarcity"),("cu","curtailment"),("mr","must_run"),
                  ("in","inertia"),("ra","ramp")]
        for short, key in labels:
            v = bd.get(key, 0)
            if v > 500:
                parts.append(f'{short}:€{v/1e3:.0f}k')
        return " ".join(parts) if parts else "—"

    rows = "\n".join(
        f'<tr>'
        f'<td>{w["window_idx"]}</td>'
        f'<td>h{w["hour_start"]}–{w["hour_end"]}</td>'
        f'<td>€{w["mean_cost"]/1e3:.0f}k</td>'
        f'<td>€{w["cvar_cost"]/1e3:.0f}k</td>'
        f'<td>€{w["fuel_cost"]/1e3:.0f}k</td>'
        f'<td>€{w["startup_cost"]/1e3:.0f}k</td>'
        f'<td style="color:#E74C3C">€{w.get("penalty_cost",0)/1e3:.0f}k</td>'
        f'<td style="font-size:0.8em;color:#666">{_pbd(w)}</td>'
        f'<td>{w["solve_time_s"]:.1f}s</td>'
        f'</tr>'
        for w in windows
    )

    # Stacked bar chart: penalty breakdown per window
    win_labels  = [f'W{w["window_idx"]}' for w in windows]
    sc_vals  = [w.get("penalty_breakdown", {}).get("scarcity",    0)/1e3 for w in windows]
    cu_vals  = [w.get("penalty_breakdown", {}).get("curtailment", 0)/1e3 for w in windows]
    mr_vals  = [w.get("penalty_breakdown", {}).get("must_run",    0)/1e3 for w in windows]
    in_vals  = [w.get("penalty_breakdown", {}).get("inertia",     0)/1e3 for w in windows]
    ra_vals  = [w.get("penalty_breakdown", {}).get("ramp",        0)/1e3 for w in windows]

    import json as _json
    pen_chart_traces = _json.dumps([
        {"type":"bar","name":"Scarcity",    "x":win_labels,"y":sc_vals,"marker":{"color":"#E74C3C"}},
        {"type":"bar","name":"Curtailment", "x":win_labels,"y":cu_vals,"marker":{"color":"#E67E22"}},
        {"type":"bar","name":"Must-run",    "x":win_labels,"y":mr_vals,"marker":{"color":"#9B59B6"}},
        {"type":"bar","name":"Inertia",     "x":win_labels,"y":in_vals,"marker":{"color":"#1ABC9C"}},
        {"type":"bar","name":"Ramp",        "x":win_labels,"y":ra_vals,"marker":{"color":"#3498DB"}},
    ])
    pen_chart_layout = _json.dumps({
        "barmode": "stack",
        "title":   "Penalty Cost Breakdown per Window (€k)",
        "yaxis":   {"title": "€k"},
        "xaxis":   {"title": "Window"},
        "legend":  {"orientation": "h", "y": -0.2},
        "margin":  {"t": 40, "b": 80},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "font": {"color": "#2C3E50"},
    })

    return f"""
<section class="summary-section">
  <h2>Executive Summary</h2>
  <div class="summary-cards">{card_html}</div>

  {pen_section}

  <h3 style="margin-top:2rem">Penalty Breakdown per Window</h3>
  <div id="pen_chart" style="height:300px;margin-bottom:1.5rem"></div>
  <script>
    Plotly.newPlot('pen_chart', {pen_chart_traces}, {pen_chart_layout},
                   {{responsive:true, displayModeBar:false}});
  </script>

  <h3 style="margin-top:2rem">Window Breakdown</h3>
  <div class="table-wrapper">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Hours</th><th>E[Cost]</th><th>CVaR 5%</th>
          <th>Fuel</th><th>Startup</th><th>Penalty</th><th>Penalty detail</th><th>Solve</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</section>
"""


# ── Master HTML template ───────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  background: #F4F6F9;
  color: #2C3E50;
  line-height: 1.5;
}
.navbar {
  background: #1A252F;
  color: #ECF0F1;
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.navbar h1 { font-size: 1.3rem; font-weight: 700; letter-spacing: 1px; }
.navbar .subtitle { color: #95A5A6; font-size: 0.85rem; }
.navbar .badge {
  background: #27AE60; color: white;
  padding: 0.2rem 0.7rem; border-radius: 99px;
  font-size: 0.75rem; font-weight: 600;
}
.container { max-width: 1400px; margin: 0 auto; padding: 1.5rem; }
.toc {
  background: white;
  border-radius: 8px;
  padding: 1.2rem 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.toc h3 { font-size: 0.9rem; color: #7F8C8D; margin-bottom: 0.8rem; }
.toc-links { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.toc-links a {
  padding: 0.3rem 0.8rem;
  border-radius: 5px;
  font-size: 0.8rem;
  font-weight: 600;
  text-decoration: none;
  color: white;
}
.summary-section {
  background: white;
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.summary-section h2 {
  font-size: 1.1rem;
  color: #2C3E50;
  margin-bottom: 1.2rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #ECF0F1;
}
.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
}
.summary-card {
  background: #FAFAFA;
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
}
.card-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.card-label { font-size: 0.72rem; color: #7F8C8D; text-transform: uppercase; letter-spacing: 0.5px; }
.card-value { font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; }
.table-wrapper { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
thead { background: #2C3E50; color: white; }
th { padding: 0.6rem 0.8rem; text-align: left; font-weight: 600; }
td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #ECF0F1; }
tr:hover td { background: #F8F9FA; }
.country-section {
  background: white;
  border-radius: 10px;
  padding: 1.2rem 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.country-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem 0.8rem;
  background: #F8F9FA;
  border-radius: 6px;
  margin-bottom: 1rem;
}
.country-header h2 { font-size: 1rem; font-weight: 700; }
.region-tag {
  background: #ECF0F1;
  color: #7F8C8D;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
}
.country-stats {
  display: flex;
  gap: 1.5rem;
  font-size: 0.8rem;
  color: #555;
  margin-left: auto;
}
.country-stats b { color: #2C3E50; }
.chart-row {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1rem;
}
.chart-full { width: 100%; }
.gantt-chart { margin-top: 0.8rem; }
h3 { font-size: 0.95rem; color: #2C3E50; margin: 1rem 0 0.5rem; }
footer {
  text-align: center;
  padding: 2rem;
  color: #95A5A6;
  font-size: 0.8rem;
}
@media (max-width: 700px) {
  .chart-row { grid-template-columns: 1fr; }
  .summary-cards { grid-template-columns: repeat(2, 1fr); }
}
"""


def build_report(data: dict, title: str = "SUCCES Energy Simulation Report") -> str:
    regions  = _get_regions(data)
    hours    = _hours(data)
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_hours = data.get("total_hours", len(hours))

    # TOC links
    toc_links = "\n".join(
        f'<a href="#country_{r}" style="background:{REGION_COLOURS.get(r,"#555")}">'
        f'{COUNTRY_NAMES.get(r, r)} ({r})</a>'
        for r in regions
    )
    toc_extras = """
      <a href="#flows_chart" style="background:#555">⚡ Flows</a>
      <a href="#conv_chart" style="background:#888">🧬 GA Conv.</a>
    """

    # Summary
    summary_html = _summary_html(data, regions)

    # Per-country sections
    region_data = data.get("region_data", {})
    country_sections = []
    for r in regions:
        rd = region_data.get(r, {})
        section = f'<div id="country_{r}">' + _country_section(r, rd, hours) + '</div>'
        country_sections.append(section)

    flows_section = _flow_section(data, hours)
    conv_section  = _convergence_section(data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>{CSS}</style>
</head>
<body>

<nav class="navbar">
  <div>
    <div class="navbar h1">⚡ SUCCES</div>
    <div class="subtitle">Stochastic Unit Commitment · Core Energy Simulation</div>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:0.8rem">
    <span class="badge">{len(regions)} Regions</span>
    <span class="badge" style="background:#2980B9">{total_hours}h</span>
    <span style="color:#95A5A6;font-size:0.8rem">Generated {ts}</span>
  </div>
</nav>

<div class="container">

  <div class="toc">
    <h3>🗺 NAVIGATION</h3>
    <div class="toc-links">
      <a href="#summary" style="background:#2C3E50">📋 Summary</a>
      {toc_links}
      {toc_extras}
    </div>
  </div>

  <div id="summary">
    {summary_html}
  </div>

  {''.join(country_sections)}

  {flows_section}

  {conv_section}

</div>

<footer>
  SUCCES v0.2 · Stochastic Unit Commitment Coupled Energy Simulation ·
  Results generated {ts}
</footer>

</body>
</html>"""


# ── Public API (called automatically by solver) ───────────────────────────────

def generate_report(
    data:     dict,
    out_path: "str | Path | None" = None,
    title:    str = "SUCCES — Energy Simulation Report",
    verbose:  bool = True,
) -> Path:
    """
    Generate an HTML report from a results dict and write it to disk.

    Called automatically at the end of every solver run. Also callable
    directly:

        from succes.reporter import generate_report
        generate_report(results_dict, out_path="my_run.html")

    Parameters
    ----------
    data     : results dict as returned by Results.to_dict() (with region_data)
    out_path : output file path; defaults to "succes_report_<timestamp>.html"
               in the current working directory
    title    : HTML page title
    verbose  : print path and size after writing

    Returns
    -------
    Path to the written HTML file.
    """
    from datetime import datetime

    if out_path is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"succes_report_{ts}.html")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = build_report(data, title=title)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    if verbose:
        size_kb = out_path.stat().st_size / 1024
        print(f"\n  ✓ Report → {out_path.absolute()}  ({size_kb:.0f} KB)")

    return out_path


# ── CLI  (python -m succes.reporter  OR  python succes/reporter.py) ───────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from a SUCCES results JSON file.",
        prog="python -m succes.reporter",
    )
    parser.add_argument("results", help="Path to results JSON file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML path (default: <results>.html)")
    parser.add_argument("--title", default="SUCCES — Energy Simulation Report")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else results_path.with_suffix(".html")
    data     = load_results(results_path)
    generate_report(data, out_path=out_path, title=args.title, verbose=True)


if __name__ == "__main__":
    main()
