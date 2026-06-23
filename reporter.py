"""Generate a self-contained HTML report from a validator run JSON file."""

from __future__ import annotations

import json
import math
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.0f}%"

def _score(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"

def _ms(v) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.1f}s"

def _color(score) -> str:
    """Return a CSS color class based on score."""
    if score is None:
        return "gray"
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "amber"
    return "red"

def _pass_badge(passed) -> str:
    if passed is True:
        return '<span class="badge pass">PASS</span>'
    if passed is False:
        return '<span class="badge fail">FAIL</span>'
    return '<span class="badge gray">—</span>'

def _score_bar(score) -> str:
    if score is None:
        return ""
    w = max(0, min(100, score * 100))
    cls = _color(score)
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar {cls}" style="width:{w:.0f}%"></div>'
        f'<span class="bar-label">{score:.2f}</span>'
        f'</div>'
    )

def _esc(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))

def _collapsible(summary: str, content: str, open: bool = False) -> str:
    """Wrap content in a <details> block. summary is shown as the clickable label."""
    attr = " open" if open else ""
    return f'<details{attr}><summary>{summary}</summary>{content}</details>'


# ── section builders ──────────────────────────────────────────────────────────

def _header_section(run: dict) -> str:
    ts = run.get("run_timestamp", "")[:19].replace("T", " ")
    agg = run.get("aggregated_scores") or {}
    ar = agg.get("agent_response") or {}
    total = agg.get("total", 0)
    errors = agg.get("errors", 0)
    passed = ar.get("passed", 0)
    mean = ar.get("mean")
    pass_rate = ar.get("pass_rate")
    latency = agg.get("avg_duration_ms")

    stat = lambda label, value: f'<div class="stat"><div class="stat-val">{value}</div><div class="stat-lbl">{label}</div></div>'

    return f"""
<div class="header">
  <h1>Analytics Agent Validator</h1>
  <div class="meta">
    Run <strong>{_esc(run.get("run_id",""))}</strong> &nbsp;·&nbsp;
    {_esc(ts)} &nbsp;·&nbsp;
    env: <strong>{_esc(run.get("env") or "—")}</strong> &nbsp;·&nbsp;
    {_esc(run.get("base_url",""))}
  </div>
  <div class="stats">
    {stat("Questions", total)}
    {stat("Errors", errors)}
    {stat("Agent pass", _pct(pass_rate))}
    {stat("Agent mean", _score(mean))}
    {stat("Avg latency", _ms(latency))}
  </div>
</div>"""


def _metrics_section(agg: dict) -> str:
    rows = ""
    for key, label in [
        ("agent_response", "Agent Response"),
        ("interpretation_coverage", "Interpretation Coverage"),
        ("sql_correctness", "SQL Correctness"),
    ]:
        m = agg.get(key) or {}
        n = m.get("n", 0)
        if n == 0:
            continue
        mean = m.get("mean")
        pass_rate = m.get("pass_rate")
        cls = _color(mean)
        rows += f"""
<tr>
  <td>{label}</td>
  <td>{n}</td>
  <td class="{cls}">{_score_bar(mean)}</td>
  <td class="{cls}">{_pct(pass_rate)}</td>
</tr>"""

    return f"""
<section>
  <h2>Metric Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>n</th><th>Mean Score</th><th>Pass Rate</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>"""


def _flow_section(agg: dict) -> str:
    by_flow = agg.get("by_flow") or {}
    if not by_flow:
        return ""
    rows = ""
    for flow, d in sorted(by_flow.items()):
        mean = d.get("agent_response_mean")
        pass_rate = d.get("agent_response_pass_rate")
        cls = _color(mean)
        rows += f"""
<tr>
  <td><span class="flow-tag">{_esc(flow)}</span></td>
  <td>{d.get("total", 0)}</td>
  <td class="{cls}">{_score_bar(mean)}</td>
  <td class="{cls}">{_pct(pass_rate)}</td>
</tr>"""

    return f"""
<section>
  <h2>By Flow</h2>
  <table>
    <thead><tr><th>Flow</th><th>n</th><th>Agent Mean</th><th>Pass Rate</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>"""


def _question_card(r: dict) -> str:
    qid = r.get("question_id", "?")
    flow = r.get("flow", "")
    complexity = r.get("complexity", "")
    question = _esc(r.get("question", ""))
    agent_resp_raw = r.get("agent_response") or ""
    duration = r.get("duration_ms")
    error = r.get("error")
    ev = r.get("eval") or {}

    ar = ev.get("agent_response") or {}
    ic = ev.get("interpretation_coverage") or {}
    tc = ev.get("tool_compliance")
    traces = r.get("cortex_traces") or []
    per_trace = ev.get("per_trace") or []
    sql_scores = [t for t in per_trace if t.get("sql_score") is not None]

    ar_score = ar.get("score")
    ar_passed = ar.get("passed")
    ic_score = ic.get("score") if ic else None
    sql_mean = (
        sum(t["sql_score"] for t in sql_scores) / len(sql_scores)
        if sql_scores else None
    )

    card_cls = "card"
    if ar_passed is False:
        card_cls += " card-fail"
    elif ar_passed is True:
        card_cls += " card-pass"

    # ── traces (collapsible) ──────────────────────────────────────────────────
    traces_html = ""
    for i, t in enumerate(traces):
        status = t.get("sql_status", "")
        interp = _esc(t.get("interpretation") or "")
        sql_full = _esc(t.get("generated_sql") or "")
        rows_count = t.get("sql_row_count_full")
        err = t.get("error")
        sql_results = t.get("sql_results") or []
        status_cls = "green" if status == "SUCCEEDED" else "red" if status == "FAILED" else "amber"
        pt = per_trace[i] if i < len(per_trace) else {}
        sql_score_val = pt.get("sql_score")
        sql_reason = _esc(pt.get("sql_reason") or "")

        sql_block = (
            f'<div class="trace-sql">'
            + _collapsible(f"SQL ({status})", f"<pre><code>{sql_full}</code></pre>")
            + "</div>"
        ) if sql_full else ""

        results_block = ""
        if sql_results:
            rows_preview = _esc(json.dumps(sql_results[:5], indent=2, default=str))
            label = f"{len(sql_results)} rows shown" + (" (truncated)" if rows_count and rows_count > len(sql_results) else "")
            results_block = _collapsible(
                f"Results ({label})",
                f"<pre><code>{rows_preview}</code></pre>"
            )

        reason_block = (
            _collapsible("SQL judge reason", f'<div class="judge-inner">{sql_reason}</div>')
            if sql_reason and sql_reason != "skipped — missing SQL or interpretation" else ""
        )

        header = (
            f'<span class="trace-status {status_cls}">{status}</span> '
            f'<span class="trace-interp">{interp}</span>'
            + (f' <span class="trace-rows">{rows_count} rows</span>' if rows_count is not None else "")
            + (f' <span class="trace-sql-score">sql {_score(sql_score_val)}</span>' if sql_score_val is not None else "")
        )
        err_html = f'<div class="trace-error">{_esc(str(err))}</div>' if err else ""

        traces_html += f"""
<div class="trace">
  <div class="trace-header">{header}</div>
  {err_html}
  {sql_block}
  {results_block}
  {reason_block}
</div>"""

    traces_section = (
        _collapsible(f"SQL Traces ({len(traces)})", f'<div class="traces">{traces_html}</div>', open=True)
        if traces_html else ""
    )

    # ── agent response (collapsible) ─────────────────────────────────────────
    preview = _esc(agent_resp_raw[:120]) + ("…" if len(agent_resp_raw) > 120 else "")
    full_resp = _esc(agent_resp_raw)
    response_section = _collapsible(
        f"Agent Response — {preview}",
        f'<div class="response-full">{full_resp}</div>',
        open=ar_passed is False,  # auto-open on failure to highlight the problem
    )

    # ── judge reason (collapsible) ────────────────────────────────────────────
    ar_reason = ar.get("reason") or ""
    judge_section = (
        _collapsible("Judge Reason", f'<div class="judge-inner">{_esc(ar_reason)}</div>', open=True)
        if ar_reason else ""
    )

    # ── ic reason (collapsible) ───────────────────────────────────────────────
    ic_reason = ic.get("reason") or "" if ic else ""
    ic_section = (
        _collapsible("Interpretation Judge Reason", f'<div class="judge-inner">{_esc(ic_reason)}</div>')
        if ic_reason else ""
    )

    # ── tool compliance badge ─────────────────────────────────────────────────
    tc_html = ""
    if tc is not None:
        if tc.get("compliant"):
            tc_html = '<span class="tc-pass">✓ tools OK</span>'
        else:
            missing = tc.get("missing") or []
            extra = tc.get("extra") or []
            detail = ""
            if missing:
                detail += f" missing: {', '.join(missing)}"
            if extra:
                detail += f" extra: {', '.join(extra)}"
            tc_html = f'<span class="tc-fail">✗ tools{detail}</span>'

    error_html = f'<div class="error-box">Error: {_esc(str(error))}</div>' if error else ""

    return f"""
<div class="{card_cls}">
  <div class="card-header">
    <span class="qid">q{qid}</span>
    <span class="flow-tag">{_esc(flow)}</span>
    <span class="complexity-tag">{_esc(complexity)}</span>
    <span class="question">{question}</span>
    <span class="duration">{_ms(duration)}</span>
    {_pass_badge(ar_passed)}
    {tc_html}
  </div>
  <div class="card-scores">
    <div class="score-item">
      <span class="score-lbl">Agent Response</span>
      {_score_bar(ar_score)}
    </div>
    {"<div class='score-item'><span class='score-lbl'>Interpretation</span>"+_score_bar(ic_score)+"</div>" if ic_score is not None else ""}
    {"<div class='score-item'><span class='score-lbl'>SQL Correctness</span>"+_score_bar(sql_mean)+"</div>" if sql_mean is not None else ""}
  </div>
  {error_html}
  {traces_section}
  {response_section}
  {judge_section}
  {ic_section}
</div>"""


def _questions_section(results: list) -> str:
    cards = "".join(_question_card(r) for r in results)
    return f"""
<section>
  <h2>Questions ({len(results)})</h2>
  {cards}
</section>"""


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       font-size: 13px; background: #f5f5f5; color: #333; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }

.header { background: #1e2a3a; color: #fff; padding: 20px 24px; border-radius: 8px; margin-bottom: 20px; }
.header h1 { font-size: 20px; margin-bottom: 6px; }
.header .meta { color: #aac; font-size: 12px; margin-bottom: 14px; }
.stats { display: flex; gap: 24px; }
.stat { text-align: center; }
.stat-val { font-size: 22px; font-weight: 700; color: #fff; }
.stat-lbl { font-size: 11px; color: #99b; text-transform: uppercase; letter-spacing: .5px; }

section { background: #fff; border-radius: 8px; padding: 18px 20px; margin-bottom: 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); }
h2 { font-size: 15px; color: #1e2a3a; margin-bottom: 14px; }

table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 7px 10px; background: #f0f2f5;
     font-size: 11px; text-transform: uppercase; letter-spacing: .4px; color: #666; }
td { padding: 8px 10px; border-bottom: 1px solid #f0f2f5; vertical-align: middle; }
tr:last-child td { border-bottom: none; }

.bar-wrap { display: flex; align-items: center; gap: 8px; }
.bar { height: 10px; border-radius: 5px; min-width: 2px; }
.bar.green { background: #22c55e; }
.bar.amber { background: #f59e0b; }
.bar.red   { background: #ef4444; }
.bar.gray  { background: #d1d5db; }
.bar-label { font-size: 12px; color: #555; min-width: 32px; }

.green { color: #16a34a; }
.amber { color: #b45309; }
.red   { color: #dc2626; }
.gray  { color: #9ca3af; }

.badge { display: inline-block; padding: 1px 7px; border-radius: 10px;
         font-size: 10px; font-weight: 700; letter-spacing: .4px; }
.badge.pass { background: #dcfce7; color: #16a34a; }
.badge.fail { background: #fee2e2; color: #dc2626; }
.badge.gray { background: #f3f4f6; color: #9ca3af; }

.flow-tag { display: inline-block; padding: 1px 7px; border-radius: 10px;
            background: #eff6ff; color: #1d4ed8; font-size: 11px; font-weight: 600; }
.complexity-tag { color: #9ca3af; font-size: 11px; }

.card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px;
        margin-bottom: 12px; }
.card-pass { border-left: 4px solid #22c55e; }
.card-fail { border-left: 4px solid #ef4444; }

.card-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.qid { font-weight: 700; color: #555; min-width: 36px; font-size: 12px; }
.question { flex: 1; font-weight: 600; }
.duration { color: #9ca3af; font-size: 11px; margin-left: auto; }

.card-scores { display: flex; gap: 20px; margin-bottom: 10px; flex-wrap: wrap; }
.score-item { display: flex; align-items: center; gap: 8px; }
.score-lbl { font-size: 11px; color: #666; min-width: 120px; }

.traces { background: #f8fafc; border-radius: 6px; padding: 8px 10px;
          margin-bottom: 10px; font-size: 11px; }
.trace { padding: 6px 0; border-bottom: 1px solid #e5e7eb; }
.trace:last-child { border-bottom: none; }
.trace-header { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 4px; }
.trace-status { font-weight: 700; font-size: 10px; padding: 1px 5px; border-radius: 3px; }
.trace-status.green { background: #dcfce7; color: #16a34a; }
.trace-status.red   { background: #fee2e2; color: #dc2626; }
.trace-status.amber { background: #fef3c7; color: #b45309; }
.trace-interp { flex: 1; color: #374151; font-size: 12px; }
.trace-rows { color: #6b7280; font-size: 11px; }
.trace-sql-score { color: #7c3aed; font-weight: 600; font-size: 11px; }
.trace-error { color: #dc2626; margin: 4px 0; font-size: 11px; }
.trace-sql { margin: 4px 0; }

details { margin: 4px 0; }
summary { cursor: pointer; font-size: 12px; color: #4b5563; padding: 3px 0;
          user-select: none; list-style: disclosure-closed; }
summary:hover { color: #1d4ed8; }
details[open] summary { list-style: disclosure-open; }
details pre { margin: 4px 0 0; }
details code { background: #1e293b; color: #94a3b8; padding: 6px 10px;
               border-radius: 4px; display: block; white-space: pre-wrap;
               font-size: 11px; line-height: 1.5; overflow-x: auto; }

.response-full { color: #374151; line-height: 1.6; white-space: pre-wrap;
                 padding: 8px 0; font-size: 13px; }
.judge-inner { color: #6b7280; font-style: italic; font-size: 12px;
               line-height: 1.5; padding: 6px 0; }
.tc-pass { color: #16a34a; font-size: 11px; font-weight: 600; }
.tc-fail { color: #dc2626; font-size: 11px; font-weight: 600; }
.error-box { background: #fee2e2; color: #dc2626; padding: 6px 10px;
             border-radius: 4px; margin-bottom: 8px; font-size: 12px; }
"""


# ── main ──────────────────────────────────────────────────────────────────────

def generate_html_report(run_path: Path) -> Path:
    """Generate an HTML report from a validator run JSON file.

    Returns the path to the generated HTML file.
    """
    run = json.loads(run_path.read_text())
    agg = run.get("aggregated_scores") or {}
    results = run.get("results") or []

    body = (
        _header_section(run)
        + _metrics_section(agg)
        + _flow_section(agg)
        + _questions_section(results)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Validator Report — {_esc(run.get("run_id", ""))}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>"""

    out_path = run_path.with_suffix(".html")
    out_path.write_text(html, encoding="utf-8")
    return out_path
