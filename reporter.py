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

def _overall_status_badge(status: str | None) -> str:
    if not status:
        return ""
    cls = "reg-pass" if status == "PASSED" else "reg-regression"
    return f'<span class="overall-badge {cls}">{_esc(status)}</span>'


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
    overall_status = run.get("overall_status")

    stat = lambda label, value: f'<div class="stat"><div class="stat-val">{value}</div><div class="stat-lbl">{label}</div></div>'

    return f"""
<div class="header">
  <h1>Analytics Agent Validator {_overall_status_badge(overall_status)}</h1>
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


def _performance_table(performance: dict) -> str:
    """Render the run-level performance comparison — no pass/fail, trend only."""
    if not performance:
        return ""
    rows = ""
    for entry in performance.values():
        # Use pre-computed delta/delta_pct from comparator — no recomputation.
        if "delta_pct" in entry:
            delta_val = entry["delta_pct"]
            delta_str = f"{delta_val:+.1f}%"
        else:
            delta_val = entry.get("delta", 0)
            delta_str = f"{delta_val:+.2f}"
        arrow = "▲" if delta_val > 0 else ("▼" if delta_val < 0 else "—")
        # Higher latency/tokens/calls = worse → amber; lower = green.
        cls = "amber" if delta_val > 0 else "green"
        baseline_fmt = f"{baseline:,.1f}" if baseline >= 100 else f"{baseline:.2f}"
        current_fmt = f"{current:,.1f}" if current >= 100 else f"{current:.2f}"
        rows += (
            f"<tr>"
            f"<td>{_esc(entry.get('label',''))}</td>"
            f"<td>{baseline_fmt}</td>"
            f"<td>{current_fmt}</td>"
            f'<td class="{cls}">{arrow} {delta_str}</td>'
            f"</tr>"
        )
    return f"""
<h3 class="reg-heading">Performance vs Baseline</h3>
<table>
  <thead><tr>
    <th>Metric</th><th>Baseline mean</th><th>This run</th><th>Change</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _regression_section(run: dict) -> str:
    """Render the comparison / regression block from the run JSON, if present."""
    comparison = run.get("comparison")
    if not comparison:
        return ""

    overall_status = comparison.get("overall_status", "PASSED")
    baseline_runs = comparison.get("baseline_runs", 0)
    regressions = comparison.get("regressions") or []
    improvements = comparison.get("improvements") or []

    if baseline_runs == 0:
        return ""  # first run — no baseline yet, nothing to show

    status_cls = "reg-regression" if overall_status == "REGRESSION" else "reg-pass"
    status_html = f'<span class="overall-badge {status_cls}">{_esc(overall_status)}</span>'
    q_with = comparison.get("questions_with_baseline")
    q_without = comparison.get("questions_without_baseline")
    coverage_html = ""
    if q_with is not None and q_without is not None:
        coverage_html = (
            f" &nbsp;·&nbsp; "
            f'<span class="reg-coverage">{q_with} question(s) compared</span>'
            + (
                f', <span class="reg-no-baseline">{q_without} without enough history</span>'
                if q_without else ""
            )
        )
    meta = f'<div class="reg-meta">Baseline: {baseline_runs} run(s) &nbsp;·&nbsp; {status_html}{coverage_html}</div>'

    def _findings_table(findings: list[dict], heading: str, delta_cls_fn) -> str:
        if not findings:
            return ""
        rows = ""
        for f in findings:
            delta = f.get("delta", 0)
            dcls = delta_cls_fn(delta)
            rows += (
                f'<tr>'
                f'<td><code>q{f["question_id"]}</code></td>'
                f'<td>{_esc(f.get("metric",""))}</td>'
                f'<td>{_score(f.get("baseline_mean"))}</td>'
                f'<td>{_score(f.get("current_score"))}</td>'
                f'<td class="{dcls}">{delta:+.3f}</td>'
                f'</tr>'
            )
        return f"""
<h3 class="reg-heading">{heading} ({len(findings)})</h3>
<table>
  <thead><tr><th>Question</th><th>Metric</th><th>Baseline mean</th><th>Current score</th><th>Delta</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""

    reg_table = _findings_table(regressions, "Regressions", lambda d: "red")
    imp_table = _findings_table(improvements, "Improvements", lambda d: "green")

    if not reg_table and not imp_table:
        no_change = '<p class="reg-ok">No regressions or improvements vs baseline.</p>'
    else:
        no_change = ""

    perf_table = _performance_table(comparison.get("performance") or {})

    return f"""
<section class="reg-section">
  <h2>Regression Report</h2>
  {meta}
  {reg_table}
  {imp_table}
  {no_change}
  {perf_table}
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
    od_fdl_results = r.get("od_fdl_results") or []
    sub_reasoning = r.get("sub_reasoning") or {}
    per_trace = ev.get("per_trace") or []
    sql_scores = [pr for pr in per_trace if pr.get("sql_score") is not None]

    ar_score = ar.get("score")
    ar_passed = ar.get("passed")
    ic_score = ic.get("score") if ic else None
    ic_passed = ic.get("passed") if ic else None
    sql_mean = (
        sum(pr["sql_score"] for pr in sql_scores) / len(sql_scores)
        if sql_scores else None
    )

    regression_flag = r.get("regression_flag", False)
    card_cls = "card"
    if regression_flag:
        card_cls += " card-regression"
    elif ar_passed is False:
        card_cls += " card-fail"
    elif ar_passed is True:
        card_cls += " card-pass"

    # ── od/fdl query results + SQL scores (one per result set) ────────────────
    results_html = ""
    for i, res in enumerate(od_fdl_results):
        interp = _esc(res.get("interpretation") or "")
        schema = _esc(res.get("schema") or "")
        sql_raw = res.get("sql") or ""
        rows = res.get("rows") or []

        pt = per_trace[i] if i < len(per_trace) else {}
        sql_score_val = pt.get("sql_score")
        sql_reason = _esc(pt.get("sql_reason") or "")
        status_cls = "green" if pt.get("sql_passed") is True else "red" if pt.get("sql_passed") is False else "amber"

        sql_block = (
            _collapsible("SQL", f"<pre><code>{_esc(sql_raw)}</code></pre>")
            if sql_raw else ""
        )
        rows_block = ""
        if rows:
            rows_preview = _esc(json.dumps(rows[:5], indent=2, default=str))
            rows_block = _collapsible(
                f"Results ({len(rows)} rows)",
                f"<pre><code>{rows_preview}</code></pre>"
            )
        reason_block = (
            _collapsible("SQL judge reason", f'<div class="judge-inner">{sql_reason}</div>')
            if sql_reason and "skipped" not in sql_reason else ""
        )

        score_tag = (
            f' <span class="trace-sql-score">sql {_score(sql_score_val)}</span>'
            if sql_score_val is not None else ""
        )
        header = (
            f'<span class="trace-status {status_cls}">SQL</span> '
            f'<span class="trace-interp">{interp}</span>'
            + (f' <span class="trace-rows">{schema}</span>' if schema else "")
            + (f' <span class="trace-rows">{len(rows)} rows</span>' if rows else "")
            + score_tag
        )

        results_html += f"""
<div class="trace">
  <div class="trace-header">{header}</div>
  {sql_block}
  {rows_block}
  {reason_block}
</div>"""

    # ── sub-agent reasoning (one collapsible per source) ─────────────────────
    reasoning_html = ""
    for src, text in sub_reasoning.items():
        if not text.strip():
            continue
        reasoning_html += _collapsible(
            f"{src} reasoning",
            f"<pre><code>{_esc(text.strip())}</code></pre>",
        )

    traces_section = ""
    if results_html or reasoning_html:
        n_traces = len(od_fdl_results)
        n_sources = len([s for s, t in sub_reasoning.items() if t.strip()])
        label = f"SQL Traces ({n_traces})"
        if n_sources:
            label += f" + Reasoning ({n_sources} source{'s' if n_sources > 1 else ''})"
        traces_section = _collapsible(
            label,
            f'<div class="traces">{results_html}{reasoning_html}</div>',
            open=True,
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

    # ── ic reason (collapsible, score in summary like SQL) ────────────────────
    ic_reason = ic.get("reason") or "" if ic else ""
    if ic_reason:
        ic_score_tag = (
            f'<span class="trace-sql-score">{_score(ic_score)}</span> '
            if ic_score is not None else ""
        )
        ic_badge = _pass_badge(ic_passed) if ic_score is not None else ""
        ic_section = _collapsible(
            f"Interpretation Coverage &nbsp; {ic_score_tag}{ic_badge}",
            f'<div class="judge-inner">{_esc(ic_reason)}</div>',
            open=ic_passed is False,
        )
    else:
        ic_section = ""

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

    regression_badge = '<span class="badge regression">REGRESSION</span>' if regression_flag else ""
    error_html = f'<div class="error-box">Error: {_esc(str(error))}</div>' if error else ""

    # ── non-SQL tool results (RCA, AO) ────────────────────────────────────────
    tool_results = r.get("tool_results") or {}
    tool_data_section = ""
    if tool_results:
        parts = []
        for tool_name, payload in tool_results.items():
            raw = _esc(json.dumps(payload, indent=2, default=str))
            parts.append(_collapsible(f"{tool_name} data", f"<pre><code>{raw}</code></pre>"))
        tool_data_section = _collapsible(
            f"Tool Data ({', '.join(tool_results)})",
            "".join(parts),
        )

    return f"""
<div class="{card_cls}">
  <div class="card-header">
    <span class="qid">q{qid}</span>
    <span class="flow-tag">{_esc(flow)}</span>
    <span class="complexity-tag">{_esc(complexity)}</span>
    <span class="question">{question}</span>
    <span class="duration">{_ms(duration)}</span>
    {_pass_badge(ar_passed)}
    {regression_badge}
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
  {tool_data_section}
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


# ── Chain section builders ────────────────────────────────────────────────────

def _chain_turn_card(turn: dict) -> str:
    idx = turn.get("turn_index", 0)
    question = _esc(turn.get("question", ""))
    agent_resp_raw = turn.get("agent_response") or ""
    duration = turn.get("duration_ms")
    error = turn.get("error")
    ev = turn.get("eval") or {}

    tr = ev.get("turn_response") or {}
    cr = ev.get("context_retention") or {}
    per_trace = ev.get("per_trace") or []
    od_fdl_results = turn.get("od_fdl_results") or []
    sub_reasoning = turn.get("sub_reasoning") or {}

    tr_score = tr.get("score")
    tr_passed = tr.get("passed")
    cr_score = cr.get("score") if cr else None
    sql_scores = [pr for pr in per_trace if pr.get("sql_score") is not None]
    sql_mean = (
        sum(pr["sql_score"] for pr in sql_scores) / len(sql_scores)
        if sql_scores else None
    )

    # SQL traces
    results_html = ""
    for i, res in enumerate(od_fdl_results):
        interp = _esc(res.get("interpretation") or "")
        schema = _esc(res.get("schema") or "")
        sql_raw = res.get("sql") or ""
        rows = res.get("rows") or []
        pt = per_trace[i] if i < len(per_trace) else {}
        sql_score_val = pt.get("sql_score")
        sql_reason = _esc(pt.get("sql_reason") or "")
        status_cls = "green" if pt.get("sql_passed") is True else "red" if pt.get("sql_passed") is False else "amber"

        sql_block = _collapsible("SQL", f"<pre><code>{_esc(sql_raw)}</code></pre>") if sql_raw else ""
        rows_block = (
            _collapsible(f"Results ({len(rows)} rows)", f"<pre><code>{_esc(json.dumps(rows[:5], indent=2, default=str))}</code></pre>")
            if rows else ""
        )
        reason_block = (
            _collapsible("SQL judge reason", f'<div class="judge-inner">{sql_reason}</div>')
            if sql_reason and "skipped" not in sql_reason else ""
        )
        score_tag = f' <span class="trace-sql-score">sql {_score(sql_score_val)}</span>' if sql_score_val is not None else ""
        header = (
            f'<span class="trace-status {status_cls}">SQL</span> '
            f'<span class="trace-interp">{interp}</span>'
            + (f' <span class="trace-rows">{schema}</span>' if schema else "")
            + (f' <span class="trace-rows">{len(rows)} rows</span>' if rows else "")
            + score_tag
        )
        results_html += f'<div class="trace"><div class="trace-header">{header}</div>{sql_block}{rows_block}{reason_block}</div>'

    reasoning_html = "".join(
        _collapsible(f"{src} reasoning", f"<pre><code>{_esc(text.strip())}</code></pre>")
        for src, text in sub_reasoning.items() if text.strip()
    )
    traces_section = ""
    if results_html or reasoning_html:
        label = f"SQL Traces ({len(od_fdl_results)})"
        if reasoning_html:
            n_src = sum(1 for t in sub_reasoning.values() if t.strip())
            label += f" + Reasoning ({n_src})"
        traces_section = _collapsible(label, f'<div class="traces">{results_html}{reasoning_html}</div>', open=True)

    preview = _esc(agent_resp_raw[:100]) + ("…" if len(agent_resp_raw) > 100 else "")
    response_section = _collapsible(
        f"Response — {preview}",
        f'<div class="response-full">{_esc(agent_resp_raw)}</div>',
        open=tr_passed is False,
    )
    tr_reason = tr.get("reason") or ""
    cr_reason = cr.get("reason") or "" if cr else ""
    judge_section = (
        _collapsible("Turn Response Judge", f'<div class="judge-inner">{_esc(tr_reason)}</div>', open=True)
        if tr_reason else ""
    )
    cr_section = (
        _collapsible("Context Retention Judge", f'<div class="judge-inner">{_esc(cr_reason)}</div>')
        if cr_reason else ""
    )
    error_html = f'<div class="error-box">Error: {_esc(str(error))}</div>' if error else ""

    turn_cls = "chain-turn"
    if tr_passed is False:
        turn_cls += " chain-turn-fail"
    elif tr_passed is True:
        turn_cls += " chain-turn-pass"

    return f"""
<div class="{turn_cls}">
  <div class="chain-turn-header">
    <span class="turn-num">Turn {idx + 1}</span>
    <span class="question">{question}</span>
    <span class="duration">{_ms(duration)}</span>
    {_pass_badge(tr_passed)}
  </div>
  <div class="card-scores">
    <div class="score-item"><span class="score-lbl">Turn Response</span>{_score_bar(tr_score)}</div>
    {"<div class='score-item'><span class='score-lbl'>Context Retention</span>"+_score_bar(cr_score)+"</div>" if cr_score is not None else ""}
    {"<div class='score-item'><span class='score-lbl'>SQL Correctness</span>"+_score_bar(sql_mean)+"</div>" if sql_mean is not None else ""}
  </div>
  {error_html}
  {traces_section}
  {response_section}
  {judge_section}
  {cr_section}
</div>"""


def _chain_card(chain: dict) -> str:
    chain_id = chain.get("chain_id", "?")
    name = _esc(chain.get("name", ""))
    flow = _esc(chain.get("flow", ""))
    turns = chain.get("turns") or []
    ce = chain.get("chain_eval") or {}

    tr_mean = ce.get("turn_response_mean")
    tr_pass_rate = ce.get("turn_response_pass_rate")
    cr_mean = ce.get("context_retention_mean")
    cr_pass_rate = ce.get("context_retention_pass_rate")

    overall_passed = tr_pass_rate is not None and tr_pass_rate >= 0.7
    overall_passed_val = True if (tr_pass_rate is not None and tr_pass_rate >= 0.7) else (False if tr_pass_rate is not None else None)

    turns_html = "".join(_chain_turn_card(t) for t in turns)

    return f"""
<div class="chain-card">
  <div class="chain-header">
    <span class="chain-id">chain{chain_id}</span>
    <span class="flow-tag">{flow}</span>
    <span class="chain-name">{name}</span>
    <span class="duration">{len(turns)} turns</span>
    {_pass_badge(overall_passed_val)}
  </div>
  <div class="card-scores" style="margin-bottom:12px;">
    <div class="score-item"><span class="score-lbl">Turn Response mean</span>{_score_bar(tr_mean)}</div>
    <div class="score-item"><span class="score-lbl">Turn Response pass</span><span class="bar-label">{_pct(tr_pass_rate)}</span></div>
    {"<div class='score-item'><span class='score-lbl'>Context Retention mean</span>"+_score_bar(cr_mean)+"</div>" if cr_mean is not None else ""}
    {"<div class='score-item'><span class='score-lbl'>Context Retention pass</span><span class='bar-label'>"+_pct(cr_pass_rate)+"</span></div>" if cr_pass_rate is not None else ""}
  </div>
  <div class="chain-turns">{turns_html}</div>
</div>"""


def _chains_section(results: list) -> str:
    chains = [r for r in results if "chain_id" in r]
    if not chains:
        return ""
    cards = "".join(_chain_card(c) for c in chains)
    return f"""
<section>
  <h2>Chains ({len(chains)})</h2>
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
.badge.pass       { background: #dcfce7; color: #16a34a; }
.badge.fail       { background: #fee2e2; color: #dc2626; }
.badge.gray       { background: #f3f4f6; color: #9ca3af; }
.badge.regression { background: #fef3c7; color: #92400e; }

.overall-badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
                 font-size: 12px; font-weight: 700; letter-spacing: .5px; vertical-align: middle; }
.overall-badge.reg-pass       { background: #dcfce7; color: #15803d; }
.overall-badge.reg-regression { background: #fef3c7; color: #92400e; }

.flow-tag { display: inline-block; padding: 1px 7px; border-radius: 10px;
            background: #eff6ff; color: #1d4ed8; font-size: 11px; font-weight: 600; }
.complexity-tag { color: #9ca3af; font-size: 11px; }

.card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px;
        margin-bottom: 12px; }
.card-pass       { border-left: 4px solid #22c55e; }
.card-fail       { border-left: 4px solid #ef4444; }
.card-regression { border-left: 4px solid #f59e0b; background: #fffbeb; }

.reg-section { border: 1px solid #fde68a; }
.reg-meta { font-size: 12px; color: #6b7280; margin-bottom: 14px; }
.reg-heading { font-size: 13px; font-weight: 600; color: #374151; margin: 14px 0 8px; }
.reg-ok { font-size: 13px; color: #6b7280; font-style: italic; margin-top: 8px; }
.reg-coverage { color: #374151; }
.reg-no-baseline { color: #b45309; }

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

.chain-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px;
              margin-bottom: 12px; }
.chain-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
                margin-bottom: 10px; border-bottom: 1px solid #f0f2f5; padding-bottom: 10px; }
.chain-id { font-weight: 700; color: #555; font-size: 12px; min-width: 52px; }
.chain-name { flex: 1; font-weight: 600; }

.chain-turns { display: flex; flex-direction: column; gap: 8px; }
.chain-turn { border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px 12px; }
.chain-turn-pass { border-left: 4px solid #22c55e; }
.chain-turn-fail { border-left: 4px solid #ef4444; }
.chain-turn-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
                     margin-bottom: 8px; }
.turn-num { font-size: 11px; font-weight: 700; color: #6b7280; min-width: 44px; }
"""


# ── main ──────────────────────────────────────────────────────────────────────

def generate_html_report(run_path: Path) -> Path:
    """Generate an HTML report from a validator run JSON file.

    Returns the path to the generated HTML file.
    """
    run = json.loads(run_path.read_text())
    agg = run.get("aggregated_scores") or {}
    # Only render single-turn question cards; chain records don't have question_id.
    results = [r for r in (run.get("results") or []) if "question_id" in r]

    all_results = run.get("results") or []
    chain_results = [r for r in all_results if "chain_id" in r]

    body = (
        _header_section(run)
        + _metrics_section(agg)
        + _flow_section(agg)
        + _regression_section(run)
        + _questions_section(results)
        + _chains_section(chain_results)
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
