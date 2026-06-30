"""Streamlit UI for the Analytics Agent Validator.

Run:
    streamlit run tests/validator/ui.py
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import streamlit as st

HARNESS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = HARNESS_DIR / "results"
CHAINS_PATH = HARNESS_DIR / "test_chains.json"
QUESTIONS_PATH = HARNESS_DIR / "test_questions.json"
PYTHON = sys.executable
PASS_THRESHOLD = 0.7
VALID_FLOWS = ["fdl", "od", "ao", "rca", "advisory", "mixed", "context", "refresh", "conversational"]


# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="Validator",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Analytics Agent Validator")


# -- Load catalogs -------------------------------------------------------------

_questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if QUESTIONS_PATH.exists() else []
_chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8")) if CHAINS_PATH.exists() else []
_all_items = (
    [("q", q) for q in _questions] +
    [("c", c) for c in _chains]
)


# -- Helpers -------------------------------------------------------------------

def _ck(kind: str, item_id: int) -> str:
    return f"{kind}_{item_id}"


def _get_selected() -> dict:
    """Return {'q': [ids], 'c': [ids]} of currently checked items."""
    qs = [q["id"] for q in _questions if st.session_state.get(_ck("q", q["id"]), False)]
    cs = [c["chain_id"] for c in _chains if st.session_state.get(_ck("c", c["chain_id"]), False)]
    return {"q": qs, "c": cs}


def _apply_preset(preset: str) -> None:
    if preset == "smoke":
        by_flow: dict = {}
        for q in _questions:
            by_flow.setdefault(q.get("flow", "other"), []).append(q)
        selected_qids = set()
        for qs in by_flow.values():
            p = [q for q in qs if q.get("priority")]
            selected_qids.add((p[0] if p else qs[0])["id"])
        for q in _questions:
            st.session_state[_ck("q", q["id"])] = q["id"] in selected_qids
        for c in _chains:
            st.session_state[_ck("c", c["chain_id"])] = False

    elif preset == "priority":
        for q in _questions:
            st.session_state[_ck("q", q["id"])] = bool(q.get("priority"))
        for c in _chains:
            st.session_state[_ck("c", c["chain_id"])] = bool(c.get("priority"))

    elif preset == "all_q":
        for q in _questions:
            st.session_state[_ck("q", q["id"])] = True

    elif preset == "none":
        for q in _questions:
            st.session_state[_ck("q", q["id"])] = False
        for c in _chains:
            st.session_state[_ck("c", c["chain_id"])] = False


def _build_args(out_path: Path, selected: dict) -> list[str]:
    q_ids = selected.get("q") or []
    c_ids = selected.get("c") or []
    base = [PYTHON, "-u", "-m", "tests.validator.run", "--output", str(out_path),
            "--base-url", st.session_state.get("base_url", "http://localhost:80")]

    _env = st.session_state.get("env_name", "staging")
    if st.session_state.get("use_tunnel"):
        base += ["--env", _env, "--tunnel"]
    elif _env:
        base += ["--env", _env]

    if st.session_state.get("skip_eval"):
        base.append("--skip-eval")

    if c_ids and not q_ids:
        base += ["--chains", "--chain-ids", ",".join(str(i) for i in sorted(c_ids))]
    elif q_ids:
        base += ["--ids", ",".join(str(i) for i in sorted(q_ids))]

    return base


def _run_env() -> dict:
    env_vars = os.environ.copy()
    env_vars["PYTHONIOENCODING"] = "utf-8"
    env_vars["PYTHONUTF8"] = "1"
    env_vars["PYTHONUNBUFFERED"] = "1"  # flush every print() immediately to the pipe
    token = st.session_state.get("_resolved_token", "")
    tenant = st.session_state.get("_resolved_tenant", "")
    if token:
        env_vars["EVAL_BEARER_TOKEN"] = token
    if tenant:
        env_vars["EVAL_TENANT_ID"] = tenant
    return env_vars


def _auth_login(email: str, password: str, env_name: str) -> dict:
    url = f"https://na1.{env_name}.nice-incontact.com/public/authentication/v1/login"
    resp = requests.post(url, json={"email": email, "password": password}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token", "")
    if not token:
        raise ValueError("Empty token returned from auth API")
    return {"token": token, "tenant_id": data.get("tenantId", "")}


def _push_status(run_path: Path) -> tuple[str, str]:
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except Exception:
        return "warning", "Could not read result file."
    agg = run.get("aggregated_scores") or {}
    total = agg.get("total", 0)
    errors = agg.get("errors", 0)
    ar = agg.get("agent_response") or {}
    pass_rate = ar.get("pass_rate")
    mean = ar.get("mean")
    if total == 0:
        return "warning", "No questions ran."
    if errors and errors / total > 0.3:
        return "error", f"HIGH ERROR RATE: {errors}/{total} questions errored."
    if pass_rate is None:
        return "warning", "No scores yet. Re-run without --skip-eval to assess quality."
    mean_str = f"  (mean {mean:.2f})" if mean is not None else ""
    if pass_rate >= 0.8:
        return "success", f"Quality looks good - {pass_rate*100:.0f}% passed{mean_str}"
    elif pass_rate >= 0.6:
        return "warning", f"Some quality issues - {pass_rate*100:.0f}% passed{mean_str}. Review failing questions."
    else:
        return "error", f"Low quality detected - only {pass_rate*100:.0f}% passed{mean_str}."


def _start_proc(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=_run_env(),
        cwd=str(HARNESS_DIR.parents[1]),
    )


def _reader_thread(proc: subprocess.Popen, out_queue: queue.Queue) -> None:
    """Background thread: stream stdout lines into the queue."""
    for line in proc.stdout:
        out_queue.put(line.rstrip())
    out_queue.put(None)  # sentinel: process finished


# -- Session state init --------------------------------------------------------

for _k, _v in [
    ("proc", None), ("run_output", []), ("last_run_path", None),
    ("out_queue", None), ("reader_done", False),
    ("base_url", "http://localhost:80"), ("env_name", "staging"),
    ("use_tunnel", False), ("skip_eval", False),
    ("_resolved_token", ""), ("_resolved_tenant", ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "picker_initialized" not in st.session_state:
    st.session_state.picker_initialized = True
    _apply_preset("smoke")


# -- Sidebar -------------------------------------------------------------------

with st.sidebar:
    # ---- Auth ----------------------------------------------------------------
    st.header("Authentication")
    auth_mode = st.radio("Method", ["Bearer token", "Email + Password"], horizontal=True)

    if auth_mode == "Bearer token":
        bearer = st.text_input("Bearer Token", type="password")
        tenant_override = st.text_input("Tenant ID", help="Optional - decoded from JWT if omitted.")
        if st.button("Apply token"):
            st.session_state._resolved_token = bearer.strip()
            st.session_state._resolved_tenant = tenant_override.strip()
            st.success("Token applied.")
    else:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_env = st.selectbox("Environment", ["staging", "dev", "production"], key="login_env")
        if st.button("Log in", type="primary"):
            with st.spinner("Authenticating..."):
                try:
                    result = _auth_login(email, password, login_env)
                    st.session_state._resolved_token = result["token"]
                    st.session_state._resolved_tenant = result.get("tenant_id", "")
                    st.success(f"Logged in. tenant={result.get('tenant_id','')[:8]}...")
                except Exception as exc:
                    st.error(f"Login failed: {exc}")

    if st.session_state._resolved_token:
        st.caption("Token: active")

    st.divider()

    # ---- Service -------------------------------------------------------------
    st.header("Service")
    st.session_state.base_url = st.text_input("Base URL", value=st.session_state.base_url)
    st.session_state.env_name = st.selectbox("Environment", ["staging", "dev", "production"],
                                             index=["staging", "dev", "production"].index(st.session_state.env_name))
    st.session_state.use_tunnel = st.checkbox("Tunnel (SSM)", value=st.session_state.use_tunnel)
    st.session_state.skip_eval = st.checkbox("Skip eval", value=st.session_state.skip_eval,
                                             help="Collect traces only, no LLM scoring.")

    st.divider()

    # ---- Test selection ------------------------------------------------------
    st.header("Test selection")

    sel = _get_selected()
    n_q = len(sel["q"])
    n_c = len(sel["c"])
    n_total_sel = n_q + n_c
    st.caption(f"Selected: {n_total_sel} ({n_q} questions, {n_c} chains)")

    p_cols = st.columns(3)
    if p_cols[0].button("Smoke", use_container_width=True):
        _apply_preset("smoke"); st.rerun()
    if p_cols[1].button("Priority", use_container_width=True):
        _apply_preset("priority"); st.rerun()
    if p_cols[2].button("Clear all", use_container_width=True):
        _apply_preset("none"); st.rerun()

    # Questions grouped by flow
    by_flow: dict = {}
    for q in _questions:
        by_flow.setdefault(q.get("flow", "other"), []).append(q)

    with st.expander(f"Questions ({n_q}/{len(_questions)} selected)", expanded=True):
        sel_all_q = st.checkbox("All questions", value=n_q == len(_questions), key="sel_all_q")
        if sel_all_q != (n_q == len(_questions)):
            for q in _questions:
                st.session_state[_ck("q", q["id"])] = sel_all_q
            st.rerun()
        for flow_name in sorted(by_flow.keys()):
            qs_in_flow = by_flow[flow_name]
            n_sel_flow = sum(1 for q in qs_in_flow if st.session_state.get(_ck("q", q["id"]), False))
            st.markdown(f"**{flow_name}** ({n_sel_flow}/{len(qs_in_flow)})")
            for q in qs_in_flow:
                key = _ck("q", q["id"])
                priority_tag = " [P]" if q.get("priority") else ""
                label = f"#{q['id']} {q.get('question','')[:45]}...  `{q.get('complexity','')}`{priority_tag}"
                st.checkbox(label, key=key)

    # Chains
    if _chains:
        n_sel_c = sum(1 for c in _chains if st.session_state.get(_ck("c", c["chain_id"]), False))
        with st.expander(f"Chains ({n_sel_c}/{len(_chains)} selected)", expanded=False):
            for c in _chains:
                key = _ck("c", c["chain_id"])
                turns = len(c.get("turns") or [])
                label = f"#{c['chain_id']} {c.get('name','')}  `{c.get('flow','')}` ({turns} turns)"
                st.checkbox(label, key=key)


# -- Re-compute selected after sidebar interactions ---------------------------

_selected = _get_selected()
_n_selected = len(_selected["q"]) + len(_selected["c"])


# -- Tabs ----------------------------------------------------------------------

tab_run, tab_results, tab_history, tab_catalog = st.tabs(["Run", "Results", "History", "Catalog"])


# -- Tab: Run ------------------------------------------------------------------

with tab_run:
    # Controls
    if not st.session_state._resolved_token:
        st.warning("No token - log in or paste a bearer token in the sidebar.")

    preview_path = RESULTS_DIR / "preview.json"
    cmd_args = _build_args(preview_path, _selected)
    with st.expander("CLI command"):
        st.code(" ".join(cmd_args), language="bash")

    ctrl_cols = st.columns([3, 1, 1])
    with ctrl_cols[0]:
        start = st.button(
            f"Run {_n_selected} test{'s' if _n_selected != 1 else ''}",
            type="primary", width="stretch",
            disabled=bool(st.session_state.proc) or _n_selected == 0 or not st.session_state._resolved_token,
        )
    with ctrl_cols[1]:
        stop = st.button("Stop", width="stretch", disabled=not bool(st.session_state.proc))
    with ctrl_cols[2]:
        if st.button("New run", width="stretch", disabled=bool(st.session_state.proc)):
            st.session_state.last_run_path = None
            st.session_state.run_output = []
            st.session_state.reader_done = False
            st.rerun()

    # Start process
    if start:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = "chains" if (_selected["c"] and not _selected["q"]) else "run"
        out_path = RESULTS_DIR / f"{prefix}_{ts}.json"
        args = _build_args(out_path, _selected)
        st.session_state.last_run_path = out_path
        st.session_state.run_output = [f"$ {' '.join(args)}", ""]
        st.session_state.reader_done = False
        proc = _start_proc(args)
        st.session_state.proc = proc
        q: queue.Queue = queue.Queue()
        st.session_state.out_queue = q
        threading.Thread(target=_reader_thread, args=(proc, q), daemon=True).start()
        st.rerun()

    # Stop process
    if stop and st.session_state.proc:
        st.session_state.proc.terminate()
        st.session_state.proc = None
        st.session_state.run_output.append("--- stopped ---")
        st.rerun()

    output_area = st.empty()

    # Blocking streaming loop: sends real-time WebSocket updates to the browser
    # without Streamlit's "running" spinner overlay (no st.rerun() during streaming).
    if st.session_state.proc is not None and st.session_state.out_queue is not None:
        q_ref = st.session_state.out_queue
        while True:
            # Drain available lines from the background reader thread
            for _ in range(100):
                try:
                    line = q_ref.get_nowait()
                    if line is None:
                        # Sentinel: process has exited
                        st.session_state.proc = None
                        st.session_state.reader_done = True
                        break
                    st.session_state.run_output.append(line)
                except queue.Empty:
                    break

            # Push current output to browser immediately (incremental update)
            output_area.code(
                "\n".join(st.session_state.run_output),
                language="text",
            )

            if st.session_state.proc is None:
                break  # process finished, exit loop

            time.sleep(0.2)

        # One final rerun to render the status banner and HTML report
        st.rerun()

    # After run: status banner -> collapsed log -> HTML report
    if st.session_state.reader_done and st.session_state.last_run_path:
        run_path = Path(st.session_state.last_run_path)

        if run_path.exists():
            status_type, status_msg = _push_status(run_path)
            if status_type == "success":
                st.success(status_msg)
            elif status_type == "warning":
                st.warning(status_msg)
            else:
                st.error(status_msg)

        with st.expander("Run log", expanded=False):
            st.code("\n".join(st.session_state.run_output), language="text")

        html_path = run_path.with_suffix(".html") if run_path.exists() else None
        if html_path and html_path.exists():
            st.subheader("Report")
            st.iframe(html_path.read_text(encoding="utf-8"), height=950)
        else:
            st.info("No HTML report yet (run without --skip-eval to generate one).")

    elif st.session_state.run_output:
        # Not yet started streaming (between start click and first loop iteration)
        output_area.code("\n".join(st.session_state.run_output), language="text")


# -- Tab: Results --------------------------------------------------------------

with tab_results:
    all_result_files = sorted(
        list(RESULTS_DIR.glob("run_*.json")) + list(RESULTS_DIR.glob("chains_*.json")),
        key=lambda p: p.name, reverse=True,
    )

    if not all_result_files:
        st.info("No results yet. Run an evaluation first.")
    else:
        selected_file = st.selectbox(
            "Select result file", all_result_files,
            format_func=lambda p: p.name, index=0,
        )

        if selected_file and selected_file.exists():
            run = json.loads(selected_file.read_text(encoding="utf-8"))
            agg = run.get("aggregated_scores") or {}
            ar = agg.get("agent_response") or {}

            # Status + re-run eval button
            r_cols = st.columns([4, 1])
            with r_cols[0]:
                status_type, status_msg = _push_status(selected_file)
                if status_type == "success":
                    st.success(status_msg)
                elif status_type == "warning":
                    st.warning(status_msg)
                else:
                    st.error(status_msg)
            with r_cols[1]:
                if st.button("Re-run eval", width="stretch",
                             disabled=bool(st.session_state.proc),
                             help="Score this run file with the LLM judge."):
                    args = [PYTHON, "-m", "tests.validator.run",
                            "--eval-only", str(selected_file)]
                    st.session_state.last_run_path = selected_file
                    st.session_state.run_output = [f"$ {' '.join(args)}", ""]
                    st.session_state.reader_done = False
                    proc = _start_proc(args)
                    st.session_state.proc = proc
                    q: queue.Queue = queue.Queue()
                    st.session_state.out_queue = q
                    threading.Thread(target=_reader_thread, args=(proc, q), daemon=True).start()
                    st.rerun()

            st.divider()

            # Metrics
            meta_cols = st.columns(4)
            meta_cols[0].metric("Questions", agg.get("total", len(run.get("results", []))))
            meta_cols[1].metric("Errors", agg.get("errors", 0))
            avg_lat = agg.get("avg_duration_ms")
            meta_cols[2].metric("Avg latency", f"{avg_lat/1000:.1f}s" if avg_lat else "-")
            meta_cols[3].metric("Timestamp", run.get("run_timestamp", "")[:16].replace("T", " "))

            st.divider()

            # Score metrics
            score_cols = st.columns(3)
            for col, (key, label) in zip(score_cols, [
                ("agent_response", "Agent Response"),
                ("interpretation_coverage", "Interpretation"),
                ("sql_correctness", "SQL Correctness"),
            ]):
                m = agg.get(key) or {}
                if m.get("mean") is not None:
                    col.metric(label, f"{m['mean']:.3f}",
                               f"{m['pass_rate']*100:.0f}% pass" if m.get("pass_rate") is not None else None)

            st.divider()

            # By flow
            by_flow_agg = agg.get("by_flow") or {}
            if by_flow_agg:
                st.subheader("By flow")
                rows = [
                    {
                        "Flow": flow,
                        "n": str(d.get("total", 0)),
                        "Mean": f"{d['agent_response_mean']:.3f}" if d.get("agent_response_mean") is not None else "-",
                        "Pass %": f"{d['agent_response_pass_rate']*100:.0f}%" if d.get("agent_response_pass_rate") is not None else "-",
                    }
                    for flow, d in sorted(by_flow_agg.items())
                ]
                st.dataframe(rows, width="stretch")

            st.divider()

            # Per-question table
            st.subheader("Questions")
            results = run.get("results") or []
            table_rows = []
            for r in results:
                ev = r.get("eval") or {}
                ar_r = ev.get("agent_response") or {}
                table_rows.append({
                    "ID": str(r.get("question_id", "")),
                    "Flow": r.get("flow", ""),
                    "Question": (r.get("question") or "")[:70],
                    "Score": f"{ar_r['score']:.3f}" if ar_r.get("score") is not None else "-",
                    "Pass": "yes" if ar_r.get("passed") else ("no" if ar_r.get("passed") is False else "-"),
                    "SQL calls": str(len(r.get("cortex_traces") or [])),
                    "Error": "!" if r.get("error") else "",
                })
            if table_rows:
                st.dataframe(table_rows, width="stretch", height=400)

            # HTML report
            html_path = selected_file.with_suffix(".html")
            if html_path.exists():
                st.divider()
                with st.expander("Full HTML report", expanded=False):
                    st.iframe(html_path.read_text(encoding="utf-8"), height=800)


# -- Tab: History --------------------------------------------------------------

with tab_history:
    st.subheader("Run history")
    all_files = sorted(
        list(RESULTS_DIR.glob("run_*.json")) + list(RESULTS_DIR.glob("chains_*.json")),
        key=lambda p: p.name, reverse=True,
    )

    if not all_files:
        st.info("No previous runs found.")
    else:
        history_rows = []
        for f in all_files[:30]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                agg = data.get("aggregated_scores") or {}
                ar = agg.get("agent_response") or {}
                history_rows.append({
                    "File": f.name,
                    "Timestamp": data.get("run_timestamp", "")[:16].replace("T", " "),
                    "n": str(agg.get("total", "-")),
                    "Agent mean": f"{ar['mean']:.3f}" if ar.get("mean") is not None else "-",
                    "Pass rate": f"{ar['pass_rate']*100:.0f}%" if ar.get("pass_rate") is not None else "-",
                    "Errors": str(agg.get("errors", 0)),
                    "Status": "PASS" if (ar.get("pass_rate") or 0) >= 0.7 else ("FAIL" if ar.get("pass_rate") is not None else "-"),
                })
            except Exception:
                pass

        if history_rows:
            st.dataframe(history_rows, width="stretch", height=300)

        # Trend chart
        trend_data = [(r["Timestamp"], float(r["Agent mean"])) for r in history_rows if r["Agent mean"] not in ("-", "")]
        if len(trend_data) >= 2:
            st.divider()
            st.subheader("Score trend")
            import pandas as pd
            df = pd.DataFrame(trend_data[::-1], columns=["Run", "Score"])
            df = df.set_index("Run")
            st.line_chart(df)
            st.caption(f"Pass threshold: {PASS_THRESHOLD}")

        # Report viewer
        st.divider()
        st.subheader("View report")
        selected_hist = st.selectbox(
            "Select run to view",
            all_files,
            format_func=lambda p: p.name,
            key="history_selected",
        )
        if selected_hist:
            html_hist = selected_hist.with_suffix(".html")
            if html_hist.exists():
                st.iframe(html_hist.read_text(encoding="utf-8"), height=900)
            else:
                st.info("No HTML report for this run (was run with --skip-eval).")


# -- Tab: Catalog --------------------------------------------------------------

def _save_questions(questions: list[dict]) -> None:
    QUESTIONS_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")

def _save_chains(chains: list[dict]) -> None:
    CHAINS_PATH.write_text(json.dumps(chains, indent=2, ensure_ascii=False), encoding="utf-8")

def _next_id(items: list[dict], id_key: str) -> int:
    return max((item.get(id_key, 0) for item in items), default=0) + 1


with tab_catalog:
    cat_q_tab, cat_c_tab = st.tabs(["Questions", "Chains"])

    # -- Questions --------------------------------------------------------------
    with cat_q_tab:
        questions_live = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if QUESTIONS_PATH.exists() else []

        # Filter bar
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 3])
        f_flow = fc1.selectbox("Flow", ["(all)"] + sorted(VALID_FLOWS), key="cat_flow")
        f_complex = fc2.selectbox("Complexity", ["(all)", "basic", "intermediate", "advanced"], key="cat_complex")
        f_priority = fc3.selectbox("Priority", ["(all)", "yes", "no"], key="cat_priority")
        f_search = fc4.text_input("Search", placeholder="keyword in question text", key="cat_search")

        filtered = questions_live
        if f_flow != "(all)":
            filtered = [q for q in filtered if q.get("flow") == f_flow]
        if f_complex != "(all)":
            filtered = [q for q in filtered if q.get("complexity") == f_complex]
        if f_priority == "yes":
            filtered = [q for q in filtered if q.get("priority")]
        elif f_priority == "no":
            filtered = [q for q in filtered if not q.get("priority")]
        if f_search.strip():
            kw = f_search.strip().lower()
            filtered = [q for q in filtered if kw in q.get("question", "").lower()]

        st.caption(f"Showing {len(filtered)} / {len(questions_live)} questions")

        # Table overview
        table_data = [
            {
                "ID": q["id"],
                "Flow": q.get("flow", ""),
                "Complexity": q.get("complexity", ""),
                "P": "yes" if q.get("priority") else "",
                "Question": q.get("question", "")[:80],
                "Tags": ", ".join(q.get("tags") or []),
            }
            for q in filtered
        ]
        st.dataframe(table_data, width="stretch", height=280)

        st.divider()

        # Edit / Add form
        all_ids = [q["id"] for q in questions_live]
        edit_options = ["-- Add new --"] + [f"#{q['id']} {q.get('question','')[:50]}" for q in filtered]
        edit_choice = st.selectbox("Edit or add", edit_options, key="cat_edit_choice")

        if edit_choice == "-- Add new --":
            editing_q = {
                "id": _next_id(questions_live, "id"),
                "question": "",
                "flow": "fdl",
                "complexity": "basic",
                "priority": False,
                "expected_tools": [],
                "golden_answer": None,
                "tags": [],
                "multi_turn": False,
            }
            is_new_q = True
        else:
            sel_id = int(edit_choice.split(" ")[0][1:])
            editing_q = next((q for q in questions_live if q["id"] == sel_id), None) or {}
            is_new_q = False

        with st.form("question_form"):
            col_a, col_b = st.columns([1, 2])
            q_id = col_a.number_input("ID", value=int(editing_q.get("id", 1)),
                                      disabled=not is_new_q)
            q_text = col_b.text_area("Question", value=editing_q.get("question", ""), height=80)

            col_c, col_d, col_e = st.columns(3)
            q_flow = col_c.selectbox("Flow", VALID_FLOWS,
                                     index=VALID_FLOWS.index(editing_q.get("flow", "fdl")) if editing_q.get("flow") in VALID_FLOWS else 0)
            q_complex = col_d.selectbox("Complexity", ["basic", "intermediate", "advanced"],
                                         index=["basic","intermediate","advanced"].index(editing_q.get("complexity","basic")))
            q_priority = col_e.checkbox("Priority", value=bool(editing_q.get("priority")))

            q_tools = st.text_input("Expected tools (comma-separated)",
                                    value=", ".join(editing_q.get("expected_tools") or []))
            q_tags = st.text_input("Tags (comma-separated)",
                                   value=", ".join(editing_q.get("tags") or []))
            q_golden = st.text_area("Golden answer (optional)", value=editing_q.get("golden_answer") or "", height=60)
            q_multi = st.checkbox("Multi-turn", value=bool(editing_q.get("multi_turn")))

            save_col, del_col = st.columns([3, 1])
            saved = save_col.form_submit_button("Save question", type="primary")
            deleted = del_col.form_submit_button("Delete", disabled=is_new_q)

        if saved:
            new_q = {
                "id": int(q_id),
                "question": q_text.strip(),
                "flow": q_flow,
                "complexity": q_complex,
                "priority": q_priority,
                "expected_tools": [t.strip() for t in q_tools.split(",") if t.strip()],
                "golden_answer": q_golden.strip() or None,
                "tags": [t.strip() for t in q_tags.split(",") if t.strip()],
                "multi_turn": q_multi,
            }
            if not new_q["question"]:
                st.error("Question text is required.")
            elif is_new_q and int(q_id) in all_ids:
                st.error(f"ID {q_id} already exists. Choose a different ID.")
            else:
                updated = [q for q in questions_live if q["id"] != new_q["id"]] + [new_q]
                updated.sort(key=lambda q: q["id"])
                _save_questions(updated)
                st.success(f"Question #{new_q['id']} saved.")
                st.rerun()

        if deleted and not is_new_q:
            updated = [q for q in questions_live if q["id"] != editing_q["id"]]
            _save_questions(updated)
            st.success(f"Question #{editing_q['id']} deleted.")
            st.rerun()

    # -- Chains -----------------------------------------------------------------
    with cat_c_tab:
        chains_live = json.loads(CHAINS_PATH.read_text(encoding="utf-8")) if CHAINS_PATH.exists() else []

        # Overview table
        chain_table = [
            {
                "ID": c["chain_id"],
                "Name": c.get("name", ""),
                "Flow": c.get("flow", ""),
                "Turns": len(c.get("turns") or []),
                "P": "yes" if c.get("priority") else "",
            }
            for c in chains_live
        ]
        st.dataframe(chain_table, width="stretch", height=220)
        st.divider()

        # Edit / Add
        chain_opts = ["-- Add new --"] + [f"#{c['chain_id']} {c.get('name','')}" for c in chains_live]
        chain_choice = st.selectbox("Edit or add chain", chain_opts, key="cat_chain_choice")

        if chain_choice == "-- Add new --":
            editing_c = {
                "chain_id": _next_id(chains_live, "chain_id"),
                "name": "",
                "flow": "fdl",
                "priority": False,
                "turns": [
                    {"question": "", "expected_tools": [], "golden_answer": None},
                    {"question": "", "expected_tools": [], "golden_answer": None},
                ],
            }
            is_new_c = True
        else:
            sel_cid = int(chain_choice.split(" ")[0][1:])
            editing_c = next((c for c in chains_live if c["chain_id"] == sel_cid), None) or {}
            is_new_c = False

        with st.form("chain_form"):
            cc1, cc2, cc3, cc4 = st.columns([1, 3, 2, 1])
            c_id = cc1.number_input("ID", value=int(editing_c.get("chain_id", 1)), disabled=not is_new_c)
            c_name = cc2.text_input("Name", value=editing_c.get("name", ""))
            c_flow = cc3.selectbox("Flow", VALID_FLOWS,
                                   index=VALID_FLOWS.index(editing_c.get("flow","fdl")) if editing_c.get("flow") in VALID_FLOWS else 0)
            c_priority = cc4.checkbox("Priority", value=bool(editing_c.get("priority")))

            st.markdown("**Turns** (one per row - at least 2 required)")
            turns = editing_c.get("turns") or []
            n_turns = st.number_input("Number of turns", min_value=2, max_value=10,
                                      value=max(len(turns), 2), step=1)
            turn_data = []
            for i in range(int(n_turns)):
                existing = turns[i] if i < len(turns) else {}
                t_cols = st.columns([5, 3])
                t_q = t_cols[0].text_input(f"Turn {i+1} question", value=existing.get("question",""),
                                            key=f"ct_q_{i}")
                t_tools = t_cols[1].text_input(f"Turn {i+1} expected tools",
                                               value=", ".join(existing.get("expected_tools") or []),
                                               key=f"ct_t_{i}")
                turn_data.append({
                    "question": t_q.strip(),
                    "expected_tools": [t.strip() for t in t_tools.split(",") if t.strip()],
                    "golden_answer": None,
                })

            cs_col, cd_col = st.columns([3, 1])
            c_saved = cs_col.form_submit_button("Save chain", type="primary")
            c_deleted = cd_col.form_submit_button("Delete", disabled=is_new_c)

        if c_saved:
            filled_turns = [t for t in turn_data if t["question"]]
            if len(filled_turns) < 2:
                st.error("At least 2 turns with questions are required.")
            elif not c_name.strip():
                st.error("Chain name is required.")
            else:
                new_c = {
                    "chain_id": int(c_id),
                    "name": c_name.strip(),
                    "flow": c_flow,
                    "priority": c_priority,
                    "turns": turn_data,
                }
                updated_c = [c for c in chains_live if c["chain_id"] != new_c["chain_id"]] + [new_c]
                updated_c.sort(key=lambda c: c["chain_id"])
                _save_chains(updated_c)
                st.success(f"Chain #{new_c['chain_id']} saved.")
                st.rerun()

        if c_deleted and not is_new_c:
            updated_c = [c for c in chains_live if c["chain_id"] != editing_c["chain_id"]]
            _save_chains(updated_c)
            st.success(f"Chain #{editing_c['chain_id']} deleted.")
            st.rerun()
