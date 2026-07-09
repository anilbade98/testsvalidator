"""Streamlit UI for the Analytics Agent Validator.

Run:
    streamlit run tests/validator/ui.py
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path so tests.validator.* imports work inside
# the Streamlit process (e.g. the History tab's "Persist to DB" button).
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import requests
import streamlit as st
import streamlit.components.v1 as _stc

HARNESS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = HARNESS_DIR / "results"
CHAINS_PATH = HARNESS_DIR / "test_chains.json"
QUESTIONS_PATH = HARNESS_DIR / "test_questions.json"
PYTHON = sys.executable
PASS_THRESHOLD = 0.7
VALID_FLOWS = ["fdl", "od", "ao", "rca", "advisory", "mixed", "context", "refresh", "conversational"]


# - Page config --------------------------------

st.set_page_config(
    page_title="Validator",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Analytics Agent Validator")


# - Load catalogs -------------------------------

_questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if QUESTIONS_PATH.exists() else []
_chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8")) if CHAINS_PATH.exists() else []
_all_items = (
    [("q", q) for q in _questions] +
    [("c", c) for c in _chains]
)


# - Helpers ----------------------------------

def _ck(kind: str, item_id: int) -> str:
    return f"{kind}_{item_id}"


def _get_selected() -> dict:
    """Return {'q': [ids], 'c': [ids]} of currently checked items."""
    qs = [q["id"] for q in _questions if st.session_state.get(_ck("q", q["id"]), False)]
    cs = [c["chain_id"] for c in _chains if st.session_state.get(_ck("c", c["chain_id"]), False)]
    return {"q": qs, "c": cs}


def _apply_preset(preset: str) -> None:
    if preset == "smoke":
        # 1 question per flow (prefer priority=true), 1 chain per flow (prefer priority=true)
        by_flow_q: dict = {}
        for q in _questions:
            by_flow_q.setdefault(q.get("flow", "other"), []).append(q)
        selected_qids = set()
        for qs in by_flow_q.values():
            p = [q for q in qs if q.get("priority")]
            selected_qids.add((p[0] if p else qs[0])["id"])
        for q in _questions:
            st.session_state[_ck("q", q["id"])] = q["id"] in selected_qids

        by_flow_c: dict = {}
        for c in _chains:
            by_flow_c.setdefault(c.get("flow", "other"), []).append(c)
        selected_cids = set()
        for cs in by_flow_c.values():
            p = [c for c in cs if c.get("priority")]
            selected_cids.add((p[0] if p else cs[0])["chain_id"])
        for c in _chains:
            st.session_state[_ck("c", c["chain_id"])] = c["chain_id"] in selected_cids

    elif preset == "priority":
        for q in _questions:
            st.session_state[_ck("q", q["id"])] = bool(q.get("priority"))
        for c in _chains:
            st.session_state[_ck("c", c["chain_id"])] = bool(c.get("priority"))

    elif preset == "all_q":
        for q in _questions:
            st.session_state[_ck("q", q["id"])] = True
        for c in _chains:
            st.session_state[_ck("c", c["chain_id"])] = True

    elif preset == "none":
        for q in _questions:
            st.session_state[_ck("q", q["id"])] = False
        for c in _chains:
            st.session_state[_ck("c", c["chain_id"])] = False


def _base_args(out_path: Path) -> list[str]:
    args = [PYTHON, "-u", "-m", "tests.validator.run",
            "--output", str(out_path),
            "--base-url", st.session_state.get("base_url", "http://localhost:80")]
    _env = st.session_state.get("env_name", "staging")
    _tunnel = st.session_state.get("service_mode", "local") == "local"
    if _tunnel:
        args += ["--env", _env, "--tunnel"]
    elif _env:
        args += ["--env", _env]
    if st.session_state.get("skip_eval"):
        args.append("--skip-eval")
    if st.session_state.get("use_persist") and not st.session_state.get("skip_eval"):
        args.append("--persist")
    _workers = int(st.session_state.get("workers", 1))
    if _workers > 1:
        args += ["--workers", str(_workers)]
    _ref_date = (st.session_state.get("reference_date") or "").strip()
    if _ref_date:
        args += ["--reference-date", _ref_date]
    return args


def _build_run_plan(ts: str, selected: dict) -> list[dict]:
    """Return a single-step plan: one subprocess handles both questions and chains."""
    q_ids = selected.get("q") or []
    c_ids = selected.get("c") or []
    if not q_ids and not c_ids:
        return []
    out = RESULTS_DIR / f"run_{ts}.json"
    args = _base_args(out)
    if q_ids:
        args += ["--ids", ",".join(str(i) for i in sorted(q_ids))]
    if c_ids:
        args += ["--chain-ids", ",".join(str(i) for i in sorted(c_ids))]
    return [{"label": "Run", "args": args, "out_path": out}]


def _run_env() -> dict:
    env_vars = os.environ.copy()
    env_vars["PYTHONIOENCODING"] = "utf-8"
    env_vars["PYTHONUTF8"] = "1"
    env_vars["PYTHONUNBUFFERED"] = "1"

    # When logged in via email+password, pass credentials so the runner can
    # re-authenticate automatically when the token expires during a long run.
    # EVAL_BEARER_TOKEN disables auto-refresh (treated as a static override),
    # so we only set it when there are no re-auth credentials available.
    email = st.session_state.get("_login_email", "")
    password = st.session_state.get("_login_password", "")
    login_env = st.session_state.get("_login_env", "")

    if email and password and login_env:
        # Pass credentials so the runner can call the auth API again on 401
        env_vars["EVAL_AUTH_EMAIL"] = email
        env_vars["EVAL_AUTH_PASSWORD"] = password
        env_vars["EVAL_AUTH_ENV"] = login_env
        env_vars["EVAL_TENANT_ID"] = st.session_state.get("_resolved_tenant", "")
        # Also set the initial token so the first requests don't need to re-login
        initial_token = st.session_state.get("_resolved_token", "")
        if initial_token:
            env_vars["EVAL_BEARER_TOKEN"] = initial_token
    else:
        # Bearer-token-only mode: pass the token (auto-refresh won't work if it expires)
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

    # All scored metrics must be above threshold for an overall PASS.
    failing = []
    for key, label in [
        ("agent_response", "agent response"),
        ("interpretation_coverage", "interpretation"),
        ("sql_correctness", "SQL correctness"),
    ]:
        m = agg.get(key) or {}
        if m.get("n", 0) > 0 and (m.get("pass_rate") or 0) < PASS_THRESHOLD:
            failing.append(f"{label} {(m['pass_rate'] or 0)*100:.0f}%")

    if failing:
        return "error", f"Quality issues — {', '.join(failing)} below {PASS_THRESHOLD*100:.0f}% threshold."
    if pass_rate >= 0.8:
        return "success", f"Quality looks good — {pass_rate*100:.0f}% passed{mean_str}"
    elif pass_rate >= 0.6:
        return "warning", f"Some quality issues — {pass_rate*100:.0f}% passed{mean_str}. Review failing questions."
    else:
        return "error", f"Low quality — only {pass_rate*100:.0f}% passed{mean_str}."


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


_HTML_INLINE_LIMIT = 400_000  # bytes — components.html WebSocket payload cap


def _html_report_section(html_path: Path, height: int = 900, key_suffix: str = "") -> None:
    """Render an HTML report file inline, with a download button always shown.

    Uses streamlit.components.v1.html() for inline preview.  Reports larger than
    _HTML_INLINE_LIMIT bytes are not embedded — they are too big for a single
    WebSocket payload and cause 'websocket.send after websocket.close' errors.
    Use the download button to open them in a browser instead.
    """
    html_content = html_path.read_text(encoding="utf-8")
    size_kb = len(html_content) // 1024
    st.download_button(
        f"⬇ Download HTML report ({size_kb} KB)",
        html_content,
        file_name=html_path.name,
        mime="text/html",
        key=f"dl_{html_path.name}_{key_suffix}",
    )
    if len(html_content) <= _HTML_INLINE_LIMIT:
        _stc.html(html_content, height=height, scrolling=True)
    else:
        st.info(
            f"Report is {size_kb} KB — too large to preview inline. "
            "Use the download button above to open it in your browser."
        )


# - Session state init ----------------------------

for _k, _v in [
    ("proc", None), ("run_output", []), ("last_run_path", None),
    ("out_queue", None), ("reader_done", False), ("pending_args", []),
    ("prog_phase", ""), ("prog_n", 0), ("prog_total", 0),
    ("service_mode", "local"), ("env_name", "staging"),
    ("base_url", "http://localhost:80"),
    ("skip_eval", False), ("use_persist", False), ("workers", 1), ("reference_date", ""),
    ("_resolved_token", ""), ("_resolved_tenant", ""),
    ("_login_email", ""), ("_login_password", ""), ("_login_env", ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "picker_initialized" not in st.session_state:
    st.session_state.picker_initialized = True
    _apply_preset("smoke")


# - Sidebar ----------------------------------

with st.sidebar:
    # -- Auth + Environment (together) ----------------
    st.header("Authentication")

    env_names = ["staging", "dev", "production"]
    st.session_state.env_name = st.selectbox(
        "Environment",
        env_names,
        index=env_names.index(st.session_state.env_name) if st.session_state.env_name in env_names else 0,
        help="Used for auth, tunnel targets, and DynamoDB table names.",
    )

    auth_mode = st.radio("Method", ["Bearer token", "Email + Password"], horizontal=True)

    if auth_mode == "Bearer token":
        bearer = st.text_input("Bearer Token", type="password")
        tenant_override = st.text_input("Tenant ID", help="Optional - decoded from JWT if omitted.")
        if st.button("Apply token"):
            st.session_state._resolved_token = bearer.strip()
            st.session_state._resolved_tenant = tenant_override.strip()
            st.session_state._login_email = ""
            st.session_state._login_password = ""
            st.success("Token applied.")
    else:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Log in", type="primary"):
            login_env = st.session_state.env_name
            with st.spinner("Authenticating..."):
                try:
                    result = _auth_login(email, password, login_env)
                    st.session_state._resolved_token = result["token"]
                    st.session_state._resolved_tenant = result.get("tenant_id", "")
                    st.session_state._login_email = email
                    st.session_state._login_password = password
                    st.session_state._login_env = login_env
                    st.success(f"Logged in. tenant={result.get('tenant_id','')[:8]}...")
                except Exception as exc:
                    st.error(f"Login failed: {exc}")

    if st.session_state._resolved_token:
        st.caption("Token: active")

    st.divider()

    # -- Service location ------------------------------
    st.header("Service")

    st.session_state.service_mode = st.radio(
        "Location",
        ["local", "deployed"],
        format_func=lambda m: "Local (python main.py)" if m == "local" else "Deployed (via ALB tunnel)",
        horizontal=True,
    )

    if st.session_state.service_mode == "local":
        st.session_state.base_url = st.text_input(
            "URL", value=st.session_state.base_url,
            help="URL of the locally running service.",
        )
        use_tunnel = True
        st.caption("SSM tunnels: auto-enabled for FDL/OD/AO/RCA")
    else:
        _default_deployed = "http://localhost:8080"
        if "localhost" not in st.session_state.base_url or st.session_state.base_url == "http://localhost:80":
            st.session_state.base_url = _default_deployed
        st.session_state.base_url = st.text_input(
            "URL", value=st.session_state.base_url,
            help="localhost:8080 when ALB tunnel is open. Change port if needed.",
        )
        use_tunnel = False
        st.caption(f"Open ALB tunnel first: `/tunnel {st.session_state.env_name}` in a terminal")

    st.divider()
    st.header("Run options")
    st.session_state.skip_eval = st.checkbox("Skip eval",
                                             value=st.session_state.skip_eval,
                                             help="Collect traces only, no LLM scoring.")
    st.session_state.use_persist = st.checkbox(
        "Persist to DB", value=st.session_state.use_persist,
        help="Write results to DynamoDB after scoring.",
        disabled=st.session_state.skip_eval,
    )
    st.session_state.workers = st.slider(
        "Workers", min_value=1, max_value=10,
        value=st.session_state.workers,
        help="Parallel workers for both collection and scoring. 1 = sequential. 3-5 is a good balance.",
    )
    st.session_state.reference_date = st.text_input(
        "Reference date",
        value=st.session_state.reference_date,
        placeholder="YYYY-MM-DD  (leave empty for today)",
        help="Pin 'today' to a fixed date so the agent generates date filters that match the static test dataset.",
    )

    st.divider()

    # -- Test selection ---------------------------
    st.header("Test selection")

    sel = _get_selected()
    n_q = len(sel["q"])
    n_c = len(sel["c"])
    n_total_sel = n_q + n_c
    st.caption(f"Selected: {n_total_sel} ({n_q} questions, {n_c} chains)")

    p_cols = st.columns(4)
    if p_cols[0].button("Smoke", use_container_width=True):
        _apply_preset("smoke"); st.rerun()
    if p_cols[1].button("Priority", use_container_width=True):
        _apply_preset("priority"); st.rerun()
    if p_cols[2].button("All", use_container_width=True):
        _apply_preset("all_q"); st.rerun()
    if p_cols[3].button("Clear", use_container_width=True):
        _apply_preset("none"); st.rerun()

    # Questions grouped by flow
    by_flow: dict = {}
    for q in _questions:
        by_flow.setdefault(q.get("flow", "other"), []).append(q)

    n_sel_c = sum(1 for c in _chains if st.session_state.get(_ck("c", c["chain_id"]), False))
    header = f"Tests ({n_q}/{len(_questions)} questions, {n_sel_c}/{len(_chains)} chains selected)"
    with st.expander(header, expanded=True):
        for flow_name in sorted(by_flow.keys()):
            qs_in_flow = by_flow[flow_name]
            n_sel_flow = sum(1 for q in qs_in_flow if st.session_state.get(_ck("q", q["id"]), False))
            st.markdown(f"**{flow_name}** ({n_sel_flow}/{len(qs_in_flow)})")
            for q in qs_in_flow:
                key = _ck("q", q["id"])
                priority_tag = " [P]" if q.get("priority") else ""
                label = f"#{q['id']} {q.get('question','')[:45]}...  `{q.get('complexity','')}`{priority_tag}"
                st.checkbox(label, key=key)

        if _chains:
            st.markdown(f"**chains** ({n_sel_c}/{len(_chains)})")
            for c in _chains:
                key = _ck("c", c["chain_id"])
                turns = len(c.get("turns") or [])
                priority_tag = " [P]" if c.get("priority") else ""
                label = f"#{c['chain_id']} {c.get('name','')}  `{c.get('flow','')}` ({turns} turns){priority_tag}"
                st.checkbox(label, key=key)


# - Re-compute selected after sidebar interactions --------------

_selected = _get_selected()
_n_selected = len(_selected["q"]) + len(_selected["c"])


# - Tabs -----------------------------------

tab_run, tab_history, tab_catalog = st.tabs(["Run", "History", "Catalog"])


# - Tab: Run ---------------------------------

with tab_run:
    # Controls
    if not st.session_state._resolved_token:
        st.warning("No token - log in or paste a bearer token in the sidebar.")

    preview_plan = _build_run_plan("preview", _selected)
    with st.expander("CLI command"):
        if preview_plan:
            for step in preview_plan:
                st.caption(step["label"])
                st.code(" ".join(step["args"]), language="bash")
        else:
            st.caption("No tests selected.")

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
            st.session_state.pending_args = []
            st.session_state.prog_phase = ""
            st.session_state.prog_n = 0
            st.session_state.prog_total = 0
            st.rerun()

    # Start process (first step; remaining steps stored in pending_args)
    if start:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        plan = _build_run_plan(ts, _selected)
        if not plan:
            st.warning("Nothing to run.")
        else:
            first = plan[0]
            st.session_state.pending_args = plan[1:]
            st.session_state.last_run_path = first["out_path"]
            st.session_state.run_output = [
                f"$ {' '.join(first['args'])}",
                "",
            ]
            st.session_state.reader_done = False
            st.session_state.prog_phase = ""
            st.session_state.prog_n = 0
            st.session_state.prog_total = 0
            proc = _start_proc(first["args"])
            st.session_state.proc = proc
            q: queue.Queue = queue.Queue()
            st.session_state.out_queue = q
            threading.Thread(target=_reader_thread, args=(proc, q), daemon=True).start()
            st.rerun()

    # Stop process (also clears pending steps)
    if stop and st.session_state.proc:
        st.session_state.proc.terminate()
        st.session_state.proc = None
        st.session_state.pending_args = []
        st.session_state.run_output.append("-- stopped --")
        st.rerun()

    _COLLECT_RE = re.compile(r"^\[(\d+)/(\d+)\]")
    _EVAL_RE = re.compile(r"^\[eval (\d+)/(\d+)\]")

    prog_bar = st.empty()
    output_area = st.empty()

    def _parse_progress(line: str) -> None:
        m = _EVAL_RE.match(line)
        if m:
            st.session_state.prog_phase = "Scoring"
            st.session_state.prog_n = int(m.group(1))
            st.session_state.prog_total = int(m.group(2))
            return
        m = _COLLECT_RE.match(line)
        if m:
            st.session_state.prog_phase = "Collecting"
            st.session_state.prog_n = int(m.group(1))
            st.session_state.prog_total = int(m.group(2))

    def _refresh_prog() -> None:
        n, total, phase = st.session_state.prog_n, st.session_state.prog_total, st.session_state.prog_phase
        if total > 0:
            pct = min(n / total, 1.0)
            prog_bar.progress(pct, text=f"{phase}: {n} / {total}")
        else:
            prog_bar.empty()

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
                        # Sentinel: process has exited — start next step if any
                        st.session_state.proc = None
                        pending = st.session_state.get("pending_args", [])
                        if pending:
                            nxt = pending[0]
                            st.session_state.pending_args = pending[1:]
                            st.session_state.last_run_path = nxt["out_path"]
                            st.session_state.run_output += [
                                "",
                                f"=== {nxt['label']} ===",
                                f"$ {' '.join(nxt['args'])}",
                                "",
                            ]
                            next_proc = _start_proc(nxt["args"])
                            st.session_state.proc = next_proc
                            nxt_q: queue.Queue = queue.Queue()
                            st.session_state.out_queue = nxt_q
                            threading.Thread(
                                target=_reader_thread, args=(next_proc, nxt_q), daemon=True,
                            ).start()
                            q_ref = nxt_q
                        else:
                            st.session_state.reader_done = True
                        break
                    _parse_progress(line)
                    st.session_state.run_output.append(line)
                except queue.Empty:
                    break

            # Push current output and progress to browser
            _refresh_prog()
            output_area.code(
                "\n".join(st.session_state.run_output),
                language="text",
            )

            if st.session_state.proc is None and st.session_state.reader_done:
                break  # all steps finished, exit loop

            if st.session_state.proc is None and st.session_state.pending_args:
                # Next process was just queued but not yet started (race guard)
                time.sleep(0.05)
                continue

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
            _html_report_section(html_path, height=950, key_suffix="run")
        else:
            st.info("No HTML report yet (run without -skip-eval to generate one).")

    elif st.session_state.run_output:
        # Not yet started streaming (between start click and first loop iteration)
        output_area.code("\n".join(st.session_state.run_output), language="text")


# - Tab: History (combined overview + details) ----------------

with tab_history:
    all_files = sorted(
        list(RESULTS_DIR.glob("run_*.json")) + list(RESULTS_DIR.glob("chains_*.json")),
        key=lambda p: p.name, reverse=True,
    )

    if not all_files:
        st.info("No runs found. Start an evaluation from the Run tab.")
    else:
        # ?? Summary table ??????????????????????????????????????????????????????
        history_rows = []
        for f in all_files[:50]:
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
            st.dataframe(history_rows, width="stretch", height=220)

        # Trend chart
        trend_data = [(r["Timestamp"], float(r["Agent mean"])) for r in history_rows if r["Agent mean"] not in ("-", "")]
        if len(trend_data) >= 2:
            import pandas as pd
            df = pd.DataFrame(trend_data[::-1], columns=["Run", "Score"])
            df = df.set_index("Run")
            st.line_chart(df)
            st.caption(f"Pass threshold: {PASS_THRESHOLD}")

        st.divider()

        # ?? Run details (combined Results) ?????????????????????????????????????
        selected_file = st.selectbox(
            "Select run to inspect",
            all_files,
            format_func=lambda p: p.name,
            index=0,
            key="history_selected",
        )

        if selected_file and selected_file.exists():
            run = json.loads(selected_file.read_text(encoding="utf-8"))
            agg = run.get("aggregated_scores") or {}
            ar = agg.get("agent_response") or {}

            # Status banner + action buttons
            banner_col, btn1, btn2 = st.columns([4, 1, 1])
            with banner_col:
                status_type, status_msg = _push_status(selected_file)
                if status_type == "success":
                    st.success(status_msg)
                elif status_type == "warning":
                    st.warning(status_msg)
                else:
                    st.error(status_msg)

            if btn1.button("Re-run eval", width="stretch",
                           disabled=bool(st.session_state.proc),
                           help="Re-score with the LLM judge (skips already-scored records)."):
                args = [PYTHON, "-u", "-m", "tests.validator.run",
                        "--eval-only", str(selected_file), "--skip-scored"]
                st.session_state.last_run_path = selected_file
                st.session_state.run_output = [f"$ {' '.join(args)}", ""]
                st.session_state.reader_done = False
                proc = _start_proc(args)
                st.session_state.proc = proc
                q_ref: queue.Queue = queue.Queue()
                st.session_state.out_queue = q_ref
                threading.Thread(target=_reader_thread, args=(proc, q_ref), daemon=True).start()
                st.rerun()

            if btn2.button("Persist to DB", width="stretch",
                           disabled=bool(st.session_state.proc),
                           help="Write scored results to DynamoDB (requires scores to exist)."):
                try:
                    from tests.validator.persistence import persist_run_file
                    persist_run_file(selected_file)
                    st.success(f"Persisted {selected_file.name} to DynamoDB.")
                except Exception as exc:
                    st.error(f"Persist failed: {exc}")

            st.divider()

            # Key metrics
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

            # By flow
            by_flow_agg = agg.get("by_flow") or {}
            if by_flow_agg:
                st.divider()
                flow_rows = [
                    {
                        "Flow": flow,
                        "n": str(d.get("total", 0)),
                        "Mean": f"{d['agent_response_mean']:.3f}" if d.get("agent_response_mean") is not None else "-",
                        "Pass %": f"{d['agent_response_pass_rate']*100:.0f}%" if d.get("agent_response_pass_rate") is not None else "-",
                    }
                    for flow, d in sorted(by_flow_agg.items())
                ]
                st.dataframe(flow_rows, width="stretch")

            # Per-question table
            results = run.get("results") or []
            if results:
                st.divider()
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
                        "SQL results": str(len(r.get("od_fdl_results") or [])),
                        "Error": "!" if r.get("error") else "",
                    })
                st.dataframe(table_rows, width="stretch", height=350)

            # HTML report
            html_path = selected_file.with_suffix(".html")
            if html_path.exists():
                st.divider()
                with st.expander("Full HTML report", expanded=True):
                    _html_report_section(html_path, height=900, key_suffix="history")


# - Tab: Catalog -------------------------------

def _save_questions(questions: list[dict]) -> None:
    QUESTIONS_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")

def _save_chains(chains: list[dict]) -> None:
    CHAINS_PATH.write_text(json.dumps(chains, indent=2, ensure_ascii=False), encoding="utf-8")

def _next_id(items: list[dict], id_key: str) -> int:
    return max((item.get(id_key, 0) for item in items), default=0) + 1


with tab_catalog:
    cat_q_tab, cat_c_tab = st.tabs(["Questions", "Chains"])

    # - Questions -------------------------------
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
        edit_options = ["- Add new -"] + [f"#{q['id']} {q.get('question','')[:50]}" for q in filtered]
        edit_choice = st.selectbox("Edit or add", edit_options, key="cat_edit_choice")

        if edit_choice == "- Add new -":
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

    # - Chains ---------------------------------
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
        chain_opts = ["- Add new -"] + [f"#{c['chain_id']} {c.get('name','')}" for c in chains_live]
        chain_choice = st.selectbox("Edit or add chain", chain_opts, key="cat_chain_choice")

        if chain_choice == "- Add new -":
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
