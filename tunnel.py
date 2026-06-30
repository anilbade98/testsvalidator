"""SSM port-forwarding tunnels for local validator runs.

When running the analytics-agentic service locally (python main.py), it
needs to reach FDL, OD, AO, and RCA via their backends.
This module manages those tunnels automatically, started on demand just
before the first question that needs each service.

Tunnel topology (matches appsettings.Local.json):

  fdl: localhost:9007 -> fdl-svc-alb.{env}-illum.internal:9007
       via: {env}-illum-jump-server (internal ALB, HTTP)

  od:  localhost:9009 -> od-analytics-dq-alb.{env}-illum.internal:80
       via: {env}-illum-jump-server (internal ALB, HTTP)

  ao:  localhost:8081 -> {ecs_task_private_ip}:80
       via: any online SSM instance (direct-to-task, bypasses the HTTPS ALB)
       The AO ALB is HTTPS-only, but the ECS container itself speaks plain HTTP
       on port 80. Task IP is looked up fresh each time (changes on every deploy).
       FastMCP sessions are pinned per-replica, so direct-to-task is required.

  rca: localhost:9010 -> eai-anomaly-detection.illuminate.service.nicecxone-staging.com:80
       via: {env}-illum-jump-server (public hostname, HTTP port 80 confirmed working)

Usage from run.py:
    pool = TunnelPool.for_env("staging")
    pool.ensure_for(["get_fdl_data"])           # starts fdl tunnel
    pool.ensure_for(["ask_automation_opportunity"])  # starts ao tunnel
    pool.ensure_for(["query_anomalies"])        # starts rca tunnel
    pool.stop_all()
"""

from __future__ import annotations

import atexit
import json
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field


# ── Tool -> service mapping ─────────────────────────────────────────────────────

TOOL_SERVICE = {
    "get_fdl_data":               "fdl",
    "get_od_data":                "od",
    "get_athena_data":            None,   # boto3 direct, no tunnel needed
    "ask_automation_opportunity": "ao",   # direct-to-task on port 80 (container HTTP)
    "query_anomalies":            "rca",
    "get_current_data":           None,   # local tool, no tunnel needed
}

# Local ports match appsettings.Local.json:
#   fdl.service_url = http://localhost:9007
#   od.service_url  = http://localhost:9009
#   ao.mcp_url      = http://localhost:8081/mcp/
#   rca.service_url = http://localhost:9010
_DEFAULT_PORTS = {"fdl": 9007, "od": 9009, "ao": 8081, "rca": 9010}

# Remote ports per service
# ao: 80 because we bypass the HTTPS ALB and connect directly to the ECS container
_REMOTE_PORTS = {"fdl": 9007, "od": 80, "ao": 80, "rca": 80}

# AO ECS cluster and service names
_AO_CLUSTER = "{env}-illum-autoinsights-analytics-fargate"
_AO_SERVICE = "ao-agent-svc"

# RCA hostnames by environment (public hostname resolves inside VPC via Route53)
_RCA_HOSTS = {
    "staging": "eai-anomaly-detection.illuminate.service.nicecxone-staging.com",
    # TODO: verify for other envs
}

_READY_TIMEOUT = 60
_READY_POLL = 0.5


# ── Single-service tunnel ──────────────────────────────────────────────────────

@dataclass
class ServiceTunnel:
    name: str
    remote_host: str        # hostname or IP to forward to
    remote_port: int
    local_port: int
    relay_instance_id: str  # EC2 instance used as SSM relay
    aws_profile: str
    _proc: subprocess.Popen | None = field(default=None, repr=False, init=False)

    def _port_open(self) -> bool:
        try:
            with socket.create_connection(("localhost", self.local_port), timeout=1):
                return True
        except OSError:
            return False

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and self._port_open()

    def start(self) -> None:
        if self.is_alive():
            return

        parameters = json.dumps({
            "portNumber": [str(self.remote_port)],
            "localPortNumber": [str(self.local_port)],
            "host": [self.remote_host],
        })
        cmd = [
            "aws", "ssm", "start-session",
            "--target", self.relay_instance_id,
            "--document-name", "AWS-StartPortForwardingSessionToRemoteHost",
            "--parameters", parameters,
            "--profile", self.aws_profile,
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.time() + _READY_TIMEOUT
        while time.time() < deadline:
            if self._port_open():
                return
            time.sleep(_READY_POLL)

        self.stop()
        raise TimeoutError(
            f"[{self.name}] tunnel did not open on localhost:{self.local_port} "
            f"within {_READY_TIMEOUT}s. "
            f"(relay={self.relay_instance_id} -> {self.remote_host}:{self.remote_port})"
        )

    def ensure(self) -> None:
        if not self.is_alive():
            if self._proc is not None:
                print(f"  [{self.name}] tunnel timed out — restarting ...")
                self.stop()
            self.start()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


# ── Instance / endpoint discovery ─────────────────────────────────────────────

def _find_jump_server(env: str, profile: str) -> str:
    """Return the first SSM-connected jump server for this environment."""
    tag = f"{env}-illum-jump-server"
    cmd = [
        "aws", "ec2", "describe-instances",
        "--profile", profile,
        "--filters",
        f"Name=tag:Name,Values={tag}",
        "Name=instance-state-name,Values=running",
        "--query", "Reservations[*].Instances[*].InstanceId",
        "--output", "text",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list jump servers: {result.stderr.strip()}")
    instance_ids = [i for i in result.stdout.strip().split() if i and i != "None"]
    if not instance_ids:
        raise RuntimeError(f"No running jump server found with tag Name={tag!r}.")

    for iid in instance_ids:
        probe = subprocess.run(
            [
                "aws", "ssm", "describe-instance-information",
                "--filters", f"Key=InstanceIds,Values={iid}",
                "--query", "InstanceInformationList[?PingStatus==`Online`].InstanceId | [0]",
                "--output", "text",
                "--profile", profile,
            ],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode == 0 and probe.stdout.strip() == iid:
            return iid

    raise RuntimeError(
        f"None of the {len(instance_ids)} jump server(s) for {tag!r} are SSM-connected: "
        + ", ".join(instance_ids)
    )


def _find_any_ssm_relay(profile: str) -> str:
    """Return any online SSM-managed instance (used as AO relay)."""
    result = subprocess.run(
        [
            "aws", "ssm", "describe-instance-information",
            "--query", "InstanceInformationList[?PingStatus==`Online`].InstanceId",
            "--output", "text",
            "--profile", profile,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list SSM instances: {result.stderr.strip()}")
    ids = [i for i in result.stdout.strip().split() if i and i != "None"]
    if not ids:
        raise RuntimeError("No online SSM-managed instances found for AO relay.")
    return ids[0]


def _find_ao_task_ip(env: str, profile: str) -> str:
    """Return the private IP of the running AO MCP ECS task.

    AO bypasses the ALB because FastMCP sessions are pinned per-replica.
    Task IPs are ephemeral — re-fetched on every tunnel creation.
    """
    cluster = _AO_CLUSTER.format(env=env)
    task_result = subprocess.run(
        [
            "aws", "ecs", "list-tasks",
            "--cluster", cluster,
            "--service-name", _AO_SERVICE,
            "--query", "taskArns[0]",
            "--output", "text",
            "--profile", profile,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if task_result.returncode != 0:
        raise RuntimeError(f"Failed to list AO tasks: {task_result.stderr.strip()}")
    task_arn = task_result.stdout.strip()
    if not task_arn or task_arn == "None":
        raise RuntimeError(
            f"No running AO task found in cluster {cluster!r}, service {_AO_SERVICE!r}."
        )
    ip_result = subprocess.run(
        [
            "aws", "ecs", "describe-tasks",
            "--cluster", cluster,
            "--tasks", task_arn,
            "--query", "tasks[0].attachments[0].details[?name==`privateIPv4Address`].value",
            "--output", "text",
            "--profile", profile,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if ip_result.returncode != 0:
        raise RuntimeError(f"Failed to describe AO task: {ip_result.stderr.strip()}")
    ip = ip_result.stdout.strip()
    if not ip or ip == "None":
        raise RuntimeError(f"Could not resolve private IP for AO task {task_arn}.")
    return ip


# ── Multi-tunnel pool ─────────────────────────────────────────────────────────

class TunnelPool:
    """Manages per-service SSM tunnels for a single environment."""

    def __init__(
        self,
        env: str,
        fdl_port: int = _DEFAULT_PORTS["fdl"],
        od_port: int = _DEFAULT_PORTS["od"],
        ao_port: int = _DEFAULT_PORTS["ao"],
        rca_port: int = _DEFAULT_PORTS["rca"],
    ):
        self.env = env
        self._profile = f"{env}-illum"
        self._ports = {"fdl": fdl_port, "od": od_port, "ao": ao_port, "rca": rca_port}
        self._tunnels: dict[str, ServiceTunnel] = {}
        self._jump_id: str | None = None

    def _get_jump_id(self) -> str:
        if not self._jump_id:
            print(f"  Looking up jump server for env={self.env!r} ...")
            self._jump_id = _find_jump_server(self.env, self._profile)
            print(f"  Jump server: {self._jump_id}")
        return self._jump_id

    def _make_tunnel(self, service: str) -> ServiceTunnel:
        if service == "ao":
            print("  [ao] resolving ECS task IP ...")
            task_ip = _find_ao_task_ip(self.env, self._profile)
            relay = _find_any_ssm_relay(self._profile)
            print(f"  [ao] task IP={task_ip} relay={relay}")
            return ServiceTunnel(
                name="ao",
                remote_host=task_ip,
                remote_port=_REMOTE_PORTS["ao"],
                local_port=self._ports["ao"],
                relay_instance_id=relay,
                aws_profile=self._profile,
            )
        if service == "rca":
            host = _RCA_HOSTS.get(self.env)
            if not host:
                raise RuntimeError(
                    f"RCA hostname not configured for env={self.env!r}. "
                    f"Add it to _RCA_HOSTS in tunnel.py."
                )
        else:
            host = {
                "fdl": f"fdl-svc-alb.{self.env}-illum.internal",
                "od":  f"od-analytics-dq-alb.{self.env}-illum.internal",
            }[service]
        return ServiceTunnel(
            name=service,
            remote_host=host,
            remote_port=_REMOTE_PORTS[service],
            local_port=self._ports[service],
            relay_instance_id=self._get_jump_id(),
            aws_profile=self._profile,
        )

    def ensure_for(self, tools: list[str]) -> None:
        services = {
            TOOL_SERVICE[t]
            for t in tools
            if t in TOOL_SERVICE and TOOL_SERVICE[t] is not None
        }
        for service in sorted(services):
            if service not in self._tunnels:
                self._tunnels[service] = self._make_tunnel(service)
            tunnel = self._tunnels[service]
            if not tunnel.is_alive():
                action = "restarting" if tunnel._proc is not None else "starting"
                print(f"  [{service}] {action} SSM tunnel "
                      f"localhost:{tunnel.local_port} -> {tunnel.remote_host}:{tunnel.remote_port}")
                tunnel.start()
                print(f"  [{service}] ready")

    def stop_all(self) -> None:
        for t in self._tunnels.values():
            t.stop()
        if self._tunnels:
            print("  All SSM tunnels closed.")
        self._tunnels.clear()

    def __enter__(self) -> "TunnelPool":
        return self

    def __exit__(self, *_) -> None:
        self.stop_all()

    @classmethod
    def for_env(
        cls,
        env: str,
        fdl_port: int = _DEFAULT_PORTS["fdl"],
        od_port: int = _DEFAULT_PORTS["od"],
        ao_port: int = _DEFAULT_PORTS["ao"],
        rca_port: int = _DEFAULT_PORTS["rca"],
    ) -> "TunnelPool":
        pool = cls(env=env, fdl_port=fdl_port, od_port=od_port, ao_port=ao_port, rca_port=rca_port)
        atexit.register(pool.stop_all)
        return pool


def install_sigint_handler(pool: TunnelPool) -> None:
    original = signal.getsignal(signal.SIGINT)

    def _handler(sig, frame):
        pool.stop_all()
        if callable(original):
            original(sig, frame)
        else:
            sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
