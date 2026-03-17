"""
agent.py
Rule-based DevOps Agent.
Given an anomalous metric row + list of flagged metrics,
it suggests concrete remediation actions that a human (or automation)
could execute to resolve the issue.
"""

import pandas as pd


# ── thresholds ────────────────────────────────────────────────────────────────
CPU_CRITICAL     = 85   # %
RAM_CRITICAL     = 90   # %
LATENCY_CRITICAL = 300  # ms
DISK_CRITICAL    = 90   # %
REQRATE_CRITICAL = 900  # req/s


class DevOpsAgent:
    """
    A lightweight rule-based agent that maps infrastructure signals
    to human-readable remediation recommendations.

    In a production system this layer would call Kubernetes / AWS / GCP APIs,
    fire PagerDuty alerts, or trigger Ansible playbooks.
    """

    def recommend(self, row: pd.Series, flags: list[str]) -> list[str]:
        """
        Analyse a metric row and return a prioritised action list.

        Parameters
        ----------
        row   : one row of infrastructure metrics
        flags : list of metric names already flagged as anomalous

        Returns
        -------
        List of recommendation strings.
        """
        actions = []

        # ── CPU spike ────────────────────────────────────────────────────────
        if "CPU" in flags or row["cpu_usage"] > CPU_CRITICAL:
            actions += [
                "🔺 [CPU SPIKE] Scale out: add 2 additional compute instances.",
                "🔺 [CPU SPIKE] Check for runaway processes: `kubectl top pods --all-namespaces`.",
                "🔺 [CPU SPIKE] Consider enabling Horizontal Pod Autoscaler (HPA).",
            ]

        # ── Memory leak / RAM pressure ────────────────────────────────────────
        if "RAM" in flags or row["ram_usage"] > RAM_CRITICAL:
            actions += [
                "💾 [RAM LEAK] Restart affected containers: `kubectl rollout restart deployment/<name>`.",
                "💾 [RAM LEAK] Inspect heap dumps for the top memory-consuming service.",
                "💾 [RAM LEAK] Set memory limits in pod spec and enable OOMKill eviction.",
            ]

        # ── Network latency burst ─────────────────────────────────────────────
        if "Latency" in flags or row["network_latency"] > LATENCY_CRITICAL:
            actions += [
                "🌐 [LATENCY BURST] Redistribute traffic: update load-balancer weights.",
                "🌐 [LATENCY BURST] Enable circuit-breaker pattern (e.g. Istio / Envoy).",
                "🌐 [LATENCY BURST] Check DNS resolution time and CDN cache-hit ratio.",
            ]

        # ── Disk I/O saturation ───────────────────────────────────────────────
        if "DiskIO" in flags or row["disk_io"] > DISK_CRITICAL:
            actions += [
                "💿 [DISK I/O] Move hot data to in-memory cache (Redis / Memcached).",
                "💿 [DISK I/O] Throttle batch jobs competing for disk bandwidth.",
            ]

        # ── Request rate spike (possible DDoS / flash crowd) ─────────────────
        if "RequestRate" in flags or row["request_rate"] > REQRATE_CRITICAL:
            actions += [
                "📈 [HIGH REQ RATE] Enable rate-limiting at the API gateway.",
                "📈 [HIGH REQ RATE] Activate auto-scaling group warm-up policy.",
                "📈 [HIGH REQ RATE] Verify WAF rules to filter malicious traffic.",
            ]

        # ── Composite / unknown anomaly ───────────────────────────────────────
        if not actions:
            actions = [
                "⚠️  [COMPOSITE] Multiple metrics deviated simultaneously.",
                "⚠️  Trigger full incident response runbook.",
                "⚠️  Notify on-call engineer via PagerDuty.",
            ]

        return actions


if __name__ == "__main__":
    # Quick smoke-test
    import pandas as pd
    agent = DevOpsAgent()

    sample = pd.Series({
        "cpu_usage": 92.0, "ram_usage": 88.0,
        "disk_io": 45.0, "network_latency": 95.0, "request_rate": 250.0,
    })
    for a in agent.recommend(sample, ["CPU", "RAM"]):
        print(a)
