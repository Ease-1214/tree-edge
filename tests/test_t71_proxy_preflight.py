"""Regression for T71 Step-5 reflector initialization failure (missing SOCKS support)."""
from pathlib import Path

from agent_system.memory.experience_updater import ExperienceUpdater


def test_launcher_removes_inherited_socks_before_reflector_preflight():
    root = Path(__file__).resolve().parents[1]
    script = (root / "examples/grpo_trainer/run_t71.sh").read_text()
    assert "unset ALL_PROXY all_proxy" in script
    assert "T71_REFLECTOR_CLIENT_PREFLIGHT_OK" in script
    assert script.index("unset ALL_PROXY all_proxy") < script.index("ExperienceUpdater(max_new_experiences_per_update=5)")
    assert script.index("ExperienceUpdater(max_new_experiences_per_update=5)") < script.index("exec bash examples/grpo_trainer/run_t062_perf.sh")


def test_client_can_init_under_real_http_proxy_without_socksio(monkeypatch):
    # The production machine inherits both an HTTP proxy and a global SOCKS
    # proxy. T71 only clears the latter; this is the exact failing path.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:17892")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:17892")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:17892")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:17892")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)

    updater = ExperienceUpdater(
        api_key="unit-test-only",
        base_url="http://127.0.0.1:29663/v1",
        model="unit-test-model",
        max_new_experiences_per_update=5,
    )
    assert updater.client is not None
    updater.client.close()
