import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[2] / "src" / "middleware" / "skills_sandbox_recovery.py"
    spec = importlib.util.spec_from_file_location("skills_sandbox_recovery_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_async_skills_middleware_recovers_missing_sandbox_once(monkeypatch):
    module = _load_module()

    class SkillsMiddleware:
        def __init__(self):
            self.calls = 0
            self.recovered = False

        async def abefore_agent(self, state, runtime):
            self.calls += 1
            if not self.recovered:
                raise RuntimeError("Sandbox not found")
            return {"skills": ["ok"]}

        async def areattach(self, state, runtime):
            self.recovered = True

    fake_middleware_module = types.SimpleNamespace(SkillsMiddleware=SkillsMiddleware)
    monkeypatch.setitem(sys.modules, "langchain", types.ModuleType("langchain"))
    monkeypatch.setitem(sys.modules, "langchain.agents", types.ModuleType("langchain.agents"))
    monkeypatch.setitem(sys.modules, "langchain.agents.middleware", fake_middleware_module)

    module.patch_skills_middleware_missing_sandbox()

    middleware = SkillsMiddleware()
    result = asyncio.run(middleware.abefore_agent({}, None))

    assert result == {"skills": ["ok"]}
    assert middleware.calls == 2


def test_async_skills_middleware_fails_open_after_retry(monkeypatch):
    module = _load_module()

    class NotFoundError(Exception):
        status_code = 404

    class SkillsMiddleware:
        def __init__(self):
            self.calls = 0

        async def abefore_agent(self, state, runtime):
            self.calls += 1
            raise NotFoundError("Sandbox not found")

    fake_middleware_module = types.SimpleNamespace(SkillsMiddleware=SkillsMiddleware)
    monkeypatch.setitem(sys.modules, "langchain", types.ModuleType("langchain"))
    monkeypatch.setitem(sys.modules, "langchain.agents", types.ModuleType("langchain.agents"))
    monkeypatch.setitem(sys.modules, "langchain.agents.middleware", fake_middleware_module)

    module.patch_skills_middleware_missing_sandbox()

    middleware = SkillsMiddleware()
    result = asyncio.run(middleware.abefore_agent({}, None))

    assert result is None
    assert middleware.calls == 2
