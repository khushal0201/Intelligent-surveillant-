# PROMPT: write pytest fixtures that build a fresh in-memory SQLite app
# instance and a TestClient bound to it for each test, ensuring the global
# engine in app.db is replaced before models import.
# CHANGES MADE: monkeypatch INTEL_DB_URL via env before importing app, and
# disable bootstrap by pointing INTEL_BOOTSTRAP_GLOB at a non-existent file.

from __future__ import annotations
import os
import tempfile
import pytest

@pytest.fixture()
def client(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("INTEL_DB_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("INTEL_BOOTSTRAP_GLOB", str(tmp_path / "nope.jsonl"))

    # Fresh import of app modules so they bind to the test DB URL.
    import importlib, sys
    for m in list(sys.modules):
        if m.startswith("intelligence_api.app"):
            del sys.modules[m]
    main = importlib.import_module("intelligence_api.app.main")
    main.init_db()

    from fastapi.testclient import TestClient
    return TestClient(main.app)
