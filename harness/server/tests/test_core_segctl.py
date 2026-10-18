"""segctl over a real segd server (uvicorn on an ephemeral port)."""
from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import uvicorn

import segd.data
from segd.app import create_app
from segd.segctl import main as segctl_main
from _helpers import install_fake_data, install_stubs


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_healthz(url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url + "/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("segd server did not come up")


def test_segctl_round_trip(tmp_path, monkeypatch, capsys):
    install_stubs(monkeypatch, force=True)
    install_fake_data(monkeypatch)
    app = create_app()
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_healthz(url)

        rc = segctl_main(["--url", url, "open-case", "phantom", "--run-id", "r1",
                          "--out-dir", str(tmp_path / "results")])
        assert rc == 0
        assert "opened phantom" in capsys.readouterr().out

        rc = segctl_main(["--url", url, "op", "phantom", "view", "--save-images", str(tmp_path / "imgs"),
                          "--args", json.dumps({"axis": "axial", "slice": 40, "marks": "fine"})])
        assert rc == 0
        out = capsys.readouterr()
        assert "axial 40/" in out.out
        assert (tmp_path / "imgs" / "000.png").exists()

        rc = segctl_main(["--url", url, "status", "phantom"])
        assert rc == 0
        st = json.loads(capsys.readouterr().out)
        assert st["handle"] == "phantom" and st["done"] is False

        # agent-facing error via CLI: exit code 1, message on stderr
        rc = segctl_main(["--url", url, "op", "phantom", "view", "--args", '{"axis": "oops"}'])
        assert rc == 1
        err = capsys.readouterr().err
        assert "axis" in err

        rc = segctl_main(["--url", url, "close", "phantom"])
        assert rc == 0
        assert "closed phantom" in capsys.readouterr().out

        rc = segctl_main(["--url", url, "status", "phantom"])
        assert rc == 1
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_segctl_bad_args(tmp_path, monkeypatch, capsys):
    rc = segctl_main(["op", "phantom", "view", "--args", "{not json}"])
    assert rc == 2
    rc = segctl_main(["--url", "http://127.0.0.1:1", "status", "x"])
    assert rc == 1
