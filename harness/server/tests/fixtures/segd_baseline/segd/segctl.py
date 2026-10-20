"""segctl: thin CLI over the segd HTTP API for debugging and human use.

Commands:
  segctl open-case <case> --run-id X --out-dir D [--nni-url URL]
  segctl op <case> <op> --args '{json}' [--save-images DIR]
  segctl status <case>
  segctl close <case>

Prints the response text to stdout (errors to stderr, exit code 1); with
--save-images DIR, response images are written as DIR/NNN.png.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx

from segd.config import HOST, PORT


def _client(url: str) -> httpx.Client:
    return httpx.Client(base_url=url.rstrip("/"), timeout=httpx.Timeout(600.0))


def _print_response(resp: dict, save_images: str | None) -> int:
    if not resp.get("ok"):
        print(resp.get("error", "unknown error"), file=sys.stderr)
        return 1
    print(resp.get("text", ""))
    if save_images:
        out = Path(save_images)
        out.mkdir(parents=True, exist_ok=True)
        for i, b64 in enumerate(resp.get("images", [])):
            (out / f"{i:03d}.png").write_bytes(base64.b64decode(b64))
        n = len(resp.get("images", []))
        if n:
            print(f"[saved {n} image(s) to {out}]", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="segctl", description="thin CLI over the segd HTTP API")
    p.add_argument("--url", default=f"http://{HOST}:{PORT}", help="segd base URL")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_open = sub.add_parser("open-case", help="open a case in segd")
    p_open.add_argument("case")
    p_open.add_argument("--run-id", required=True)
    p_open.add_argument("--out-dir", required=True)
    p_open.add_argument("--nni-url", default=None)
    p_open.add_argument("--budgets", default=None, help="JSON, e.g. '{\"max_tool_calls\":40}'")

    p_op = sub.add_parser("op", help="run one op on an open case")
    p_op.add_argument("case")
    p_op.add_argument("op")
    p_op.add_argument("--args", default="{}", help="JSON object of op arguments")
    p_op.add_argument("--save-images", default=None, dest="save_images")

    p_status = sub.add_parser("status", help="case counters and state")
    p_status.add_argument("case")

    p_close = sub.add_parser("close", help="close a case (flush logs, free the nni session)")
    p_close.add_argument("case")

    a = p.parse_args(argv)
    try:
        with _client(a.url) as http:
            if a.cmd == "open-case":
                body = {"case_id": a.case, "run_id": a.run_id, "out_dir": a.out_dir}
                if a.nni_url:
                    body["nni_url"] = a.nni_url
                if a.budgets:
                    body["budgets"] = json.loads(a.budgets)
                resp = http.post("/cases", json=body)
                data = resp.json()
                if resp.status_code == 200 and data.get("ok"):
                    print(f"opened {data['handle']}")
                    return 0
                print(data.get("detail") or data.get("error") or resp.text, file=sys.stderr)
                return 1
            if a.cmd == "op":
                try:
                    args = json.loads(a.args)
                except json.JSONDecodeError as e:
                    print(f"--args is not valid JSON: {e}", file=sys.stderr)
                    return 2
                if not isinstance(args, dict):
                    print("--args must be a JSON object", file=sys.stderr)
                    return 2
                resp = http.post(f"/cases/{a.case}/op", json={"op": a.op, "args": args}).json()
                return _print_response(resp, a.save_images)
            if a.cmd == "status":
                resp = http.get(f"/cases/{a.case}/status")
                if resp.status_code != 200:
                    print(resp.json().get("detail", resp.text), file=sys.stderr)
                    return 1
                print(json.dumps(resp.json(), indent=2))
                return 0
            if a.cmd == "close":
                resp = http.delete(f"/cases/{a.case}")
                if resp.status_code != 200:
                    print(resp.json().get("detail", resp.text), file=sys.stderr)
                    return 1
                print(f"closed {a.case}")
                return 0
    except httpx.HTTPError as e:
        print(f"segd at {a.url}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
