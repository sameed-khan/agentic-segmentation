# Runs ON the Colab VM via `colab exec -s nnint -f harness/colab/start_server.py`.
# Starts nninteractive-server bound to loopback (reached only through the SSH
# port-forward) and then blocks, streaming the log, so the Jupyter kernel stays
# BUSY. Colab keeps a VM alive based on kernel activity; a busy kernel is the
# documented way to avoid idle reclamation.
import subprocess, time, os, sys

PORT = int(os.environ.get("NNI_PORT", "1527"))
LOG = "/content/nninteractive-server.log"
MODEL = os.environ.get("NNI_MODEL", "nnInteractive_v1.0")

# Idempotent: if a server is already listening, just follow its log.
already = subprocess.run(["bash", "-lc", f"ss -ltn | grep -q ':{PORT} '"]).returncode == 0
if not already:
    log = open(LOG, "ab")
    subprocess.Popen(
        ["nninteractive-server", "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
         "--device", "cuda:0", "--max-sessions", os.environ.get("NNI_MAX_SESSIONS", "4")],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    print(f"started nninteractive-server on 127.0.0.1:{PORT}, log={LOG}", flush=True)
else:
    print(f"server already listening on {PORT}", flush=True)

# Block forever, echoing new log lines (keeps the kernel busy).
pos = 0
while True:
    try:
        with open(LOG, "rb") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        if chunk:
            sys.stdout.write(chunk.decode(errors="replace"))
            sys.stdout.flush()
    except FileNotFoundError:
        pass
    time.sleep(5)
