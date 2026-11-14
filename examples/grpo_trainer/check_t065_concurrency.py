import time, json, requests, statistics
from pathlib import Path

root = Path("/home/zehao/echo-collab/T065")
s = requests.Session()
s.trust_env = False
records = []


def retrieval(label):
    t = time.monotonic()
    try:
        r = s.post(
            "http://127.0.0.1:29661/retrieve",
            json={
                "queries": ["Wikipedia population city"],
                "topk": 3,
                "return_scores": True,
            },
            timeout=90,
        )
        result = dict(label=label, seconds=time.monotonic() - t, status=r.status_code)
    except Exception as e:
        result = dict(label=label, seconds=time.monotonic() - t, error=type(e).__name__)
    with open(root / "retrieval-latency.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


for _ in range(3):
    retrieval("before_concurrency")
wait = time.time()
while time.time() - wait < 1200:
    try:
        rows = [
            json.loads(x)
            for x in (root / "embedding-requests-private.jsonl")
            .read_text()
            .splitlines()
        ]
        eligible = [x for x in rows if x["rows"] >= 64]
        if eligible:
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(5)
else:
    raise RuntimeError("No real smoke request corpus available")
started = time.time()
last_probe = started
while time.time() - started < 600:
    rows = [
        json.loads(x)
        for x in (root / "embedding-requests-private.jsonl").read_text().splitlines()
    ]
    eligible = [x for x in rows if x["rows"] >= 64]
    request = eligible[len(records) % len(eligible)]
    t = time.monotonic()
    try:
        r = s.post(
            "http://127.0.0.1:29700/embed",
            json=dict(
                texts=request["texts"], normalize=True, client="t065-second-client"
            ),
            timeout=180,
        )
        r.raise_for_status()
        body = r.json()
        record = dict(
            time=time.time(),
            rows=request["rows"],
            wall_seconds=time.monotonic() - t,
            encode_seconds=body["seconds"],
            peak_bytes=body["peak_bytes"],
        )
    except Exception as e:
        record = dict(time=time.time(), rows=request["rows"], error=type(e).__name__)
    records.append(record)
    (root / "embedding-concurrency.json").write_text(
        json.dumps(
            dict(started=started, elapsed=time.time() - started, records=records),
            indent=2,
        )
    )
    if time.time() - last_probe >= 60:
        retrieval("during_concurrency")
        last_probe = time.time()
    time.sleep(5)
for _ in range(3):
    retrieval("after_concurrency")
(root / "embedding-concurrency.done").write_text(str(time.time()))
