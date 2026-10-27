"""A blind test page served from this computer: the page of an Artifact blind test (its HTML with the data in it) with
the videos from the disk, so that they load at once, and its answers kept in answers.json next to it. The page talks
to the same small db as on claude.ai (collection("answers").doc(id).set / delete, onSnapshot), here a shim over
POST /db. The folder holds page.html (VIDEOS maps a video id to "v/<id>.mp4") and v/.

    python scripts/reports/blind_server.py <folder> [--port 8770]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import argparse
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from wan_dit_lab.harness.config import located

SHIM = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<script>
// the db of an Artifact page, over this computer's server: answers.json next to the page
(function () {
  const api = (body) => fetch("/db", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}).then((r) => r.json());
  const collection = (name) => ({
    doc: (id) => ({
      set: (data) => api({op: "set", collection: name, id, data}),
      delete: () => api({op: "delete", collection: name, id}),
    }),
    onSnapshot: (fn) => { api({op: "list", collection: name}).then((docs) => fn({docs: Object.entries(docs).map(([id, d]) => ({id, data: () => d}))})); return () => {}; },
  });
  window.claude = {use: async (what) => what === "db" ? {collection} : null};
})();
</script>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=located)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    folder = args.folder.resolve()
    store, lock = folder / "answers.json", threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(folder), **kw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = (folder / "page.html").read_text(encoding="utf-8")
                body = (SHIM + page).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != "/db":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                db = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
                docs = db.setdefault(req["collection"], {})
                if req["op"] == "set":
                    docs[req["id"]] = req["data"]
                elif req["op"] == "delete":
                    docs.pop(req["id"], None)
                if req["op"] != "list":
                    store.write_text(json.dumps(db, ensure_ascii=False, indent=1), encoding="utf-8")
                out = json.dumps(docs if req["op"] == "list" else {"ok": True}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    print(f"http://localhost:{args.port}/  ({folder})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
