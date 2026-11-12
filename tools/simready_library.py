"""Full-page SimReady asset browser served beside the scene editor.

viser serves only its own client, so the browser runs a small HTTP server in the editor process
(tools/simready_library.html plus a JSON API). It never edits scene.yaml itself: "添加到场景" calls
back into the editor, which downloads/wraps through shared.simready and adds the entity with the
same apply_edits as every other edit. The editor binds it to its own host, on the next free port.

  GET  /                          the page
  GET  /api/catalog               every asset {key, publisher, name, category, local, bytes} + sources
  GET  /api/asset?key=pub/name    details: files present/missing, wrapper, scene entities, suggested name
  GET  /api/status                the current add job (download progress, result)
  GET  /preview/<size>/<pub>/<name>.jpg   official white-background preview (sizes in SIZES)
  POST /api/add  {key, name, position}    hand an asset to the editor

The editor's add tab embeds the asset shelf (tools/asset_shelf.html, every asset source, not only
SimReady) from the editor's own origin; it calls these with CORS allowed for that origin only:

  GET  /api/shelf/items | state    shelf cards and categories; selection, detail, placement, add job
  POST /api/shelf/select | option | arm | cancel | drop    (see Editor.shelf_post)
  GET  /thumb/<key>.png            an editor sketch of a catalog asset
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback
import urllib.parse

from shared import simready

PAGE = Path(__file__).with_name('simready_library.html')
SIZES = (320, 768)
_fetching = threading.BoundedSemaphore(6)  # previews not yet cached hit Hugging Face at most 6 at a time


class Library:
    """What the pages need from the editor: ``catalog()``, ``asset(key)``, ``status()``,
    ``add(key, name, position)``, ``shelf_get(name, query)``, ``shelf_post(name, request)``,
    ``shelf_picture(key)`` and ``origin_port`` (raising ValueError for a request it rejects)."""

    def __init__(self, host, port, editor, attempts=20):
        handler = type('Handler', (_Handler,), {'editor': editor})
        for candidate in range(port, port + attempts):
            try:
                self.server = ThreadingHTTPServer((host, candidate), handler)
                break
            except OSError:
                continue
        else:
            raise OSError(f'端口 {port}–{port + attempts - 1} 都已占用')
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


class _Handler(BaseHTTPRequestHandler):
    editor = None

    def log_message(self, *args):  # quiet: the editor's terminal shows its own reports
        pass

    def _send(self, status, body, kind='application/json; charset=utf-8', cache=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', cache or 'no-store')
        self._allow_origin()
        self.end_headers()
        self.wfile.write(body)

    def _allow_origin(self):
        """The shelf runs on the editor's origin (same host name, the viser port)."""
        origin = self.headers.get('Origin')
        if origin and urllib.parse.urlsplit(origin).port == self.editor.origin_port:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')

    def do_OPTIONS(self):  # noqa: N802 - http.server API (CORS preflight of the shelf's JSON posts)
        self.send_response(204)
        self._allow_origin()
        self.send_header('Access-Control-Allow-Methods', 'GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):  # noqa: N802 - http.server API
        url = urllib.parse.urlsplit(self.path)
        query = dict(urllib.parse.parse_qsl(url.query))
        try:
            if url.path in ('/', '/index.html'):
                return self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            if url.path == '/api/catalog':
                return self._send(200, self.editor.catalog())
            if url.path == '/api/asset':
                return self._send(200, self.editor.asset(query.get('key', '')))
            if url.path == '/api/status':
                return self._send(200, self.editor.status())
            if url.path.startswith('/preview/'):
                return self._preview(url.path)
            if url.path.startswith('/api/shelf/'):
                return self._send(200, self.editor.shelf_get(url.path[11:], query))
            if url.path.startswith('/thumb/') and url.path.endswith('.png'):
                return self._send(200, self.editor.shelf_picture(urllib.parse.unquote(url.path[7:-4])), 'image/png',
                                  cache='no-cache')
        except (KeyError, ValueError) as error:
            return self._send(404, {'error': str(error)})
        except Exception as error:  # noqa: BLE001 - reported to the page
            traceback.print_exc()
            return self._send(500, {'error': str(error)})
        self._send(404, {'error': 'not found'})

    def _preview(self, path):
        parts = urllib.parse.unquote(path).split('/')  # '', 'preview', size, publisher, name.jpg
        if len(parts) != 5 or not parts[2].isdigit() or int(parts[2]) not in SIZES or not parts[4].endswith('.jpg') \
                or parts[3] not in simready.SOURCES or '..' in parts[4]:
            return self._send(404, {'error': 'not found'})
        size, publisher, name = int(parts[2]), parts[3], parts[4][:-4]
        image = simready.preview(publisher, name, size, online=False)
        if image is None:
            with _fetching:
                try:
                    image = simready.preview(publisher, name, size)
                except Exception:  # noqa: BLE001 - offline or no preview: the page shows a placeholder
                    image = None
        if image is None:
            return self._send(404, {'error': 'no preview'})
        self._send(200, image.read_bytes(), 'image/jpeg', cache='max-age=86400')

    def do_POST(self):  # noqa: N802 - http.server API
        url = urllib.parse.urlsplit(self.path)
        try:
            request = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            if url.path == '/api/add':
                return self._send(200, self.editor.add(request.get('key', ''), str(request.get('name', '')).strip(),
                                                       request.get('position') or [0, 0, 0]))
            if url.path.startswith('/api/shelf/'):
                return self._send(200, self.editor.shelf_post(url.path[11:], request))
        except (KeyError, ValueError) as error:
            return self._send(400, {'error': str(error)})
        except Exception as error:  # noqa: BLE001 - reported to the page
            traceback.print_exc()
            return self._send(500, {'error': str(error)})
        self._send(404, {'error': 'not found'})
