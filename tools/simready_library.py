"""Full-page SimReady asset browser served beside the scene editor.

viser serves only its own client, so the browser runs a small HTTP server in the editor process
(tools/simready_library.html plus a JSON API). It never edits scene.yaml itself: "添加到场景" calls
back into the editor, which downloads/wraps through shared.simready and adds the entity with the
same apply_edits as every other edit. The editor binds it to its own host, on the next free port.

  GET  /library                   the page
  GET  / (/editor/)               the editor: viser's page patched (tools/viser_client), with the app
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

and the scene tab's outliner (tools/scene_outliner.html):

  GET  /api/outliner/state?revision=n   tree rows with their switches (only the revision when unchanged)
  POST /api/outliner/select | focus | toggle   (see Editor.outliner_post)

and the editor app that replaces viser's panels (tools/editor_ui/):

  GET  /ui/<file>                  app.js, app.css, shelf.html (the asset shelf with this port)
  GET  /api/ui/state | inspector   top bar; the selection's properties (Editor.ui_get)
  POST /api/ui/<action>            save, undo, pose, physics, initial, ... (Editor.ui_post)
  GET  /api/ui/camera.jpg          the selected camera's view, rendered once edits settle
  POST /api/import/file?batch=&path=   one file of an import (raw body; a folder is uploaded file by file)
  POST /api/import/needs {batch, main, xacro, files}   which files of a chosen folder the import uses
  POST /api/import/finish {batch, kind, name, fixed}   convert and add it (shared.asset_import)
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback
import urllib.parse

from shared import simready

PAGE = Path(__file__).with_name('simready_library.html')
APP = Path(__file__).with_name('editor_ui')
APP_FILES = {'app.js': 'text/javascript; charset=utf-8', 'app.css': 'text/css; charset=utf-8'}
# viser's client with the editor's patches (tools/viser_client/rlinf.patch, built by tools/build_viser_client.py):
# the editor's page at / (and /editor/), connected to viser's websocket on its own port.
VIEWER = Path(__file__).with_name('viser_client') / 'viewer.html'


def viewer_problem():
    """Why the patched page cannot be served (None when it can): it speaks the protocol of the viser it was
    built from."""
    import viser
    record = VIEWER.with_name('BUILD.json')
    if not VIEWER.is_file() or not record.is_file():
        return 'tools/viser_client/viewer.html 未构建：python tools/build_viser_client.py'
    built = json.loads(record.read_text()).get('viser')
    if built != viser.__version__:
        return f'改进版视图按 viser {built} 构建，当前是 {viser.__version__}：python tools/build_viser_client.py'
    return None


def viewer_page(viser_port):
    """The patched viser page, told where viser's websocket is (the same host, viser's port)."""
    script = ("<script>window.__RLINF_WS__ = (location.protocol === 'https:' ? 'wss://' : 'ws://') + "
              f"location.hostname + ':{int(viser_port)}';</script>"
              # The editor app with the page (served here too), so viser's own panels never show while it loads;
              # the loader viser's GUI injects later finds it already there.
              "<link rel='stylesheet' href='/ui/app.css'><script id='rl-app-js' src='/ui/app.js' data-api='/' defer></script>")
    return VIEWER.read_text().replace('<head>', '<head>' + script, 1).encode()
SIZES = (320, 768)
_fetching = threading.BoundedSemaphore(6)  # previews not yet cached hit Hugging Face at most 6 at a time


class Library:
    """What the pages need from the editor: ``catalog()``, ``asset(key)``, ``status()``,
    ``add(key, name, position)``, ``shelf_get(name, query)``, ``shelf_post(name, request)``,
    ``shelf_picture(key)``, ``outliner_get(name, query)``, ``outliner_post(name, request)`` and
    ``origin_port`` (raising ValueError for a request it rejects)."""

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
            if url.path in ('/library', '/library/'):
                return self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            if url.path in ('/', '/index.html', '/editor', '/editor/'):
                if viewer_problem():
                    return self._send(404, {'error': viewer_problem()})
                return self._send(200, viewer_page(self.editor.origin_port), 'text/html; charset=utf-8', cache='no-cache')
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
            if url.path.startswith('/api/outliner/'):
                return self._send(200, self.editor.outliner_get(url.path[14:], query))
            if url.path == '/api/ui/camera.jpg':
                frame = self.editor.ui_camera()
                return self._send(200, frame, 'image/jpeg') if frame else self._send(204, b'', 'image/jpeg')
            if url.path.startswith('/api/ui/'):
                return self._send(200, self.editor.ui_get(url.path[8:], query))
            if url.path.startswith('/ui/'):
                name = url.path[4:]
                if name in APP_FILES:
                    return self._send(200, (APP / name).read_bytes(), APP_FILES[name], cache='no-cache')
                if name == 'shelf.html':
                    page = Path(__file__).with_name('asset_shelf.html').read_text()
                    page = page.replace('__LIBRARY_PORT__', str(self.server.server_address[1]))
                    return self._send(200, page.encode(), 'text/html; charset=utf-8', cache='no-cache')
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
            if url.path == '/api/import/file':  # raw bytes of one file, streamed to disk
                query = dict(urllib.parse.parse_qsl(url.query))
                return self._send(200, self.editor.import_file(query.get('batch', ''), query.get('path', ''), self.rfile,
                                                               int(self.headers.get('Content-Length') or 0)))
            request = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            if url.path == '/api/import/finish':
                return self._send(200, self.editor.import_finish(request))
            if url.path == '/api/import/needs':
                return self._send(200, self.editor.import_needs(request))
            if url.path == '/api/add':
                return self._send(200, self.editor.add(request.get('key', ''), str(request.get('name', '')).strip(),
                                                       request.get('position') or [0, 0, 0]))
            if url.path.startswith('/api/shelf/'):
                return self._send(200, self.editor.shelf_post(url.path[11:], request))
            if url.path.startswith('/api/outliner/'):
                return self._send(200, self.editor.outliner_post(url.path[14:], request))
            if url.path.startswith('/api/ui/'):
                return self._send(200, self.editor.ui_post(url.path[8:], request))
        except (KeyError, ValueError) as error:
            return self._send(400, {'error': str(error)})
        except Exception as error:  # noqa: BLE001 - reported to the page
            traceback.print_exc()
            return self._send(500, {'error': str(error)})
        self._send(404, {'error': 'not found'})
