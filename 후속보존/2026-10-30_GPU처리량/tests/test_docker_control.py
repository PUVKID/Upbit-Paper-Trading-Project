"""공유 컨테이너의 프로젝트 경계와 읽기 전용 중계를 검사합니다."""

import base64
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from tools import docker_control as control


def container_data():
    return {"Name": "/2026", "Config": {"Labels": {"io.yunseok.managed": "docker2026-v1"}},
            "State": {"Running": True}, "Mounts": [
                {"Type": "bind", "Source": str(control.ROOT.parent),
                 "Destination": "/workspace/2026", "RW": True}]}


class OwnershipTests(unittest.TestCase):
    def inspect(self, data):
        return patch.object(control, "run", return_value=json.dumps([data]))

    def test_shared_container_accepted(self):
        data = container_data()
        with self.inspect(data):
            self.assertEqual(control.owned_container(), data)

    def test_label_source_destination_and_rw_fail_closed(self):
        for field, value in (("label", "foreign"), ("Source", "D:/other"),
                             ("Destination", "/foreign"), ("RW", False), ("Type", "volume")):
            with self.subTest(field=field):
                data = container_data()
                if field == "label":
                    data["Config"]["Labels"]["io.yunseok.managed"] = value
                else:
                    data["Mounts"][0][field] = value
                with self.inspect(data), self.assertRaises(RuntimeError):
                    control.owned_container()

    def test_nested_foreign_mount_rejected(self):
        data = container_data()
        data["Mounts"].append({"Type": "bind", "Source": "D:/foreign", "RW": True,
                               "Destination": control.project_root() + "/work"})
        with self.inspect(data), self.assertRaises(RuntimeError):
            control.owned_container()

    def test_missing_container_no_mutation(self):
        with patch.object(control, "run", return_value="[]") as call:
            with self.assertRaises(RuntimeError):
                control.owned_container()
            self.assertEqual(call.call_args.args[0], ["inspect", "2026"])
            self.assertEqual(call.call_count, 1)

    def test_stopped_container_no_mutation(self):
        data = container_data()
        data["State"]["Running"] = False
        with self.inspect(data) as call, self.assertRaises(RuntimeError):
            control.execute("start")
        self.assertEqual(call.call_count, 1)

    def test_dynamic_root_and_all_actions_use_exec(self):
        with patch.object(control, "ROOT", Path("D:/2026/Upbit 5-Coin Project")), \
                patch.object(control, "owned_container"), patch.object(control, "run", return_value="{}") as run:
            root = "/workspace/2026/Upbit 5-Coin Project"
            for action in ("start", "status", "stop", "gpu", "test"):
                control.execute(action)
                argv = run.call_args.args[0]
                self.assertEqual(argv[:6], ["exec", "-w", root, "2026", ".venv/bin/python", "-B"])
                self.assertNotIn("run", argv)
                self.assertNotIn("build", argv)
                if action in ("start", "status", "stop"):
                    self.assertEqual(argv[-2:], ["tools/container_runtime.py", action])
                elif action == "gpu":
                    self.assertEqual(argv[-2:], ["engine.py", "check-gpu"])
                else:
                    self.assertEqual(argv[-6:], ["-m", "unittest", "discover", "-s", "tests", "-v"])


class GatewayTests(unittest.TestCase):
    @contextmanager
    def gateway(self):
        server = control.create_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertEqual(server.server_address[0], "127.0.0.1")
            yield server.server_address[1]
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def request(self, port, method, path):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_allowed_get_and_security_headers(self):
        value = {"status": 200, "headers": {"Content-Type": "application/json", "Cache-Control": "no-store",
                 "Content-Security-Policy": "default-src 'self'", "Set-Cookie": "private=1"},
                 "body_base64": base64.b64encode(b'{"ok":true}').decode()}
        with self.gateway() as port, patch.object(control, "relay", return_value=value):
            status, headers, body = self.request(port, "GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"ok":true}')
        self.assertEqual(headers["Content-Security-Policy"], "default-src 'self'")
        self.assertNotIn("Set-Cookie", headers)

    def test_private_query_and_traversal_never_relayed(self):
        with self.gateway() as port, patch.object(control, "relay") as relay:
            for path, expected in (("/work/data/candles.sqlite3", 404), ("/../engine.py", 404),
                                   ("/%2e%2e/engine.py", 404), ("/api/status?x=1", 400),
                                   ("/api/status?", 400), ("//private", 404)):
                self.assertEqual(self.request(port, "GET", path)[0], expected)
            relay.assert_not_called()

    def test_non_get_methods_are_405(self):
        with self.gateway() as port, patch.object(control, "relay") as relay:
            for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"):
                self.assertEqual(self.request(port, method, "/api/status")[0], 405)
            relay.assert_not_called()

    def test_upstream_forbidden_is_preserved(self):
        value = {"status": 403, "headers": {}, "body_base64": ""}
        with self.gateway() as port, patch.object(control, "relay", return_value=value):
            self.assertEqual(self.request(port, "GET", "/api/status")[0], 403)

    def test_failures_are_visible_503(self):
        with self.gateway() as port, patch.object(control, "relay", side_effect=RuntimeError("relay failed")):
            status, _, body = self.request(port, "GET", "/")
        self.assertEqual(status, 503)
        self.assertIn(b"relay failed", body)

    def test_invalid_and_oversize_body_are_503(self):
        for value in ({"status": 200, "headers": {}, "body_base64": "!"},
                      {"status": 200, "headers": {"Content-Type": "x\r\nInjected: yes"}, "body_base64": ""},
                      {"status": 200, "headers": {}, "body_base64": base64.b64encode(b"x" * 33).decode()}):
            with self.gateway() as port, patch.object(control, "relay", return_value=value), \
                    patch.object(control, "MAX_BYTES", 32):
                self.assertEqual(self.request(port, "GET", "/")[0], 503)

    def test_relay_has_fixed_argv_and_rejects_private(self):
        value = {"status": 200, "headers": {}, "body_base64": ""}
        with patch.object(control, "owned_container"), patch.object(control, "run", return_value=json.dumps(value)) as run:
            self.assertEqual(control.relay("/api/status"), value)
            self.assertEqual(run.call_args.args[0][-3:], ["tools/container_runtime.py", "relay", "/api/status"])
            with self.assertRaises(ValueError):
                control.relay("/engine.py")
            self.assertEqual(run.call_count, 1)

    def test_bounded_subprocess_and_timeout(self):
        with patch.object(control, "docker_path", return_value=sys.executable):
            self.assertEqual(control.run(["-c", "print('ok')"]).strip(), "ok")
            with patch.object(control, "MAX_BYTES", 64), self.assertRaises(RuntimeError):
                control.run(["-c", "print('x'*1000)"])
            with patch.object(control, "COMMAND_TIMEOUT", 0.05), self.assertRaises(RuntimeError):
                control.run(["-c", "import time; time.sleep(2)"])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        from tools import container_runtime
        self.runtime = container_runtime

    def test_allowlist_matches_dashboard_without_import(self):
        paths = self.runtime.allowed_paths()
        self.assertIn("/assets/doge.png", paths)
        self.assertIn("/fonts/NanumGothic-Bold.ttf", paths)
        self.assertNotIn("/work/paper/account.json", paths)
        for path in ("/api/status?x=1", "/../engine.py", "/%2e%2e/engine.py", "http://evil/api/status"):
            self.assertNotEqual(self.runtime.path_status(path), 200)

    def test_proc_requires_exact_cwd_and_supervisor_command(self):
        runtime = self.runtime
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            root = proc / "project"
            root.mkdir()
            for pid in ("11", "12", "13", "14"):
                (proc / pid).mkdir()
            commands = {11: [str(root / ".venv/bin/python"), "-u", str(root / "supervisor.py")],
                        12: ["python", str(root / "supervisor.py")],
                        13: ["python", "other/supervisor.py"], 14: ["echo", str(root / "supervisor.py")]}
            for pid, command in commands.items():
                (proc / str(pid) / "cmdline").write_bytes(b"\0".join(x.encode() for x in command) + b"\0")
            original_resolve = Path.resolve
            def resolve(path, *args, **kwargs):
                if path.name == "cwd":
                    return root if path.parent.name != "12" else proc
                return original_resolve(path, *args, **kwargs)
            with patch.object(runtime, "ROOT", root), patch.object(runtime, "PROC", proc), \
                    patch.object(Path, "resolve", resolve):
                self.assertEqual([item["pid"] for item in runtime.processes()], [11])

    def test_stop_only_signals_revalidated_supervisor(self):
        runtime = self.runtime
        with patch.object(runtime, "processes", return_value=[{"pid": 42}]), \
                patch.object(runtime, "is_supervisor", return_value=True), \
                patch.object(runtime.os, "pidfd_open", create=True, return_value=7), \
                patch.object(runtime.os, "close"), \
                patch.object(runtime.signal, "pidfd_send_signal", create=True) as send:
            self.assertEqual(runtime.stop()["signalled"], [42])
            send.assert_called_once_with(7, signal.SIGTERM)

    def test_stop_rejects_changed_process_identity(self):
        runtime = self.runtime
        with patch.object(runtime, "processes", return_value=[{"pid": 42}]), \
                patch.object(runtime, "is_supervisor", return_value=False), \
                patch.object(runtime.os, "pidfd_open", create=True, return_value=7), \
                patch.object(runtime.os, "close"), \
                patch.object(runtime.signal, "pidfd_send_signal", create=True) as send:
            self.assertEqual(runtime.stop()["signalled"], [])
            send.assert_not_called()

    def test_start_is_detached_and_duplicate_safe(self):
        runtime = self.runtime
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, "ROOT", Path(directory)), \
                patch.object(runtime, "processes", side_effect=[[], [{"pid": 42}]]), \
                patch.object(runtime.subprocess, "Popen") as popen:
            popen.return_value.pid = 42
            popen.return_value.poll.return_value = None
            with patch.object(runtime, "is_supervisor", return_value=True):
                runtime.start()
                runtime.start()
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertFalse(popen.call_args.kwargs["shell"])
            self.assertTrue((Path(directory) / "work/service/supervisor-console.log").exists())

    def test_runtime_relay_is_http_only_and_bounded(self):
        runtime = self.runtime
        response = Mock(status=200)
        response.getheaders.return_value = [("Content-Type", "text/plain"), ("Set-Cookie", "secret")]
        response.read.return_value = b"hello"
        with patch.object(runtime.http.client, "HTTPConnection") as connection:
            connection.return_value.getresponse.return_value = response
            value = runtime.relay("/api/status")
            connection.assert_called_once_with("127.0.0.1", 8765, timeout=10)
            connection.return_value.request.assert_called_once_with("GET", "/api/status", headers={"Accept-Encoding": "identity"})
            self.assertEqual(base64.b64decode(value["body_base64"]), b"hello")
            self.assertNotIn("Set-Cookie", value["headers"])
            with self.assertRaises(ValueError):
                runtime.relay("/work/data/candles.sqlite3")
            with patch.object(runtime, "MAX_BYTES", 4), self.assertRaises(RuntimeError):
                runtime.relay("/")


if __name__ == "__main__":
    unittest.main()
