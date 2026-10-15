"""Protocol/SSE tests: intentionally no model downloads, MLX imports or microphone."""
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error

spec = importlib.util.spec_from_file_location("openless_textstream_worker", Path(__file__).with_name("worker.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FakeAudio(list):
    @property
    def data(self):
        return bytearray(len(self) * 4)
    def astype(self, _dtype):
        return self
    def __truediv__(self, _value):
        return self
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return FakeAudio(result) if isinstance(key, slice) else result


class FakeSession:
    calls = []
    def __init__(self, model):
        self.calls.append(("load", model))
    def init_streaming(self, **kwargs):
        self.calls.append(("init", kwargs))
        return SimpleNamespace(stable_text="", text="")
    def feed_audio(self, _audio, _state):
        self.calls.append(("feed",))
        return SimpleNamespace(stable_text="你好", text="你好世")
    def finish_streaming(self, _state):
        self.calls.append(("finish",))
        return SimpleNamespace(stable_text="你好，世界。", text="你好，世界。")


class WorkerTests(unittest.TestCase):
    def test_import_does_not_initialize_inference_or_capture(self):
        self.assertNotIn("mlx.core", sys.modules)
        self.assertNotIn("sounddevice", sys.modules)
        self.assertNotIn("mlx_qwen3_asr", sys.modules)

    def test_draft_is_revisable_and_final_comes_from_decoder(self):
        snapshots = worker.Snapshots()
        self.assertEqual(snapshots.update("你", "你好吗")['draft'], "好吗")
        self.assertEqual(snapshots.update("你", "你好")['draft'], "好")
        self.assertEqual(snapshots.update("你好", "你好")['draft'], "")
        final = snapshots.update("你好", "你好！", terminal=True)
        self.assertEqual((final['finalized'], final['draft']), ("你好！", ""))
        with self.assertRaises(ValueError):
            snapshots.update("cannot match", "different")

    def run_fake(self, finish=True):
        FakeSession.calls = []
        request = {"type": "audio", "pcm": base64.b64encode(b'\x01\x00' * 32000).decode()}
        lines = json.dumps(request) + "\n" + ('{"type":"finish"}\n' if finish else '')
        output = io.StringIO()
        modules = {
            "numpy": SimpleNamespace(frombuffer=lambda raw, dtype: FakeAudio([0] * (len(raw) // 2)), float32=object(), ascontiguousarray=lambda audio: audio),
            "mlx_qwen3_asr": SimpleNamespace(Session=FakeSession),
            "silero_vad_lite": SimpleNamespace(SileroVAD=lambda _rate: SimpleNamespace(process=lambda _frame: 1.0)),
        }
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "model.safetensors").write_bytes(b'weights')
            (root / "config.json").write_text('{}')
            with patch.dict(sys.modules, modules), patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(lines.encode()))), patch.object(sys, "stdout", output), patch.object(worker, "parent_watch", lambda _parent: None), patch.object(worker, "sse_server", lambda *_args: SimpleNamespace(shutdown=lambda: None, server_close=lambda: None)):
                worker.run(root, 0, os.getpid())
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_feed_produces_draft_before_finish_and_final_is_refined(self):
        events = self.run_fake()
        self.assertEqual([event['type'] for event in events], ['ready', 'stream', 'final'])
        self.assertEqual(events[1]['draft'], '世')
        self.assertEqual(events[-1]['finalized'], '你好，世界。')
        self.assertEqual(events[-1]['draft'], '')
        self.assertIsNone(FakeSession.calls[1][1]['language'], 'do not hard-code English')

    def test_eof_does_not_promote_a_draft(self):
        events = self.run_fake(finish=False)
        self.assertEqual([event['type'] for event in events], ['ready', 'stream'])
        self.assertNotIn(('finish',), FakeSession.calls)

    def test_missing_files_fail_before_any_server_or_model(self):
        with tempfile.TemporaryDirectory() as root, patch.object(worker, "sse_server") as server:
            with self.assertRaisesRegex(ValueError, 'missing'):
                worker.run(Path(root), 0, os.getpid())
            server.assert_not_called()

    def test_subscribers_are_bounded_and_slow_clients_receive_latest_snapshot(self):
        events = worker.Broadcast()
        clients = [events.subscribe() for _ in range(16)]
        self.assertIsNone(events.subscribe())
        events.publish({'type': 'stream', 'finalized': 'old', 'draft': 'wrong'})
        events.publish({'type': 'stream', 'finalized': 'new', 'draft': ''})
        self.assertEqual(clients[0].get_nowait()['finalized'], 'new')

    def test_sse_is_loopback_only_and_rejects_cross_site_controls(self):
        events = worker.Broadcast()
        server = worker.sse_server(events, 0)
        url = 'http://127.0.0.1:%d' % server.server_port
        try:
            for request in [urllib.request.Request(url + '/stream', headers={'Origin': 'https://evil.example'}), urllib.request.Request(url + '/stream', headers={'Host': 'evil.example'}), urllib.request.Request(url + '/stop')]:
                with self.assertRaises(urllib.error.HTTPError) as failure:
                    urllib.request.urlopen(request, timeout=2)
                self.assertIn(failure.exception.code, [403, 404])
            with urllib.request.urlopen(url + '/stream', timeout=2) as response:
                self.assertEqual(response.headers['Content-Type'], 'text/event-stream; charset=utf-8')
                events.publish({'type': 'stream', 'finalized': '你好', 'draft': '世界'})
                first = json.loads(response.readline().decode()[6:])
                self.assertEqual(first['draft'], '世界')
                response.readline()
                events.publish({'type': 'final', 'finalized': '你好，世界。', 'draft': ''})
                final = json.loads(response.readline().decode()[6:])
                self.assertEqual(final['type'], 'stream')
                self.assertTrue(final['done'])
                self.assertEqual(final['draft'], '')
        finally:
            events.closed.set()
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
