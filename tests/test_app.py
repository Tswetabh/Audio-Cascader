import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app import (
    EventHub,
    IncrementalTranscriber,
    LiveSession,
    PlaybackBuffer,
    RmsEndpointDetector,
    SessionRecorder,
    align_and_score,
    auto_convert_settings,
    cable_check_verdict,
    chunk_text,
    create_web_app,
    decode_audio,
    device_extra_settings,
    DiagnosticsSession,
    SegmentDiagnostics,
    stable_prefix,
    summarize_diagnostics,
    sample_system_usage,
    voice_cache_is_current,
)
from unittest import mock


class AppHelpersTests(unittest.TestCase):
    def test_empty_diagnostics_session_is_json_safe(self):
        session = DiagnosticsSession(session_id="session-1", started_at=10.0)

        snapshot = session.to_dict()

        self.assertEqual(snapshot["session_id"], "session-1")
        self.assertEqual(snapshot["segments"], [])
        self.assertEqual(snapshot["system_samples"], [])
        self.assertEqual(snapshot["counters"]["segment_count"], 0)

    def test_diagnostics_preserves_segment_order_and_nested_timings(self):
        session = DiagnosticsSession(session_id="session-1", started_at=10.0)
        session.segments.extend(
            [
                SegmentDiagnostics(segment_id=1, started_at=1.0, ended_at=2.0, stt={"latency_ms": 20.0}),
                SegmentDiagnostics(segment_id=2, started_at=3.0, ended_at=4.0, stt={"latency_ms": 40.0}),
            ]
        )

        snapshot = session.to_dict()

        self.assertEqual([row["segment_id"] for row in snapshot["segments"]], [1, 2])
        self.assertEqual(snapshot["segments"][1]["stt"]["latency_ms"], 40.0)

    def test_diagnostics_summary_reports_average_p50_and_p95(self):
        snapshot = {
            "segments": [
                {"timings": {"stt_latency_ms": 10.0, "end_to_end_latency_ms": 100.0}},
                {"timings": {"stt_latency_ms": 20.0, "end_to_end_latency_ms": 200.0}},
                {"timings": {"stt_latency_ms": 30.0, "end_to_end_latency_ms": 300.0}},
            ]
        }

        summary = summarize_diagnostics(snapshot)

        self.assertEqual(summary["stt_latency_ms"]["average"], 20.0)
        self.assertEqual(summary["stt_latency_ms"]["p50"], 20.0)
        self.assertEqual(summary["stt_latency_ms"]["p95"], 29.0)
        self.assertEqual(summary["end_to_end_latency_ms"]["p50"], 200.0)

    def test_live_session_records_and_returns_diagnostics(self):
        session = LiveSession()
        session.begin_diagnostics({"stt_model": "turbo"})
        session.dropped_blocks = 3
        session.record_segment_diagnostics(
            SegmentDiagnostics(
                segment_id=1,
                started_at=1.0,
                ended_at=2.0,
                timings={"stt_latency_ms": 80.0},
            )
        )

        snapshot = session.diagnostics_snapshot()

        self.assertEqual(snapshot["config"]["stt_model"], "turbo")
        self.assertEqual(snapshot["counters"]["segment_count"], 1)
        self.assertEqual(snapshot["counters"]["input_queue_drops"], 3)
        self.assertEqual(snapshot["segments"][0]["timings"]["stt_latency_ms"], 80.0)

    def test_system_sample_has_stable_cpu_memory_and_gpu_fields(self):
        sample = sample_system_usage()

        self.assertIn("cpu_percent", sample)
        self.assertIn("process_rss_mb", sample)
        self.assertIn("gpu_utilization_percent", sample)
        self.assertIn("gpu_vram_used_mb", sample)
        self.assertIn("gpu_vram_total_mb", sample)
        self.assertIn("gpu_temperature_c", sample)

    def test_chunk_text_preserves_punctuation_and_word_cap(self):
        self.assertEqual(
            chunk_text("Hello there. This is a longer sentence with more words", max_words=4),
            ["Hello there.", "This is a longer", "sentence with more words"],
        )


    def test_stable_prefix_emits_only_new_complete_words(self):
        self.assertEqual(stable_prefix("hello wor", "hello world today"), "hello")
        self.assertEqual(stable_prefix("hello world", "hello world today"), "hello world")

    def test_rms_endpoint_emits_after_silence(self):
        detector = RmsEndpointDetector(
            sample_rate=1000, block_size=100, silence_ms=200, min_speech_ms=100, calibrate_ms=0
        )
        self.assertEqual(detector.push(np.ones(100, dtype=np.float32) * 0.2), [])
        self.assertEqual(detector.push(np.zeros(100, dtype=np.float32)), [])
        emitted = detector.push(np.zeros(100, dtype=np.float32))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0]), 300)

    def test_live_endpoint_defaults_keep_natural_pauses_in_one_segment(self):
        detector = RmsEndpointDetector(sample_rate=1000, block_size=100, calibrate_ms=0)
        self.assertEqual(detector.silence_blocks, 5)
        self.assertEqual(detector.max_samples, 5000)

    def test_calibrated_threshold_rejects_ambient_then_detects_speech(self):
        detector = RmsEndpointDetector(
            sample_rate=1000, block_size=100, silence_ms=200, min_speech_ms=100, calibrate_ms=500
        )
        quiet = np.ones(100, dtype=np.float32) * 0.01
        for _ in range(5):
            self.assertEqual(detector.push(quiet), [])
        self.assertAlmostEqual(detector.threshold, 0.03, places=4)
        speech = np.ones(100, dtype=np.float32) * 0.2
        self.assertEqual(detector.push(speech), [])
        self.assertEqual(detector.push(np.zeros(100, dtype=np.float32)), [])
        emitted = detector.push(np.zeros(100, dtype=np.float32))
        self.assertEqual(len(emitted), 1)

    def test_pre_roll_retained_before_speech(self):
        detector = RmsEndpointDetector(
            sample_rate=1000, block_size=100, silence_ms=200, min_speech_ms=100, calibrate_ms=0
        )
        detector.push(np.zeros(100, dtype=np.float32))
        detector.push(np.zeros(100, dtype=np.float32))
        detector.push(np.ones(100, dtype=np.float32) * 0.2)
        detector.push(np.zeros(100, dtype=np.float32))
        emitted = detector.push(np.zeros(100, dtype=np.float32))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0]), 400)

    def test_in_speech_reflects_open_segment(self):
        detector = RmsEndpointDetector(
            sample_rate=1000, block_size=100, silence_ms=200, min_speech_ms=100, calibrate_ms=0
        )
        self.assertFalse(detector.in_speech)
        detector.push(np.ones(100, dtype=np.float32) * 0.2)
        self.assertTrue(detector.in_speech)
        detector.push(np.zeros(100, dtype=np.float32))
        detector.push(np.zeros(100, dtype=np.float32))
        self.assertFalse(detector.in_speech)

    def test_incremental_sequence_emits_each_word_once(self):
        tracker = IncrementalTranscriber()
        passes = ["I just had an", "I just had an idea", "I just had an idea today"]
        output = [tracker.update(text) for text in passes]
        output.append(tracker.flush("I just had an idea today and"))
        self.assertEqual(" ".join(item for item in output if item), "I just had an idea today and")

    def test_incremental_reset_starts_new_segment(self):
        tracker = IncrementalTranscriber()
        tracker.update("first pass words")
        tracker.update("first pass words more")
        tracker.reset()
        self.assertEqual(tracker.update("second segment"), "")
        self.assertEqual(tracker.flush("second segment"), "second segment")

    def test_incremental_revision_does_not_duplicate_committed_words(self):
        tracker = IncrementalTranscriber()
        tracker.update("I scream")
        self.assertEqual(tracker.update("I scream"), "I scream")
        self.assertEqual(tracker.update("ice cream cones"), "")
        self.assertEqual(tracker.flush("ice cream cones"), "")
        self.assertEqual(tracker.flush("I scream you scream"), "you scream")

    def test_incremental_empty_input(self):
        tracker = IncrementalTranscriber()
        self.assertEqual(tracker.update(""), "")
        self.assertEqual(tracker.flush(""), "")

    def test_playback_guard_drops_oldest_frames(self):
        buffer = PlaybackBuffer(max_samples=100)
        buffer.push(np.zeros(40, dtype=np.float32))
        buffer.push(np.ones(40, dtype=np.float32))
        buffer.push(np.full(40, 2.0, dtype=np.float32))
        self.assertEqual(buffer.dropped_frames, 1)
        self.assertEqual(buffer.samples, 80)
        first = buffer.pop()
        self.assertEqual(float(first[0]), 1.0)
        second = buffer.pop()
        self.assertEqual(float(second[0]), 2.0)
        self.assertIsNone(buffer.pop())

    def test_playback_clear_resets_backlog(self):
        buffer = PlaybackBuffer(max_samples=1000)
        buffer.push(np.zeros(100, dtype=np.float32))
        buffer.clear()
        self.assertEqual(buffer.samples, 0)
        self.assertIsNone(buffer.pop())

    def test_playback_pop_frames_spans_tts_chunks_without_silence(self):
        buffer = PlaybackBuffer(max_samples=1000)
        buffer.push(np.ones(3, dtype=np.float32))
        buffer.push(np.full(3, 2.0, dtype=np.float32))

        first = buffer.pop_frames(5)
        second = buffer.pop_frames(1)

        np.testing.assert_array_equal(first, np.array([1, 1, 1, 2, 2], dtype=np.float32))
        np.testing.assert_array_equal(second, np.array([2], dtype=np.float32))

    def test_voice_cache_current_only_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "voice.pt"
            manifest = root / "manifest.json"
            cache.write_bytes(b"state")
            manifest.write_text(json.dumps({"sha256_reference": "abc"}), encoding="utf-8")
            self.assertTrue(voice_cache_is_current(cache, manifest, "abc"))
            self.assertFalse(voice_cache_is_current(cache, manifest, "other"))
            manifest.write_text("{not json", encoding="utf-8")
            self.assertFalse(voice_cache_is_current(cache, manifest, "abc"))
            manifest.write_text(json.dumps({"sha256_reference": "abc"}), encoding="utf-8")
            cache.unlink()
            self.assertFalse(voice_cache_is_current(cache, manifest, "abc"))

    def test_event_hub_fans_out_and_unsubscribes(self):
        hub = EventHub()
        first = hub.subscribe()
        second = hub.subscribe()
        hub.publish({"kind": "state", "state": "running"})
        self.assertEqual(first.get_nowait()["state"], "running")
        self.assertEqual(second.get_nowait()["state"], "running")
        hub.unsubscribe(second)
        hub.publish({"kind": "state", "state": "ready"})
        self.assertEqual(first.get_nowait()["state"], "ready")
        self.assertTrue(second.empty())

    def test_session_recorder_writes_wav_and_metrics(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionRecorder()
            recorder.add_audio(np.zeros(2400, dtype=np.float32))
            recorder.add_audio(np.ones(2400, dtype=np.float32) * 0.5)
            recorder.add_event("spoken", "hello world")
            recorder.add_event("converted", "hello world")
            result = recorder.finalize(Path(tmp), 24000)
            self.assertTrue(Path(result["wav"]).is_file())
            self.assertTrue(Path(result["metrics"]).is_file())
            self.assertAlmostEqual(result["duration_s"], 0.2, places=3)
            payload = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))
            self.assertEqual([event["kind"] for event in payload["events"]], ["spoken", "converted"])
            audio, rate = sf.read(result["wav"])
            self.assertEqual(rate, 24000)
            self.assertEqual(len(audio), 4800)

    def test_session_recorder_handles_empty_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = SessionRecorder().finalize(Path(tmp), 24000)
            self.assertTrue(Path(result["metrics"]).is_file())

    def test_align_and_score_identical_signals(self):
        signal = np.random.default_rng(7).standard_normal(4800).astype(np.float32)
        score = align_and_score(signal, signal.copy())
        self.assertEqual(score["lag_samples"], 0)
        self.assertGreater(score["peak_correlation"], 0.99)

    def test_align_and_score_detects_delay_and_tolerates_scale(self):
        signal = np.random.default_rng(7).standard_normal(4800).astype(np.float32)
        recorded = np.concatenate([np.zeros(240, dtype=np.float32), signal * 0.5])
        score = align_and_score(signal, recorded)
        self.assertEqual(score["lag_samples"], 240)
        self.assertGreater(score["peak_correlation"], 0.99)

    def test_align_and_score_advanced_signal_reports_negative_lag(self):
        signal = np.random.default_rng(7).standard_normal(4800).astype(np.float32)
        recorded = np.concatenate([signal[240:], np.zeros(240, dtype=np.float32)])
        score = align_and_score(signal, recorded)
        self.assertEqual(score["lag_samples"], -240)

    def test_align_and_score_silence_has_zero_correlation(self):
        signal = np.random.default_rng(7).standard_normal(4800).astype(np.float32)
        score = align_and_score(signal, np.zeros(4800, dtype=np.float32))
        self.assertEqual(score["peak_correlation"], 0.0)

    def test_cable_check_verdict_branches(self):
        self.assertEqual(cable_check_verdict(0.95, 0.2), "pass")
        self.assertEqual(cable_check_verdict(0.0, 0.0), "silent")
        self.assertEqual(cable_check_verdict(0.3, 0.2), "mismatch")

    def test_auto_convert_settings_only_for_wasapi_hosts(self):
        self.assertIsNone(auto_convert_settings("MME"))
        self.assertIsNone(auto_convert_settings("Windows DirectSound"))
        settings = auto_convert_settings("Windows WASAPI")
        self.assertIsNotNone(settings)
        self.assertNotEqual(settings._streaminfo.flags, 0)

    def test_device_extra_settings_picks_wasapi_auto_convert(self):
        hostapis = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]
        # Indexed WASAPI device -> auto-convert settings; non-WASAPI -> None.
        with mock.patch("sounddevice.query_devices", return_value={"name": "CABLE Input", "hostapi": 2}), \
             mock.patch("sounddevice.query_hostapis", side_effect=lambda index=None: hostapis if index is None else hostapis[index]):
            self.assertIsNotNone(device_extra_settings(19, "output"))
        with mock.patch("sounddevice.query_devices", return_value={"name": "Speakers", "hostapi": 0}), \
             mock.patch("sounddevice.query_hostapis", side_effect=lambda index=None: hostapis if index is None else hostapis[index]):
            self.assertIsNone(device_extra_settings(5, "output"))

    def test_device_extra_settings_default_device_uses_kind(self):
        hostapis = [{"name": "Windows WASAPI"}]
        with mock.patch("sounddevice.query_devices", return_value={"name": "Default Mic", "hostapi": 0}) as query, \
             mock.patch("sounddevice.query_hostapis", side_effect=lambda index=None: hostapis if index is None else hostapis[index]):
            self.assertIsNotNone(device_extra_settings(None, "input"))
            query.assert_called_once_with(kind="input")

    def test_decode_audio_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            decode_audio(Path("does_not_exist.wav"))


class WebAppTests(unittest.TestCase):
    def _client(self) -> "object":
        from fastapi.testclient import TestClient

        return TestClient(create_web_app())

    def test_health_endpoint(self):
        response = self._client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_react_frontend_serves_from_production_build(self):
        response = self._client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="root"', response.text)

    def test_missing_react_build_explains_how_to_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("app.ROOT", Path(tmp)):
                response = self._client().get("/")

        self.assertEqual(response.status_code, 503)
        self.assertIn("npm run build", response.text)

    def test_process_upload_rejects_garbage_with_400(self):
        # Regression: `from __future__ import annotations` stringifies the
        # /api/process signature; FastAPI must be able to resolve UploadFile,
        # otherwise every upload fails with a PydanticUserError 500.
        response = self._client().post(
            "/api/process",
            files={"upload": ("bad.wav", b"this is not audio", "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("PydanticUserError", response.text)

    def test_diagnostics_endpoint_returns_empty_snapshot(self):
        response = self._client().get("/api/diagnostics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["segments"], [])
        self.assertIn("summary", response.json())

    def test_diagnostics_json_export_is_an_explicit_download(self):
        response = self._client().get("/api/diagnostics/export?format=json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertEqual(response.json()["segments"], [])

    def test_diagnostics_csv_export_has_stable_headers(self):
        response = self._client().get("/api/diagnostics/export?format=csv")

        self.assertEqual(response.status_code, 200)
        self.assertIn("segment_id", response.text.splitlines()[0])
        self.assertIn("attachment", response.headers["content-disposition"])

    def test_diagnostics_export_rejects_unknown_format(self):
        response = self._client().get("/api/diagnostics/export?format=xml")

        self.assertEqual(response.status_code, 400)

    def test_decode_audio_garbage_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.wav"
            bad.write_bytes(b"this is not audio")
            with self.assertRaises(ValueError):
                decode_audio(bad)


if __name__ == "__main__":
    unittest.main()
