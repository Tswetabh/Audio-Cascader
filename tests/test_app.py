import unittest
from pathlib import Path

import numpy as np

from app import (
    IncrementalTranscriber,
    PlaybackBuffer,
    RmsEndpointDetector,
    chunk_text,
    decode_audio,
    stable_prefix,
)


class AppHelpersTests(unittest.TestCase):
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

    def test_decode_audio_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            decode_audio(Path("does_not_exist.wav"))


if __name__ == "__main__":
    unittest.main()
