import unittest
from pathlib import Path

import numpy as np

from app import RmsEndpointDetector, chunk_text, decode_audio, stable_prefix


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
        detector = RmsEndpointDetector(sample_rate=1000, block_size=100, silence_ms=200, min_speech_ms=100)
        self.assertEqual(detector.push(np.ones(100, dtype=np.float32) * 0.2), [])
        self.assertEqual(detector.push(np.zeros(100, dtype=np.float32)), [])
        emitted = detector.push(np.zeros(100, dtype=np.float32))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0]), 300)

    def test_decode_audio_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            decode_audio(Path("does_not_exist.wav"))


if __name__ == "__main__":
    unittest.main()
