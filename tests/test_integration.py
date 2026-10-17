"""GPU integration smoke test for the file pipeline (Phase 3A).

Requires CUDA, the local Whisper weights under weights/, and the Pocket TTS
weights in the Hugging Face cache. Run with:

    .venv\\Scripts\\python.exe -m unittest tests.test_integration -v
"""

import json
import unittest
from pathlib import Path

from app import process_file

ROOT = Path(__file__).resolve().parents[1]


class FilePipelineSmokeTest(unittest.TestCase):
    def test_sample_produces_transcript_text_and_wav(self):
        result = process_file(ROOT / "samples" / "anoop_voice_line.wav")
        self.assertTrue(result["transcript"]["text"].strip(), "transcript must not be empty")
        self.assertTrue(result["converted_text"], "chunker must emit at least one text chunk")
        output = Path(result["output"])
        self.assertTrue(output.is_file(), f"missing output WAV: {output}")
        self.assertGreater(output.stat().st_size, 44, "output WAV is an empty header only")
        self.assertGreater(result["output_duration_s"], 0.5, "converted audio is too short")
        metrics = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))
        for key in ("gpu", "transcript", "converted_text", "end_to_end_latency_ms"):
            self.assertIn(key, metrics)

    def test_compressed_input_produces_output(self):
        result = process_file(ROOT / "samples" / "anoop_voice_line.m4a")
        self.assertTrue(result["transcript"]["text"].strip(), "transcript must not be empty")
        self.assertGreater(result["output_duration_s"], 0.5, "converted audio is too short")


if __name__ == "__main__":
    unittest.main()
