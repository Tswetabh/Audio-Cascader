# Patches applied to `pocket_tts/`

The vendored engine is kept close to upstream; only minimal fixes are applied,
each justified by a crash or failing behavior observed in this repo.

## 1. `models/tts_model.py` — decoder timing guard (2026-10-17)

- In `_decode_audio_worker`, the completion guard was
  `if hasattr(self, "_active_generation_timing"):` — changed to
  `if hasattr(self, "_active_generation_timing") and self._active_generation_timing is not None:`
- Why: after any `generate_audio()` call the attribute is set to `None`. A
  later standalone `generate_audio_stream()` (now used for streaming playback)
  then crashed the decoder thread with
  `TypeError: 'NoneType' object does not support item assignment`.
- Reproduced by `tests/test_integration.py::test_pocket_stream_yields_audio_frames`
  when it runs after the file-pipeline tests (shared model instance via
  `get_pocket_runner()`).
- Every other timing site in the file already uses the `is not None` guard;
  this change simply brings the decoder worker in line.
