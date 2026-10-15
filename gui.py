"""Audio Cascader — Real-Time GUI (Tkinter).

Supports both Live Microphone Mode (continuous streaming with WASAPI + Silero VAD)
and File Mode (batch audio file conversion) with American voice cloning.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.audio_convert import convert_to_wav
from core.audio_stream import (
    AudioCapture,
    StreamingAudioPlayer,
    StreamingVADSegmenter,
    list_audio_devices,
)
from core.chunker import chunk_text
from core.stt_engine import STTEngine
from core.tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AudioCascaderGUI")


class AudioCascaderApp(tk.Tk):
    """Main desktop application window."""

    def __init__(self):
        super().__init__()
        self.title("Audio Cascader — Real-Time S2S Voice Converter (Indian -> American)")
        self.geometry("960x780")
        self.minsize(800, 650)
        self.configure(bg="#121214")

        # Engine objects
        self.stt_engine: Optional[STTEngine] = None
        self.tts_engine: Optional[TTSEngine] = None
        self.capture: Optional[AudioCapture] = None
        self.segmenter: Optional[StreamingVADSegmenter] = None
        self.player: Optional[StreamingAudioPlayer] = None

        # State flags & queues
        self.is_live = False
        self.live_stop_event = threading.Event()
        self.stt_queue: queue.Queue = queue.Queue(maxsize=20)
        self.tts_queue: queue.Queue = queue.Queue(maxsize=20)
        self.ui_update_queue: queue.Queue = queue.Queue()

        self.last_converted_file: Optional[Path] = None

        self._init_theme()
        self._build_ui()
        self._populate_audio_devices()
        self.after(100, self._process_ui_queue)

        # Pre-load engines in background thread
        threading.Thread(target=self._preload_models_background, daemon=True).start()

    def _init_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        
        # Colors
        bg_card = "#1e1e24"
        accent_blue = "#3b82f6"
        accent_green = "#10b981"
        fg_text = "#f3f4f6"

        style.configure(".", background="#121214", foreground=fg_text)
        style.configure("TFrame", background="#121214")
        style.configure("Card.TFrame", background=bg_card, relief="flat")
        style.configure("TLabel", background="#121214", foreground=fg_text, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=bg_card, foreground=fg_text, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#121214", foreground="#60a5fa", font=("Segoe UI", 16, "bold"))
        style.configure("SubHeader.TLabel", background="#121214", foreground="#9ca3af", font=("Segoe UI", 10))
        style.configure("Badge.TLabel", background="#2563eb", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        
        style.configure("Primary.TButton", background="#2563eb", foreground="#ffffff", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Primary.TButton", background=[("active", "#1d4ed8")])

        style.configure("Success.TButton", background="#059669", foreground="#ffffff", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Success.TButton", background=[("active", "#047857")])

        style.configure("Danger.TButton", background="#dc2626", foreground="#ffffff", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Danger.TButton", background=[("active", "#b91c1c")])

        style.configure("TNotebook", background="#121214", tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", background="#27272a", foreground="#d4d4d8", padding=[16, 6], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#3b82f6")], foreground=[("selected", "#ffffff")])

    def _build_ui(self):
        # 1. Header Bar
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", padx=20, pady=(15, 10))

        ttk.Label(header_frame, text="Audio Cascader", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header_frame, text="Real-Time Speech-to-Speech Voice Converter • Indian English to American Accent", style="SubHeader.TLabel").pack(anchor="w")

        # Status & Metrics Bar
        self.status_bar = ttk.Frame(self, style="Card.TFrame")
        self.status_bar.pack(fill="x", padx=20, pady=(0, 10))
        
        self.lbl_status = ttk.Label(self.status_bar, text="Status: Initializing models on GPU...", style="Card.TLabel", font=("Segoe UI", 10, "italic"))
        self.lbl_status.pack(side="left", padx=12, pady=8)

        self.lbl_metrics = ttk.Label(self.status_bar, text="STT: -- ms | TTFB: -- ms | Glass-to-Glass: -- ms", style="Card.TLabel", foreground="#38bdf8")
        self.lbl_metrics.pack(side="right", padx=12, pady=8)

        # 2. Main Tabbed Notebook
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=20, pady=5)

        # Tab 1: Live Mic Mode
        self.tab_live = ttk.Frame(notebook)
        notebook.add(self.tab_live, text="  🎙️ Live Mic Mode  ")
        self._build_live_tab()

        # Tab 2: File Conversion Mode
        self.tab_file = ttk.Frame(notebook)
        notebook.add(self.tab_file, text="  📁 File Mode  ")
        self._build_file_tab()

        # 3. Bottom Transcripts Feed
        feed_frame = ttk.Frame(self, style="Card.TFrame")
        feed_frame.pack(fill="both", expand=True, padx=20, pady=(5, 15))

        lbl_feed_title = ttk.Label(feed_frame, text="Live Transcription & Output Stream Feed", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        lbl_feed_title.pack(anchor="w", padx=12, pady=(8, 4))

        self.txt_feed = tk.Text(feed_frame, bg="#141416", fg="#e2e8f0", insertbackground="#38bdf8", relief="flat", font=("Consolas", 10), wrap="word")
        self.txt_feed.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    def _build_live_tab(self):
        card = ttk.Frame(self.tab_live, style="Card.TFrame")
        card.pack(fill="x", padx=10, pady=10)

        # Device selection row
        dev_frame = ttk.Frame(card, style="Card.TFrame")
        dev_frame.pack(fill="x", padx=12, pady=10)

        ttk.Label(dev_frame, text="Microphone:", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.combo_mic = ttk.Combobox(dev_frame, state="readonly", width=38)
        self.combo_mic.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ttk.Label(dev_frame, text="Speaker:", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.combo_speaker = ttk.Combobox(dev_frame, state="readonly", width=38)
        self.combo_speaker.grid(row=0, column=3, sticky="w")

        # VU Meters row
        vu_frame = ttk.Frame(card, style="Card.TFrame")
        vu_frame.pack(fill="x", padx=12, pady=(0, 10))

        ttk.Label(vu_frame, text="Mic Level:", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.prog_mic = ttk.Progressbar(vu_frame, length=240, mode="determinate")
        self.prog_mic.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ttk.Label(vu_frame, text="Out Level:", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.prog_out = ttk.Progressbar(vu_frame, length=240, mode="determinate")
        self.prog_out.grid(row=0, column=3, sticky="w")

        # Control buttons row
        btn_frame = ttk.Frame(card, style="Card.TFrame")
        btn_frame.pack(fill="x", padx=12, pady=(0, 12))

        self.btn_live_toggle = ttk.Button(btn_frame, text="🎙️ Start Live Conversion", style="Primary.TButton", command=self._toggle_live_conversion)
        self.btn_live_toggle.pack(side="left", padx=(0, 10))

        ttk.Label(btn_frame, text="Conditioned Voice: American Master (embeddings/american_accent.pt)", style="Card.TLabel", foreground="#10b981").pack(side="left")

    def _build_file_tab(self):
        card = ttk.Frame(self.tab_file, style="Card.TFrame")
        card.pack(fill="x", padx=10, pady=10)

        # File picker row
        file_frame = ttk.Frame(card, style="Card.TFrame")
        file_frame.pack(fill="x", padx=12, pady=12)

        ttk.Label(file_frame, text="Select Audio File (.wav, .m4a, .mp3):", style="Card.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        
        self.ent_filepath = tk.Entry(file_frame, bg="#141416", fg="#f3f4f6", insertbackground="#38bdf8", relief="flat", font=("Segoe UI", 10), width=65)
        self.ent_filepath.grid(row=1, column=0, sticky="w", padx=(0, 8), ipady=4)

        btn_browse = ttk.Button(file_frame, text="Browse...", command=self._browse_file)
        btn_browse.grid(row=1, column=1, sticky="w")

        # Action buttons row
        btn_row = ttk.Frame(card, style="Card.TFrame")
        btn_row.pack(fill="x", padx=12, pady=(0, 12))

        self.btn_convert = ttk.Button(btn_row, text="⚡ Convert to American Accent", style="Success.TButton", command=self._start_file_conversion)
        self.btn_convert.pack(side="left", padx=(0, 10))

        self.btn_play_result = ttk.Button(btn_row, text="🔊 Play Output Audio", command=self._play_output_file, state="disabled")
        self.btn_play_result.pack(side="left")

    def _populate_audio_devices(self):
        try:
            in_devs, out_devs = list_audio_devices()
            self.input_devices = in_devs
            self.output_devices = out_devs

            mic_names = [f"[{d['index']}] {d['name']}" for d in in_devs]
            spk_names = [f"[{d['index']}] {d['name']}" for d in out_devs]

            self.combo_mic["values"] = mic_names
            self.combo_speaker["values"] = spk_names

            for idx, d in enumerate(in_devs):
                if d["is_default_in"]:
                    self.combo_mic.current(idx)
                    break
            if not self.combo_mic.get() and mic_names:
                self.combo_mic.current(0)

            for idx, d in enumerate(out_devs):
                if d["is_default_out"]:
                    self.combo_speaker.current(idx)
                    break
            if not self.combo_speaker.get() and spk_names:
                self.combo_speaker.current(0)
        except Exception as e:
            logger.warning("Could not enumerate audio devices: %s", e)

    def _preload_models_background(self):
        """Warm up Faster-Whisper and Pocket TTS on GPU."""
        try:
            self.ui_update_queue.put(("status", "Loading Faster-Whisper Turbo on GPU..."))
            self.stt_engine = STTEngine()
            self.stt_engine.load()

            self.ui_update_queue.put(("status", "Loading Kyutai Pocket TTS on GPU..."))
            self.tts_engine = TTSEngine()
            self.tts_engine.load()

            self.ui_update_queue.put(("status", "System ready. Models primed on GPU."))
        except Exception as e:
            self.ui_update_queue.put(("status", f"Error loading models: {e}"))
            logger.error("Preload error: %s", e)

    def _browse_file(self):
        f = filedialog.askopenfilename(
            title="Select Spoken Audio File",
            filetypes=[("Audio Files", "*.wav *.m4a *.mp3 *.flac *.ogg"), ("All Files", "*.*")],
        )
        if f:
            self.ent_filepath.delete(0, tk.END)
            self.ent_filepath.insert(0, f)

    def _start_file_conversion(self):
        f_path = self.ent_filepath.get().strip()
        if not f_path or not Path(f_path).exists():
            messagebox.showerror("File Error", "Please select a valid audio file.")
            return

        self.btn_convert.config(state="disabled")
        self.btn_play_result.config(state="disabled")
        self.lbl_status.config(text=f"Converting {Path(f_path).name}...")

        def _worker():
            try:
                from pipeline import run_pipeline
                out = run_pipeline(f_path)
                self.last_converted_file = out
                self.ui_update_queue.put(("file_done", out))
            except Exception as e:
                self.ui_update_queue.put(("file_error", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _play_output_file(self):
        if self.last_converted_file and self.last_converted_file.exists():
            try:
                import sounddevice as sd
                data, sr = sf.read(str(self.last_converted_file), dtype="float32")
                sd.play(data, sr)
            except Exception as e:
                messagebox.showerror("Playback Error", f"Failed to play audio: {e}")

    def _toggle_live_conversion(self):
        if not self.is_live:
            self._start_live()
        else:
            self._stop_live()

    def _start_live(self):
        self.is_live = True
        self.live_stop_event.clear()
        self.btn_live_toggle.config(text="⏹️ Stop Live Conversion", style="Danger.TButton")
        self.lbl_status.config(text="Live conversion active • Listening for Indian English speech...")

        mic_sel = self.combo_mic.current()
        spk_sel = self.combo_speaker.current()
        mic_idx = self.input_devices[mic_sel]["index"] if mic_sel >= 0 else None
        spk_idx = self.output_devices[spk_sel]["index"] if spk_sel >= 0 else None

        self.segmenter = StreamingVADSegmenter(speech_threshold=0.5, min_silence_duration_ms=300, max_speech_duration_s=3.5)
        self.capture = AudioCapture(device_index=mic_idx, vu_callback=self._on_mic_vu)
        self.player = StreamingAudioPlayer(sample_rate=24000, device_index=spk_idx, vu_callback=self._on_out_vu)

        self.player.start()
        self.capture.start()

        # Background worker threads
        threading.Thread(target=self._live_vad_worker, daemon=True).start()
        threading.Thread(target=self._live_stt_worker, daemon=True).start()
        threading.Thread(target=self._live_tts_worker, daemon=True).start()

    def _stop_live(self):
        self.is_live = False
        self.live_stop_event.set()
        if self.capture:
            self.capture.stop()
        if self.player:
            self.player.stop()

        self.prog_mic["value"] = 0
        self.prog_out["value"] = 0
        self.btn_live_toggle.config(text="🎙️ Start Live Conversion", style="Primary.TButton")
        self.lbl_status.config(text="Live conversion stopped.")

    def _on_mic_vu(self, level: float):
        val = min(100, int(level * 400))
        self.ui_update_queue.put(("vu_mic", val))

    def _on_out_vu(self, level: float):
        val = min(100, int(level * 400))
        self.ui_update_queue.put(("vu_out", val))

    def _live_vad_worker(self):
        while not self.live_stop_event.is_set():
            chunk = self.capture.get_chunk(timeout=0.05)
            if chunk is None:
                continue
            res = self.segmenter.process_chunk(chunk)
            if res is not None:
                speech_audio, start_t, end_t = res
                try:
                    self.stt_queue.put_nowait((speech_audio, start_t, end_t))
                except queue.Full:
                    pass

    def _live_stt_worker(self):
        while not self.live_stop_event.is_set():
            try:
                audio_seg, start_t, speech_end_t = self.stt_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            res = self.stt_engine.transcribe(audio_seg, language="en", beam_size=1)
            stt_latency = (time.perf_counter() - t0) * 1000.0
            text = res.get("text", "").strip()

            cleaned = "".join(c for c in text if c.isalnum() or c.isspace()).strip()
            if not cleaned or len(cleaned) < 2:
                continue

            # Log transcription in feed
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.ui_update_queue.put(("feed", f"[{timestamp}] [Indian English Input]: {text} (STT: {stt_latency:.0f}ms)\n"))

            # Send chunks to TTS queue
            chunks = chunk_text(text, max_words=10)
            for c in chunks:
                try:
                    self.tts_queue.put_nowait((c, speech_end_t, stt_latency))
                except queue.Full:
                    pass

    def _live_tts_worker(self):
        voice_state = self.tts_engine.get_or_create_voice_state()
        while not self.live_stop_event.is_set():
            try:
                clause, speech_end_t, stt_latency = self.tts_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            first_bit = False

            try:
                for chunk_np in self.tts_engine.synthesize_stream(clause, voice_state):
                    if not first_bit:
                        first_bit = True
                        ttfb = (time.perf_counter() - t0) * 1000.0
                        total_g2g = (time.perf_counter() - speech_end_t) * 1000.0
                        self.ui_update_queue.put(("metrics", (stt_latency, ttfb, total_g2g)))

                    self.player.push_audio(chunk_np)

                timestamp = datetime.now().strftime("%H:%M:%S")
                self.ui_update_queue.put(("feed", f"[{timestamp}]  -> [American Accent Output]: {clause}\n"))
            except Exception as e:
                logger.error("TTS generation error: %s", e)

    def _process_ui_queue(self):
        while not self.ui_update_queue.empty():
            msg_type, val = self.ui_update_queue.get_nowait()
            if msg_type == "status":
                self.lbl_status.config(text=f"Status: {val}")
            elif msg_type == "metrics":
                stt_ms, ttfb_ms, g2g_ms = val
                self.lbl_metrics.config(text=f"STT: {stt_ms:.0f} ms | TTFB: {ttfb_ms:.0f} ms | Glass-to-Glass: {g2g_ms:.0f} ms")
            elif msg_type == "feed":
                self.txt_feed.insert(tk.END, val)
                self.txt_feed.see(tk.END)
            elif msg_type == "vu_mic":
                self.prog_mic["value"] = val
            elif msg_type == "vu_out":
                self.prog_out["value"] = val
            elif msg_type == "file_done":
                self.btn_convert.config(state="normal")
                self.btn_play_result.config(state="normal")
                self.lbl_status.config(text=f"Successfully converted to {val.name}!")
                self.txt_feed.insert(tk.END, f"\n[File Mode Complete] Saved output to: {val}\n")
                self.txt_feed.see(tk.END)
            elif msg_type == "file_error":
                self.btn_convert.config(state="normal")
                self.lbl_status.config(text="Conversion error.")
                messagebox.showerror("Error", f"Conversion failed: {val}")

        self.after(50, self._process_ui_queue)


def main():
    app = AudioCascaderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
