#!/usr/bin/env python3
"""S2S Voice & Accent Studio - Desktop GUI.

A graphical interface for the Speech-to-Speech (S2S) Voice & Accent
Conversion Pipeline:
  Audio Input (.m4a / recording) -> Faster-Whisper (STT) -> Kyutai Pocket TTS (TTS)
"""

from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import List, Optional

# Auto-redirect to dedicated venv if invoked with system Python
ROOT = Path(__file__).resolve().parents[1]
TARGET_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not TARGET_PYTHON.exists():
    TARGET_PYTHON = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU" / ".venv" / "Scripts" / "python.exe"

if TARGET_PYTHON.exists():
    curr_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(str(TARGET_PYTHON.resolve()))
    if curr_exe != target_exe:
        result = subprocess.run([str(TARGET_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:], check=False)
        sys.exit(result.returncode)

# Ensure clean UTF-8 terminal output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import psutil
except ImportError:
    psutil = None

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.utils.audio_convert import convert_to_wav
from models.stt.faster_whisper.adapter import FasterWhisperAdapter
import soundfile as sf
import numpy as np
import torch
from pocket_tts.models.tts_model import TTSModel

DEFAULT_RECORDINGS_DIR = Path(
    os.getenv("SOUND_RECORDINGS_DIR", r"C:\Users\tripa\Documents\Sound Recordings")
)
DEFAULT_REFERENCE_VOICE = (
    (ROOT / "Testinput" / "american accent.wav")
    if (ROOT / "Testinput" / "american accent.wav").exists()
    else (ROOT / "references" / "american accent.wav")
)
DEFAULT_FALLBACK_AUDIO = (
    (ROOT / "Testinput" / "Anoop voice line.wav")
    if (ROOT / "Testinput" / "Anoop voice line.wav").exists()
    else (ROOT / "test_inputs" / "Anoop voice line.wav")
)
DEFAULT_OUTPUT_DIR = (
    (ROOT / "Output")
    if (ROOT / "Output").exists()
    else (ROOT / "outputs")
)


class S2SApp(tk.Tk):
    """Main Application Window for S2S Voice & Accent Studio."""

    def __init__(self):
        super().__init__()

        self.title("🎙️ S2S Voice & Accent Studio")
        self.geometry("1020x820")
        self.minsize(960, 700)

        # Apply dark theme styling
        self._configure_styles()

        self.work_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False

        self.latest_output_file: Optional[Path] = None
        self.recordings_cache: List[Path] = []

        # Latency & Telemetry state
        self.show_metrics_var = tk.BooleanVar(value=True)
        self._stop_telemetry = threading.Event()
        self._telemetry_lock = threading.Lock()
        self.latest_telemetry = {}
        self.run_cpu_samples: List[float] = []
        self.run_gpu_samples: List[float] = []
        self.peak_telemetry_cpu = 0.0
        self.peak_telemetry_gpu = 0.0
        self.peak_telemetry_ram = 0.0
        self.peak_telemetry_vram = 0.0
        self.live_engine = None
        self.is_live_streaming = False

        self._build_ui()
        self._refresh_recordings()
        self._check_queue()


        # Start background telemetry thread and schedule UI updates
        self.telemetry_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
        self.telemetry_thread.start()
        self._schedule_telemetry_update()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _configure_styles(self):
        """Set up dark theme palette and TTK widget styles."""
        self.bg_color = "#181825"       # Dark charcoal
        self.card_bg = "#1e1e2e"        # Deep slate
        self.card_border = "#313244"    # Subtle border
        self.fg_color = "#cdd6f4"       # Light text
        self.accent_color = "#89b4fa"   # Soft blue
        self.accent_hover = "#b4befe"
        self.success_color = "#a6e3a1"  # Mint green
        self.warn_color = "#f9e2af"     # Amber
        self.danger_color = "#f38ba8"   # Coral red
        self.entry_bg = "#11111b"       # Inner dark

        self.configure(bg=self.bg_color)

        style = ttk.Style(self)
        style.theme_use("clam")

        # Base configurations
        style.configure(".", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10))
        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.card_bg, relief="solid", borderwidth=1)
        style.configure("TLabelframe", background=self.card_bg, foreground=self.accent_color, font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=self.card_bg, foreground=self.accent_color, font=("Segoe UI", 10, "bold"))

        # Buttons
        style.configure(
            "Primary.TButton",
            background=self.accent_color,
            foreground="#11111b",
            font=("Segoe UI", 11, "bold"),
            padding=(16, 8),
            borderwidth=0,
        )
        style.map("Primary.TButton", background=[("active", self.accent_hover), ("disabled", "#45475a")])

        style.configure(
            "Secondary.TButton",
            background="#313244",
            foreground=self.fg_color,
            font=("Segoe UI", 9),
            padding=(10, 5),
            borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#45475a"), ("disabled", "#1e1e2e")])

        style.configure(
            "Success.TButton",
            background=self.success_color,
            foreground="#11111b",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
            borderwidth=0,
        )
        style.map("Success.TButton", background=[("active", "#94e2d5")])

        style.configure(
            "Danger.TButton",
            background=self.danger_color,
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
            borderwidth=0,
        )
        style.map("Danger.TButton", background=[("active", "#eba0ac")])


        style.configure("TCombobox", fieldbackground=self.entry_bg, background=self.card_bg, foreground=self.fg_color)
        style.map("TCombobox", fieldbackground=[("readonly", self.entry_bg)])

        style.configure("TProgressbar", background=self.accent_color, troughcolor="#313244")
        style.configure("CPU.Horizontal.TProgressbar", background=self.accent_color, troughcolor="#313244")
        style.configure("RAM.Horizontal.TProgressbar", background=self.success_color, troughcolor="#313244")
        style.configure("GPU.Horizontal.TProgressbar", background="#cba6f7", troughcolor="#313244")
        style.configure("VRAM.Horizontal.TProgressbar", background="#fab387", troughcolor="#313244")

    def _build_ui(self):
        """Build responsive two-column interface."""
        # Top Header Banner
        header = tk.Frame(self, bg=self.card_bg, height=65)
        header.pack(fill="x", padx=12, pady=(10, 6))

        title_label = tk.Label(
            header,
            text="🎙️ S2S Voice & Accent Studio",
            font=("Segoe UI", 16, "bold"),
            bg=self.card_bg,
            fg=self.accent_color,
        )
        title_label.pack(side="left", padx=16, pady=10)

        gpu_name = "CUDA Available (RTX 3050)" if torch.cuda.is_available() else "CPU Mode"
        badge = tk.Label(
            header,
            text=f"● {gpu_name}",
            font=("Segoe UI", 9, "bold"),
            bg="#313244",
            fg=self.success_color if torch.cuda.is_available() else self.warn_color,
            padx=10,
            pady=4,
        )
        badge.pack(side="right", padx=16)

        # Toggle option for latency and system usage
        chk_metrics = tk.Checkbutton(
            header,
            text="📊 Show Latency & System Usage",
            variable=self.show_metrics_var,
            command=self._on_toggle_metrics,
            bg=self.card_bg,
            fg=self.accent_color,
            selectcolor="#313244",
            activebackground=self.card_bg,
            activeforeground=self.accent_hover,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        chk_metrics.pack(side="right", padx=(0, 10))

        # Main Content Layout (Left: Inputs/Config, Right: Text & Outputs)
        main_pane = tk.PanedWindow(self, orient="horizontal", bg=self.bg_color, sashrelief="flat", sashwidth=4)
        main_pane.pack(fill="both", expand=True, padx=12, pady=6)

        left_frame = ttk.Frame(main_pane)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, minsize=440)
        main_pane.add(right_frame, minsize=440)

        # ==========================================
        # LEFT COLUMN: INPUTS & SETTINGS
        # ==========================================

        # --- Card 1: Input Audio Source ---
        in_group = ttk.LabelFrame(left_frame, text=" 1. Input Audio Source ")
        in_group.pack(fill="x", pady=6)

        self.input_mode = tk.StringVar(value="newest")
        rb_frame = tk.Frame(in_group, bg=self.card_bg)
        rb_frame.pack(fill="x", padx=10, pady=4)

        tk.Radiobutton(
            rb_frame,
            text="Newest Recording",
            variable=self.input_mode,
            value="newest",
            bg=self.card_bg,
            fg=self.fg_color,
            selectcolor="#313244",
            activebackground=self.card_bg,
            activeforeground=self.accent_color,
            command=self._on_input_mode_change,
        ).pack(side="left", padx=4)

        tk.Radiobutton(
            rb_frame,
            text="Pick Recording",
            variable=self.input_mode,
            value="list",
            bg=self.card_bg,
            fg=self.fg_color,
            selectcolor="#313244",
            activebackground=self.card_bg,
            activeforeground=self.accent_color,
            command=self._on_input_mode_change,
        ).pack(side="left", padx=4)

        tk.Radiobutton(
            rb_frame,
            text="Custom File",
            variable=self.input_mode,
            value="custom",
            bg=self.card_bg,
            fg=self.fg_color,
            selectcolor="#313244",
            activebackground=self.card_bg,
            activeforeground=self.accent_color,
            command=self._on_input_mode_change,
        ).pack(side="left", padx=4)

        # Dropdown for Pick Recording
        self.rec_combo_var = tk.StringVar()
        self.rec_combo = ttk.Combobox(in_group, textvariable=self.rec_combo_var, state="readonly", width=42)
        self.rec_combo.pack(fill="x", padx=10, pady=4)

        # Custom File Picker Row
        custom_frame = tk.Frame(in_group, bg=self.card_bg)
        custom_frame.pack(fill="x", padx=10, pady=4)
        self.custom_path_var = tk.StringVar()
        self.custom_entry = tk.Entry(
            custom_frame,
            textvariable=self.custom_path_var,
            bg=self.entry_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            relief="flat",
        )
        self.custom_entry.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=3)
        self.btn_browse_in = ttk.Button(custom_frame, text="Browse...", style="Secondary.TButton", command=self._browse_custom_audio)
        self.btn_browse_in.pack(side="right")

        # Status & Preview Row
        in_action_row = tk.Frame(in_group, bg=self.card_bg)
        in_action_row.pack(fill="x", padx=10, pady=(2, 8))
        self.lbl_detected_input = tk.Label(
            in_action_row, text="Detected: Scanning...", bg=self.card_bg, fg="#a6adc8", font=("Segoe UI", 9)
        )
        self.lbl_detected_input.pack(side="left")
        ttk.Button(in_action_row, text="🔄 Refresh", style="Secondary.TButton", command=self._refresh_recordings).pack(side="right", padx=(4, 0))
        ttk.Button(in_action_row, text="▶ Preview Input", style="Secondary.TButton", command=self._preview_input_audio).pack(side="right")

        # --- Card 2: STT Options ---
        stt_group = ttk.LabelFrame(left_frame, text=" 2. Speech-to-Text (STT) ")
        stt_group.pack(fill="x", pady=6)

        stt_grid = tk.Frame(stt_group, bg=self.card_bg)
        stt_grid.pack(fill="x", padx=10, pady=6)

        tk.Label(stt_grid, text="STT Engine:", bg=self.card_bg, fg=self.fg_color).grid(row=0, column=0, sticky="w", pady=4)
        self.stt_model_var = tk.StringVar(value="Faster-Whisper (tiny, Fastest CUDA)")
        self.stt_combo = ttk.Combobox(
            stt_grid,
            textvariable=self.stt_model_var,
            values=[
                "Faster-Whisper (tiny, Fastest CUDA)",
                "Faster-Whisper (turbo, High Accuracy)",
                "Sherpa-ONNX Zipformer (20M Streaming)",
                "Useful Sensors Moonshine (ONNX CPU)",
                "NVIDIA Nemotron 3.5 (NeMo ASR)",
                "NVIDIA Parakeet (NeMo ASR)",
            ],
            state="readonly",
            width=32,
        )
        self.stt_combo.grid(row=0, column=1, sticky="w", padx=8, pady=4)

        tk.Label(stt_grid, text="Inference Device:", bg=self.card_bg, fg=self.fg_color).grid(row=1, column=0, sticky="w", pady=4)
        self.device_var = tk.StringVar(value="auto")
        self.device_combo = ttk.Combobox(
            stt_grid, textvariable=self.device_var, values=["auto", "cuda", "cpu"], state="readonly", width=12
        )
        self.device_combo.grid(row=1, column=1, sticky="w", padx=8, pady=4)

        # --- Card 3: Target Voice & Accent Cloning ---
        tts_group = ttk.LabelFrame(left_frame, text=" 3. Target Voice & Accent Cloning ")
        tts_group.pack(fill="x", pady=6)

        tts_grid = tk.Frame(tts_group, bg=self.card_bg)
        tts_grid.pack(fill="x", padx=10, pady=6)

        # TTS Engine Selector
        tk.Label(tts_grid, text="TTS Engine:", bg=self.card_bg, fg=self.fg_color).grid(row=0, column=0, sticky="w", pady=4)
        self.tts_engine_var = tk.StringVar(value="Pocket TTS (Kyutai FlowLM Voice Clone)")
        self.tts_engine_combo = ttk.Combobox(
            tts_grid,
            textvariable=self.tts_engine_var,
            values=[
                "Pocket TTS (Kyutai FlowLM Voice Clone)",
                "Kokoro v1.0 (82M Ultra-Natural)",
                "Piper (Fast Offline CPU)",
                "MeloTTS (Multi-Accent English)",
                "StyleTTS 2 (Diffusion Adaptation)",
            ],
            state="readonly",
            width=32,
        )
        self.tts_engine_combo.grid(row=0, column=1, sticky="w", padx=8, pady=4)
        self.tts_engine_combo.bind("<<ComboboxSelected>>", self._on_tts_engine_selected)

        tk.Label(tts_grid, text="Reference Voice:", bg=self.card_bg, fg=self.fg_color).grid(row=1, column=0, sticky="w", pady=4)
        self.voice_preset_var = tk.StringVar(value="american accent.wav (Default)")
        self.voice_presets = [
            "american accent.wav (Default)",
            "Anoop voice line.wav",
            "Built-in: alba",
            "Built-in: cosette",
            "Built-in: marius",
            "Built-in: javert",
            "Built-in: fantine",
            "Custom WAV File...",
        ]
        self.voice_combo = ttk.Combobox(tts_grid, textvariable=self.voice_preset_var, values=self.voice_presets, state="readonly", width=28)
        self.voice_combo.grid(row=1, column=1, sticky="w", padx=8, pady=4)
        self.voice_combo.bind("<<ComboboxSelected>>", self._on_voice_selected)

        self.custom_ref_path = tk.StringVar(value="")
        self.btn_browse_ref = ttk.Button(tts_grid, text="Browse...", style="Secondary.TButton", command=self._browse_custom_ref)
        self.btn_browse_ref.grid(row=1, column=2, padx=4)

        # Container for Engine-specific tuning parameters
        self.tuning_container = tk.Frame(tts_grid, bg=self.card_bg)
        self.tuning_container.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)

        # Pocket TTS Tuning Frame
        self.frame_pocket_tuning = tk.Frame(self.tuning_container, bg=self.card_bg)
        tk.Label(self.frame_pocket_tuning, text="LSD Steps (Clarity):", bg=self.card_bg, fg=self.fg_color).grid(row=0, column=0, sticky="w", pady=2)
        self.lsd_steps_var = tk.IntVar(value=3)
        self.lsd_spin = tk.Spinbox(
            self.frame_pocket_tuning, from_=1, to=6, textvariable=self.lsd_steps_var, width=6, bg=self.entry_bg, fg=self.fg_color, relief="flat"
        )
        self.lsd_spin.grid(row=0, column=1, sticky="w", padx=8, pady=2)

        tk.Label(self.frame_pocket_tuning, text="Temperature:", bg=self.card_bg, fg=self.fg_color).grid(row=1, column=0, sticky="w", pady=2)
        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_spin = tk.Spinbox(
            self.frame_pocket_tuning, from_=0.1, to=1.0, increment=0.05, textvariable=self.temp_var, width=6, bg=self.entry_bg, fg=self.fg_color, relief="flat"
        )
        self.temp_spin.grid(row=1, column=1, sticky="w", padx=8, pady=2)
        self.frame_pocket_tuning.pack(fill="x")

        # StyleTTS 2 Tuning Frame
        self.frame_styletts_tuning = tk.Frame(self.tuning_container, bg=self.card_bg)
        tk.Label(self.frame_styletts_tuning, text="Diffusion Steps:", bg=self.card_bg, fg=self.fg_color).grid(row=0, column=0, sticky="w", pady=2)
        self.diff_steps_var = tk.IntVar(value=5)
        self.diff_spin = tk.Spinbox(
            self.frame_styletts_tuning, from_=1, to=15, textvariable=self.diff_steps_var, width=6, bg=self.entry_bg, fg=self.fg_color, relief="flat"
        )
        self.diff_spin.grid(row=0, column=1, sticky="w", padx=8, pady=2)

        tk.Label(self.frame_styletts_tuning, text="Timbre Alpha:", bg=self.card_bg, fg=self.fg_color).grid(row=1, column=0, sticky="w", pady=2)
        self.alpha_var = tk.DoubleVar(value=0.3)
        self.alpha_spin = tk.Spinbox(
            self.frame_styletts_tuning, from_=0.0, to=1.0, increment=0.05, textvariable=self.alpha_var, width=6, bg=self.entry_bg, fg=self.fg_color, relief="flat"
        )
        self.alpha_spin.grid(row=1, column=1, sticky="w", padx=8, pady=2)

        tk.Label(self.frame_styletts_tuning, text="Prosody Beta:", bg=self.card_bg, fg=self.fg_color).grid(row=2, column=0, sticky="w", pady=2)
        self.beta_var = tk.DoubleVar(value=0.7)
        self.beta_spin = tk.Spinbox(
            self.frame_styletts_tuning, from_=0.0, to=1.0, increment=0.05, textvariable=self.beta_var, width=6, bg=self.entry_bg, fg=self.fg_color, relief="flat"
        )
        self.beta_spin.grid(row=2, column=1, sticky="w", padx=8, pady=2)

        ttk.Button(tts_grid, text="▶ Preview Voice", style="Secondary.TButton", command=self._preview_ref_audio).grid(row=2, column=2, padx=4, sticky="n")

        # --- Card 4: Action Buttons ---
        act_frame = tk.Frame(left_frame, bg=self.bg_color)
        act_frame.pack(fill="x", pady=10)

        self.btn_run_all = ttk.Button(act_frame, text="🚀 Convert Speech (STT + TTS)", style="Primary.TButton", command=self._start_full_pipeline)
        self.btn_run_all.pack(fill="x", pady=4)

        split_btn_row = tk.Frame(act_frame, bg=self.bg_color)
        split_btn_row.pack(fill="x", pady=2)
        self.btn_stt_only = ttk.Button(split_btn_row, text="📝 Transcribe Only (STT)", style="Secondary.TButton", command=self._start_stt_only)
        self.btn_stt_only.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_tts_only = ttk.Button(split_btn_row, text="🗣️ Synthesize Text (TTS)", style="Secondary.TButton", command=self._start_tts_only)
        self.btn_tts_only.pack(side="right", fill="x", expand=True, padx=(4, 0))

        self.btn_live_stream = ttk.Button(
            act_frame,
            text="🎙️ Live Speech-to-Speech (Streaming Mic -> Voice Clone)",
            style="Success.TButton",
            command=self._toggle_live_streaming,
        )
        self.btn_live_stream.pack(fill="x", pady=(6, 2))


        # ==========================================
        # RIGHT COLUMN: TEXT, LOGS & OUTPUT
        # ==========================================

        # --- Card 5: Transcribed Text Box ---
        self.text_group = ttk.LabelFrame(right_frame, text=" Transcribed Speech (Editable) ")
        self.text_group.pack(fill="x", pady=(6, 4))

        self.transcribed_text = tk.Text(
            self.text_group,
            height=4,
            bg=self.entry_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            font=("Segoe UI", 10),
            relief="flat",
            wrap="word",
        )
        self.transcribed_text.pack(fill="both", expand=True, padx=8, pady=6)

        # --- Card 6: Latency & System Usage Monitor ---
        self.metrics_group = ttk.LabelFrame(right_frame, text=" ⚡ Latency & System Usage Monitor ")
        self.metrics_group.pack(fill="x", pady=4)

        # 1. Live System Telemetry Sub-panel
        telemetry_box = tk.Frame(self.metrics_group, bg=self.card_bg)
        telemetry_box.pack(fill="x", padx=10, pady=(6, 3))

        t_header_row = tk.Frame(telemetry_box, bg=self.card_bg)
        t_header_row.pack(fill="x", pady=(0, 3))
        tk.Label(
            t_header_row,
            text="● LIVE SYSTEM TELEMETRY",
            bg=self.card_bg,
            fg=self.accent_color,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")
        self.lbl_telemetry_status = tk.Label(
            t_header_row,
            text="Polling: 1.5s",
            bg=self.card_bg,
            fg="#6c7086",
            font=("Segoe UI", 8),
        )
        self.lbl_telemetry_status.pack(side="right")

        # Telemetry Grid (CPU, RAM, GPU, VRAM)
        t_grid = tk.Frame(telemetry_box, bg=self.card_bg)
        t_grid.pack(fill="x", pady=2)
        t_grid.columnconfigure(1, weight=1)
        t_grid.columnconfigure(3, weight=1)

        # Row 0: CPU & GPU
        self.lbl_cpu_val = tk.Label(t_grid, text="CPU: 0%", bg=self.card_bg, fg=self.fg_color, font=("Segoe UI", 8, "bold"), width=16, anchor="w")
        self.lbl_cpu_val.grid(row=0, column=0, sticky="w", pady=2)
        self.pb_cpu = ttk.Progressbar(t_grid, style="CPU.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.pb_cpu.grid(row=0, column=1, sticky="ew", padx=(4, 12), pady=2)

        self.lbl_gpu_val = tk.Label(t_grid, text="GPU: 0%", bg=self.card_bg, fg=self.fg_color, font=("Segoe UI", 8, "bold"), width=16, anchor="w")
        self.lbl_gpu_val.grid(row=0, column=2, sticky="w", pady=2)
        self.pb_gpu = ttk.Progressbar(t_grid, style="GPU.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.pb_gpu.grid(row=0, column=3, sticky="ew", padx=(4, 0), pady=2)

        # Row 1: RAM & VRAM
        self.lbl_ram_val = tk.Label(t_grid, text="RAM: 0.0/0.0 GB (0%)", bg=self.card_bg, fg=self.fg_color, font=("Segoe UI", 8, "bold"), width=16, anchor="w")
        self.lbl_ram_val.grid(row=1, column=0, sticky="w", pady=2)
        self.pb_ram = ttk.Progressbar(t_grid, style="RAM.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.pb_ram.grid(row=1, column=1, sticky="ew", padx=(4, 12), pady=2)

        self.lbl_vram_val = tk.Label(t_grid, text="VRAM: 0/0 MB (0%)", bg=self.card_bg, fg=self.fg_color, font=("Segoe UI", 8, "bold"), width=16, anchor="w")
        self.lbl_vram_val.grid(row=1, column=2, sticky="w", pady=2)
        self.pb_vram = ttk.Progressbar(t_grid, style="VRAM.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.pb_vram.grid(row=1, column=3, sticky="ew", padx=(4, 0), pady=2)

        # Divider
        tk.Frame(self.metrics_group, bg=self.card_border, height=1).pack(fill="x", padx=10, pady=3)

        # 2. Pipeline Latency Breakdown Sub-panel
        bench_box = tk.Frame(self.metrics_group, bg=self.card_bg)
        bench_box.pack(fill="x", padx=10, pady=(2, 6))

        tk.Label(
            bench_box,
            text="⏱️ PIPELINE LATENCY & BENCHMARKS",
            bg=self.card_bg,
            fg=self.accent_color,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 3))

        tiles_frame = tk.Frame(bench_box, bg=self.card_bg)
        tiles_frame.pack(fill="x")
        tiles_frame.columnconfigure((0, 1, 2), weight=1, uniform="tile")

        def make_tile(parent, row, col, title):
            card = tk.Frame(parent, bg=self.entry_bg, relief="flat", highlightbackground=self.card_border, highlightthickness=1)
            card.grid(row=row, column=col, sticky="nsew", padx=3, pady=2)
            tk.Label(card, text=title, bg=self.entry_bg, fg="#a6adc8", font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=6, pady=(3, 1))
            val_lbl = tk.Label(card, text="--", bg=self.entry_bg, fg=self.fg_color, font=("Segoe UI", 9, "bold"))
            val_lbl.pack(anchor="w", padx=6, pady=1)
            sub_lbl = tk.Label(card, text="Awaiting run", bg=self.entry_bg, fg="#6c7086", font=("Segoe UI", 7))
            sub_lbl.pack(anchor="w", padx=6, pady=(1, 3))
            return val_lbl, sub_lbl

        # Row 0: Pipeline Stage Totals
        self.tile1_val, self.tile1_sub = make_tile(tiles_frame, 0, 0, "Ingest / Resample")
        self.tile2_val, self.tile2_sub = make_tile(tiles_frame, 0, 1, "Whisper STT (Total)")
        self.tile3_val, self.tile3_sub = make_tile(tiles_frame, 0, 2, "Pocket TTS (Total)")

        # Row 1: Conversational First Word & Time to First Bit (TTFB)
        self.tile_fw_val, self.tile_fw_sub = make_tile(tiles_frame, 1, 0, "🗣️ STT First Word")
        self.tile_fb_val, self.tile_fb_sub = make_tile(tiles_frame, 1, 1, "⚡ TTS First Bit (TTFB)")
        self.tile4_val, self.tile4_sub = make_tile(tiles_frame, 1, 2, "🎯 Total First Audio Bit")


        # --- Card 7: Output & Audio Player ---
        self.out_group = ttk.LabelFrame(right_frame, text=" Generated Cloned Output ")
        self.out_group.pack(fill="x", pady=4)

        out_info_frame = tk.Frame(self.out_group, bg=self.card_bg)
        out_info_frame.pack(fill="x", padx=10, pady=4)

        self.lbl_output_file = tk.Label(
            out_info_frame, text="No output generated yet.", bg=self.card_bg, fg="#a6adc8", anchor="w"
        )
        self.lbl_output_file.pack(fill="x", pady=1)

        self.lbl_metrics = tk.Label(
            out_info_frame, text="Latency Metrics: Ready", bg=self.card_bg, fg=self.accent_color, font=("Segoe UI", 9, "bold"), anchor="w"
        )
        self.lbl_metrics.pack(fill="x", pady=1)

        out_btn_row = tk.Frame(self.out_group, bg=self.card_bg)
        out_btn_row.pack(fill="x", padx=10, pady=(2, 6))
        self.btn_play_out = ttk.Button(out_btn_row, text="▶ Play Generated Audio", style="Success.TButton", command=self._play_output_audio)
        self.btn_play_out.pack(side="left", padx=(0, 6))
        self.btn_open_folder = ttk.Button(out_btn_row, text="📁 Open Output Folder", style="Secondary.TButton", command=self._open_output_folder)
        self.btn_open_folder.pack(side="left")

        # --- Card 8: Live Execution Logs ---
        self.log_group = ttk.LabelFrame(right_frame, text=" Pipeline Status & Logs ")
        self.log_group.pack(fill="both", expand=True, pady=4)

        self.progress_bar = ttk.Progressbar(self.log_group, mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=8, pady=(4, 3))

        self.log_text = tk.Text(
            self.log_group,
            height=5,
            bg=self.entry_bg,
            fg="#a6adc8",
            font=("Consolas", 9),
            relief="flat",
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        self.log_text.tag_config("STAGE", foreground=self.accent_color, font=("Consolas", 9, "bold"))
        self.log_text.tag_config("SUCCESS", foreground=self.success_color, font=("Consolas", 9, "bold"))
        self.log_text.tag_config("WARN", foreground=self.warn_color)
        self.log_text.tag_config("ERROR", foreground=self.danger_color, font=("Consolas", 9, "bold"))

        # --- Bottom Status Bar ---
        self.status_bar = tk.Frame(self, bg=self.card_bg, height=24)
        self.status_bar.pack(side="bottom", fill="x")
        self.lbl_status_telemetry = tk.Label(
            self.status_bar,
            text="⚡ Telemetry: Initializing...",
            bg=self.card_bg,
            fg="#a6adc8",
            font=("Segoe UI", 8),
            anchor="w",
            padx=12,
            pady=2,
        )
        self.lbl_status_telemetry.pack(side="left")

    def _on_toggle_metrics(self):
        """Show or hide the Latency & System Usage monitor panel."""
        if self.show_metrics_var.get():
            self.metrics_group.pack(fill="x", pady=4, after=self.text_group)
            self._update_telemetry_ui()
        else:
            self.metrics_group.pack_forget()
            self.lbl_status_telemetry.config(
                text="📊 Latency & System Usage Monitor: Hidden (Check 'Show Latency & System Usage' in header to display)"
            )

    def _on_window_close(self):
        """Clean up background threads and live streaming engine on window exit."""
        self._stop_telemetry.set()
        if self.live_engine is not None:
            try:
                self.live_engine.stop()
            except Exception:
                pass
            self.live_engine = None
        self.destroy()

    def _query_system_telemetry(self) -> dict:
        """Query CPU, RAM, GPU, and VRAM utilization safely."""
        cpu_pct = 0.0
        ram_used = 0.0
        ram_total = 0.0
        ram_pct = 0.0
        if psutil is not None:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                ram_used = vm.used / (1024 ** 3)
                ram_total = vm.total / (1024 ** 3)
                ram_pct = vm.percent
            except Exception:
                pass

        gpu_util = 0.0
        vram_used = 0.0
        vram_total = 0.0
        vram_pct = 0.0
        gpu_name = "N/A"
        try:
            creation_flags = 0x08000000 if sys.platform == "win32" else 0
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                timeout=0.8,
                creationflags=creation_flags,
            ).decode("utf-8").strip()
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 4:
                gpu_name = parts[0]
                gpu_util = float(parts[1])
                vram_used = float(parts[2])
                vram_total = float(parts[3])
                vram_pct = (vram_used / vram_total * 100.0) if vram_total > 0 else 0.0
        except Exception:
            if torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_used = torch.cuda.memory_allocated(0) / (1024 ** 2)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                    vram_pct = (vram_used / vram_total * 100.0) if vram_total > 0 else 0.0
                except Exception:
                    pass

        return {
            "cpu_pct": cpu_pct,
            "ram_used": ram_used,
            "ram_total": ram_total,
            "ram_pct": ram_pct,
            "gpu_name": gpu_name,
            "gpu_util": gpu_util,
            "vram_used": vram_used,
            "vram_total": vram_total,
            "vram_pct": vram_pct,
        }

    @staticmethod
    def _format_completion_summary(
        mode: str,
        t_resample: float,
        t_stt: float,
        stt_rtf: float,
        t_tts: float,
        tts_rtf: float,
        t_total: float,
        in_duration: float,
        out_duration: float,
        overall_rtf: float,
        cpu_peak: float,
        cpu_avg: float,
        ram_used: float,
        ram_peak: float,
        ram_total: float,
        gpu_peak: float,
        vram_peak: float,
        vram_total: float,
        gpu_name: str,
        stt_first_word_ms: float = 0.0,
        tts_first_bit_ms: float = 0.0,
        first_word: str = "",
    ) -> str:
        """Generate a clean, highly readable benchmark summary block."""
        vram_pct = (vram_peak / vram_total * 100.0) if vram_total > 0 else 0.0
        ram_pct = (ram_peak / ram_total * 100.0) if ram_total > 0 else 0.0

        lines = [
            "============================================================",
            f"📊 {mode.upper()} - RUN SUMMARY",
            "============================================================",
            "⏱️ LATENCY BREAKDOWN:",
        ]
        if t_resample > 0:
            lines.append(f"   • Resample / Ingest       : {t_resample:.2f}s ({t_resample * 1000:.0f} ms)")
        if t_stt > 0:
            lines.append(f"   • Faster-Whisper STT      : {t_stt:.2f}s ({t_stt * 1000:.0f} ms) | RTF: {stt_rtf:.2f}x")
        if stt_first_word_ms > 0:
            fw_label = f" ('{first_word}')" if first_word else ""
            lines.append(f"   • STT Time to First Word  : {stt_first_word_ms:.0f} ms{fw_label}")
        if t_tts > 0:
            lines.append(f"   • Kyutai Pocket TTS Clone : {t_tts:.2f}s ({t_tts * 1000:.0f} ms) | RTF: {tts_rtf:.2f}x")
        if tts_first_bit_ms > 0:
            lines.append(f"   • TTS Time to First Bit   : {tts_first_bit_ms:.0f} ms (First Audio Chunk)")
        if stt_first_word_ms > 0 and tts_first_bit_ms > 0:
            lines.append(f"   • Total First Audio Bit   : {stt_first_word_ms + tts_first_bit_ms:.0f} ms (Glass-to-First-Sound)")
        lines.append(f"   • Total Pipeline Latency  : {t_total:.2f}s ({t_total * 1000:.0f} ms) | RTF: {overall_rtf:.2f}x")
        if out_duration > 0:
            lines.append(f"   • Synthesized Audio       : {out_duration:.2f}s")
        elif in_duration > 0:
            lines.append(f"   • Source Audio Length     : {in_duration:.2f}s")

        lines.extend([
            "",
            "🖥️ SYSTEM RESOURCE USAGE:",
            f"   • CPU Utilization         : Peak {cpu_peak:.1f}% | Avg {cpu_avg:.1f}%",
            f"   • System RAM              : Peak {ram_peak:.2f} GB / {ram_total:.1f} GB ({ram_pct:.1f}%) [Used: {ram_used:.2f} GB]",
            f"   • GPU Compute Utilization : Peak {gpu_peak:.0f}% ({gpu_name})",
            f"   • GPU VRAM                : Peak {vram_peak:.0f} MB / {vram_total:.0f} MB ({vram_pct:.1f}%)",
            "   • GPU VRAM Isolation      : Enforced (STT evicted before TTS load)",
            "============================================================",
        ])
        return "\n".join(lines)

    def _telemetry_worker(self):
        """Background thread polling system stats without freezing UI."""
        while not self._stop_telemetry.is_set():
            try:
                stats = self._query_system_telemetry()
                with self._telemetry_lock:
                    self.latest_telemetry = stats
                    if self.is_running:
                        c_val = stats.get("cpu_pct", 0.0)
                        g_val = stats.get("gpu_util", 0.0)
                        r_val = stats.get("ram_used", 0.0)
                        vr_val = stats.get("vram_used", 0.0)

                        if c_val > 0:
                            self.run_cpu_samples.append(c_val)
                        if g_val > 0:
                            self.run_gpu_samples.append(g_val)

                        if c_val > self.peak_telemetry_cpu:
                            self.peak_telemetry_cpu = c_val
                        if g_val > self.peak_telemetry_gpu:
                            self.peak_telemetry_gpu = g_val
                        if r_val > self.peak_telemetry_ram:
                            self.peak_telemetry_ram = r_val
                        if vr_val > self.peak_telemetry_vram:
                            self.peak_telemetry_vram = vr_val
            except Exception:
                pass
            sleep_time = 0.25 if self.is_running else 1.2
            self._stop_telemetry.wait(sleep_time)

    def _schedule_telemetry_update(self):
        """Update live telemetry indicators periodically."""
        try:
            self._update_telemetry_ui()
        finally:
            self.after(1000, self._schedule_telemetry_update)

    def _update_telemetry_ui(self):
        """Refresh telemetry progressbars and labels on Tkinter thread."""
        with self._telemetry_lock:
            stats = dict(self.latest_telemetry)

        if not stats:
            return

        cpu = stats.get("cpu_pct", 0.0)
        ram_u = stats.get("ram_used", 0.0)
        ram_t = stats.get("ram_total", 0.0)
        ram_pct = stats.get("ram_pct", 0.0)
        gpu_u = stats.get("gpu_util", 0.0)
        vram_u = stats.get("vram_used", 0.0)
        vram_t = stats.get("vram_total", 0.0)
        vram_pct = stats.get("vram_pct", 0.0)

        if self.show_metrics_var.get():
            self.pb_cpu["value"] = cpu
            self.lbl_cpu_val.config(text=f"CPU: {cpu:.1f}%")

            self.pb_ram["value"] = ram_pct
            self.lbl_ram_val.config(text=f"RAM: {ram_u:.1f}/{ram_t:.1f}GB ({ram_pct:.0f}%)")

            self.pb_gpu["value"] = gpu_u
            self.lbl_gpu_val.config(text=f"GPU: {gpu_u:.0f}%")

            self.pb_vram["value"] = vram_pct
            self.lbl_vram_val.config(text=f"VRAM: {vram_u:.0f}/{vram_t:.0f}MB ({vram_pct:.0f}%)")

        status_text = (
            f"⚡ Telemetry: CPU: {cpu:.0f}% | RAM: {ram_u:.1f}/{ram_t:.1f} GB ({ram_pct:.0f}%) | "
            f"GPU: {gpu_u:.0f}% | VRAM: {vram_u:.0f}/{vram_t:.0f} MB ({vram_pct:.0f}%) | "
            f"VRAM Isolation: Safe"
        )
        self.lbl_status_telemetry.config(text=status_text)

    def _display_latency_metrics(self, data: dict):
        """Update latency benchmark tiles with pipeline timing results."""
        t_resample = data.get("resample_time", 0.0)
        t_stt = data.get("stt_time", 0.0)
        stt_rtf = data.get("stt_rtf", 0.0)
        t_tts = data.get("tts_time", 0.0)
        tts_rtf = data.get("tts_rtf", 0.0)
        t_total = data.get("total_time", 0.0)
        in_dur = data.get("in_duration", 0.0)
        out_dur = data.get("out_duration", 0.0)
        overall_rtf = data.get("overall_rtf", 0.0)
        peak_ram = data.get("peak_ram_gb", 0.0)
        peak_vram = data.get("peak_vram_mb", 0.0)

        # Tile 1: Preprocessing
        if t_resample > 0:
            self.tile1_val.config(text=f"{t_resample:.2f}s", fg=self.accent_color)
            self.tile1_sub.config(text=f"Input: {in_dur:.1f}s audio" if in_dur > 0 else "16kHz Resampled")
        else:
            self.tile1_val.config(text="Bypassed", fg="#a6adc8")
            self.tile1_sub.config(text="Native 16kHz WAV")

        # Tile 2: STT
        if t_stt > 0:
            self.tile2_val.config(text=f"{t_stt:.2f}s", fg=self.accent_color)
            self.tile2_sub.config(text=f"RTF: {stt_rtf:.2f}x (Faster)" if stt_rtf < 1 else f"RTF: {stt_rtf:.2f}x")
        else:
            self.tile2_val.config(text="N/A", fg="#a6adc8")
            self.tile2_sub.config(text="Direct text input")

        # Tile 3: TTS
        if t_tts > 0:
            self.tile3_val.config(text=f"{t_tts:.2f}s", fg=self.accent_color)
            self.tile3_sub.config(text=f"RTF: {tts_rtf:.2f}x (Faster)" if tts_rtf < 1 else f"RTF: {tts_rtf:.2f}x")
        else:
            self.tile3_val.config(text="N/A", fg="#a6adc8")
            self.tile3_sub.config(text="STT only run")

        cpu_peak = data.get("cpu_peak", 0.0)
        gpu_peak = data.get("gpu_peak", 0.0)

        # Row 1: Conversational First Word & Time to First Bit (TTFB)
        stt_fw = data.get("stt_first_word_ms", 0.0)
        tts_fb = data.get("tts_first_bit_ms", 0.0)
        tot_fb = data.get("total_first_bit_ms", (stt_fw + tts_fb) if (stt_fw > 0 and tts_fb > 0) else 0.0)
        first_word = data.get("first_word", "")

        # Tile: STT First Word
        if stt_fw > 0:
            fw_display = f'"{first_word}"' if first_word else "Emitted"
            self.tile_fw_val.config(text=f"{stt_fw:.0f} ms", fg=self.accent_color)
            self.tile_fw_sub.config(text=f"First word: {fw_display}")
        else:
            self.tile_fw_val.config(text="--", fg=self.fg_color)
            self.tile_fw_sub.config(text="Awaiting speech")

        # Tile: TTS First Bit (TTFB)
        if tts_fb > 0:
            self.tile_fb_val.config(text=f"{tts_fb:.0f} ms", fg=self.accent_color)
            self.tile_fb_sub.config(text="Text->1st audio bit")
        else:
            self.tile_fb_val.config(text="--", fg=self.fg_color)
            self.tile_fb_sub.config(text="Awaiting text")

        # Tile 4: Total First Audio Bit & Total Pipeline
        if tot_fb > 0:
            color = self.success_color if tot_fb < 800.0 else self.warn_color
            self.tile4_val.config(text=f"{tot_fb:.0f} ms", fg=color)
            rtf_note = f"RTF {overall_rtf:.2f}x" if overall_rtf > 0 else "Sub-second"
            self.tile4_sub.config(text=f"First-Bit Glass • {rtf_note}")
        elif t_total > 0:
            color = self.success_color if overall_rtf < 1.0 else self.warn_color
            rtf_note = "Real-Time" if overall_rtf < 1.0 else "Standard"
            self.tile4_val.config(text=f"{t_total:.2f}s ({overall_rtf:.2f}x)", fg=color)
            res_note = f"CPU {cpu_peak:.0f}% | GPU {gpu_peak:.0f}% | VRAM {peak_vram:.0f}MB"
            self.tile4_sub.config(text=f"{rtf_note} • {res_note}")
        else:
            self.tile4_val.config(text="--", fg=self.fg_color)
            self.tile4_sub.config(text="Awaiting run")

        # Update summary label in Output card
        fw_tag = f"First Word: {stt_fw:.0f}ms ('{first_word}') | " if stt_fw > 0 else ""
        fb_tag = f"TTS First Bit: {tts_fb:.0f}ms | Total First Audio: {tot_fb:.0f}ms | " if tts_fb > 0 else ""
        summary_txt = (
            f"⏱️ {fw_tag}{fb_tag}"
            f"Total: {t_total:.2f}s (RTF {overall_rtf:.2f}x) | "
            f"STT: {t_stt:.2f}s | TTS: {t_tts:.2f}s | "
            f"CPU: {cpu_peak:.0f}% | GPU: {gpu_peak:.0f}% | VRAM: {peak_vram:.0f}MB"
        )
        self.lbl_metrics.config(text=summary_txt)


    def _log(self, message: str, tag: Optional[str] = None):
        """Append a message safely to the log console."""
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")

    def _refresh_recordings(self):
        """Scan Windows Sound Recordings folder and update UI."""
        self.recordings_cache = []
        if DEFAULT_RECORDINGS_DIR.exists() and DEFAULT_RECORDINGS_DIR.is_dir():
            valid_exts = {".m4a", ".wav", ".mp3", ".ogg", ".flac", ".aac"}
            files = [f for f in DEFAULT_RECORDINGS_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            self.recordings_cache = files

        if self.recordings_cache:
            newest = self.recordings_cache[0]
            mtime = time.strftime("%I:%M %p", time.localtime(newest.stat().st_mtime))
            size_kb = newest.stat().st_size / 1024
            self.lbl_detected_input.config(
                text=f"Newest: {newest.name} ({size_kb:.0f} KB, {mtime})",
                fg=self.success_color,
            )
            # Update combobox
            display_list = [f"{f.name} ({f.stat().st_size // 1024} KB)" for f in self.recordings_cache[:20]]
            self.rec_combo["values"] = display_list
            self.rec_combo.current(0)
        else:
            self.lbl_detected_input.config(
                text=f"Fallback: {DEFAULT_FALLBACK_AUDIO.name}",
                fg=self.warn_color,
            )
            self.rec_combo["values"] = [DEFAULT_FALLBACK_AUDIO.name]
            self.rec_combo.set(DEFAULT_FALLBACK_AUDIO.name)

        self._on_input_mode_change()

    def _on_input_mode_change(self):
        """Toggle input field visibility depending on radio selection."""
        mode = self.input_mode.get()
        if mode == "newest":
            self.rec_combo.pack_forget()
            self.custom_entry.pack_forget()
            self.btn_browse_in.pack_forget()
        elif mode == "list":
            self.rec_combo.pack(fill="x", padx=10, pady=4)
            self.custom_entry.pack_forget()
            self.btn_browse_in.pack_forget()
        elif mode == "custom":
            self.rec_combo.pack_forget()
            self.custom_entry.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=3)
            self.btn_browse_in.pack(side="right")

    def _browse_custom_audio(self):
        """Browse file dialog for custom input audio."""
        f = filedialog.askopenfilename(
            title="Select Spoken Audio File",
            filetypes=[("Audio Files", "*.m4a *.wav *.mp3 *.ogg *.flac *.aac"), ("All Files", "*.*")],
        )
        if f:
            self.custom_path_var.set(f)
            self.lbl_detected_input.config(text=f"Selected: {Path(f).name}", fg=self.accent_color)

    def _on_tts_engine_selected(self, event=None):
        """Toggle tuning panels and button labels when TTS engine changes."""
        engine = self.tts_engine_var.get()
        if "StyleTTS 2" in engine:
            self.frame_pocket_tuning.pack_forget()
            self.frame_styletts_tuning.pack(fill="x")
            self.btn_tts_only.config(text="🗣️ Synthesize Text (StyleTTS 2)")
        elif "Pocket" in engine:
            self.frame_styletts_tuning.pack_forget()
            self.frame_pocket_tuning.pack(fill="x")
            self.btn_tts_only.config(text="🗣️ Synthesize Text (Pocket TTS)")
        elif "Kokoro" in engine:
            self.frame_styletts_tuning.pack_forget()
            self.frame_pocket_tuning.pack_forget()
            self.btn_tts_only.config(text="🗣️ Synthesize Text (Kokoro v1.0)")
        elif "Piper" in engine:
            self.frame_styletts_tuning.pack_forget()
            self.frame_pocket_tuning.pack_forget()
            self.btn_tts_only.config(text="🗣️ Synthesize Text (Piper)")
        elif "Melo" in engine:
            self.frame_styletts_tuning.pack_forget()
            self.frame_pocket_tuning.pack_forget()
            self.btn_tts_only.config(text="🗣️ Synthesize Text (MeloTTS)")
        else:
            self.frame_styletts_tuning.pack_forget()
            self.frame_pocket_tuning.pack_forget()
            self.btn_tts_only.config(text="🗣️ Synthesize Text (TTS)")

    def _on_voice_selected(self, event=None):
        """Handle voice preset selection."""
        val = self.voice_preset_var.get()
        if "Custom" in val:
            self.btn_browse_ref.grid()
        else:
            self.btn_browse_ref.grid_remove()

    def _browse_custom_ref(self):
        """Browse file dialog for custom reference voice."""
        f = filedialog.askopenfilename(
            title="Select Reference Voice (.wav)",
            filetypes=[("WAV Audio", "*.wav"), ("All Audio", "*.wav *.m4a *.mp3"), ("All Files", "*.*")],
        )
        if f:
            self.custom_ref_path.set(f)
            self.voice_preset_var.set(f"Custom: {Path(f).name}")

    def _resolve_input_audio(self) -> Path:
        """Resolve current input audio path."""
        mode = self.input_mode.get()
        if mode == "custom":
            path_str = self.custom_path_var.get().strip()
            if not path_str or not Path(path_str).is_file():
                raise FileNotFoundError("Please browse or select a valid custom input audio file.")
            return Path(path_str).resolve()
        elif mode == "list":
            idx = self.rec_combo.current()
            if idx >= 0 and idx < len(self.recordings_cache):
                return self.recordings_cache[idx]
            return DEFAULT_FALLBACK_AUDIO
        else:  # newest
            if self.recordings_cache:
                return self.recordings_cache[0]
            return DEFAULT_FALLBACK_AUDIO

    def _resolve_reference_voice(self) -> tuple[Optional[Path], Optional[str]]:
        """Resolve reference voice path or preset name."""
        val = self.voice_preset_var.get()
        if "american accent" in val:
            return DEFAULT_REFERENCE_VOICE, None
        elif "Anoop voice line" in val:
            return DEFAULT_FALLBACK_AUDIO, None
        elif val.startswith("Built-in:"):
            preset_name = val.split(":")[1].strip()
            return None, preset_name
        elif "Custom" in val:
            ref_str = self.custom_ref_path.get().strip()
            if not ref_str or not Path(ref_str).is_file():
                raise FileNotFoundError("Please browse a valid custom reference voice WAV file.")
            return Path(ref_str).resolve(), None
        return DEFAULT_REFERENCE_VOICE, None

    def _play_audio_file(self, path: Optional[Path]):
        """Play audio directly via sounddevice to speaker/headphones and open with player as fallback."""
        if not path or not path.exists():
            messagebox.showwarning("File Not Found", "The requested audio file does not exist.")
            return

        def _play_worker():
            try:
                import sounddevice as sd
                data, sr = sf.read(str(path), dtype="float32")
                sd.play(data, sr)
                sd.wait()
            except Exception:
                try:
                    os.startfile(str(path))
                except Exception as e:
                    self.work_queue.put({"type": "ERROR", "text": f"Audio playback error: {e}"})

        threading.Thread(target=_play_worker, daemon=True).start()

    def _preview_input_audio(self):
        try:
            p = self._resolve_input_audio()
            self._play_audio_file(p)
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    def _preview_ref_audio(self):
        try:
            p, name = self._resolve_reference_voice()
            if p:
                self._play_audio_file(p)
            else:
                messagebox.showinfo("Built-in Voice", f"'{name}' is an embedded Pocket TTS catalog voice.")
        except Exception as e:
            messagebox.showerror("Reference Error", str(e))

    def _play_output_audio(self):
        self._play_audio_file(self.latest_output_file)

    def _open_output_folder(self):
        folder = DEFAULT_OUTPUT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ==========================================
    # BACKGROUND PIPELINE WORKERS
    # ==========================================

    def _set_ui_state(self, running: bool):
        """Enable or disable controls during task execution."""
        self.is_running = running
        if running:
            self.progress_bar.start(10)
            self.btn_run_all.config(state="disabled")
            self.btn_stt_only.config(state="disabled")
            self.btn_tts_only.config(state="disabled")
        else:
            self.progress_bar.stop()
            self.btn_run_all.config(state="normal")
            self.btn_stt_only.config(state="normal")
            self.btn_tts_only.config(state="normal")

    def _check_queue(self):
        """Poll the message queue from worker threads."""
        while not self.work_queue.empty():
            msg = self.work_queue.get_nowait()
            mtype = msg.get("type")
            if mtype == "LOG":
                self._log(msg["text"], msg.get("tag"))
            elif mtype == "STT_DONE":
                self.transcribed_text.delete("1.0", "end")
                self.transcribed_text.insert("1.0", msg["text"])
            elif mtype == "TTS_DONE":
                out_path = Path(msg["output_file"])
                self.latest_output_file = out_path
                self.lbl_output_file.config(text=f"File: {out_path.name}", fg=self.fg_color)
                self.lbl_metrics.config(text=msg["metrics"])
            elif mtype == "LATENCY_UPDATE":
                self._display_latency_metrics(msg["data"])
            elif mtype == "FINISHED":
                self._set_ui_state(False)
            elif mtype == "STOP_LIVE":
                self._stop_live_streaming()
            elif mtype == "ERROR":
                self._set_ui_state(False)
                messagebox.showerror("Pipeline Error", msg["text"])

        self.after(100, self._check_queue)

    @staticmethod
    def _trim_leading_silence(audio_np: np.ndarray, sr: int, threshold: float = 0.015) -> np.ndarray:
        """Trim initial near-silent frames (such as prompt conditioning silence)."""
        if len(audio_np) == 0:
            return audio_np
        above = np.where(np.abs(audio_np) >= threshold)[0]
        if len(above) > 0 and above[0] > int(sr * 0.10):
            start_idx = max(0, above[0] - int(sr * 0.04))
            return audio_np[start_idx:]
        return audio_np

    @staticmethod
    def _normalize_audio(audio_np: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
        """Normalize peak amplitude to guarantee audible, clear volume."""
        if len(audio_np) == 0:
            return audio_np
        peak = float(np.max(np.abs(audio_np)))
        if peak > 0.001:
            return (audio_np / peak) * target_peak
        return audio_np

    def _execute_stt(self, wav_path: Path, stt_choice: str, device: str) -> dict:
        """Execute STT using chosen engine candidate and return transcription dict."""
        t_stt_start = time.perf_counter()
        stt_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
        stt_compute = "float16" if stt_device == "cuda" else "int8"

        if "Zipformer" in stt_choice:
            from stt.factory import get_stt_engine
            engine = get_stt_engine("zipformer", device=stt_device)
            engine.load()
            audio_data, sr = sf.read(str(wav_path), dtype="float32")
            if getattr(audio_data, "ndim", 1) > 1:
                audio_data = np.mean(audio_data, axis=1)
            res = engine.transcribe_detailed(audio_data, sample_rate=sr)
            engine.unload()
        elif "Moonshine" in stt_choice:
            from stt.factory import get_stt_engine
            engine = get_stt_engine("moonshine", device=stt_device)
            engine.load()
            audio_data, sr = sf.read(str(wav_path), dtype="float32")
            if getattr(audio_data, "ndim", 1) > 1:
                audio_data = np.mean(audio_data, axis=1)
            res = engine.transcribe_detailed(audio_data, sample_rate=sr)
            engine.unload()
        elif "Nemotron" in stt_choice:
            from stt.factory import get_stt_engine
            engine = get_stt_engine("nemotron", device=stt_device)
            engine.load()
            audio_data, sr = sf.read(str(wav_path), dtype="float32")
            if getattr(audio_data, "ndim", 1) > 1:
                audio_data = np.mean(audio_data, axis=1)
            res = engine.transcribe_detailed(audio_data, sample_rate=sr)
            engine.unload()
        elif "Parakeet" in stt_choice:
            from stt.factory import get_stt_engine
            engine = get_stt_engine("parakeet", device=stt_device)
            engine.load()
            audio_data, sr = sf.read(str(wav_path), dtype="float32")
            if getattr(audio_data, "ndim", 1) > 1:
                audio_data = np.mean(audio_data, axis=1)
            res = engine.transcribe_detailed(audio_data, sample_rate=sr)
            engine.unload()
        else:
            model_key = "tiny" if "tiny" in stt_choice.lower() else "turbo"
            stt = FasterWhisperAdapter(model_path=model_key, device=stt_device, compute_type=stt_compute)
            stt.load()
            res = stt.transcribe(str(wav_path))
            stt.unload()

        t_stt = time.perf_counter() - t_stt_start
        text = res.get("text", "").strip()
        first_word = res.get("first_word", "")
        first_word_time_ms = res.get("first_word_time_ms") or (t_stt * 1000.0)
        if not first_word and text:
            words = text.split()
            if words:
                first_word = words[0]

        return {
            "text": text,
            "first_word": first_word,
            "first_word_time_ms": first_word_time_ms,
            "elapsed_seconds": t_stt,
        }

    def _execute_tts(
        self,
        text: str,
        tts_choice: str,
        ref_path: Optional[Path],
        preset_name: Optional[str],
        device: str,
        lsd_steps: int = 2,
        temp: float = 0.7,
        diff_steps: int = 5,
        alpha: float = 0.3,
        beta: float = 0.7,
    ) -> tuple[np.ndarray, int, float, str]:
        """Synthesize text using chosen TTS engine, returning (audio_np, sample_rate, t_first_bit_ms, display_name)."""
        t_tts_start = time.perf_counter()
        tts_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"

        if "StyleTTS 2" in tts_choice:
            engine_name = "StyleTTS 2"
            from models.tts.styletts2.adapter import StyleTTS2Adapter
            adapter = StyleTTS2Adapter(device=tts_device)
            adapter.load()
            state = adapter.get_voice_state(ref_path) if ref_path else None
            t0 = time.perf_counter()
            audio_np, sample_rate = adapter.generate(
                text=text,
                target_voice_path=ref_path,
                ref_style=state,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diff_steps,
            )
            t_first_bit = (time.perf_counter() - t0) * 1000.0
            adapter.unload()

        elif "Kokoro" in tts_choice:
            engine_name = "Kokoro v1.0"
            from tts.kokoro import KokoroEngine
            engine = KokoroEngine(voice="af_bella", device=tts_device)
            engine.load()
            t0 = time.perf_counter()
            chunks = []
            t_first_bit = None
            for chunk in engine.synthesize_stream(text):
                if t_first_bit is None:
                    t_first_bit = (time.perf_counter() - t0) * 1000.0
                chunks.append(chunk)
            sample_rate = engine.get_sample_rate()
            audio_np = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            if t_first_bit is None:
                t_first_bit = (time.perf_counter() - t0) * 1000.0
            engine.unload()

        elif "Piper" in tts_choice:
            engine_name = "Piper"
            from tts.piper import PiperEngine
            engine = PiperEngine(voice_name="en_US-lessac-medium", device=tts_device)
            engine.load()
            t0 = time.perf_counter()
            chunks = []
            t_first_bit = None
            for chunk in engine.synthesize_stream(text):
                if t_first_bit is None:
                    t_first_bit = (time.perf_counter() - t0) * 1000.0
                chunks.append(chunk)
            sample_rate = engine.get_sample_rate()
            audio_np = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            if t_first_bit is None:
                t_first_bit = (time.perf_counter() - t0) * 1000.0
            engine.unload()

        elif "Melo" in tts_choice:
            engine_name = "MeloTTS"
            from tts.melo import MeloEngine
            engine = MeloEngine(speaker_id="EN-Default", device=tts_device)
            engine.load()
            t0 = time.perf_counter()
            chunks = []
            t_first_bit = None
            for chunk in engine.synthesize_stream(text):
                if t_first_bit is None:
                    t_first_bit = (time.perf_counter() - t0) * 1000.0
                chunks.append(chunk)
            sample_rate = engine.get_sample_rate()
            audio_np = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            if t_first_bit is None:
                t_first_bit = (time.perf_counter() - t0) * 1000.0
            engine.unload()

        else:
            engine_name = "Pocket TTS"
            tts_model = TTSModel.load_model(device=tts_device, temp=temp, lsd_decode_steps=lsd_steps)
            cached_pt = ROOT / "embeddings" / "american_accent.pt"
            if ref_path and "american accent" in str(ref_path).lower() and cached_pt.exists():
                try:
                    state = torch.load(str(cached_pt), map_location=tts_device, weights_only=False)
                except Exception:
                    state = tts_model.get_state_for_audio_prompt(str(ref_path))
            elif ref_path and tts_model.has_voice_cloning:
                state = tts_model.get_state_for_audio_prompt(str(ref_path))
            else:
                fallback = preset_name or "alba"
                state = tts_model.get_state_for_audio_prompt(fallback)

            from core.chunker import chunk_text
            clauses = chunk_text(text, max_words=12)
            if not clauses:
                clauses = [text]

            gen_start = time.perf_counter()
            chunks = []
            t_first_bit = None
            try:
                for clause in clauses:
                    for chunk_tensor in tts_model.generate_audio_stream(state, text_to_generate=clause, copy_state=True):
                        if t_first_bit is None:
                            t_first_bit = (time.perf_counter() - gen_start) * 1000.0
                        chunks.append(chunk_tensor.detach().cpu())
            except Exception:
                pass

            if chunks:
                audio_np = torch.cat(chunks, dim=-1).squeeze().numpy()
            else:
                audio = tts_model.generate_audio(state, text_to_generate=text)
                audio_np = audio.cpu().squeeze().numpy()
                if t_first_bit is None:
                    t_first_bit = (time.perf_counter() - gen_start) * 1000.0

            sample_rate = tts_model.sample_rate
            if t_first_bit is None:
                t_first_bit = (time.perf_counter() - t_tts_start) * 1000.0

        # Eliminate conditioning silence and ensure audible volume
        audio_np = self._trim_leading_silence(audio_np, sample_rate)
        audio_np = self._normalize_audio(audio_np, target_peak=0.90)

        return audio_np, sample_rate, t_first_bit, engine_name

    def _start_full_pipeline(self):
        """Launch complete STT + TTS pipeline in background thread."""
        try:
            input_file = self._resolve_input_audio()
            ref_path, preset_name = self._resolve_reference_voice()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            return

        stt_choice = self.stt_model_var.get()
        device = self.device_var.get()
        tts_choice = self.tts_engine_var.get()
        lsd_steps = int(self.lsd_steps_var.get())
        temp = float(self.temp_var.get())
        diff_steps = int(self.diff_steps_var.get())
        alpha = float(self.alpha_var.get())
        beta = float(self.beta_var.get())

        self._set_ui_state(True)
        self.log_text.delete("1.0", "end")
        self._log("=== Starting Speech-to-Speech Conversion ===", "STAGE")

        threading.Thread(
            target=self._worker_run_full,
            args=(input_file, ref_path, preset_name, stt_choice, device, lsd_steps, temp, tts_choice, diff_steps, alpha, beta),
            daemon=True,
        ).start()

    def _worker_run_full(self, input_file, ref_path, preset_name, stt_choice, device, lsd_steps, temp, tts_choice, diff_steps=5, alpha=0.3, beta=0.7):
        try:
            t_total_start = time.perf_counter()
            with self._telemetry_lock:
                self.run_cpu_samples = []
                self.run_gpu_samples = []
                self.peak_telemetry_cpu = 0.0
                self.peak_telemetry_gpu = 0.0
                self.peak_telemetry_ram = 0.0
                self.peak_telemetry_vram = 0.0

            # Stage 1: Ingestion
            self.work_queue.put({"type": "LOG", "text": f"[1/4] Ingesting: {input_file.name}", "tag": "STAGE"})
            in_duration = 0.0
            try:
                in_duration = sf.info(str(input_file)).duration
            except Exception:
                pass

            # Stage 2: Audio Conversion (.m4a -> 16kHz WAV)
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            t_resample_start = time.perf_counter()
            if input_file.suffix.lower() != ".wav":
                self.work_queue.put({"type": "LOG", "text": f"[2/4] Resampling {input_file.suffix} -> 16kHz mono WAV...", "tag": "STAGE"})
                converted_path = DEFAULT_OUTPUT_DIR / f"converted_{input_file.stem}.wav"
                wav_path = convert_to_wav(input_file, converted_path, target_sr=16000)
            else:
                wav_path = input_file
            t_resample = time.perf_counter() - t_resample_start

            # Stage 3: STT Execution
            self.work_queue.put({"type": "LOG", "text": f"[3/4] Transcribing with {stt_choice}...", "tag": "STAGE"})
            stt_res = self._execute_stt(wav_path, stt_choice, device)
            t_stt = stt_res["elapsed_seconds"]
            stt_rtf = t_stt / max(in_duration, 0.001) if in_duration > 0 else 0.0

            text = stt_res["text"]
            first_word = stt_res["first_word"]
            stt_first_word_ms = stt_res["first_word_time_ms"]

            self.work_queue.put({"type": "STT_DONE", "text": text})
            fw_log = f", First Word: \"{first_word}\" ({stt_first_word_ms:.0f}ms)" if first_word else ""
            self.work_queue.put({"type": "LOG", "text": f"Transcribed ({t_stt:.2f}s, RTF {stt_rtf:.2f}x{fw_log}): \"{text}\"", "tag": "SUCCESS"})

            if not text:
                self.work_queue.put({"type": "ERROR", "text": "No speech detected in audio."})
                return

            # Stage 4: TTS Execution
            self.work_queue.put({"type": "LOG", "text": f"[4/4] Synthesizing Voice Clone with {tts_choice}...", "tag": "STAGE"})
            t_tts_start = time.perf_counter()
            audio_np, sample_rate, tts_first_bit_ms, engine_name = self._execute_tts(
                text=text,
                tts_choice=tts_choice,
                ref_path=ref_path,
                preset_name=preset_name,
                device=device,
                lsd_steps=lsd_steps,
                temp=temp,
                diff_steps=diff_steps,
                alpha=alpha,
                beta=beta,
            )
            t_tts = time.perf_counter() - t_tts_start

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_file = DEFAULT_OUTPUT_DIR / f"pipeline_output_{ts}.wav"
            sf.write(str(out_file), audio_np, sample_rate, subtype="PCM_16")
            duration = len(audio_np) / max(sample_rate, 1)

            t_total = time.perf_counter() - t_total_start
            rtf = t_total / max(duration, 0.001)
            tts_rtf = t_tts / max(duration, 0.001)
            total_first_bit_ms = stt_first_word_ms + tts_first_bit_ms

            with self._telemetry_lock:
                c_samples = list(self.run_cpu_samples)
                g_samples = list(self.run_gpu_samples)
                cpu_peak = self.peak_telemetry_cpu
                gpu_peak = self.peak_telemetry_gpu
                ram_peak = self.peak_telemetry_ram
                vram_peak = self.peak_telemetry_vram
                st = dict(self.latest_telemetry)

            cpu_avg = sum(c_samples) / len(c_samples) if c_samples else cpu_peak
            ram_used = st.get("ram_used", ram_peak)
            ram_total = st.get("ram_total", 15.3)
            vram_total = st.get("vram_total", 6144.0)
            gpu_name = st.get("gpu_name", "NVIDIA GPU")

            summary_report = self._format_completion_summary(
                mode=f"Speech-to-Speech Voice Conversion ({engine_name})",
                t_resample=t_resample,
                t_stt=t_stt,
                stt_rtf=stt_rtf,
                t_tts=t_tts,
                tts_rtf=tts_rtf,
                t_total=t_total,
                in_duration=in_duration,
                out_duration=duration,
                overall_rtf=rtf,
                cpu_peak=cpu_peak,
                cpu_avg=cpu_avg,
                ram_used=ram_used,
                ram_peak=ram_peak,
                ram_total=ram_total,
                gpu_peak=gpu_peak,
                vram_peak=vram_peak,
                vram_total=vram_total,
                gpu_name=gpu_name,
                stt_first_word_ms=stt_first_word_ms,
                tts_first_bit_ms=tts_first_bit_ms,
                first_word=first_word,
            )

            print("\n" + summary_report, flush=True)

            latency_data = {
                "mode": f"Full Pipeline ({stt_choice} + {engine_name})",
                "resample_time": t_resample,
                "stt_time": t_stt,
                "stt_rtf": stt_rtf,
                "tts_time": t_tts,
                "tts_rtf": tts_rtf,
                "total_time": t_total,
                "in_duration": in_duration,
                "out_duration": duration,
                "overall_rtf": rtf,
                "cpu_peak": cpu_peak,
                "gpu_peak": gpu_peak,
                "peak_ram_gb": ram_peak,
                "peak_vram_mb": vram_peak,
                "stt_first_word_ms": stt_first_word_ms,
                "tts_first_bit_ms": tts_first_bit_ms,
                "total_first_bit_ms": total_first_bit_ms,
                "first_word": first_word,
            }

            metrics_str = (
                f"STT First Word: {stt_first_word_ms:.0f}ms ('{first_word}') | "
                f"TTS First Bit: {tts_first_bit_ms:.0f}ms | Total First Audio: {total_first_bit_ms:.0f}ms | "
                f"Total: {t_total:.2f}s (RTF: {rtf:.2f}x) | "
                f"CPU: {cpu_peak:.0f}% | GPU: {gpu_peak:.0f}% | VRAM: {vram_peak:.0f}MB"
            )

            self.work_queue.put({
                "type": "TTS_DONE",
                "output_file": str(out_file),
                "metrics": metrics_str,
            })
            self.work_queue.put({
                "type": "LATENCY_UPDATE",
                "data": latency_data,
            })
            self.work_queue.put({"type": "LOG", "text": f"Generated successfully -> {out_file.name}", "tag": "SUCCESS"})
            self.work_queue.put({"type": "LOG", "text": summary_report, "tag": "STAGE"})

        except Exception as e:
            self.work_queue.put({"type": "ERROR", "text": str(e)})
        finally:
            self.work_queue.put({"type": "FINISHED"})

    def _start_stt_only(self):
        """Transcribe audio without generating TTS."""
        try:
            input_file = self._resolve_input_audio()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            return

        stt_choice = self.stt_model_var.get()
        device = self.device_var.get()

        self._set_ui_state(True)
        self.log_text.delete("1.0", "end")
        self._log(f"=== Running STT Only ({stt_choice}) ===", "STAGE")

        def _work():
            try:
                with self._telemetry_lock:
                    self.run_cpu_samples = []
                    self.run_gpu_samples = []
                    self.peak_telemetry_cpu = 0.0
                    self.peak_telemetry_gpu = 0.0
                    self.peak_telemetry_ram = 0.0
                    self.peak_telemetry_vram = 0.0

                in_duration = 0.0
                try:
                    in_duration = sf.info(str(input_file)).duration
                except Exception:
                    pass

                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                t_resample_start = time.perf_counter()
                if input_file.suffix.lower() != ".wav":
                    self.work_queue.put({"type": "LOG", "text": f"Resampling {input_file.suffix} -> WAV...", "tag": "STAGE"})
                    wav_path = convert_to_wav(input_file, DEFAULT_OUTPUT_DIR / f"converted_{input_file.stem}.wav", target_sr=16000)
                else:
                    wav_path = input_file
                t_resample = time.perf_counter() - t_resample_start

                stt_res = self._execute_stt(wav_path, stt_choice, device)
                t_stt = stt_res["elapsed_seconds"]
                stt_rtf = t_stt / max(in_duration, 0.001) if in_duration > 0 else 0.0
                text = stt_res["text"]
                first_word = stt_res["first_word"]
                stt_first_word_ms = stt_res["first_word_time_ms"]

                self.work_queue.put({"type": "STT_DONE", "text": text})

                with self._telemetry_lock:
                    c_samples = list(self.run_cpu_samples)
                    g_samples = list(self.run_gpu_samples)
                    cpu_peak = self.peak_telemetry_cpu
                    gpu_peak = self.peak_telemetry_gpu
                    ram_peak = self.peak_telemetry_ram
                    vram_peak = self.peak_telemetry_vram
                    st = dict(self.latest_telemetry)

                cpu_avg = sum(c_samples) / len(c_samples) if c_samples else cpu_peak
                ram_used = st.get("ram_used", ram_peak)
                ram_total = st.get("ram_total", 15.3)
                vram_total = st.get("vram_total", 6144.0)
                gpu_name = st.get("gpu_name", "NVIDIA GPU")

                summary_report = self._format_completion_summary(
                    mode="Faster-Whisper STT (Transcription Only)",
                    t_resample=t_resample,
                    t_stt=t_stt,
                    stt_rtf=stt_rtf,
                    t_tts=0.0,
                    tts_rtf=0.0,
                    t_total=t_stt,
                    in_duration=in_duration,
                    out_duration=0.0,
                    overall_rtf=stt_rtf,
                    cpu_peak=cpu_peak,
                    cpu_avg=cpu_avg,
                    ram_used=ram_used,
                    ram_peak=ram_peak,
                    ram_total=ram_total,
                    gpu_peak=gpu_peak,
                    vram_peak=vram_peak,
                    vram_total=vram_total,
                    gpu_name=gpu_name,
                    stt_first_word_ms=stt_first_word_ms,
                    tts_first_bit_ms=0.0,
                    first_word=first_word,
                )

                print("\n" + summary_report, flush=True)

                latency_data = {
                    "mode": "STT Only (Faster-Whisper)",
                    "resample_time": t_resample,
                    "stt_time": t_stt,
                    "stt_rtf": stt_rtf,
                    "tts_time": 0.0,
                    "tts_rtf": 0.0,
                    "total_time": t_stt,
                    "in_duration": in_duration,
                    "out_duration": 0.0,
                    "overall_rtf": stt_rtf,
                    "cpu_peak": cpu_peak,
                    "gpu_peak": gpu_peak,
                    "peak_ram_gb": ram_peak,
                    "peak_vram_mb": vram_peak,
                    "stt_first_word_ms": stt_first_word_ms,
                    "tts_first_bit_ms": 0.0,
                    "total_first_bit_ms": stt_first_word_ms,
                    "first_word": first_word,
                }
                self.work_queue.put({"type": "LATENCY_UPDATE", "data": latency_data})
                fw_log = f", First Word: \"{first_word}\" ({stt_first_word_ms:.0f}ms)" if first_word else ""
                self.work_queue.put({"type": "LOG", "text": f"Transcription complete ({t_stt:.2f}s, RTF {stt_rtf:.2f}x{fw_log}): \"{text}\"", "tag": "SUCCESS"})
                self.work_queue.put({"type": "LOG", "text": summary_report, "tag": "STAGE"})

            except Exception as e:
                self.work_queue.put({"type": "ERROR", "text": str(e)})
            finally:
                self.work_queue.put({"type": "FINISHED"})

        threading.Thread(target=_work, daemon=True).start()

    def _start_tts_only(self):
        """Synthesize text from the editable text box without STT."""
        text = self.transcribed_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty Text", "Please enter or transcribe some text to synthesize.")
            return

        try:
            ref_path, preset_name = self._resolve_reference_voice()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            return

        device = self.device_var.get()
        tts_choice = self.tts_engine_var.get()
        lsd_steps = int(self.lsd_steps_var.get())
        temp = float(self.temp_var.get())
        diff_steps = int(self.diff_steps_var.get())
        alpha = float(self.alpha_var.get())
        beta = float(self.beta_var.get())

        self._set_ui_state(True)
        self._log(f"=== Running {tts_choice} Synthesis Only ===", "STAGE")

        def _work():
            try:
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                t_start = time.perf_counter()
                with self._telemetry_lock:
                    self.run_cpu_samples = []
                    self.run_gpu_samples = []
                    self.peak_telemetry_cpu = 0.0
                    self.peak_telemetry_gpu = 0.0
                    self.peak_telemetry_ram = 0.0
                    self.peak_telemetry_vram = 0.0

                audio_np, sample_rate, tts_first_bit_ms, engine_name = self._execute_tts(
                    text=text,
                    tts_choice=tts_choice,
                    ref_path=ref_path,
                    preset_name=preset_name,
                    device=device,
                    lsd_steps=lsd_steps,
                    temp=temp,
                    diff_steps=diff_steps,
                    alpha=alpha,
                    beta=beta,
                )
                t_tts = time.perf_counter() - t_start

                ts = time.strftime("%Y%m%d_%H%M%S")
                out_file = DEFAULT_OUTPUT_DIR / f"pipeline_output_{ts}.wav"
                sf.write(str(out_file), audio_np, sample_rate, subtype="PCM_16")
                duration = len(audio_np) / max(sample_rate, 1)

                tts_rtf = t_tts / max(duration, 0.001)

                with self._telemetry_lock:
                    c_samples = list(self.run_cpu_samples)
                    g_samples = list(self.run_gpu_samples)
                    cpu_peak = self.peak_telemetry_cpu
                    gpu_peak = self.peak_telemetry_gpu
                    ram_peak = self.peak_telemetry_ram
                    vram_peak = self.peak_telemetry_vram
                    st = dict(self.latest_telemetry)

                cpu_avg = sum(c_samples) / len(c_samples) if c_samples else cpu_peak
                ram_used = st.get("ram_used", ram_peak)
                ram_total = st.get("ram_total", 15.3)
                vram_total = st.get("vram_total", 6144.0)
                gpu_name = st.get("gpu_name", "NVIDIA GPU")

                summary_report = self._format_completion_summary(
                    mode=f"{engine_name} Voice Cloning (Synthesis Only)",
                    t_resample=0.0,
                    t_stt=0.0,
                    stt_rtf=0.0,
                    t_tts=t_tts,
                    tts_rtf=tts_rtf,
                    t_total=t_tts,
                    in_duration=0.0,
                    out_duration=duration,
                    overall_rtf=tts_rtf,
                    cpu_peak=cpu_peak,
                    cpu_avg=cpu_avg,
                    ram_used=ram_used,
                    ram_peak=ram_peak,
                    ram_total=ram_total,
                    gpu_peak=gpu_peak,
                    vram_peak=vram_peak,
                    vram_total=vram_total,
                    gpu_name=gpu_name,
                    stt_first_word_ms=0.0,
                    tts_first_bit_ms=tts_first_bit_ms,
                    first_word="",
                )

                print("\n" + summary_report, flush=True)

                latency_data = {
                    "mode": f"TTS Only ({engine_name})",
                    "resample_time": 0.0,
                    "stt_time": 0.0,
                    "stt_rtf": 0.0,
                    "tts_time": t_tts,
                    "tts_rtf": tts_rtf,
                    "total_time": t_tts,
                    "in_duration": 0.0,
                    "out_duration": duration,
                    "overall_rtf": tts_rtf,
                    "cpu_peak": cpu_peak,
                    "gpu_peak": gpu_peak,
                    "peak_ram_gb": ram_peak,
                    "peak_vram_mb": vram_peak,
                    "stt_first_word_ms": 0.0,
                    "tts_first_bit_ms": tts_first_bit_ms,
                    "total_first_bit_ms": tts_first_bit_ms,
                    "first_word": "",
                }
                metrics_str = (
                    f"TTS Time: {t_tts:.2f}s | TTFB: {tts_first_bit_ms:.0f}ms | Audio: {duration:.1f}s | RTF: {tts_rtf:.2f}x | "
                    f"CPU Peak: {cpu_peak:.0f}% | GPU Peak: {gpu_peak:.0f}% | VRAM Peak: {vram_peak:.0f}MB"
                )
                self.work_queue.put({
                    "type": "TTS_DONE",
                    "output_file": str(out_file),
                    "metrics": metrics_str,
                })
                self.work_queue.put({"type": "LATENCY_UPDATE", "data": latency_data})
                self.work_queue.put({"type": "LOG", "text": f"Synthesized -> {out_file.name}", "tag": "SUCCESS"})
                self.work_queue.put({"type": "LOG", "text": summary_report, "tag": "STAGE"})

            except Exception as e:
                self.work_queue.put({"type": "ERROR", "text": str(e)})
            finally:
                self.work_queue.put({"type": "FINISHED"})

        threading.Thread(target=_work, daemon=True).start()

    def _toggle_live_streaming(self):
        """Toggle live real-time speech-to-speech continuous streaming."""
        if self.is_live_streaming:
            self._stop_live_streaming()
        else:
            self._start_live_streaming()

    def _start_live_streaming(self):
        """Initialize and run LiveS2SEngine in background thread."""
        try:
            ref_path, preset_name = self._resolve_reference_voice()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            return

        ref_voice = str(ref_path) if ref_path else (preset_name or "alba")
        stt_model = "tiny" if "tiny" in self.stt_model_var.get() else "turbo"
        device = "cuda" if (self.device_var.get() == "cuda" or (self.device_var.get() == "auto" and torch.cuda.is_available())) else "cpu"

        self.is_live_streaming = True
        self.btn_live_stream.config(
            text="🛑 Stop Live Streaming (Listening...)",
            style="Danger.TButton",
        )
        self.btn_run_all.config(state="disabled")
        self.btn_stt_only.config(state="disabled")
        self.btn_tts_only.config(state="disabled")
        self.progress_bar.start(10)
        self._log("=== 🎙️ Live Speech-to-Speech Streaming Started ===", "STAGE")
        self._log(f"Config: STT={stt_model}, Device={device}, Voice={ref_voice}", "STAGE")

        def _on_status(status_str: str):
            self.work_queue.put({"type": "LOG", "text": f"[LIVE] {status_str}", "tag": "STAGE"})

        def _on_transcription(text_str: str, stt_lat: float, dur: float):
            self.work_queue.put({"type": "STT_DONE", "text": text_str})
            self.work_queue.put({
                "type": "LOG",
                "text": f"🗣️ Heard ({dur:.1f}s, {stt_lat:.0f}ms): \"{text_str}\"",
                "tag": "SUCCESS",
            })

        def _on_latency(metrics_dict: dict):
            stt_fw = metrics_dict.get("stt_first_word_ms", 0.0)
            tts_fb = metrics_dict.get("tts_first_bit_ms", 0.0)
            tot_fb = metrics_dict.get("total_first_bit_ms", stt_fw + tts_fb)
            glass = metrics_dict.get("total_latency_ms", 0.0)
            fw = metrics_dict.get("first_word", "")

            lat_data = {
                "mode": "Live S2S Stream",
                "resample_time": 0.0,
                "stt_time": metrics_dict.get("stt_latency_ms", 0.0) / 1000.0,
                "stt_rtf": 0.0,
                "tts_time": tts_fb / 1000.0,
                "tts_rtf": 0.0,
                "total_time": glass / 1000.0,
                "in_duration": 0.0,
                "out_duration": 0.0,
                "overall_rtf": 0.0,
                "stt_first_word_ms": stt_fw,
                "tts_first_bit_ms": tts_fb,
                "total_first_bit_ms": tot_fb,
                "first_word": fw,
            }
            self.work_queue.put({"type": "LATENCY_UPDATE", "data": lat_data})
            self.work_queue.put({
                "type": "LOG",
                "text": f"⚡ First Word: {stt_fw:.0f}ms ('{fw}') | First Bit (TTFB): {tts_fb:.0f}ms | Total First Audio: {tot_fb:.0f}ms | Glass: {glass:.0f}ms",
                "tag": "STAGE",
            })

        def _on_audio_saved(saved_path_str: str, dur: float):
            p = Path(saved_path_str)
            self.latest_output_file = p
            self.work_queue.put({
                "type": "TTS_DONE",
                "output_file": str(p),
                "metrics": f"Live Audio Saved: {dur:.1f}s | File: {p.name}",
            })
            self.work_queue.put({
                "type": "LOG",
                "text": f"💾 Saved live audio -> {p.name} ({dur:.1f}s)",
                "tag": "SUCCESS",
            })

        def _run_live():
            try:
                from scripts.live_engine import LiveS2SEngine
                self.live_engine = LiveS2SEngine(
                    reference_voice=ref_voice,
                    stt_model=stt_model,
                    device=device,
                    output_dir=DEFAULT_OUTPUT_DIR,
                    min_silence_duration_ms=300,
                    on_status=_on_status,
                    on_transcription=_on_transcription,
                    on_latency=_on_latency,
                    on_audio_saved=_on_audio_saved,
                )
                self.live_engine.start()
            except Exception as e:
                self.work_queue.put({"type": "ERROR", "text": f"Live streaming failed to start:\n{e}"})
                self.work_queue.put({"type": "STOP_LIVE"})

        threading.Thread(target=_run_live, daemon=True).start()

    def _stop_live_streaming(self):
        """Stop running LiveS2SEngine and restore buttons."""
        self.is_live_streaming = False
        self.btn_live_stream.config(
            text="🎙️ Live Speech-to-Speech (Streaming Mic -> Voice Clone)",
            style="Success.TButton",
        )
        self.btn_run_all.config(state="normal")
        self.btn_stt_only.config(state="normal")
        self.btn_tts_only.config(state="normal")
        self.progress_bar.stop()

        if self.live_engine is not None:
            engine = self.live_engine
            self.live_engine = None
            def _stop():
                try:
                    engine.stop()
                except Exception:
                    pass
            threading.Thread(target=_stop, daemon=True).start()

        self._log("=== 🛑 Live Speech-to-Speech Streaming Stopped ===", "STAGE")


def main():
    if "--check-only" in sys.argv:
        print("GUI_READY")
        return
    app = S2SApp()
    app.mainloop()


if __name__ == "__main__":
    main()