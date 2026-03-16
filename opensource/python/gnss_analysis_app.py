"""GNSSAnalysis App – Python/Tkinter equivalent of the MATLAB GNSSAnalysis App.

Provides a graphical interface to the full GNSS processing pipeline:
read a GnssLogger log file, fetch NASA ephemeris, compute pseudoranges,
run a Weighted Least-Squares PVT solution, and display interactive plots.

Usage
-----
    python gnss_analysis_app.py

Or from Python:
    from opensource.python.gnss_analysis_app import GnssAnalysisApp
    import tkinter as tk
    root = tk.Tk()
    GnssAnalysisApp(root).mainloop()

Author: Frank van Diggelen
Open Source code for processing Android GNSS Measurements
"""

# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import queue
import threading
import traceback
import importlib
from typing import Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import matplotlib
from matplotlib.figure import Figure

try:
    import mplcursors as _mplcursors
    _HAS_MPLCURSORS = True
except ImportError:  # pragma: no cover
    _mplcursors = None
    _HAS_MPLCURSORS = False

# ---------------------------------------------------------------------------
# Path constants – needed so sibling modules can be imported whether this
# file is run as a standalone script *or* loaded as part of the package.
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))

# ---------------------------------------------------------------------------
# Colour palette – matches the look of the original MATLAB app
# ---------------------------------------------------------------------------
_BG         = '#F0F0F0'
_HEADER_BG  = '#2C3E50'
_HEADER_FG  = '#FFFFFF'
_BTN_RUN    = '#27AE60'
_BTN_CLEAR  = '#E74C3C'
_BTN_FG     = '#FFFFFF'
_ENTRY_BG   = '#FFFFFF'
_LOG_BG     = '#1E1E1E'
_LOG_FG     = '#D4D4D4'
_LOG_INFO   = '#4FC1FF'
_LOG_WARN   = '#CE9178'
_LOG_ERR    = '#F44747'
_FONT_MONO  = ('Courier New', 9)
_FONT_LABEL = ('Helvetica', 10)
_FONT_TITLE = ('Helvetica', 13, 'bold')

# Default demo values
_DEMO_DIR  = os.path.join(os.path.dirname(__file__), '..', 'demoFiles')
_DEMO_FILE = 'pseudoranges_log_2016_06_30_21_26_07.txt'
_DEMO_LAT  = '37.422578'
_DEMO_LON  = '-122.081678'
_DEMO_ALT  = '-28'

# Tab order / labels used throughout the app
_TAB_NAMES = [
    'Pseudoranges',
    'PR Rates',
    'C/No',
    'PVT',
    'PVT States',
    'ADR',
    'ADR Residuals',
]

# RINEX quality analysis tabs (added separately from the main pipeline tabs)
_RINEX_TAB_NAMES = [
    'Visibility',
    'Availability',
    'RINEX C/N0',
    'Cycle Slips',
]

# Default demo RINEX observation file
_DEMO_RINEX = os.path.join(os.path.dirname(__file__), '..', 'demoFiles',
                           'demo_obs.rnx')


# ===========================================================================
# Main Application
# ===========================================================================

class GnssAnalysisApp:
    """GNSS Analysis GUI application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('GNSS Analysis')
        self.root.configure(bg=_BG)
        self.root.minsize(1000, 700)

        # Switch to the TkAgg backend now that Tk is running
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk,
        )
        self._plt = plt
        self._FigureCanvasTkAgg = FigureCanvasTkAgg
        self._NavigationToolbar2Tk = NavigationToolbar2Tk

        # State
        self._processing = False
        self._msg_queue: queue.Queue = queue.Queue()
        self._figures: dict[str, Figure] = {}
        self._canvases: dict[str, FigureCanvasTkAgg] = {}
        self._rinex_figures: dict[str, Figure] = {}
        self._rinex_canvases: dict[str, FigureCanvasTkAgg] = {}
        # mplcursors hover-tooltip handles (one per tab, replaced on each render)
        self._cursors: dict[str, Any] = {}
        self._rinex_cursors: dict[str, Any] = {}

        self._build_ui()
        self._poll_message_queue()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Header ---------------------------------------------------
        header = tk.Frame(self.root, bg=_HEADER_BG, height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        tk.Label(
            header,
            text='GNSS Analysis',
            font=('Helvetica', 16, 'bold'),
            bg=_HEADER_BG, fg=_HEADER_FG,
        ).pack(side=tk.LEFT, padx=14, pady=8)

        # ---- Main content area: left panel + right notebook -----------
        content = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL,
            bg=_BG, sashwidth=5, sashrelief=tk.FLAT,
        )
        content.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Left panel
        left = tk.Frame(content, bg=_BG, width=280)
        content.add(left, minsize=240)
        self._build_left_panel(left)

        # Right: top-level notebook with two tabs – main pipeline and RINEX QC
        right = tk.Frame(content, bg=_BG)
        content.add(right, minsize=600)

        # Top-level notebook: "GNSS Pipeline" | "RINEX Quality"
        self._top_notebook = ttk.Notebook(right)
        self._top_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        pipeline_frame = tk.Frame(self._top_notebook, bg=_BG)
        self._top_notebook.add(pipeline_frame, text='GNSS Pipeline')
        self._build_plot_notebook(pipeline_frame)

        rinex_frame = tk.Frame(self._top_notebook, bg=_BG)
        self._top_notebook.add(rinex_frame, text='RINEX Quality')
        self._build_rinex_notebook(rinex_frame)

        self._build_log_panel(right)

        # Bottom status bar
        self._status_var = tk.StringVar(value='Ready.')
        status_bar = tk.Label(
            self.root, textvariable=self._status_var,
            bg='#D5D5D5', anchor=tk.W, padx=8,
            font=('Helvetica', 9),
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ---- Left panel ---------------------------------------------------

    def _build_left_panel(self, parent: tk.Frame):
        pad = {'padx': 8, 'pady': 3}

        # --- Section: Input file ----------------------------------------
        self._section_label(parent, 'Input')

        # Log file
        tk.Label(parent, text='Log file:', bg=_BG,
                 font=_FONT_LABEL, anchor=tk.W).pack(fill=tk.X, **pad)
        row_file = tk.Frame(parent, bg=_BG)
        row_file.pack(fill=tk.X, padx=8)
        self._log_file_var = tk.StringVar(value=os.path.join(_DEMO_DIR, _DEMO_FILE))
        tk.Entry(
            row_file, textvariable=self._log_file_var,
            bg=_ENTRY_BG, font=_FONT_MONO, width=20,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            row_file, text='…', width=3,
            command=self._browse_log_file,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        # Output directory
        tk.Label(parent, text='Output directory:', bg=_BG,
                 font=_FONT_LABEL, anchor=tk.W).pack(fill=tk.X, padx=8, pady=(6, 2))
        row_dir = tk.Frame(parent, bg=_BG)
        row_dir.pack(fill=tk.X, padx=8)
        self._out_dir_var = tk.StringVar(value=os.path.abspath(_DEMO_DIR))
        tk.Entry(
            row_dir, textvariable=self._out_dir_var,
            bg=_ENTRY_BG, font=_FONT_MONO, width=20,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            row_dir, text='…', width=3,
            command=self._browse_out_dir,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=8)

        # --- Section: Parameters ----------------------------------------
        self._section_label(parent, 'Parameters')

        tk.Label(parent, text='True position  (optional)',
                 bg=_BG, font=_FONT_LABEL, anchor=tk.W).pack(fill=tk.X, **pad)

        for label, attr, default in (
            ('Latitude  (°N)', '_lat_var', _DEMO_LAT),
            ('Longitude (°E)', '_lon_var', _DEMO_LON),
            ('Altitude  (m)',  '_alt_var', _DEMO_ALT),
        ):
            r = tk.Frame(parent, bg=_BG)
            r.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(r, text=label, bg=_BG, font=_FONT_LABEL,
                     width=15, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            tk.Entry(r, textvariable=var, bg=_ENTRY_BG,
                     font=_FONT_MONO, width=12).pack(side=tk.LEFT, padx=(4, 0))

        tk.Label(
            parent,
            text='(leave blank if unknown)',
            bg=_BG, fg='#777', font=('Helvetica', 8),
            anchor=tk.W,
        ).pack(fill=tk.X, padx=8)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=8)

        # --- Section: RINEX Quality Analysis ----------------------------
        self._section_label(parent, 'RINEX Quality')

        tk.Label(parent, text='RINEX obs file:', bg=_BG,
                 font=_FONT_LABEL, anchor=tk.W).pack(fill=tk.X, **pad)
        row_rnx = tk.Frame(parent, bg=_BG)
        row_rnx.pack(fill=tk.X, padx=8)
        self._rinex_file_var = tk.StringVar(value=os.path.abspath(_DEMO_RINEX))
        tk.Entry(
            row_rnx, textvariable=self._rinex_file_var,
            bg=_ENTRY_BG, font=_FONT_MONO, width=20,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            row_rnx, text='…', width=3,
            command=self._browse_rinex_file,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        self._rinex_btn = tk.Button(
            parent, text='📊  Analyse RINEX',
            bg='#2980B9', fg=_BTN_FG,
            font=('Helvetica', 11, 'bold'),
            relief=tk.FLAT, pady=6,
            command=self._on_rinex_analyse,
        )
        self._rinex_btn.pack(fill=tk.X, padx=8, pady=4)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=8)

        # --- Section: Actions -------------------------------------------
        self._section_label(parent, 'Actions')

        self._run_btn = tk.Button(
            parent, text='▶  Run Processing',
            bg=_BTN_RUN, fg=_BTN_FG,
            font=('Helvetica', 11, 'bold'),
            relief=tk.FLAT, pady=6,
            command=self._on_run,
        )
        self._run_btn.pack(fill=tk.X, padx=8, pady=4)

        self._progress = ttk.Progressbar(
            parent, mode='indeterminate', length=200,
        )
        self._progress.pack(fill=tk.X, padx=8, pady=2)

        tk.Button(
            parent, text='✖  Clear Plots',
            bg=_BTN_CLEAR, fg=_BTN_FG,
            font=('Helvetica', 10),
            relief=tk.FLAT, pady=4,
            command=self._clear_plots,
        ).pack(fill=tk.X, padx=8, pady=4)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=8)

        # --- Section: About ---------------------------------------------
        self._section_label(parent, 'About')
        tk.Label(
            parent,
            text=(
                'GNSS Analysis App\n'
                'Python port of Google\n'
                'GPS Measurement Tools\n'
                'github.com/google/\n'
                'gps-measurement-tools'
            ),
            bg=_BG, font=('Helvetica', 8), fg='#555',
            justify=tk.LEFT,
        ).pack(padx=8, pady=2, anchor=tk.W)

    def _section_label(self, parent: tk.Frame, text: str):
        tk.Label(
            parent, text=text.upper(),
            bg=_BG, fg='#333',
            font=('Helvetica', 9, 'bold'),
        ).pack(fill=tk.X, padx=8, pady=(6, 2))

    # ---- Plot notebook ------------------------------------------------

    def _build_plot_notebook(self, parent: tk.Frame):
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        for name in _TAB_NAMES:
            frame = tk.Frame(self._notebook, bg=_BG)
            self._notebook.add(frame, text=name)

            fig = Figure(figsize=(8, 4), dpi=100, facecolor='white')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#F8F8F8')
            ax.text(
                0.5, 0.5, f'{name}\n(run processing to populate)',
                ha='center', va='center',
                transform=ax.transAxes,
                color='#AAAAAA', fontsize=12,
            )
            ax.axis('off')

            canvas = self._FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Navigation toolbar
            toolbar_frame = tk.Frame(frame, bg=_BG)
            toolbar_frame.pack(fill=tk.X)
            toolbar = self._NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

            self._figures[name]  = fig
            self._canvases[name] = canvas

    # ---- RINEX quality notebook ----------------------------------------

    def _build_rinex_notebook(self, parent: tk.Frame):
        self._rinex_notebook = ttk.Notebook(parent)
        self._rinex_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        for name in _RINEX_TAB_NAMES:
            frame = tk.Frame(self._rinex_notebook, bg=_BG)
            self._rinex_notebook.add(frame, text=name)

            fig = Figure(figsize=(8, 4), dpi=100, facecolor='white')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#F8F8F8')
            ax.text(
                0.5, 0.5, f'{name}\n(load a RINEX obs file to populate)',
                ha='center', va='center',
                transform=ax.transAxes,
                color='#AAAAAA', fontsize=12,
            )
            ax.axis('off')

            canvas = self._FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            toolbar_frame = tk.Frame(frame, bg=_BG)
            toolbar_frame.pack(fill=tk.X)
            toolbar = self._NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

            self._rinex_figures[name]  = fig
            self._rinex_canvases[name] = canvas

    # ---- Log panel ----------------------------------------------------

    def _build_log_panel(self, parent: tk.Frame):
        log_frame = tk.LabelFrame(
            parent, text=' Console ', bg=_BG,
            font=('Helvetica', 9), fg='#444',
        )
        log_frame.pack(fill=tk.X, padx=4, pady=(2, 4), ipady=2)

        self._log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            bg=_LOG_BG, fg=_LOG_FG,
            font=_FONT_MONO,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self._log_text.tag_config('info',  foreground=_LOG_INFO)
        self._log_text.tag_config('warn',  foreground=_LOG_WARN)
        self._log_text.tag_config('error', foreground=_LOG_ERR)

        self._log('info', 'GNSS Analysis App ready.')
        self._log('info', f'Demo directory: {os.path.abspath(_DEMO_DIR)}')

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, level: str, msg: str):
        """Append a message to the console (thread-safe via queue)."""
        self._msg_queue.put((level, msg))

    def _poll_message_queue(self):
        """Drain the message queue and update the log widget."""
        try:
            while True:
                level, msg = self._msg_queue.get_nowait()
                self._log_text.config(state=tk.NORMAL)
                self._log_text.insert(tk.END, msg + '\n', level)
                self._log_text.see(tk.END)
                self._log_text.config(state=tk.DISABLED)
                self._status_var.set(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_message_queue)

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _browse_log_file(self):
        path = filedialog.askopenfilename(
            title='Select GnssLogger log file',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
            initialdir=self._out_dir_var.get(),
        )
        if path:
            self._log_file_var.set(path)
            self._out_dir_var.set(os.path.dirname(path))

    def _browse_out_dir(self):
        path = filedialog.askdirectory(
            title='Select output directory',
            initialdir=self._out_dir_var.get(),
        )
        if path:
            self._out_dir_var.set(path)

    def _browse_rinex_file(self):
        path = filedialog.askopenfilename(
            title='Select RINEX observation file',
            filetypes=[
                ('RINEX obs files', '*.rnx *.obs *.??o *.RNX *.OBS'),
                ('All files', '*.*'),
            ],
            initialdir=os.path.abspath(_DEMO_DIR),
        )
        if path:
            self._rinex_file_var.set(path)

    # ------------------------------------------------------------------
    # RINEX quality analysis
    # ------------------------------------------------------------------

    def _on_rinex_analyse(self):
        """Start RINEX quality analysis in a background thread."""
        if self._processing:
            return
        rinex_path = self._rinex_file_var.get().strip()
        if not os.path.isfile(rinex_path):
            messagebox.showerror(
                'File not found',
                f'Cannot find RINEX obs file:\n{rinex_path}',
            )
            return

        self._processing = True
        self._rinex_btn.config(state=tk.DISABLED)
        self._progress.start(10)
        self._log('info', f'Analysing RINEX obs: {os.path.basename(rinex_path)} …')

        t = threading.Thread(
            target=self._rinex_worker,
            args=(rinex_path,),
            daemon=True,
        )
        t.start()

    def _rinex_worker(self, rinex_path):
        """Run RINEX quality analysis pipeline in a background thread."""
        try:
            self._run_rinex_pipeline(rinex_path)
        except Exception:
            tb = traceback.format_exc()
            self._log('error', 'RINEX analysis failed with exception:')
            for line in tb.splitlines():
                self._log('error', '  ' + line)
        finally:
            self._msg_queue.put(('_done_', ''))
        self.root.after(0, self._on_rinex_done)

    def _on_rinex_done(self):
        self._processing = False
        self._progress.stop()
        self._rinex_btn.config(state=tk.NORMAL)
        # Switch the top-level notebook to the RINEX Quality tab
        self._top_notebook.select(1)

    def _run_rinex_pipeline(self, rinex_path):
        """Read RINEX obs file and populate the four quality tabs."""
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)

        _pkg = __package__ or 'opensource.python'

        def _im(name):
            return importlib.import_module('.' + name, package=_pkg)

        read_rinex_obs          = _im('read_rinex_obs').read_rinex_obs
        plot_rinex_visibility   = _im('plot_rinex_quality').plot_rinex_visibility
        plot_rinex_availability = _im('plot_rinex_quality').plot_rinex_availability
        plot_rinex_cn0          = _im('plot_rinex_quality').plot_rinex_cn0
        plot_rinex_cycle_slips  = _im('plot_rinex_quality').plot_rinex_cycle_slips

        self._log('info', 'Reading RINEX observation file …')
        rinex_obs = read_rinex_obs(rinex_path)
        if rinex_obs is None:
            self._log('error', 'Failed to parse RINEX obs file.')
            return

        file_name = os.path.basename(rinex_path)
        n_ep  = len(rinex_obs['times'])
        n_sv  = len(rinex_obs['sats'])
        dur_m = rinex_obs['times'][-1] / 60.0 if n_ep > 1 else 0.0
        self._log(
            'info',
            f'  RINEX {rinex_obs["version"]:.2f}: '
            f'{n_ep} epochs, {n_sv} satellites, {dur_m:.1f} min',
        )

        self._log('info', 'Plotting satellite visibility …')
        self._render_rinex_plot('Visibility', plot_rinex_visibility,
                                rinex_obs, file_name)

        self._log('info', 'Plotting observable availability …')
        self._render_rinex_plot('Availability', plot_rinex_availability,
                                rinex_obs, file_name)

        self._log('info', 'Plotting C/N0 …')
        self._render_rinex_plot('RINEX C/N0', plot_rinex_cn0,
                                rinex_obs, file_name)

        self._log('info', 'Plotting cycle slips …')
        self._render_rinex_plot('Cycle Slips', plot_rinex_cycle_slips,
                                rinex_obs, file_name)

        self._log('info', '✔  RINEX analysis complete.')

    def _render_rinex_plot(self, tab_name: str, plot_fn, *args):
        """Like _render_plot but for the RINEX quality notebook."""
        from matplotlib._pylab_helpers import Gcf
        from matplotlib.backend_bases import FigureManagerBase

        # Remove the previous hover cursor for this tab before clearing the figure
        old_cursor = self._rinex_cursors.pop(tab_name, None)
        if old_cursor is not None and _HAS_MPLCURSORS:
            try:
                old_cursor.remove()
            except Exception as exc:
                self._log('warn', f'Could not remove old cursor for {tab_name}: {exc}')

        fig = self._rinex_figures[tab_name]
        fig.clf()

        if not hasattr(fig.canvas, '_embed_manager'):
            fig.canvas._embed_manager = FigureManagerBase(
                fig.canvas,
                len(_TAB_NAMES) + _RINEX_TAB_NAMES.index(tab_name) + 1,
            )
        Gcf._set_new_active_manager(fig.canvas._embed_manager)

        plot_fn(*args)
        fig.canvas.draw_idle()

        # Attach hover-cursor so users can inspect data values by hovering
        if _HAS_MPLCURSORS:
            self._rinex_cursors[tab_name] = _mplcursors.cursor(fig, hover=True)

        self.root.after(0, lambda f=tab_name: self._rinex_canvases[f].draw())

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._processing:
            return
        log_path = self._log_file_var.get().strip()
        if not os.path.isfile(log_path):
            messagebox.showerror(
                'File not found',
                f'Cannot find log file:\n{log_path}',
            )
            return

        self._processing = True
        self._run_btn.config(state=tk.DISABLED)
        self._progress.start(10)
        self._log('info', f'Starting processing: {os.path.basename(log_path)} …')

        # Build param dict
        param = {}
        try:
            lat = float(self._lat_var.get())
            lon = float(self._lon_var.get())
            alt = float(self._alt_var.get())
            param['llaTrueDegDegM'] = [lat, lon, alt]
        except ValueError:
            param['llaTrueDegDegM'] = []

        dir_name    = os.path.dirname(log_path)
        file_name   = os.path.basename(log_path)
        out_dir     = self._out_dir_var.get().strip() or dir_name

        t = threading.Thread(
            target=self._processing_worker,
            args=(dir_name, file_name, out_dir, param),
            daemon=True,
        )
        t.start()

    def _processing_worker(self, dir_name, file_name, out_dir, param):
        """Run the full GNSS pipeline in a background thread."""
        try:
            self._run_pipeline(dir_name, file_name, out_dir, param)
        except Exception:
            tb = traceback.format_exc()
            self._log('error', 'Processing failed with exception:')
            for line in tb.splitlines():
                self._log('error', '  ' + line)
        finally:
            self._msg_queue.put(('_done_', ''))

        # Signal the GUI to re-enable controls
        self.root.after(0, self._on_processing_done)

    def _on_processing_done(self):
        self._processing = False
        self._progress.stop()
        self._run_btn.config(state=tk.NORMAL)

    def _run_pipeline(self, dir_name, file_name, out_dir, param):
        """Execute each pipeline stage and update the plot tabs."""
        # Ensure repo root is on sys.path so importlib can find the package
        # regardless of how the app was launched (standalone script or package).
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)

        import numpy as np

        # Use importlib so imports work both when running as `python
        # gnss_analysis_app.py` (no parent package) and when imported as
        # `opensource.python.gnss_analysis_app` (package context).
        _pkg = __package__ or 'opensource.python'

        def _im(name):
            return importlib.import_module('.' + name, package=_pkg)

        set_data_filter         = _im('set_data_filter').set_data_filter
        read_gnss_logger        = _im('read_gnss_logger').read_gnss_logger
        gps2utc                 = _im('gps2utc').gps2utc
        get_nasa_hourly_ephemeris = _im('get_nasa_hourly_ephemeris').get_nasa_hourly_ephemeris
        process_gnss_meas       = _im('process_gnss_meas').process_gnss_meas
        gps_wls_pvt             = _im('gps_wls_pvt').gps_wls_pvt
        process_adr             = _im('process_adr').process_adr
        gps_adr_residuals       = _im('gps_adr_residuals').gps_adr_residuals
        plot_pseudoranges       = _im('plot_pseudoranges').plot_pseudoranges
        plot_pseudorange_rates  = _im('plot_pseudorange_rates').plot_pseudorange_rates
        plot_cno                = _im('plot_cno').plot_cno
        plot_pvt                = _im('plot_pvt').plot_pvt
        plot_pvt_states         = _im('plot_pvt_states').plot_pvt_states
        plot_adr                = _im('plot_adr').plot_adr
        plot_adr_resids         = _im('plot_adr_resids').plot_adr_resids

        # 1. Read log file
        self._log('info', 'Reading log file …')
        data_filter = set_data_filter()
        gnss_raw, gnss_analysis = read_gnss_logger(dir_name, file_name, data_filter)
        if gnss_raw is None:
            self._log('error', 'read_gnss_logger returned None – check file.')
            return

        # 2. Ephemeris
        self._log('info', 'Fetching GPS ephemeris …')
        fct_seconds = float(gnss_raw['allRxMillis'][-1]) * 1e-3
        utc_time = gps2utc(fct_seconds=fct_seconds)[0]
        all_gps_eph, _ = get_nasa_hourly_ephemeris(utc_time, dir_name)
        if not all_gps_eph:
            self._log('warn', 'No GPS ephemeris obtained – PVT plots will be skipped.')
        else:
            self._log('info', f'Loaded {len(all_gps_eph)} ephemeris records.')

        # 3. Process raw measurements
        self._log('info', 'Processing raw GNSS measurements …')
        gnss_meas = process_gnss_meas(gnss_raw)

        # 4. Pseudoranges plot
        self._log('info', 'Plotting pseudoranges …')
        colors = self._render_plot('Pseudoranges', plot_pseudoranges,
                                   gnss_meas, file_name)

        # 5. PR Rates plot
        self._log('info', 'Plotting pseudorange rates …')
        self._render_plot('PR Rates', plot_pseudorange_rates,
                          gnss_meas, file_name, colors)

        # 6. C/No plot
        self._log('info', 'Plotting C/No …')
        self._render_plot('C/No', plot_cno, gnss_meas, file_name, colors)

        if all_gps_eph:
            # 7. WLS PVT
            self._log('info', 'Computing WLS PVT solution …')
            gps_pvt = gps_wls_pvt(gnss_meas, all_gps_eph)

            # 8. PVT plot
            self._log('info', 'Plotting PVT …')
            ts = 'Raw Pseudoranges, Weighted Least Squares solution'
            self._render_plot(
                'PVT', plot_pvt,
                gps_pvt, file_name, param.get('llaTrueDegDegM'), ts,
            )

            # 9. PVT States plot
            self._log('info', 'Plotting PVT states …')
            self._render_plot('PVT States', plot_pvt_states,
                              gps_pvt, file_name)

            # 10. ADR
            adr_m = gnss_meas['AdrM']
            has_adr = bool(np.any(np.isfinite(adr_m) & (adr_m != 0)))
            if has_adr:
                self._log('info', 'Processing ADR …')
                gnss_meas = process_adr(gnss_meas)

                self._log('info', 'Plotting ADR …')
                self._render_plot('ADR', plot_adr,
                                  gnss_meas, file_name, colors)

                self._log('info', 'Computing ADR residuals …')
                lla_true = param.get('llaTrueDegDegM') or None
                adr_resid = gps_adr_residuals(gnss_meas, all_gps_eph, lla_true)

                self._log('info', 'Plotting ADR residuals …')
                self._render_plot('ADR Residuals', plot_adr_resids,
                                  adr_resid, gnss_meas, file_name, colors)
            else:
                self._log('warn',
                          'No valid ADR data found – ADR plots skipped.')

        self._log('info', '✔  Processing complete.')

    def _render_plot(self, tab_name: str, plot_fn, *args):
        """Call plot_fn(*args) into the pre-existing figure for tab_name.

        The plot functions use plt.gcf() / plt.gca(), so we temporarily
        make the tab's figure current, then redraw the canvas.

        Returns the return value of plot_fn (e.g. the ``colors`` array).
        """
        from matplotlib._pylab_helpers import Gcf
        from matplotlib.backend_bases import FigureManagerBase

        # Remove the previous hover cursor for this tab before clearing the figure
        old_cursor = self._cursors.pop(tab_name, None)
        if old_cursor is not None and _HAS_MPLCURSORS:
            try:
                old_cursor.remove()
            except Exception as exc:
                self._log('warn', f'Could not remove old cursor for {tab_name}: {exc}')

        fig = self._figures[tab_name]
        fig.clf()

        # Make this figure the current matplotlib figure so plot_fn can call
        # plt.gcf() / plt.gca() and target the correct embedded figure.
        # Figures created directly with Figure() (not via plt.figure()) are
        # not registered with pyplot and have no .number attribute in modern
        # matplotlib.  We register a headless FigureManagerBase (no extra Tk
        # window) so that Gcf.get_active() returns our figure.
        # _set_new_active_manager also wires up _cidgcf so Gcf.destroy_all
        # can clean up properly at interpreter shutdown.
        if not hasattr(fig.canvas, '_embed_manager'):
            fig.canvas._embed_manager = FigureManagerBase(
                fig.canvas, _TAB_NAMES.index(tab_name) + 1
            )
        Gcf._set_new_active_manager(fig.canvas._embed_manager)

        result = plot_fn(*args)
        fig.canvas.draw_idle()

        # Attach hover-cursor so users can inspect data values by hovering
        if _HAS_MPLCURSORS:
            self._cursors[tab_name] = _mplcursors.cursor(fig, hover=True)

        # Schedule canvas refresh on the main thread
        self.root.after(0, lambda f=tab_name: self._canvases[f].draw())

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _clear_plots(self):
        for name in _TAB_NAMES:
            fig = self._figures[name]
            fig.clf()
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f'{name}\n(run processing to populate)',
                ha='center', va='center',
                transform=ax.transAxes,
                color='#AAAAAA', fontsize=12,
            )
            ax.axis('off')
            self._canvases[name].draw()
        for name in _RINEX_TAB_NAMES:
            fig = self._rinex_figures[name]
            fig.clf()
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f'{name}\n(load a RINEX obs file to populate)',
                ha='center', va='center',
                transform=ax.transAxes,
                color='#AAAAAA', fontsize=12,
            )
            ax.axis('off')
            self._rinex_canvases[name].draw()
        self._log('info', 'Plots cleared.')

    def mainloop(self):
        self.root.mainloop()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    root = tk.Tk()
    app = GnssAnalysisApp(root)
    app.mainloop()


if __name__ == '__main__':
    # Allow running as  python gnss_analysis_app.py
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    main()
