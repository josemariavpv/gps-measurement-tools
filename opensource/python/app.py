#!/usr/bin/env python3
"""
GNSS Measurement Tools - Python GUI
====================================
Tkinter-based desktop application that provides a graphical interface for
running the GNSS measurement processing MATLAB scripts.

Usage
-----
    python app.py

Requirements
------------
    Python 3.7+  (tkinter is included in the standard library)
    MATLAB must be installed and the ``matlab`` executable must be on PATH
    for the "Run Processing" button to launch the pipeline automatically.
    If MATLAB is not available the GUI shows the equivalent MATLAB commands
    that you can paste into the MATLAB Command Window.
"""

import os
import subprocess
import sys
import textwrap
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matlab_executable() -> list[str]:
    """Return the MATLAB command as a list suitable for subprocess."""
    return ["matlab", "-nosplash", "-nodesktop", "-batch"]


def _opensource_dir() -> str:
    """Return the absolute path to the opensource/ directory."""
    # This file lives in opensource/python/, so go up one level.
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class GnssMeasurementApp:
    """Tkinter GUI for the GNSS Measurement Tools."""

    # Window geometry
    _WIN_W = 720
    _WIN_H = 560

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GNSS Measurement Tools")
        self.root.geometry(f"{self._WIN_W}x{self._WIN_H}")
        self.root.minsize(600, 480)
        self.root.resizable(True, True)

        self._build_ui()
        self._log("Welcome to GNSS Measurement Tools.")
        self._log("Select a data directory and log file, then click 'Run Processing'.")
        self._log(
            "Note: MATLAB must be installed and accessible from the command line.\n"
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(
            root_frame,
            text="GNSS Measurement Tools",
            font=("TkDefaultFont", 15, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(0, 12))

        # ---- Input Files --------------------------------------------------
        file_frame = ttk.LabelFrame(root_frame, text="Input Files", padding=8)
        file_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(1, weight=1)

        # Directory row
        ttk.Label(file_frame, text="Data Directory:").grid(
            row=0, column=0, sticky="w", pady=3
        )
        self._dir_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self._dir_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse_dir, width=9).grid(
            row=0, column=2
        )

        # Log file row
        ttk.Label(file_frame, text="Log File:").grid(row=1, column=0, sticky="w", pady=3)
        self._file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self._file_var).grid(
            row=1, column=1, sticky="ew", padx=6
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse_file, width=9).grid(
            row=1, column=2
        )

        # ---- True Position (optional) -------------------------------------
        pos_frame = ttk.LabelFrame(
            root_frame,
            text="Optional: True WGS84 Position (leave blank to skip)",
            padding=8,
        )
        pos_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))

        ttk.Label(pos_frame, text="Latitude (°):").grid(row=0, column=0, sticky="w")
        self._lat_var = tk.StringVar()
        ttk.Entry(pos_frame, textvariable=self._lat_var, width=14).grid(
            row=0, column=1, padx=(4, 16)
        )

        ttk.Label(pos_frame, text="Longitude (°):").grid(row=0, column=2, sticky="w")
        self._lon_var = tk.StringVar()
        ttk.Entry(pos_frame, textvariable=self._lon_var, width=14).grid(
            row=0, column=3, padx=(4, 16)
        )

        ttk.Label(pos_frame, text="Altitude (m):").grid(row=0, column=4, sticky="w")
        self._alt_var = tk.StringVar()
        ttk.Entry(pos_frame, textvariable=self._alt_var, width=10).grid(
            row=0, column=5, padx=(4, 0)
        )

        # ---- Buttons ------------------------------------------------------
        btn_row = ttk.Frame(root_frame)
        btn_row.grid(row=3, column=0, columnspan=3, pady=(0, 8))

        self._run_btn = ttk.Button(
            btn_row,
            text="▶  Run Processing",
            command=self._start_run,
            width=20,
        )
        self._run_btn.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(btn_row, text="Clear Output", command=self._clear_output).pack(
            side=tk.LEFT
        )

        # ---- Output area --------------------------------------------------
        out_frame = ttk.LabelFrame(root_frame, text="Output", padding=6)
        out_frame.grid(row=4, column=0, columnspan=3, sticky="nsew")
        root_frame.rowconfigure(4, weight=1)
        root_frame.columnconfigure(0, weight=1)

        self._output = scrolledtext.ScrolledText(
            out_frame,
            wrap=tk.WORD,
            font=("Courier", 10),
            state=tk.DISABLED,
            height=14,
        )
        self._output.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Browse callbacks
    # ------------------------------------------------------------------

    def _browse_dir(self) -> None:
        initial = self._dir_var.get() or os.getcwd()
        chosen = filedialog.askdirectory(
            title="Select the data directory", initialdir=initial
        )
        if chosen:
            self._dir_var.set(chosen)

    def _browse_file(self) -> None:
        initial = self._dir_var.get() or os.getcwd()
        chosen = filedialog.askopenfilename(
            title="Select GnssLogger log file",
            initialdir=initial,
            filetypes=[
                ("Log files", "*.txt *.csv *.log"),
                ("All files", "*.*"),
            ],
        )
        if chosen:
            if not self._dir_var.get():
                self._dir_var.set(os.path.dirname(chosen))
            self._file_var.set(os.path.basename(chosen))

    # ------------------------------------------------------------------
    # Run button
    # ------------------------------------------------------------------

    def _start_run(self) -> None:
        """Validate inputs and launch processing in a background thread."""
        dir_name = self._dir_var.get().strip()
        pr_file = self._file_var.get().strip()

        if not dir_name:
            messagebox.showerror("Missing Input", "Please select a data directory.")
            return
        if not pr_file:
            messagebox.showerror("Missing Input", "Please select a log file.")
            return
        if not os.path.isdir(dir_name):
            messagebox.showerror(
                "Invalid Directory", f"Directory not found:\n{dir_name}"
            )
            return

        # Parse optional true position
        lat_str = self._lat_var.get().strip()
        lon_str = self._lon_var.get().strip()
        alt_str = self._alt_var.get().strip()
        lla_cmd = "param.llaTrueDegDegM = [];"

        if lat_str or lon_str or alt_str:
            try:
                lat = float(lat_str)
                lon = float(lon_str)
                alt = float(alt_str)
                lla_cmd = f"param.llaTrueDegDegM = [{lat}, {lon}, {alt}];"
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Latitude, longitude, and altitude must all be numeric values.",
                )
                return

        self._run_btn.config(state=tk.DISABLED)
        thread = threading.Thread(
            target=self._run_processing,
            args=(dir_name, pr_file, lla_cmd),
            daemon=True,
        )
        thread.start()

    def _run_processing(self, dir_name: str, pr_file: str, lla_cmd: str) -> None:
        """Build and execute the MATLAB command (runs in a background thread)."""
        self._log("")
        self._log("=== Starting GNSS Processing ===")
        self._log(f"Directory : {dir_name}")
        self._log(f"File      : {pr_file}")
        if "[]" not in lla_cmd:
            self._log(f"True LLA  : {lla_cmd}")

        opensource_dir = _opensource_dir()

        # Build the MATLAB one-liner
        matlab_code = textwrap.dedent(f"""\
            addpath('{opensource_dir}'); \
            dirName = '{dir_name}'; \
            prFileName = '{pr_file}'; \
            {lla_cmd} \
            dataFilter = SetDataFilter; \
            [gnssRaw,gnssAnalysis] = ReadGnssLogger(dirName,prFileName,dataFilter); \
            if isempty(gnssRaw), error('Could not read log file'); end; \
            fctSeconds = 1e-3*double(gnssRaw.allRxMillis(end)); \
            utcTime = Gps2Utc([],fctSeconds); \
            allGpsEph = GetNasaHourlyEphemeris(utcTime,dirName); \
            if isempty(allGpsEph), error('Could not get ephemeris'); end; \
            [gnssMeas] = ProcessGnssMeas(gnssRaw); \
            h1=figure; colors=PlotPseudoranges(gnssMeas,prFileName); \
            h2=figure; PlotPseudorangeRates(gnssMeas,prFileName,colors); \
            h3=figure; PlotCno(gnssMeas,prFileName,colors); \
            gpsPvt = GpsWlsPvt(gnssMeas,allGpsEph); \
            h4=figure; PlotPvt(gpsPvt,prFileName,param.llaTrueDegDegM,'Raw Pseudoranges, WLS Solution'); drawnow; \
            h5=figure; PlotPvtStates(gpsPvt,prFileName); \
            if any(any(isfinite(gnssMeas.AdrM) & gnssMeas.AdrM~=0)), \
              [gnssMeas]=ProcessAdr(gnssMeas); \
              h6=figure; PlotAdr(gnssMeas,prFileName,colors); \
              [adrResid]=GpsAdrResiduals(gnssMeas,allGpsEph,param.llaTrueDegDegM); drawnow; \
              h7=figure; PlotAdrResids(adrResid,gnssMeas,prFileName,colors); \
            end\
        """)

        cmd = _matlab_executable() + [matlab_code]

        try:
            self._log("\nLaunching MATLAB…")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10-minute hard timeout
            )

            if result.stdout:
                self._log("\n--- MATLAB output ---")
                self._log(result.stdout.strip())

            if result.returncode != 0:
                stderr = result.stderr.strip()
                self._log("\n--- MATLAB errors ---")
                if stderr:
                    self._log(stderr)
                self._log(
                    "\nProcessing failed. "
                    "Make sure MATLAB is installed and the input files are valid."
                )
            else:
                self._log("\n✓ Processing complete! Check the MATLAB figure windows.")

        except FileNotFoundError:
            self._show_manual_instructions(dir_name, pr_file, lla_cmd, opensource_dir)

        except subprocess.TimeoutExpired:
            self._log(
                "\n⚠  MATLAB did not finish within 10 minutes. "
                "The process has been stopped."
            )

        except Exception as exc:  # Catch any other unexpected OS/subprocess error not covered above
            self._log(f"\n⚠  Unexpected error: {exc}")

        finally:
            # Re-enable the button on the main thread
            self.root.after(0, lambda: self._run_btn.config(state=tk.NORMAL))

    def _show_manual_instructions(
        self, dir_name: str, pr_file: str, lla_cmd: str, opensource_dir: str
    ) -> None:
        """Log manual instructions when MATLAB cannot be found automatically."""
        self._log(
            "\n⚠  MATLAB executable not found on PATH.\n"
            "   Open MATLAB and run the following commands:\n"
        )
        lines = [
            f"    addpath('{opensource_dir}');",
            f"    dirName = '{dir_name}';",
            f"    prFileName = '{pr_file}';",
            f"    {lla_cmd}",
            "    dataFilter = SetDataFilter;",
            "    [gnssRaw,gnssAnalysis] = ReadGnssLogger(dirName,prFileName,dataFilter);",
            "    fctSeconds = 1e-3*double(gnssRaw.allRxMillis(end));",
            "    utcTime = Gps2Utc([],fctSeconds);",
            "    allGpsEph = GetNasaHourlyEphemeris(utcTime,dirName);",
            "    [gnssMeas] = ProcessGnssMeas(gnssRaw);",
            "    figure; colors = PlotPseudoranges(gnssMeas,prFileName);",
            "    figure; PlotPseudorangeRates(gnssMeas,prFileName,colors);",
            "    figure; PlotCno(gnssMeas,prFileName,colors);",
            "    gpsPvt = GpsWlsPvt(gnssMeas,allGpsEph);",
            "    figure; PlotPvt(gpsPvt,prFileName,param.llaTrueDegDegM,'WLS Solution');",
            "    figure; PlotPvtStates(gpsPvt,prFileName);",
        ]
        for line in lines:
            self._log(line)
        self._log("")
        self._log(
            "   Alternatively, open ProcessGnssMeasGUI.m in MATLAB for a\n"
            "   built-in graphical interface."
        )

    # ------------------------------------------------------------------
    # Output helpers (thread-safe via root.after)
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Append *message* to the output widget (thread-safe)."""
        self.root.after(0, self._append_output, message)

    def _append_output(self, message: str) -> None:
        self._output.config(state=tk.NORMAL)
        self._output.insert(tk.END, message + "\n")
        self._output.see(tk.END)
        self._output.config(state=tk.DISABLED)

    def _clear_output(self) -> None:
        self._output.config(state=tk.NORMAL)
        self._output.delete("1.0", tk.END)
        self._output.config(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    GnssMeasurementApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

########################################################################
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
