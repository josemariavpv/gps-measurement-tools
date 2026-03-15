# GNSS Measurement Tools – Python GUI

This folder contains a **Python graphical user interface** for the GNSS
Measurement Tools.  It wraps the existing MATLAB scripts so that you can
browse for files, set parameters, and launch the processing pipeline without
editing any `.m` files.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| **Python 3.7 or later** | `tkinter` is included in the standard library and is the only dependency |
| **MATLAB** | Must be installed and the `matlab` executable must be on your `PATH` for one-click processing.  If MATLAB is not on your PATH the GUI will display the equivalent MATLAB commands that you can copy/paste. |
| **GNSS Measurement Tools on the MATLAB path** | The app automatically calls `addpath` for the `opensource/` directory, so you do not need to configure this separately. |

### Verify Python and tkinter

```bash
python3 --version        # 3.7 or later required
python3 -m tkinter       # should open a small test window
```

---

## Running the GUI

From a terminal, navigate to this directory and run:

```bash
python3 app.py
```

Or from the repository root:

```bash
python3 opensource/python/app.py
```

---

## Usage

1. **Data Directory** – click *Browse…* and select the folder that contains
   your GnssLogger `.txt` log file (or the `demoFiles/` folder to try the
   built-in examples).

2. **Log File** – click *Browse…* and select the log file inside that folder.

3. **True WGS84 Position** *(optional)* – if you know the exact position of
   the receiver, enter the latitude (degrees), longitude (degrees), and
   altitude (metres).  This is used to compute position error statistics.
   Leave all three fields blank to skip.

4. Click **▶ Run Processing**.  MATLAB will be launched in a non-interactive
   (`-batch`) session.  Progress and any errors are shown in the *Output*
   panel.

5. When processing finishes the MATLAB figure windows will remain open for
   inspection.

---

## What the processing pipeline does

| Step | MATLAB function called |
|------|------------------------|
| Read the GnssLogger log | `ReadGnssLogger` |
| Download / read GPS ephemeris | `GetNasaHourlyEphemeris` |
| Compute pseudoranges | `ProcessGnssMeas` |
| Plot pseudoranges & C/N₀ | `PlotPseudoranges`, `PlotPseudorangeRates`, `PlotCno` |
| Weighted least-squares PVT | `GpsWlsPvt` |
| Plot position solution | `PlotPvt`, `PlotPvtStates` |
| Carrier-phase processing *(if data available)* | `ProcessAdr`, `PlotAdr`, `GpsAdrResiduals`, `PlotAdrResids` |

---

## Troubleshooting

### MATLAB not found

If the GUI reports *"MATLAB executable not found on PATH"*, it will display
the equivalent MATLAB commands that you can run manually.  Alternatively, use
the **MATLAB GUI** (`opensource/ProcessGnssMeasGUI.m`) described below.

### Internet / ephemeris errors

`GetNasaHourlyEphemeris` downloads the ephemeris file from NASA's FTP
server.  If that fails, follow the manual instructions printed by the
function: download the file yourself, copy it to the same directory as your
log file, and re-run.

---

## MATLAB GUI alternative

A native MATLAB graphical application is also provided:

```matlab
% In the MATLAB Command Window:
addpath('~/gpstools/opensource');   % adjust to your actual path
ProcessGnssMeasGUI
```

This opens the same controls (file browser, true-position fields, run button,
and output log) as a native MATLAB `uifigure` app, without needing Python.
