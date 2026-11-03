#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operando XRD series viewer
-----------------------------
Loads a series of diffraction patterns (.xy, .dat, .uxd...) from an
operando experiment and offers these display modes:

  - 2D contour plot (2theta / time-index / intensity as color)
  - "Waterfall" view (stacked/offset patterns, 2D or interactive 3D)
  - Single animation (one pattern at a time) and cumulative animation
    (patterns pile up), both exportable as video (.mp4)

An optional electrochemical cycling curve (EC-Lab .mpt or generic .csv
file) can be displayed next to the plots, aligned on the same Y axis.

Available exports: PDF, SVG, PNG (vector formats for PDF/SVG), MP4 for the
animations (requires ffmpeg installed on the machine), and CSV / Excel
files of the data (for Origin, Excel...).

Author: Jean-Noël Chotard (Insight Instruments / UPJV)
License: MIT (see the LICENSE file)
"""

import os
import re
import csv
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

# ------------------------------------------------------------------
# Insight Instruments color palette
# ------------------------------------------------------------------
BLUE = "#5270FF"
BLUE_DARK = "#3A54D6"
PINK = "#FF45B2"
BG = "#F4F5FA"
CARD_BG = "#FFFFFF"
TEXT = "#20263B"
SUBTEXT = "#6B7290"
BORDER = "#E1E3EE"

HL_PALETTE = ["#FF45B2", "#E53935", "#FB8C00", "#43A047", "#00ACC1", "#5270FF", "#8E24AA", "#000000"]

ECHEM_WSPACE = 0.16          # gap between the diffraction axes and the electrochemistry panel
CONTOUR_ECHEM_WSPACE = 0.30  # wider on the contour plot, to leave room for the colorbar and its label

__version__ = "1.0.3"
PROJECT_URL = "https://github.com/jeannoelchotard/OperandoViewer"
SITE_URL = "https://jeannoelchotard.github.io/OperandoViewer/"
AUTHOR_NAME = "Jean-Noël Chotard"
AUTHOR_EMAIL = "jean-noel.chotard@u-picardie.fr"

ANIM_MODES = ("anim", "anim_cum")

INT_SCALES = ["Linear", "Square root", "Power (gamma)", "Logarithmic"]

COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo",
             "jet", "gray", "hot", "coolwarm", "rainbow"]


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def is_oscillating(col):
    """True if the column goes both up and down (e.g. Li/Na content or
    capacity across charge/discharge half-cycles), as opposed to a
    monotonic quantity like elapsed time."""
    col = np.asarray(col, dtype=float)
    d = np.diff(col)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return False
    return bool(np.any(d > 0) and np.any(d < 0))


def unwrap_cumulative(col):
    """Turns an oscillating column into a monotonically increasing sequence,
    so it can be used as a pseudo-time / sequential Y axis: cumulative sum
    of the absolute differences between consecutive points, starting from
    the first raw value. E.g. 0 -> 1 -> 0 becomes 0 -> 1 -> 2 instead of
    folding back on itself."""
    col = np.asarray(col, dtype=float)
    if len(col) == 0:
        return col
    d = np.abs(np.diff(col))
    d = np.nan_to_num(d, nan=0.0)
    cum = np.concatenate(([0.0], np.cumsum(d)))
    return col[0] + cum


def read_pattern(path):
    """Robust reader for a diffraction file (.xy, .dat, .uxd, ...).
    Skips header/comment lines and keeps only numeric (2theta, intensity)
    pairs."""
    xs, ys = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ";#!%*'\"" or line.startswith("_"):
                continue
            parts = re.split(r"[,\s;]+", line)
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _read_lines_any_encoding(path, encodings=("cp1252", "latin-1", "utf-8")):
    """Reads a text file trying several common encodings
    (EC-Lab .mpt files are usually cp1252)."""
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines(), enc
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines(), "utf-8 (ignore)"


def read_electrochem_file(path):
    """Reads an ASCII electrochemical cycling file (typically an EC-Lab
    .mpt export). Automatically handles:
      - encoding (cp1252/latin-1/utf-8)
      - number of header lines (via 'Nb header lines : N' if present,
        otherwise the first line is used as the column header)
      - comma decimal separator (European EC-Lab format)
      - trailing empty tab at the end of the header line
    Returns (list of column names, dict {name: np.array})."""
    lines, _ = _read_lines_any_encoding(path)
    if not lines:
        raise ValueError("Empty file.")

    nb_header = None
    for l in lines[:10]:
        m = re.search(r"Nb header lines\s*:\s*(\d+)", l)
        if m:
            nb_header = int(m.group(1))
            break

    header_idx = (nb_header - 1) if nb_header else 0
    data_start = nb_header if nb_header else 1
    if header_idx >= len(lines):
        raise ValueError("Header line not found: file too short.")

    header_line = lines[header_idx].rstrip("\r\n")
    delim = "\t" if "\t" in header_line else None
    raw_cols = header_line.split(delim) if delim else header_line.split()
    col_names = [c.strip() for c in raw_cols]
    while col_names and col_names[-1] == "":
        col_names.pop()  # trailing empty tab found in EC-Lab files
    n_cols = len(col_names)
    if n_cols == 0:
        raise ValueError("No column detected in the header line.")

    data_rows = []
    for l in lines[data_start:]:
        l = l.rstrip("\r\n")
        if not l.strip():
            continue
        parts = l.split(delim) if delim else l.split()
        while parts and parts[-1] == "":
            parts.pop()
        if len(parts) < n_cols:
            continue
        parts = parts[:n_cols]
        try:
            row = [float(p.replace(",", ".")) for p in parts]
        except ValueError:
            continue
        data_rows.append(row)

    if not data_rows:
        raise ValueError("No numeric data row could be read.")

    arr = np.array(data_rows, dtype=float)
    columns = {name: arr[:, i] for i, name in enumerate(col_names)}
    return col_names, columns


def read_electrochem_csv(path):
    """Reads a generic electrochemical cycling .csv export (cycler software, pandas...).
    Handles a UTF-8 BOM, the delimiter (comma, semicolon or tab), a decimal comma when the
    delimiter isn't a comma, an unnamed index column, and drops columns that hold no number
    at all (e.g. a DateTime column). Cells that aren't numeric become NaN.
    Returns (list of column names, dict {name: np.array}), like read_electrochem_file."""
    lines, _ = _read_lines_any_encoding(path, encodings=("utf-8-sig", "cp1252", "latin-1"))
    lines = [l.rstrip("\r\n") for l in lines if l.strip()]
    if not lines:
        raise ValueError("Empty file.")

    delim = max((",", ";", "\t"), key=lambda d: lines[0].count(d))
    if lines[0].count(delim) == 0:
        raise ValueError("No column separator (comma, semicolon or tab) found in the first line.")
    decimal_comma = delim != ","

    def split(line):
        return [c.strip().strip('"') for c in next(csv.reader([line], delimiter=delim))]

    def to_float(cell):
        try:
            return float(cell.replace(",", ".") if decimal_comma else cell)
        except ValueError:
            return np.nan

    first = split(lines[0])
    has_header = any(np.isnan(to_float(c)) for c in first if c != "")
    if has_header:
        names = [c or ("index" if i == 0 else f"Column {i + 1}") for i, c in enumerate(first)]
        data_lines = lines[1:]
    else:
        names = [f"Column {i + 1}" for i in range(len(first))]
        data_lines = lines
    seen = {}
    for i, nm in enumerate(names):  # duplicated names would overwrite each other in the dict
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            names[i] = f"{nm} ({seen[nm]})"

    n_cols = len(names)
    rows = []
    for l in data_lines:
        cells = split(l)
        if len(cells) < n_cols:
            continue
        rows.append([to_float(c) for c in cells[:n_cols]])
    if not rows:
        raise ValueError("No data row could be read.")

    arr = np.array(rows, dtype=float)
    keep = [i for i in range(n_cols) if not np.all(np.isnan(arr[:, i]))]
    if not keep:
        raise ValueError("No numeric column found.")
    col_names = [names[i] for i in keep]
    return col_names, {names[i]: arr[:, i] for i in keep}


def guess_echem_columns(col_names):
    """Best guess of the (potential, time) columns for the X / Y selectors."""
    low = {c: c.lower() for c in col_names}
    x = (next((c for c in col_names if low[c] in ("ecell/v", "ewe/v", "control/v", "voltage", "potential")), None)
         or next((c for c in col_names if "volt" in low[c] or "potential" in low[c]), None)
         or col_names[0])
    y = (next((c for c in col_names if low[c] in ("time/s", "test_time")), None)
         or next((c for c in col_names if "time" in low[c] and "step" not in low[c]), None)
         or col_names[-1])
    return x, y


EXCEL_MAX_COLS = 16384
EXCEL_MAX_ROWS = 1048576


def _num_str(v):
    return "" if v != v or v in (np.inf, -np.inf) else f"{v:.10g}"


def _num_cell(v):
    return None if v != v or v in (np.inf, -np.inf) else float(v)


def _echem_export_columns(snap):
    """Ordered {name: array} of the electrochemistry data to export: every numeric column of
    the loaded file, plus the unwrapped (cumulative) version of the selected Y column when that
    is what the plots display."""
    cols = dict(snap["echem_columns"])
    if snap["echem_unwrap"] and snap["echem_ycol"] in cols:
        cols[f"{snap['echem_ycol']} (cumulative, as displayed)"] = unwrap_cumulative(cols[snap["echem_ycol"]])
    return cols


def _info_rows(snap):
    xr = (float(np.nanmin(snap["two_theta"])), float(np.nanmax(snap["two_theta"])))
    rows = [("Exported by", "Operando XRD viewer"),
            ("Diffraction patterns", len(snap["filenames"])),
            ("Points per pattern", len(snap["two_theta"])),
            ("2theta range (deg)", f"{xr[0]:.4f} - {xr[1]:.4f}"),
            ("Y axis (matrix rows)", snap["ylabel"]),
            ("Intensity unit", snap["cbarlabel"])]
    if snap["echem_columns"]:
        rows += [("Electrochemistry X column (as displayed)", snap["echem_xcol"]),
                 ("Electrochemistry Y column (as displayed)",
                  snap["echem_ycol"] + (" (cumulative)" if snap["echem_unwrap"] else ""))]
    return rows


def export_data_csv(base, snap, progress=None):
    """Writes <base>_patterns.csv (2theta + one column per pattern), <base>_matrix.csv
    (intensity matrix, first row = 2theta, first column = Y value) and, if electrochemistry
    is loaded, <base>_echem.csv. Returns (files written, notes)."""
    X, Z, Y, names = snap["two_theta"], snap["intensities"], snap["y_values"], snap["filenames"]
    files, notes = [], []

    def step():
        if progress:
            progress()

    path = f"{base}_patterns.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["2theta"] + [os.path.splitext(n)[0] for n in names])
        for j in range(len(X)):
            w.writerow([_num_str(X[j])] + [_num_str(v) for v in Z[:, j]])
    files.append(path)
    step()

    path = f"{base}_matrix.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [_num_str(v) for v in X])
        for i in range(len(Y)):
            w.writerow([_num_str(Y[i])] + [_num_str(v) for v in Z[i]])
    files.append(path)
    step()

    if snap["echem_columns"]:
        cols = _echem_export_columns(snap)
        path = f"{base}_echem.csv"
        n = len(next(iter(cols.values())))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(cols))
            for i in range(n):
                w.writerow([_num_str(c[i]) for c in cols.values()])
        files.append(path)
        step()
    return files, notes


def export_data_xlsx(path, snap, progress=None):
    """Writes one Excel workbook with the sheets Patterns, Matrix, Electrochemistry (if loaded)
    and Info. Sheets that would exceed Excel's size limits are skipped (see the returned notes)."""
    from openpyxl import Workbook
    X, Z, Y, names = snap["two_theta"], snap["intensities"], snap["y_values"], snap["filenames"]
    notes = []

    def step():
        if progress:
            progress()

    wb = Workbook(write_only=True)
    n_fr, n_pt = Z.shape

    if n_fr + 1 <= EXCEL_MAX_COLS and n_pt + 1 <= EXCEL_MAX_ROWS:
        ws = wb.create_sheet("Patterns")
        ws.append(["2theta"] + [os.path.splitext(n)[0] for n in names])
        for j in range(n_pt):
            ws.append([_num_cell(X[j])] + [_num_cell(v) for v in Z[:, j]])
    else:
        notes.append("Sheet 'Patterns' skipped: too large for Excel (use the CSV export).")
    step()

    if n_pt + 1 <= EXCEL_MAX_COLS and n_fr + 1 <= EXCEL_MAX_ROWS:
        ws = wb.create_sheet("Matrix")
        ws.append([None] + [_num_cell(v) for v in X])
        for i in range(n_fr):
            ws.append([_num_cell(Y[i])] + [_num_cell(v) for v in Z[i]])
    else:
        notes.append(f"Sheet 'Matrix' skipped: {n_pt} points per pattern exceed Excel's "
                     f"{EXCEL_MAX_COLS}-column limit (use the CSV export).")
    step()

    if snap["echem_columns"]:
        cols = _echem_export_columns(snap)
        n = len(next(iter(cols.values())))
        if n + 1 <= EXCEL_MAX_ROWS:
            ws = wb.create_sheet("Electrochemistry")
            ws.append(list(cols))
            for i in range(n):
                ws.append([_num_cell(c[i]) for c in cols.values()])
        else:
            notes.append("Sheet 'Electrochemistry' skipped: too many rows for Excel (use the CSV export).")
        step()

    ws = wb.create_sheet("Info")
    for k, v in _info_rows(snap):
        ws.append([k, v])
    wb.save(path)
    return [path], notes


class XRDSeries:
    """Holds the series of diffraction patterns aligned on a common
    2theta grid."""

    def __init__(self):
        self.filenames = []
        self.two_theta = None      # common grid (n_points,)
        self.intensities = None    # matrix (n_frames, n_points)
        self.y_values = None       # (n_frames,) Y axis values (index, time, ...)

    def load(self, paths, progress_cb=None):
        raw = []
        ref_x = None
        for i, p in enumerate(paths):
            x, y = read_pattern(p)
            if len(x) == 0:
                continue
            if ref_x is None:
                ref_x = x
            raw.append((x, y))
            if progress_cb:
                progress_cb(i + 1, len(paths))

        if not raw:
            raise ValueError("No usable data found in the selected files.")

        # common grid = the one from the file with the most points, to lose nothing
        ref_x = max((r[0] for r in raw), key=len)
        mat = np.zeros((len(raw), len(ref_x)))
        for i, (x, y) in enumerate(raw):
            if len(x) == len(ref_x) and np.allclose(x, ref_x):
                mat[i] = y
            else:
                mat[i] = np.interp(ref_x, x, y, left=np.nan, right=np.nan)

        self.two_theta = ref_x
        self.intensities = mat
        self.filenames = [os.path.basename(p) for p in paths]
        self.y_values = np.arange(len(raw), dtype=float)

    @property
    def n_frames(self):
        return 0 if self.intensities is None else self.intensities.shape[0]


class OperandoViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Operando XRD viewer v{__version__}")
        self.geometry("1280x820")
        self.minsize(1080, 700)
        self.configure(bg=BG)

        self.series = XRDSeries()
        self.echem_columns = {}
        self.echem_colnames = []
        self.echem_loaded = False
        self.echem_xcol = tk.StringVar(value="")
        self.echem_ycol = tk.StringVar(value="")
        self.echem_xlabel = tk.StringVar(value="Potential (V)")
        self.echem_ylabel = tk.StringVar(value="")
        self.echem_unwrap_y = tk.BooleanVar(value=False)
        self.view_mode = tk.StringVar(value="contour")
        self.playing = False
        self.current_frame = tk.IntVar(value=0)
        self._anim_xlim = None  # remembers manual zoom on the animation plot between frames
        self._anim_ylim = None
        self._cum_xlim = None  # same, for the cumulative animation
        self._cum_ylim = None
        self.anim_widgets = {}  # per animation mode: its slider, play buttons, speed box, export button
        self._slider_updating = False
        self._anim_echem_ax = None  # electrochemistry axes of the animation currently drawn (drag target)
        self._anim_drawn_frame = 0
        self._marker_drag = False
        self._cursor_is_hand = False

        # plot dressing variables
        self.title_var = tk.StringVar(value="")
        self.xlabel_var = tk.StringVar(value="2θ (°)")
        self.ylabel_var = tk.StringVar(value="Frame")
        self.cbarlabel_var = tk.StringVar(value="Intensity (a.u.)")
        self.fs_title = tk.IntVar(value=14)
        self.fs_labels = tk.IntVar(value=12)
        self.fs_ticks = tk.IntVar(value=10)

        # contour
        self.cmap_contour = tk.StringVar(value="viridis")
        self.smooth_contour = tk.BooleanVar(value=True)
        self.n_levels = tk.IntVar(value=100)
        self.int_scale = tk.StringVar(value=INT_SCALES[0])
        self.int_gamma = tk.DoubleVar(value=0.5)
        self.int_plow = tk.DoubleVar(value=0.0)     # lower clip, percentile of the intensities
        self.int_phigh = tk.DoubleVar(value=100.0)  # upper clip, percentile of the intensities
        self.int_vmin = tk.StringVar(value="")      # absolute min (blank = automatic)
        self.int_vmax = tk.StringVar(value="")      # absolute max (blank = automatic)
        self.zoomx_mode = tk.BooleanVar(value=False)
        self.zoomx_min = tk.DoubleVar(value=0.0)
        self.zoomx_max = tk.DoubleVar(value=0.0)
        self._x_span = None
        self._echem_ax = None
        self._echem_xlim_full = None

        # waterfall
        self.wf_offset = tk.DoubleVar(value=1.0)
        self.wf_xshift = tk.DoubleVar(value=0.0)
        self.wf_step = tk.IntVar(value=1)
        self.wf_cmap = tk.StringVar(value="viridis")
        self.wf_lw = tk.DoubleVar(value=1.0)
        self.wf_3d = tk.BooleanVar(value=False)
        self.wf_hl_text = tk.StringVar(value="")
        self.wf_hl_colors = {}  # 0-based frame index -> hex color
        self._wf_hl_rendered = None  # frame set currently shown in the color rows
        self.wf_hl_lw = tk.DoubleVar(value=2.5)
        self.wf_elev = tk.DoubleVar(value=15.0)
        self.wf_azim = tk.DoubleVar(value=-90.0)
        self._last_3d_view = None  # remembers manual rotation between redraws
        self._applied_3d = (15.0, -90.0)  # last elevation/azimuth taken from the two fields

        # animation
        self.anim_speed = tk.IntVar(value=150)  # ms between frames
        self.anim_dpi = tk.IntVar(value=150)

        # export
        self.export_dpi = tk.IntVar(value=300)

        self._build_style()
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=SUBTEXT, font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=CARD_BG, foreground=BLUE_DARK, font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", background=BG, foreground=SUBTEXT, font=("Segoe UI", 9))
        style.configure("Accent.TButton", background=BLUE, foreground="white",
                         font=("Segoe UI", 10, "bold"), padding=7, borderwidth=0)
        style.map("Accent.TButton", background=[("active", BLUE_DARK)])
        style.configure("Ghost.TButton", background=CARD_BG, foreground=BLUE_DARK,
                         font=("Segoe UI", 9), padding=6, borderwidth=1)
        style.map("Ghost.TButton", background=[("active", "#EEF0FF")])
        style.configure("GhostCompact.TButton", background=CARD_BG, foreground=BLUE_DARK,
                         font=("Segoe UI", 7), padding=4, borderwidth=1)
        style.map("GhostCompact.TButton", background=[("active", "#EEF0FF")])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(6, 4), font=("Segoe UI", 8))
        style.layout("Hidden.TNotebook.Tab", [])  # tabs are driven by our own button bar
        style.configure("TCheckbutton", background=CARD_BG, font=("Segoe UI", 9))
        style.configure("TRadiobutton", background=CARD_BG, font=("Segoe UI", 9))

    # ------------------------------------------------------------------
    def _build_ui(self):
        header = ttk.Frame(self, padding=(20, 14, 20, 6))
        header.pack(fill="x")
        ttk.Label(header, text="Operando XRD viewer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Contour plot • waterfall view • animation exportable as video",
                  style="Subtitle.TLabel").pack(anchor="w")

        body = ttk.Frame(self, padding=(20, 0, 20, 10))
        body.pack(fill="both", expand=True)

        # -------- left panel: control tabs --------
        left = tk.Frame(body, bg=CARD_BG, width=360, highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)

        # custom tab bar (two rows of equally sized buttons) driving a notebook whose own
        # tab strip is hidden: native tabs can't be made equal and wide enough for 7 names
        self.tab_var = tk.StringVar(value="0")
        tabbar = tk.Frame(left, bg=CARD_BG)
        tabbar.pack(fill="x", padx=8, pady=(8, 0))
        self.notebook = ttk.Notebook(left, style="Hidden.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        self.tab_data = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.tab_view = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.tab_contour = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.tab_waterfall = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.tab_anim = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.tab_anim_cum = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.tab_export = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        self.tab_contact = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)

        tab_defs = [(self.tab_data, "Data"), (self.tab_view, "Display"),
                    (self.tab_contour, "Contour"), (self.tab_waterfall, "Waterfall"),
                    (self.tab_anim, "Single\nanimation"), (self.tab_anim_cum, "Cumulative\nanimation"),
                    (self.tab_export, "Export"), (self.tab_contact, "Contact")]
        n_cols = 4
        for idx, (frame, label) in enumerate(tab_defs):
            self.notebook.add(frame, text=label.replace("\n", " "))
            tk.Radiobutton(
                tabbar, text=label, variable=self.tab_var, value=str(idx), indicatoron=False,
                command=lambda idx=idx: self.notebook.select(idx),
                font=("Segoe UI", 8), bg="#E8EAF5", fg=TEXT, selectcolor="#C9D2FF",
                activebackground="#DDE1F5", relief="flat", bd=0, highlightthickness=0, pady=4
            ).grid(row=idx // n_cols, column=idx % n_cols, sticky="nsew", padx=1, pady=1)
        for c in range(n_cols):
            tabbar.grid_columnconfigure(c, weight=1, uniform="tabcol")
        for r in range((len(tab_defs) + n_cols - 1) // n_cols):
            tabbar.grid_rowconfigure(r, weight=1, uniform="tabrow")

        self._build_tab_data()
        self._build_tab_view()
        self._build_tab_contour()
        self._build_tab_waterfall()
        self._build_tab_anim()
        self._build_tab_anim_cum()
        self._build_tab_export()
        self._build_tab_contact()

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # -------- right panel: plot --------
        right = tk.Frame(body, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="left", fill="both", expand=True)

        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(6, 0))
        toolbar_frame = tk.Frame(right, bg=CARD_BG)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self._make_toolbar_exclusive_with_zoomx()
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("button_press_event", self._on_marker_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_marker_motion)
        self.canvas.mpl_connect("button_release_event", self._on_marker_release)

        self._draw_placeholder()

        # -------- status bar --------
        footer = ttk.Frame(self, padding=(20, 0, 20, 12))
        footer.pack(fill="x")
        self.status_var = tk.StringVar(value="Load a series of files to get started.")
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="determinate", length=220)
        self.progress.pack(side="right")

    # ---------------------------- Data tab ----------------------------
    def _make_scrollable(self, parent):
        """Wraps a tab in a scrollable canvas (useful for tabs that
        accumulate many controls, like Data)."""
        canvas = tk.Canvas(parent, bg=CARD_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Card.TFrame")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)
        inner.bind("<MouseWheel>", _on_mousewheel)
        return inner

    def _build_tab_data(self):
        t = self._make_scrollable(self.tab_data)
        ttk.Label(t, text="Load the series", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Button(t, text="From a .buf file", style="Accent.TButton",
                   command=self.load_from_buf).pack(fill="x", pady=3)
        ttk.Button(t, text="Select files…", style="Ghost.TButton",
                   command=self.load_from_folder).pack(fill="x", pady=3)

        self.data_info = ttk.Label(t, text="No data loaded.", style="Card.TLabel",
                                    foreground=SUBTEXT, wraplength=310, justify="left")
        self.data_info.pack(anchor="w", pady=(10, 14))

        ttk.Separator(t).pack(fill="x", pady=6)
        ttk.Label(t, text="Y axis (time / potential...)", style="Section.TLabel").pack(anchor="w", pady=(6, 8))

        row = ttk.Frame(t, style="Card.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Start", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.y_start = tk.DoubleVar(value=0.0)
        tk.Entry(row, textvariable=self.y_start, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(row, text="Step", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.y_step = tk.DoubleVar(value=1.0)
        tk.Entry(row, textvariable=self.y_step, width=8).grid(row=0, column=3, padx=4)
        ttk.Button(t, text="Apply (linear)", style="Ghost.TButton",
                   command=self.apply_linear_y).pack(fill="x", pady=(6, 3))
        ttk.Button(t, text="Load from a CSV (1 value/line)", style="Ghost.TButton",
                   command=self.load_y_from_csv).pack(fill="x", pady=3)

        ttk.Separator(t).pack(fill="x", pady=8)
        ttk.Label(t, text="Electrochemical cycling (.mpt / .csv)", style="Section.TLabel").pack(anchor="w", pady=(4, 8))
        load_row = ttk.Frame(t, style="Card.TFrame")
        load_row.pack(fill="x", pady=3)
        ttk.Button(load_row, text="Load .mpt (EC-Lab)…", style="Accent.TButton",
                   command=lambda: self.load_electrochem_file("mpt")).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(load_row, text="Load .csv…", style="Accent.TButton",
                   command=lambda: self.load_electrochem_file("csv")).pack(side="left", expand=True, fill="x")

        self.echem_info = ttk.Label(t, text="No electrochemical data loaded.", style="Card.TLabel",
                                     foreground=SUBTEXT, wraplength=310, justify="left")
        self.echem_info.pack(anchor="w", pady=(8, 8))

        rowx = ttk.Frame(t, style="Card.TFrame")
        rowx.pack(fill="x", pady=2)
        ttk.Label(rowx, text="X column", style="Card.TLabel", width=10).pack(side="left")
        self.echem_xcombo = ttk.Combobox(rowx, textvariable=self.echem_xcol, values=[],
                                          state="disabled", width=20)
        self.echem_xcombo.pack(side="left", fill="x", expand=True)
        self.echem_xcombo.bind("<<ComboboxSelected>>", self.on_echem_xcol_change)

        rowy = ttk.Frame(t, style="Card.TFrame")
        rowy.pack(fill="x", pady=2)
        ttk.Label(rowy, text="Y column", style="Card.TLabel", width=10).pack(side="left")
        self.echem_ycombo = ttk.Combobox(rowy, textvariable=self.echem_ycol, values=[],
                                          state="disabled", width=20)
        self.echem_ycombo.pack(side="left", fill="x", expand=True)
        self.echem_ycombo.bind("<<ComboboxSelected>>", self.on_echem_ycol_change)

        self.echem_unwrap_check = ttk.Checkbutton(
            t, text="Unwrap (cumulative, always increasing)", variable=self.echem_unwrap_y,
            command=self.on_echem_unwrap_toggle)
        self.echem_unwrap_check.pack(anchor="w", pady=(2, 4))
        ttk.Label(t, text="Use for columns that oscillate with charge/discharge half-cycles "
                           "(e.g. x, Capacity/mA.h, (Q-Qo)/mA.h) instead of increasing steadily "
                           "like time. Auto-detected when you pick the Y column.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 4))

        rowlblx = ttk.Frame(t, style="Card.TFrame")
        rowlblx.pack(fill="x", pady=2)
        ttk.Label(rowlblx, text="X axis label", style="Card.TLabel", width=10).pack(side="left")
        tk.Entry(rowlblx, textvariable=self.echem_xlabel).pack(side="left", fill="x", expand=True)

        rowlbly = ttk.Frame(t, style="Card.TFrame")
        rowlbly.pack(fill="x", pady=2)
        ttk.Label(rowlbly, text="Y axis label", style="Card.TLabel", width=10).pack(side="left")
        tk.Entry(rowlbly, textvariable=self.echem_ylabel).pack(side="left", fill="x", expand=True)

        echem_btn_row = ttk.Frame(t, style="Card.TFrame")
        echem_btn_row.pack(fill="x", pady=(6, 2))
        self.echem_apply_btn = ttk.Button(echem_btn_row, text="Apply", style="GhostCompact.TButton",
                                           command=self.apply_echem_columns, state="disabled")
        self.echem_apply_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.echem_remove_btn = ttk.Button(echem_btn_row, text="Remove", style="GhostCompact.TButton",
                                            command=self.remove_echem_data, state="disabled")
        self.echem_remove_btn.pack(side="left", fill="x", expand=True)
        ttk.Label(t, text="The XRD series Y axis is automatically rescaled to match the full range of "
                           "the chosen Y column whenever you change it (assumes frames are evenly "
                           "spaced in time).",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(6, 0))

    def load_from_buf(self):
        buf_path = filedialog.askopenfilename(title="Choose the .buf file",
                                               filetypes=[(".buf file", "*.buf"), ("All", "*.*")])
        if not buf_path:
            return
        base_dir = os.path.dirname(buf_path)
        with open(buf_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]

        paths = []
        for l in lines:
            if os.path.isabs(l) and os.path.isfile(l):
                paths.append(l)
            else:
                candidate = os.path.join(base_dir, l)
                paths.append(candidate)
        self._load_paths(paths)

    def load_from_folder(self):
        paths = filedialog.askopenfilenames(
            title="Select the series files (.xy, .dat, .uxd, ...)",
            filetypes=[("XRD data files", "*.xy *.dat *.uxd"),
                       ("XY files", "*.xy"), ("DAT files", "*.dat"), ("UXD files", "*.uxd"),
                       ("All files", "*.*")])
        if not paths:
            return
        paths = sorted(paths, key=lambda p: natural_key(os.path.basename(p)))
        self._load_paths(list(paths))

    def _load_paths(self, paths):
        if not paths:
            messagebox.showwarning("No file", "No file to load.")
            return
        self.status_var.set(f"Loading {len(paths)} files…")
        self.progress["maximum"] = len(paths)
        self.progress["value"] = 0
        self.update_idletasks()

        def cb(i, n):
            self.progress["value"] = i
            self.update_idletasks()

        try:
            self.series.load(paths, progress_cb=cb)
        except Exception as e:
            messagebox.showerror("Loading error", str(e))
            self.status_var.set("Error while loading.")
            return

        n = self.series.n_frames
        xr = (self.series.two_theta.min(), self.series.two_theta.max())
        self.data_info.config(
            text=f"{n} diffraction patterns loaded.\n2θ range: {xr[0]:.2f}° – {xr[1]:.2f}°")
        self.current_frame.set(0)
        for w in self.anim_widgets.values():
            w["slider"].config(to=max(n - 1, 0))
        self._anim_xlim = None  # a new series invalidates any previously remembered zoom
        self._anim_ylim = None
        self._cum_xlim = None
        self._cum_ylim = None
        self.zoomx_min.set(round(float(xr[0]), 4))
        self.zoomx_max.set(round(float(xr[1]), 4))
        self.status_var.set(f"{n} files loaded successfully.")
        self.progress["value"] = 0
        self.refresh_plot()

    def apply_linear_y(self):
        if self.series.n_frames == 0:
            return
        n = self.series.n_frames
        self.series.y_values = self.y_start.get() + self.y_step.get() * np.arange(n)
        self.refresh_plot()

    def load_y_from_csv(self):
        if self.series.n_frames == 0:
            messagebox.showinfo("Info", "Load a series of diffraction patterns first.")
            return
        path = filedialog.askopenfilename(title="Y values CSV file",
                                           filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return
        vals = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                try:
                    vals.append(float(row[0]))
                except ValueError:
                    continue
        if len(vals) != self.series.n_frames:
            messagebox.showwarning(
                "Inconsistent count",
                f"{len(vals)} values read for {self.series.n_frames} diffraction patterns. Not applied.")
            return
        self.series.y_values = np.array(vals)
        self.refresh_plot()

    # ---------------------------- Electrochemistry (.mpt) ----------------------------
    def load_electrochem_file(self, fmt="mpt"):
        if fmt == "csv":
            path = filedialog.askopenfilename(
                title="Choose an electrochemical cycling .csv file",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
            reader = read_electrochem_csv
        else:
            path = filedialog.askopenfilename(
                title="Choose an electrochemical cycling .mpt file",
                filetypes=[("EC-Lab .mpt", "*.mpt"), ("Text files", "*.txt"), ("All files", "*.*")])
            reader = read_electrochem_file
        if not path:
            return
        try:
            col_names, columns = reader(path)
        except Exception as e:
            messagebox.showerror("Reading error", str(e))
            return

        self.echem_colnames = col_names
        self.echem_columns = columns
        self.echem_loaded = True

        self.echem_xcombo.config(values=col_names, state="readonly")
        self.echem_ycombo.config(values=col_names, state="readonly")

        x_guess, y_guess = guess_echem_columns(col_names)
        self.echem_xcol.set(x_guess)
        self.echem_ycol.set(y_guess)

        self.echem_apply_btn.config(state="normal")
        self.echem_remove_btn.config(state="normal")

        n = len(next(iter(columns.values())))
        self.echem_info.config(
            text=f"{n} points loaded from {os.path.basename(path)}.\n"
                 f"{len(col_names)} columns available — pick X and Y below.")
        self.status_var.set("Electrochemical data loaded.")

        # apply the guessed Y column right away: auto-label + auto-align + redraw
        self.on_echem_ycol_change()

    def on_echem_xcol_change(self, event=None):
        self.refresh_plot()

    def on_echem_ycol_change(self, event=None):
        """Called automatically whenever the electrochemistry Y column changes:
        auto-names the Y axis label, auto-detects whether the column oscillates
        (e.g. Li/Na content or capacity across half-cycles) and pre-checks
        'Unwrap' accordingly, rescales the XRD series Y axis to match, then
        redraws."""
        if not self.echem_loaded:
            return
        col_name = self.echem_ycol.get()
        if not col_name or col_name not in self.echem_columns:
            return
        self.echem_ylabel.set(col_name)  # automatic naming; user can still edit it
        self.echem_unwrap_y.set(is_oscillating(self.echem_columns[col_name]))
        self._align_y_with_echem()
        self.refresh_plot()

    def on_echem_unwrap_toggle(self):
        if not self.echem_loaded:
            return
        self._align_y_with_echem()
        self.refresh_plot()

    def _get_echem_y_series(self):
        """Returns the currently selected electrochemistry Y column, unwrapped
        into a monotonically increasing sequence if requested."""
        if not self.echem_loaded or not self.echem_ycol.get():
            return None
        col = self.echem_columns.get(self.echem_ycol.get())
        if col is None:
            return None
        return unwrap_cumulative(col) if self.echem_unwrap_y.get() else col

    def _echem_panel_ready(self):
        """True if there is a fully configured electrochemistry column pair
        to display alongside a diffraction plot."""
        return bool(self.echem_loaded and self.echem_xcol.get() and self.echem_ycol.get()
                    and self.echem_xcol.get() in self.echem_columns
                    and self.echem_ycol.get() in self.echem_columns)

    def _get_echem_marker_position(self, i):
        """Finds the electrochemistry data point whose Y value is closest to
        the given XRD frame's Y position, so a marker can show where that
        diffraction pattern falls on the E vs. Y curve. Uses nearest-value
        matching rather than interpolation, since the Y column may not be
        strictly monotonic when 'Unwrap' isn't used."""
        if not self._echem_panel_ready() or self.series.y_values is None:
            return None
        ex = self.echem_columns.get(self.echem_xcol.get())
        ey = self._get_echem_y_series()
        if ex is None or ey is None or len(ey) == 0:
            return None
        target = self.series.y_values[i]
        idx = int(np.nanargmin(np.abs(ey - target)))
        return float(ex[idx]), float(ey[idx])

    def _align_y_with_echem(self):
        if self.series.n_frames == 0:
            return
        col = self._get_echem_y_series()
        if col is None:
            return
        n = self.series.n_frames
        self.series.y_values = np.linspace(float(np.nanmin(col)), float(np.nanmax(col)), n)
        suffix = " (cumulative)" if self.echem_unwrap_y.get() else ""
        self.status_var.set(
            f"XRD series Y axis aligned on '{self.echem_ycol.get()}{suffix}' "
            f"({np.nanmin(col):.3g} → {np.nanmax(col):.3g}).")

    def apply_echem_columns(self):
        if not self.echem_loaded or not self.echem_xcol.get() or not self.echem_ycol.get():
            return
        self.refresh_plot()

    def remove_echem_data(self):
        self.echem_loaded = False
        self.echem_columns = {}
        self.echem_colnames = []
        self.echem_xcombo.config(values=[], state="disabled")
        self.echem_ycombo.config(values=[], state="disabled")
        self.echem_apply_btn.config(state="disabled")
        self.echem_remove_btn.config(state="disabled")
        self.echem_info.config(text="No electrochemical data loaded.")
        self.echem_unwrap_y.set(False)
        self.status_var.set("Electrochemical data removed.")
        self.refresh_plot()

    # ---------------------------- Display tab ----------------------------
    def _build_tab_view(self):
        t = self.tab_view
        ttk.Label(t, text="Display mode", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        for val, label in [("contour", "2D contour plot"), ("waterfall", "Waterfall (stacked) view"),
                            ("anim", "Single animation"),
                            ("anim_cum", "Cumulative animation")]:
            ttk.Radiobutton(t, text=label, value=val, variable=self.view_mode,
                             command=self._on_view_change).pack(anchor="w", pady=2)

        ttk.Separator(t).pack(fill="x", pady=10)
        ttk.Label(t, text="Plot dressing", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        self._labeled_entry(t, "Title", self.title_var)
        self._labeled_entry(t, "X axis label", self.xlabel_var)
        self._labeled_entry(t, "Y axis label", self.ylabel_var)
        self._labeled_entry(t, "Colorbar label", self.cbarlabel_var)

        ttk.Label(t, text="Font sizes", style="Card.TLabel").pack(anchor="w", pady=(8, 2))
        self._labeled_spin(t, "Title", self.fs_title, 6, 30)
        self._labeled_spin(t, "Axes", self.fs_labels, 6, 30)
        self._labeled_spin(t, "Ticks", self.fs_ticks, 6, 24)

        ttk.Button(t, text="Refresh plot", style="Accent.TButton",
                   command=self.refresh_plot).pack(fill="x", pady=(14, 0))

    def _labeled_entry(self, parent, label, var):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Card.TLabel", width=16).pack(side="left")
        tk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def _labeled_spin(self, parent, label, var, lo, hi):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Card.TLabel", width=16).pack(side="left")
        tk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=6).pack(side="left")

    def _on_view_change(self):
        self._set_anim_controls_state("disabled" if getattr(self, "_mp4_exporting", False) else "normal")
        mode = self.view_mode.get()
        if mode in ANIM_MODES:
            # the two animation tabs have their own slider: keep the active one on the current frame
            self._slider_updating = True
            self.anim_widgets[mode]["slider"].set(self.current_frame.get())
            self._slider_updating = False
        self.refresh_plot()

    def _on_tab_changed(self, event=None):
        """Clicking the Contour / Waterfall / Animation tab directly selects
        the matching display mode, without needing to visit the Display tab."""
        self.tab_var.set(str(self.notebook.index("current")))  # keep the button bar in sync
        if not hasattr(self, "canvas"):
            return  # notebook not fully built yet
        selected = self.notebook.nametowidget(self.notebook.select())
        mapping = {self.tab_contour: "contour", self.tab_waterfall: "waterfall",
                   self.tab_anim: "anim", self.tab_anim_cum: "anim_cum"}
        mode = mapping.get(selected)
        if mode is None:
            return
        self.view_mode.set(mode)
        self._on_view_change()

    # ---------------------------- Contour tab ----------------------------
    def _build_tab_contour(self):
        t = self._make_scrollable(self.tab_contour)
        ttk.Label(t, text="Contour plot settings", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(t, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Colormap", style="Card.TLabel", width=14).pack(side="left")
        ttk.Combobox(row, textvariable=self.cmap_contour, values=COLORMAPS, width=14,
                     state="readonly").pack(side="left")

        ttk.Checkbutton(t, text="Smooth rendering (pcolormesh)", variable=self.smooth_contour,
                         command=self.refresh_plot).pack(anchor="w", pady=4)

        row2 = ttk.Frame(t, style="Card.TFrame")
        row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="Levels (if contourf)", style="Card.TLabel", width=16).pack(side="left")
        tk.Spinbox(row2, from_=5, to=500, textvariable=self.n_levels, width=6).pack(side="left")

        ttk.Separator(t).pack(fill="x", pady=10)
        ttk.Label(t, text="Intensity scale", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(t, text="Bring out weak peaks and background when a few very intense peaks "
                           "flatten everything else.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 6))

        row_sc = ttk.Frame(t, style="Card.TFrame")
        row_sc.pack(fill="x", pady=3)
        ttk.Label(row_sc, text="Scale", style="Card.TLabel", width=16).pack(side="left")
        cb = ttk.Combobox(row_sc, textvariable=self.int_scale, values=INT_SCALES, width=14, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plot())

        for label, var, lo, hi, inc in [("Gamma (power)", self.int_gamma, 0.05, 5, 0.05),
                                         ("Clip low (%)", self.int_plow, 0, 100, 0.5),
                                         ("Clip high (%)", self.int_phigh, 0, 100, 0.5)]:
            r = ttk.Frame(t, style="Card.TFrame")
            r.pack(fill="x", pady=3)
            ttk.Label(r, text=label, style="Card.TLabel", width=16).pack(side="left")
            sp = tk.Spinbox(r, from_=lo, to=hi, increment=inc, textvariable=var, width=8,
                            command=self.refresh_plot)
            sp.pack(side="left")
            sp.bind("<Return>", lambda e: self.refresh_plot())

        for label, var in [("Min value (blank = auto)", self.int_vmin),
                           ("Max value (blank = auto)", self.int_vmax)]:
            r = ttk.Frame(t, style="Card.TFrame")
            r.pack(fill="x", pady=3)
            ttk.Label(r, text=label, style="Card.TLabel", width=22).pack(side="left")
            e = tk.Entry(r, textvariable=var, width=10)
            e.pack(side="left")
            e.bind("<Return>", lambda e: self.refresh_plot())

        ttk.Label(t, text="Clip high (%) = 99 shows everything above the 99th percentile in the top "
                           "color: the usual fix for a few dominant peaks. Min/Max values override "
                           "the percentiles.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(4, 2))
        self.int_info = ttk.Label(t, text="", style="Card.TLabel", foreground=BLUE_DARK, wraplength=310)
        self.int_info.pack(anchor="w", pady=(2, 4))

        btn_row = ttk.Frame(t, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(2, 0))
        ttk.Button(btn_row, text="Apply", style="Ghost.TButton",
                   command=self.refresh_plot).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(btn_row, text="Reset scale", style="Ghost.TButton",
                   command=self.reset_intensity_scale).pack(side="left", expand=True, fill="x")

        ttk.Button(t, text="Refresh", style="Accent.TButton",
                   command=self.refresh_plot).pack(fill="x", pady=(14, 0))

        ttk.Separator(t).pack(fill="x", pady=10)
        ttk.Label(t, text="Zoom", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        ttk.Label(t, text="Set X range directly", style="Card.TLabel").pack(anchor="w", pady=(0, 3))
        rowxz = ttk.Frame(t, style="Card.TFrame")
        rowxz.pack(fill="x", pady=2)
        ttk.Label(rowxz, text="X min", style="Card.TLabel", width=6).pack(side="left")
        tk.Entry(rowxz, textvariable=self.zoomx_min, width=8).pack(side="left", padx=(0, 10))
        ttk.Label(rowxz, text="X max", style="Card.TLabel", width=6).pack(side="left")
        tk.Entry(rowxz, textvariable=self.zoomx_max, width=8).pack(side="left")
        ttk.Button(t, text="Apply X range", style="Ghost.TButton",
                   command=self.apply_x_zoom).pack(fill="x", pady=(4, 10))

        ttk.Checkbutton(t, text="Zoom X only (drag on the plot)", variable=self.zoomx_mode,
                         command=self._on_zoomx_toggle).pack(anchor="w", pady=2)
        ttk.Label(t, text="Keeps the Y axis at full scale. Turn off the toolbar's Pan/Zoom "
                           "tool below the plot first, so the drag isn't captured twice.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 8))
        ttk.Label(t, text="For zooming both X and Y together, use the magnifying-glass "
                           "Pan/Zoom tool in the toolbar below the plot. The electrochemistry "
                           "panel's Y axis follows automatically; its X axis (potential) always "
                           "stays at full scale.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 8))
        ttk.Button(t, text="Reset zoom (full scale)", style="Ghost.TButton",
                   command=self.reset_contour_zoom).pack(fill="x", pady=2)

    # ---------------------------- Waterfall tab ----------------------------
    def _build_tab_waterfall(self):
        t = self._make_scrollable(self.tab_waterfall)
        ttk.Label(t, text="Waterfall view settings", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        self.wf_offset_row = ttk.Frame(t, style="Card.TFrame")
        self.wf_offset_row.pack(fill="x", pady=3)
        ttk.Label(self.wf_offset_row, text="Vertical offset", style="Card.TLabel", width=16).pack(side="left")
        self.wf_offset_spin = tk.Spinbox(self.wf_offset_row, from_=0, to=1e9, increment=0.1,
                                          textvariable=self.wf_offset, width=8)
        self.wf_offset_spin.pack(side="left")

        self.wf_xshift_row = ttk.Frame(t, style="Card.TFrame")
        self.wf_xshift_row.pack(fill="x", pady=3)
        ttk.Label(self.wf_xshift_row, text="Horizontal offset", style="Card.TLabel", width=16).pack(side="left")
        self.wf_xshift_spin = tk.Spinbox(self.wf_xshift_row, from_=0, to=100, increment=0.05,
                                          textvariable=self.wf_xshift, width=8)
        self.wf_xshift_spin.pack(side="left")

        row3 = ttk.Frame(t, style="Card.TFrame")
        row3.pack(fill="x", pady=3)
        ttk.Label(row3, text="1 curve out of N", style="Card.TLabel", width=16).pack(side="left")
        tk.Spinbox(row3, from_=1, to=1000, textvariable=self.wf_step, width=8).pack(side="left")

        row4 = ttk.Frame(t, style="Card.TFrame")
        row4.pack(fill="x", pady=3)
        ttk.Label(row4, text="Colormap (gradient)", style="Card.TLabel", width=16).pack(side="left")
        ttk.Combobox(row4, textvariable=self.wf_cmap, values=COLORMAPS, width=12,
                     state="readonly").pack(side="left")

        row5 = ttk.Frame(t, style="Card.TFrame")
        row5.pack(fill="x", pady=3)
        ttk.Label(row5, text="Line width", style="Card.TLabel", width=16).pack(side="left")
        tk.Spinbox(row5, from_=0.2, to=5, increment=0.2, textvariable=self.wf_lw, width=8).pack(side="left")

        ttk.Separator(t).pack(fill="x", pady=8)
        ttk.Label(t, text="Highlighted curves", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(t, text="Frame numbers (1 = first), separated by commas; ranges allowed "
                           "(e.g. 5, 12, 20-25). Drawn even if hidden by '1 curve out of N'.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 4))

        row_hl = ttk.Frame(t, style="Card.TFrame")
        row_hl.pack(fill="x", pady=3)
        ttk.Label(row_hl, text="Frames", style="Card.TLabel", width=16).pack(side="left")
        hl_entry = tk.Entry(row_hl, textvariable=self.wf_hl_text)
        hl_entry.pack(side="left", fill="x", expand=True)
        hl_entry.bind("<Return>", lambda e: self.refresh_plot())

        ttk.Label(t, text="One color per highlighted frame (click a swatch to change it):",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(4, 2))
        self.wf_hl_rows = ttk.Frame(t, style="Card.TFrame")
        self.wf_hl_rows.pack(fill="x")

        row_hlw = ttk.Frame(t, style="Card.TFrame")
        row_hlw.pack(fill="x", pady=3)
        ttk.Label(row_hlw, text="Line width", style="Card.TLabel", width=16).pack(side="left")
        tk.Spinbox(row_hlw, from_=0.2, to=10, increment=0.2, textvariable=self.wf_hl_lw, width=8).pack(side="left")

        ttk.Separator(t).pack(fill="x", pady=8)
        ttk.Label(t, text="3D viewpoint", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Checkbutton(t, text="Interactive 3D view (mouse rotation)", variable=self.wf_3d,
                         command=self._on_wf_3d_toggle).pack(anchor="w", pady=(0, 4))
        ttk.Label(t, text="Left-click and drag on the plot to rotate in real time. Typing an angle "
                           "(Enter) or using the arrows applies it immediately.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 6))

        for label, var, lo, hi in (("Initial elevation", self.wf_elev, -90, 90),
                                   ("Initial azimuth", self.wf_azim, -180, 180)):
            r = ttk.Frame(t, style="Card.TFrame")
            r.pack(fill="x", pady=3)
            ttk.Label(r, text=label, style="Card.TLabel", width=16).pack(side="left")
            sp = tk.Spinbox(r, from_=lo, to=hi, textvariable=var, width=8, command=self._apply_3d_angles)
            sp.pack(side="left")
            sp.bind("<Return>", self._apply_3d_angles)
            sp.bind("<KP_Enter>", self._apply_3d_angles)
            sp.bind("<FocusOut>", self._apply_3d_angles)

        ttk.Button(t, text="Reset view", style="Ghost.TButton",
                   command=self.reset_3d_view).pack(fill="x", pady=(6, 0))

        ttk.Button(t, text="Refresh", style="Accent.TButton",
                   command=self.refresh_plot).pack(fill="x", pady=(14, 0))

        self._on_wf_3d_toggle()  # applies the initial state (offset/xshift disabled in 3D)

    # ---------------------------- Animation tab ----------------------------
    def _build_tab_anim(self):
        self._build_anim_controls(self._make_scrollable(self.tab_anim), "anim")

    def _build_tab_anim_cum(self):
        self._build_anim_controls(self._make_scrollable(self.tab_anim_cum), "anim_cum")

    def _build_anim_controls(self, t, mode):
        """Playback + video-export controls shared by the two animation tabs.
        The widgets of each tab are kept in self.anim_widgets[mode]."""
        cumulative = (mode == "anim_cum")
        widgets = {}
        self.anim_widgets[mode] = widgets

        if cumulative:
            ttk.Label(t, text="Each diffraction pattern is added on top of the previous ones "
                               "(kept on screen, waterfall style); the electrochemistry point moves "
                               "along with the current pattern.",
                      style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 8))

        ttk.Label(t, text="Playback", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        widgets["slider"] = tk.Scale(t, from_=0, to=0, orient="horizontal", bg=CARD_BG,
                                      highlightthickness=0, command=self._on_slider_move, state="disabled")
        widgets["slider"].pack(fill="x", pady=4)

        btn_row = ttk.Frame(t, style="Card.TFrame")
        btn_row.pack(fill="x", pady=4)
        widgets["loop"] = ttk.Button(btn_row, text="▶ Loop", style="Accent.TButton",
                                      command=lambda: self.play_animation(loop=True, mode=mode),
                                      state="disabled")
        widgets["loop"].pack(side="left", expand=True, fill="x", padx=(0, 4))
        widgets["once"] = ttk.Button(btn_row, text="▶ Once", style="Accent.TButton",
                                      command=lambda: self.play_animation(loop=False, mode=mode),
                                      state="disabled")
        widgets["once"].pack(side="left", expand=True, fill="x", padx=(0, 4))
        widgets["pause"] = ttk.Button(btn_row, text="⏸ Pause", style="Ghost.TButton",
                                       command=self.pause_animation, state="disabled")
        widgets["pause"].pack(side="left", expand=True, fill="x")

        row = ttk.Frame(t, style="Card.TFrame")
        row.pack(fill="x", pady=(10, 3))
        ttk.Label(row, text="Speed (ms/frame)", style="Card.TLabel", width=16).pack(side="left")
        widgets["speed"] = tk.Spinbox(row, from_=10, to=5000, increment=10,
                                       textvariable=self.anim_speed, width=8, state="disabled")
        widgets["speed"].pack(side="left")
        ttk.Label(t, text="The smaller the value, the faster the animation.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(2, 4))
        ttk.Label(t, text="With an electrochemistry file loaded, you can also grab the round marker "
                           "on the curve and drag it up or down to scrub through the patterns.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 10))

        if cumulative:
            ttk.Separator(t).pack(fill="x", pady=6)
            ttk.Label(t, text="Stacking (shared with Waterfall)", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
            for label, var, lo, hi, inc in [("Vertical offset", self.wf_offset, 0, 1e9, 0.1),
                                             ("Horizontal offset", self.wf_xshift, 0, 100, 0.05),
                                             ("Line width", self.wf_lw, 0.2, 5, 0.2)]:
                r = ttk.Frame(t, style="Card.TFrame")
                r.pack(fill="x", pady=3)
                ttk.Label(r, text=label, style="Card.TLabel", width=16).pack(side="left")
                tk.Spinbox(r, from_=lo, to=hi, increment=inc, textvariable=var, width=8).pack(side="left")
            r = ttk.Frame(t, style="Card.TFrame")
            r.pack(fill="x", pady=3)
            ttk.Label(r, text="Colormap (gradient)", style="Card.TLabel", width=16).pack(side="left")
            ttk.Combobox(r, textvariable=self.wf_cmap, values=COLORMAPS, width=12,
                         state="readonly").pack(side="left")
            ttk.Label(t, text="Highlighted curves (frames, colors, width) are set in the Waterfall tab "
                               "and appear here as soon as their frame is reached.",
                      style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(4, 0))
            ttk.Button(t, text="Refresh", style="Ghost.TButton",
                       command=self.refresh_plot).pack(fill="x", pady=(8, 10))

        ttk.Button(t, text="Reset zoom (full scale)", style="Ghost.TButton",
                   command=self.reset_anim_zoom).pack(fill="x", pady=(0, 6))
        ttk.Label(t, text="Zooming on the plot (toolbar below) stays in place while playing "
                           "or stepping through frames.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 6))

        ttk.Separator(t).pack(fill="x", pady=6)
        ttk.Label(t, text="Video export", style="Section.TLabel").pack(anchor="w", pady=(6, 8))
        row2 = ttk.Frame(t, style="Card.TFrame")
        row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="Video DPI", style="Card.TLabel", width=16).pack(side="left")
        tk.Spinbox(row2, from_=72, to=300, textvariable=self.anim_dpi, width=8).pack(side="left")

        widgets["export"] = ttk.Button(t, text="Export as MP4…", style="Accent.TButton",
                                        command=lambda: self.export_mp4(mode))
        widgets["export"].pack(fill="x", pady=(10, 0))
        ttk.Label(t, text="Requires ffmpeg installed and available in the system PATH.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(6, 0))

    def _on_slider_move(self, _val):
        if self._slider_updating:
            return
        self.current_frame.set(int(float(_val)))
        if self.view_mode.get() in ANIM_MODES:
            self._draw_anim_frame(self.current_frame.get())

    def _draw_anim_frame(self, i):
        if self.view_mode.get() == "anim_cum":
            self.draw_cumulative_frame(i)
        else:
            self.draw_animation_frame(i)

    # ---------------------------- Export tab ----------------------------
    def _build_tab_export(self):
        t = self._make_scrollable(self.tab_export)
        ttk.Label(t, text="Static figure export", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(t, text="(currently displayed Contour or Waterfall view)",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 10))

        row = ttk.Frame(t, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="PNG resolution (dpi)", style="Card.TLabel", width=16).pack(side="left")
        tk.Spinbox(row, from_=72, to=1200, textvariable=self.export_dpi, width=8).pack(side="left")

        ttk.Button(t, text="Export as PDF (vector)", style="Accent.TButton",
                   command=lambda: self.export_figure("pdf")).pack(fill="x", pady=4)
        ttk.Button(t, text="Export as SVG (vector)", style="Accent.TButton",
                   command=lambda: self.export_figure("svg")).pack(fill="x", pady=4)
        ttk.Button(t, text="Export as PNG", style="Ghost.TButton",
                   command=lambda: self.export_figure("png")).pack(fill="x", pady=4)

        ttk.Separator(t).pack(fill="x", pady=12)
        ttk.Label(t, text="Data export (Origin / Excel)", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Label(t, text="Exports the loaded series, with the Y axis as currently set, and the "
                           "electrochemistry data if loaded. CSV files import directly into Origin "
                           "(drag and drop): patterns as columns, plus the intensity matrix for "
                           "contour plots.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 8))
        self.export_csv_btn = ttk.Button(t, text="Export CSV files (for Origin)…", style="Accent.TButton",
                                          command=lambda: self.export_data("csv"))
        self.export_csv_btn.pack(fill="x", pady=4)
        self.export_xlsx_btn = ttk.Button(t, text="Export Excel workbook (.xlsx)…", style="Accent.TButton",
                                           command=lambda: self.export_data("xlsx"))
        self.export_xlsx_btn.pack(fill="x", pady=4)
        ttk.Label(t, text="CSV: pick a base name, e.g. 'mydata' creates mydata_patterns.csv, "
                           "mydata_matrix.csv and mydata_echem.csv. Excel: one file with the sheets "
                           "Patterns, Matrix, Electrochemistry and Info.",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(4, 0))

    def _data_snapshot(self):
        """Everything the export needs, read on the main thread so the worker never touches Tk."""
        s = self.series
        echem = self.echem_loaded and bool(self.echem_columns)
        return dict(
            two_theta=s.two_theta, intensities=s.intensities, y_values=s.y_values, filenames=list(s.filenames),
            ylabel=self.ylabel_var.get(), cbarlabel=self.cbarlabel_var.get(),
            echem_columns=dict(self.echem_columns) if echem else {},
            echem_xcol=self.echem_xcol.get(), echem_ycol=self.echem_ycol.get(),
            echem_unwrap=bool(self.echem_unwrap_y.get()))

    def export_data(self, fmt):
        if self.series.n_frames == 0:
            messagebox.showinfo("Info", "Load a series of diffraction patterns first.")
            return
        if getattr(self, "_data_exporting", False):
            messagebox.showinfo("Info", "A data export is already in progress.")
            return
        if fmt == "xlsx":
            try:
                import openpyxl  # noqa: F401
            except ImportError:
                messagebox.showerror("Excel export unavailable",
                                     "The 'openpyxl' library is required for Excel export.\n"
                                     "Install it with: pip install openpyxl")
                return
            path = filedialog.asksaveasfilename(defaultextension=".xlsx", title="Excel workbook",
                                                 filetypes=[("Excel workbook", "*.xlsx")])
            if not path:
                return
            targets = [path]
        else:
            path = filedialog.asksaveasfilename(defaultextension=".csv", title="Base name for the CSV files",
                                                 filetypes=[("CSV", "*.csv")])
            if not path:
                return
            path = os.path.splitext(path)[0]
            targets = [f"{path}_patterns.csv", f"{path}_matrix.csv"]
            if self.echem_loaded and self.echem_columns:
                targets.append(f"{path}_echem.csv")
            existing = [t for t in targets if os.path.exists(t)]
            if existing and not messagebox.askyesno(
                    "Overwrite?", "These files already exist and will be overwritten:\n\n" +
                    "\n".join(os.path.basename(t) for t in existing) + "\n\nContinue?"):
                return

        snap = self._data_snapshot()
        self._data_exporting = True
        self.export_csv_btn.config(state="disabled")
        self.export_xlsx_btn.config(state="disabled")
        self.status_var.set("Data export in progress…")
        self.progress["maximum"] = len(targets) if fmt == "csv" else 4
        self.progress["value"] = 0
        threading.Thread(target=self._export_data_worker, args=(fmt, path, snap), daemon=True).start()

    def _export_data_worker(self, fmt, path, snap):
        # No Tk access here: UI updates go through self.after(0, ...)
        def tick():
            self.after(0, lambda: self.progress.config(value=self.progress["value"] + 1))
        try:
            if fmt == "xlsx":
                files, notes = export_data_xlsx(path, snap, progress=tick)
            else:
                files, notes = export_data_csv(path, snap, progress=tick)
            self.after(0, lambda: self._finish_data_export(files, notes, None))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self._finish_data_export([], [], err))

    def _finish_data_export(self, files, notes, error):
        self._data_exporting = False
        self.progress["value"] = 0
        self.export_csv_btn.config(state="normal")
        self.export_xlsx_btn.config(state="normal")
        if error:
            self.status_var.set("Data export failed.")
            messagebox.showerror("Export error", error)
            return
        self.status_var.set(f"Data exported: {files[0]}" + (f" (+{len(files) - 1} more)" if len(files) > 1 else ""))
        msg = "Data exported successfully:\n" + "\n".join(files)
        if notes:
            msg += "\n\nNote:\n" + "\n".join(notes)
        messagebox.showinfo("Export complete", msg)

    # ---------------------------- Contact tab ----------------------------
    def _build_tab_contact(self):
        t = self.tab_contact
        ttk.Label(t, text="Contact", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(t, text="Questions, bug reports, or ideas for new features or changes? "
                           "Get in touch:",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 12))

        ttk.Label(t, text=AUTHOR_NAME, style="Card.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(t, text=AUTHOR_EMAIL, style="Card.TLabel", foreground=BLUE_DARK,
                  font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 12))

        ttk.Button(t, text="Send an email", style="Accent.TButton",
                   command=lambda: webbrowser.open(f"mailto:{AUTHOR_EMAIL}?subject=Operando XRD viewer")
                   ).pack(fill="x", pady=3)
        ttk.Button(t, text="Copy the email address", style="Ghost.TButton",
                   command=self._copy_author_email).pack(fill="x", pady=3)

        ttk.Separator(t).pack(fill="x", pady=12)
        ttk.Label(t, text="Website (download, quick start) and GitHub page (new versions, issue tracker):",
                  style="Card.TLabel", foreground=SUBTEXT, wraplength=310).pack(anchor="w", pady=(0, 6))
        ttk.Button(t, text="Open the website", style="Ghost.TButton",
                   command=lambda: webbrowser.open(SITE_URL)).pack(fill="x", pady=3)
        ttk.Button(t, text="Open the GitHub page", style="Ghost.TButton",
                   command=lambda: webbrowser.open(PROJECT_URL)).pack(fill="x", pady=3)
        ttk.Label(t, text=f"Operando XRD viewer  v{__version__}  ·  MIT license",
                  style="Card.TLabel", foreground=SUBTEXT).pack(anchor="w", pady=(10, 0))

    def _copy_author_email(self):
        self.clipboard_clear()
        self.clipboard_append(AUTHOR_EMAIL)
        self.status_var.set(f"Email address copied: {AUTHOR_EMAIL}")

    def export_figure(self, fmt):
        if self.series.n_frames == 0:
            messagebox.showinfo("Info", "Load a series of diffraction patterns first.")
            return
        if self.view_mode.get() in ANIM_MODES:
            messagebox.showinfo("Info", "Switch to Contour or Waterfall view to export a static image.")
            return
        path = filedialog.asksaveasfilename(defaultextension=f".{fmt}",
                                             filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path:
            return
        try:
            self.figure.savefig(path, format=fmt, dpi=self.export_dpi.get(), bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Export error", str(e))
            return
        self.status_var.set(f"Figure exported: {path}")
        messagebox.showinfo("Export complete", f"The figure was exported successfully:\n{path}")

    # ---------------------------- Plot rendering ----------------------------
    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Load a series to display the plot",
                      ha="center", va="center", color=SUBTEXT, fontsize=12,
                      transform=self.ax.transAxes)
        self.ax.axis("off")
        self.canvas.draw()

    def refresh_plot(self):
        if self.series.n_frames == 0:
            self._draw_placeholder()
            return
        mode = self.view_mode.get()
        if mode == "contour":
            self._draw_contour(self.ax)
        elif mode == "waterfall":
            self._draw_waterfall(self.ax)
        else:
            self._draw_anim_frame(self.current_frame.get())
        self.canvas.draw()

    def _apply_common_labels(self, ax, cbar=None):
        if self.title_var.get():
            ax.set_title(self.title_var.get(), fontsize=self.fs_title.get())
        ax.set_xlabel(self.xlabel_var.get(), fontsize=self.fs_labels.get())
        ax.set_ylabel(self.ylabel_var.get(), fontsize=self.fs_labels.get())
        ax.tick_params(labelsize=self.fs_ticks.get())
        if cbar is not None:
            cbar.set_label(self.cbarlabel_var.get(), fontsize=self.fs_labels.get())
            cbar.ax.tick_params(labelsize=self.fs_ticks.get())

    def _draw_contour(self, ax):
        self.figure.clear()
        show_echem = (self.echem_loaded and self.echem_xcol.get() and self.echem_ycol.get()
                      and self.echem_xcol.get() in self.echem_columns
                      and self.echem_ycol.get() in self.echem_columns)

        if show_echem:
            gs = self.figure.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=CONTOUR_ECHEM_WSPACE)
            ax = self.figure.add_subplot(gs[0, 0])
            ax_echem = self.figure.add_subplot(gs[0, 1])  # NO sharey: the contour must
        else:                                             # never depend on the echem scale
            ax = self.figure.add_subplot(111)
            ax_echem = None

        X = self.series.two_theta
        Y = self.series.y_values
        Z = self.series.intensities.copy()
        Z = np.nan_to_num(Z, nan=0.0)

        if self.int_scale.get() == "Logarithmic":
            Z = np.clip(Z, 1e-6, None)

        vmin, vmax = self._intensity_limits(Z)
        norm = self._make_norm(vmin, vmax)
        z_min, z_max = float(Z.min()), float(Z.max())
        clip_low, clip_high = vmin > z_min, vmax < z_max
        extend = "both" if (clip_low and clip_high) else "min" if clip_low else "max" if clip_high else "neither"
        self._update_intensity_info(Z)

        if self.smooth_contour.get():
            mesh = ax.pcolormesh(X, Y, Z, cmap=self.cmap_contour.get(), shading="auto", norm=norm)
        else:
            # levels evenly spaced in the chosen scale, so linear/sqrt/log all get a full color range
            levels = np.asarray(np.ma.filled(norm.inverse(np.linspace(0.0, 1.0, self.n_levels.get() + 1)), np.nan))
            mesh = ax.contourf(X, Y, Z, levels=levels, cmap=self.cmap_contour.get(), norm=norm, extend=extend)

        # the contour always stays at full scale on its own Y data range,
        # whatever happens on the electrochemistry panel side
        ax.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))

        cbar = self.figure.colorbar(mesh, ax=ax, extend=extend) if self.smooth_contour.get() \
            else self.figure.colorbar(mesh, ax=ax)
        self._apply_common_labels(ax, cbar)

        if ax_echem is not None:
            ex = self.echem_columns[self.echem_xcol.get()]
            ey = self._get_echem_y_series()
            ax_echem.plot(ex, ey, color=PINK, linewidth=1.3)
            # same numeric range as the contour, imposed explicitly (never the other way around)
            ax_echem.set_ylim(ax.get_ylim())
            # potential (or whatever X quantity) decreasing left to right, as requested
            ax_echem.set_xlim(float(np.nanmax(ex)), float(np.nanmin(ex)))
            ax_echem.set_xlabel(self.echem_xlabel.get() or self.echem_xcol.get(),
                                 fontsize=self.fs_labels.get())
            ax_echem.set_ylabel(self.echem_ylabel.get() or self.echem_ycol.get(),
                                 fontsize=self.fs_labels.get())
            ax_echem.yaxis.set_label_position("right")
            ax_echem.yaxis.tick_right()  # real tick values shown on the right side
            ax_echem.tick_params(labelsize=self.fs_ticks.get())
            ax_echem.grid(alpha=0.3)

            if self.echem_unwrap_y.get():
                # Y position uses the unwrapped (monotonic) scale, but the tick
                # LABELS show the original oscillating value at that position,
                # via interpolation on the unwrapped->raw mapping. Applied to
                # both axes since they share the same numeric Y domain.
                from matplotlib.ticker import FuncFormatter
                raw_full = self.echem_columns[self.echem_ycol.get()]
                ey_full = unwrap_cumulative(raw_full)

                def _raw_value_formatter(val, pos, _ey=ey_full, _raw=raw_full):
                    return f"{np.interp(val, _ey, _raw):.3g}"

                fmt = FuncFormatter(_raw_value_formatter)
                ax.yaxis.set_major_formatter(fmt)
                ax_echem.yaxis.set_major_formatter(fmt)

            # fixed margins: tight_layout handles gridspecs with independent axes poorly
            self.figure.subplots_adjust(left=0.11, right=0.90, top=0.93, bottom=0.13,
                                        wspace=CONTOUR_ECHEM_WSPACE)
            self._echem_ax = ax_echem
            self._echem_xlim_full = ax_echem.get_xlim()
        else:
            self.figure.tight_layout()
            self._echem_ax = None
            self._echem_xlim_full = None

        self.ax = ax
        self._contour_mesh = mesh
        self._contour_X = X
        self._contour_Y = Y
        self._contour_Z = Z

        # keep the electrochemistry panel's Y axis synced to the contour's, and
        # rescale the color/intensity scale to whatever is currently visible,
        # whatever triggers the change (toolbar zoom/pan, or the X-only zoom below)
        ax.callbacks.connect("xlim_changed", self._on_contour_view_changed)
        ax.callbacks.connect("ylim_changed", self._on_contour_view_changed)

        # dedicated "zoom X only, keep Y full scale" interaction
        Y_full = (float(np.nanmin(Y)), float(np.nanmax(Y)))
        self._x_span = SpanSelector(
            ax, lambda xmin, xmax: self._on_x_zoom_select(xmin, xmax, Y_full),
            direction="horizontal", useblit=False,
            props=dict(alpha=0.15, facecolor=BLUE_DARK), interactive=False)
        self._x_span.set_active(self.zoomx_mode.get())

    def _sync_echem_axes(self, ax):
        """Mirrors the contour's current Y range onto the electrochemistry
        panel, while forcing its X axis (potential) to stay at full scale."""
        if self._echem_ax is None:
            return
        self._echem_ax.set_ylim(ax.get_ylim())
        if self._echem_xlim_full is not None:
            self._echem_ax.set_xlim(self._echem_xlim_full)

    def _rescale_intensity_to_view(self, ax):
        """Recomputes the color/intensity scale from only the data currently
        visible within the contour's X/Y limits, so a zoomed-in region always
        shows its own full dynamic range instead of the whole series' range."""
        mesh = getattr(self, "_contour_mesh", None)
        Xd = getattr(self, "_contour_X", None)
        Yd = getattr(self, "_contour_Y", None)
        Zd = getattr(self, "_contour_Z", None)
        if mesh is None or Xd is None or Yd is None or Zd is None:
            return
        xlo, xhi = sorted(ax.get_xlim())
        ylo, yhi = sorted(ax.get_ylim())
        x_mask = (Xd >= xlo) & (Xd <= xhi)
        y_mask = (Yd >= ylo) & (Yd <= yhi)
        if not np.any(x_mask) or not np.any(y_mask):
            return
        sub = Zd[np.ix_(y_mask, x_mask)]
        if sub.size == 0:
            return
        vmin, vmax = self._intensity_limits(sub)
        try:
            mesh.set_clim(vmin, vmax)
        except Exception:
            pass

    # ---- intensity scale (contour) ----
    @staticmethod
    def _opt_float(var):
        """Float from a StringVar, or None when it is blank/invalid (= automatic)."""
        try:
            s = var.get().strip().replace(",", ".")
            return float(s) if s else None
        except (tk.TclError, ValueError):
            return None

    def _intensity_limits(self, data):
        """(vmin, vmax) of the color scale for `data`: percentile clipping first, then the
        optional absolute Min/Max values, which take precedence."""
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        try:
            plo, phi = float(self.int_plow.get()), float(self.int_phigh.get())
        except (tk.TclError, ValueError):
            plo, phi = 0.0, 100.0
        plo, phi = min(max(plo, 0.0), 100.0), min(max(phi, 0.0), 100.0)
        if phi <= plo:
            plo, phi = 0.0, 100.0
        vmin = float(np.min(finite)) if plo <= 0 else float(np.percentile(finite, plo))
        vmax = float(np.max(finite)) if phi >= 100 else float(np.percentile(finite, phi))
        a, b = self._opt_float(self.int_vmin), self._opt_float(self.int_vmax)
        if a is not None:
            vmin = a
        if b is not None:
            vmax = b
        log = self.int_scale.get() == "Logarithmic"
        if log:
            vmin = max(vmin, 1e-6)
        if vmax <= vmin:
            vmax = vmin * 10 if log else vmin + 1e-9
        return vmin, vmax

    def _make_norm(self, vmin, vmax):
        from matplotlib.colors import Normalize, LogNorm, PowerNorm
        kind = self.int_scale.get()
        if kind == "Logarithmic":
            return LogNorm(vmin=vmin, vmax=vmax)
        if kind == "Square root":
            return PowerNorm(0.5, vmin=vmin, vmax=vmax)
        if kind == "Power (gamma)":
            try:
                gamma = max(float(self.int_gamma.get()), 0.01)
            except (tk.TclError, ValueError):
                gamma = 0.5
            return PowerNorm(gamma, vmin=vmin, vmax=vmax)
        return Normalize(vmin=vmin, vmax=vmax)

    def _update_intensity_info(self, Z):
        """Shows the data range so sensible Min/Max values can be picked."""
        sample = Z.ravel()[::max(1, Z.size // 500000)]
        p50, p99 = np.percentile(sample, [50, 99])
        self.int_info.config(text=f"Data: min {Z.min():.3g} · median {p50:.3g} · "
                                   f"99th pct {p99:.3g} · max {Z.max():.3g}")

    def reset_intensity_scale(self):
        self.int_scale.set(INT_SCALES[0])
        self.int_gamma.set(0.5)
        self.int_plow.set(0.0)
        self.int_phigh.set(100.0)
        self.int_vmin.set("")
        self.int_vmax.set("")
        self.refresh_plot()

    def _on_contour_view_changed(self, ax):
        if getattr(self, "_syncing_echem", False):
            return
        self._syncing_echem = True
        try:
            self._sync_echem_axes(ax)
            self._rescale_intensity_to_view(ax)
            self.canvas.draw_idle()
        finally:
            self._syncing_echem = False

    def _on_x_zoom_select(self, xmin, xmax, y_full_range):
        if xmin == xmax:
            return
        ax = self.ax
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(*y_full_range)  # X-only zoom: Y always stays at full scale
        self._sync_echem_axes(ax)
        self._rescale_intensity_to_view(ax)
        self.canvas.draw_idle()

    def _on_zoomx_toggle(self):
        if self.zoomx_mode.get():
            # the toolbar's own Pan/Zoom tool holds an exclusive lock on the canvas
            # while active, which silently blocks our span selector -- release it
            if self.toolbar.mode == "zoom rect":
                self.toolbar.zoom()
            elif self.toolbar.mode == "pan/zoom":
                self.toolbar.pan()
        if self._x_span is not None:
            self._x_span.set_active(self.zoomx_mode.get())

    def _make_toolbar_exclusive_with_zoomx(self):
        """Wraps the toolbar's zoom()/pan() so that activating either one
        automatically turns off our custom 'Zoom X only' mode -- the two
        drag interactions must never be active on the canvas at the same
        time, or the toolbar silently wins and a normal X+Y zoom happens
        instead of the X-only one."""
        original_zoom = self.toolbar.zoom
        original_pan = self.toolbar.pan

        def wrapped_zoom(*args, **kwargs):
            original_zoom(*args, **kwargs)
            if self.toolbar.mode == "zoom rect" and self.zoomx_mode.get():
                self.zoomx_mode.set(False)
                if self._x_span is not None:
                    self._x_span.set_active(False)

        def wrapped_pan(*args, **kwargs):
            original_pan(*args, **kwargs)
            if self.toolbar.mode == "pan/zoom" and self.zoomx_mode.get():
                self.zoomx_mode.set(False)
                if self._x_span is not None:
                    self._x_span.set_active(False)

        self.toolbar.zoom = wrapped_zoom
        self.toolbar.pan = wrapped_pan

    def reset_contour_zoom(self):
        if self.view_mode.get() == "contour":
            self.refresh_plot()

    def apply_x_zoom(self):
        if self.series.n_frames == 0:
            messagebox.showinfo("Info", "Load a series of diffraction patterns first.")
            return
        if self.view_mode.get() != "contour":
            messagebox.showinfo("Info", "Switch to the Contour view first.")
            return
        try:
            xmin = float(self.zoomx_min.get())
            xmax = float(self.zoomx_max.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid range", "Please enter valid numeric X min/max values.")
            return
        if xmin >= xmax:
            messagebox.showwarning("Invalid range", "X min must be smaller than X max.")
            return
        ax = self.ax
        Y = self.series.y_values
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))  # keep Y at full scale, like the X-only drag zoom
        self._sync_echem_axes(ax)
        self._rescale_intensity_to_view(ax)
        self.canvas.draw_idle()

    def _draw_waterfall(self, ax):
        self.figure.clear()
        X = self.series.two_theta
        Z = self.series.intensities
        n = self.series.n_frames
        step = max(1, self.wf_step.get())
        highlights = self._parse_highlight_frames(n)
        idxs = sorted(set(range(0, n, step)) | highlights)
        cmap = plt.get_cmap(self.wf_cmap.get())
        Yv = self.series.y_values
        self._rebuild_hl_rows(highlights)
        hl_lw = self.wf_hl_lw.get()
        pending_sync = None

        if self.wf_3d.get():
            ax, ax_echem = self._make_anim_axes(self.figure, projection="3d", wspace=0.3)
            for k, i in enumerate(idxs):
                y_curve = np.nan_to_num(Z[i], nan=0.0)
                if i in highlights:
                    ax.plot(X, [Yv[i]] * len(X), y_curve, color=self._hl_color_for(i),
                            linewidth=hl_lw, zorder=10)
                else:
                    color = cmap(k / max(1, len(idxs) - 1))
                    ax.plot(X, [Yv[i]] * len(X), y_curve, color=color, linewidth=self.wf_lw.get())

            ax.set_xlabel(self.xlabel_var.get(), fontsize=self.fs_labels.get())
            ax.set_ylabel(self.ylabel_var.get(), fontsize=self.fs_labels.get())
            ax.set_zlabel(self.cbarlabel_var.get(), fontsize=self.fs_labels.get())
            if self.title_var.get():
                ax.set_title(self.title_var.get(), fontsize=self.fs_title.get())
            ax.tick_params(labelsize=self.fs_ticks.get())

            if self._last_3d_view is not None:
                elev, azim = self._last_3d_view
            else:
                elev, azim = self.wf_elev.get(), self.wf_azim.get()
                self._applied_3d = (elev, azim)
            ax.view_init(elev=elev, azim=azim)
        else:
            ax, ax_echem = self._make_anim_axes(self.figure)
            zmax = np.nanmax(Z) if n else 1.0
            offset = self.wf_offset.get() * zmax * 0.02 if self.wf_offset.get() else 0.0
            xshift = self.wf_xshift.get()

            for k, i in enumerate(idxs):
                y = np.nan_to_num(Z[i], nan=0.0) + k * offset
                x = X + k * xshift
                if i in highlights:
                    ax.plot(x, y, color=self._hl_color_for(i), linewidth=hl_lw, zorder=10)
                else:
                    color = cmap(k / max(1, len(idxs) - 1))
                    ax.plot(x, y, color=color, linewidth=self.wf_lw.get())

            self._apply_common_labels(ax)

            y0, y1 = float(Yv[idxs[0]]), float(Yv[idxs[-1]])
            if ax_echem is not None and offset > 0 and len(idxs) > 1 and y0 != y1:
                # same vertical alignment as the contour view: the stack (frame k sits at
                # k*offset) and the electrochemistry curve share the same frame range
                span = (len(idxs) - 1) * offset

                def sync_echem_y(_ax=None):
                    lo, hi = ax.get_ylim()
                    ax_echem.set_ylim(y0 + lo / span * (y1 - y0), y0 + hi / span * (y1 - y0))

                ax.callbacks.connect("ylim_changed", sync_echem_y)
                pending_sync = sync_echem_y

        if ax_echem is not None:
            self._style_echem_panel(ax_echem)
            if pending_sync is not None:  # after styling, so the aligned limits are the ones kept
                pending_sync()
        if self._echem_panel_ready():
            if self.wf_3d.get():
                self.figure.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.08)
            else:
                self.figure.subplots_adjust(left=0.11, right=0.90, top=0.92, bottom=0.13, wspace=ECHEM_WSPACE)
        else:
            self.figure.tight_layout()
        self.ax = ax

    def _hl_color_for(self, i):
        if i not in self.wf_hl_colors:
            self.wf_hl_colors[i] = HL_PALETTE[len(self.wf_hl_colors) % len(HL_PALETTE)]
        return self.wf_hl_colors[i]

    def _pick_hl_color(self, i):
        from tkinter import colorchooser
        _, hex_color = colorchooser.askcolor(color=self._hl_color_for(i),
                                              title=f"Color for frame {i + 1}")
        if not hex_color:
            return
        self.wf_hl_colors[i] = hex_color
        self._wf_hl_rendered = None
        self.refresh_plot()

    def _rebuild_hl_rows(self, frames):
        """One row (label + color swatch) per highlighted frame."""
        frames = sorted(frames)
        if frames == self._wf_hl_rendered:
            return
        self._wf_hl_rendered = frames
        for w in self.wf_hl_rows.winfo_children():
            w.destroy()
        for i in frames:
            color = self._hl_color_for(i)
            row = ttk.Frame(self.wf_hl_rows, style="Card.TFrame")
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"Frame {i + 1}", style="Card.TLabel", width=16).pack(side="left")
            tk.Button(row, width=8, relief="solid", bd=1, bg=color, activebackground=color,
                      command=lambda i=i: self._pick_hl_color(i)).pack(side="left")

    def _parse_highlight_frames(self, n):
        """Turns '5, 12, 20-25' (1-based) into a set of valid 0-based frame indices."""
        out = set()
        for tok in re.split(r"[,;\s]+", self.wf_hl_text.get().strip()):
            if not tok:
                continue
            try:
                if "-" in tok:
                    a, b = tok.split("-", 1)
                    lo, hi = sorted((int(a), int(b)))
                else:
                    lo = hi = int(tok)
            except ValueError:
                continue
            out.update(i - 1 for i in range(lo, hi + 1) if 1 <= i <= n)
        return out

    def _on_wf_3d_toggle(self):
        # vertical/horizontal offset only makes sense in 2D (in 3D, separation
        # happens naturally along the Y axis) — so disable them in 3D
        state = "disabled" if self.wf_3d.get() else "normal"
        self.wf_offset_spin.config(state=state)
        self.wf_xshift_spin.config(state=state)
        self._last_3d_view = None
        if hasattr(self, "canvas"):  # not created yet during the initial tab build
            self.refresh_plot()

    def reset_3d_view(self):
        self._last_3d_view = None
        self.refresh_plot()

    def _apply_3d_angles(self, event=None):
        """Applies the typed elevation / azimuth at once, replacing the view left by the last
        mouse rotation. Invalid or half-typed values are ignored."""
        try:
            elev, azim = float(self.wf_elev.get()), float(self.wf_azim.get())
        except (tk.TclError, ValueError):
            return
        if event is not None and event.type == tk.EventType.FocusOut and (elev, azim) == self._applied_3d:
            return  # leaving a field without editing it must not undo a mouse rotation
        self._applied_3d = (elev, azim)
        self._last_3d_view = None
        ax = getattr(self, "ax", None)
        if (self.series.n_frames and self.view_mode.get() == "waterfall" and self.wf_3d.get()
                and hasattr(ax, "view_init")):
            ax.view_init(elev=elev, azim=azim)
            self.canvas.draw_idle()

    def _on_canvas_release(self, event):
        """Remembers the view angle after a manual mouse rotation, so it
        isn't lost on the next redraw (e.g. clicking Refresh)."""
        if self.view_mode.get() == "waterfall" and self.wf_3d.get():
            elev = getattr(self.ax, "elev", None)
            azim = getattr(self.ax, "azim", None)
            if elev is not None and azim is not None:
                self._last_3d_view = (elev, azim)

    def _make_anim_axes(self, fig, projection=None, wspace=ECHEM_WSPACE):
        """Diffraction axes, plus the electrochemistry axes on the right when available."""
        if self._echem_panel_ready():
            gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=wspace)
            return fig.add_subplot(gs[0, 0], projection=projection), fig.add_subplot(gs[0, 1])
        return fig.add_subplot(111, projection=projection), None

    def _style_echem_panel(self, ax_echem):
        """Draws the static electrochemistry curve and dresses its axes."""
        ex = self.echem_columns[self.echem_xcol.get()]
        ey = self._get_echem_y_series()
        ax_echem.plot(ex, ey, color=PINK, linewidth=1.0, alpha=0.8)
        # potential (or whatever X quantity) decreasing left to right, as requested
        ax_echem.set_xlim(float(np.nanmax(ex)), float(np.nanmin(ex)))
        ax_echem.set_xlabel(self.echem_xlabel.get() or self.echem_xcol.get(),
                             fontsize=self.fs_labels.get())
        ax_echem.set_ylabel(self.echem_ylabel.get() or self.echem_ycol.get(),
                             fontsize=self.fs_labels.get())
        ax_echem.yaxis.set_label_position("right")
        ax_echem.yaxis.tick_right()
        ax_echem.tick_params(labelsize=self.fs_ticks.get())
        ax_echem.grid(alpha=0.3)

        if self.echem_unwrap_y.get():
            from matplotlib.ticker import FuncFormatter
            raw_full = self.echem_columns[self.echem_ycol.get()]
            ey_full = unwrap_cumulative(raw_full)

            def _raw_value_formatter(val, pos, _ey=ey_full, _raw=raw_full):
                return f"{np.interp(val, _ey, _raw):.3g}"

            ax_echem.yaxis.set_major_formatter(FuncFormatter(_raw_value_formatter))

    def _plot_echem_marker(self, ax_echem, i):
        """Marker showing where diffraction pattern i falls on the E vs. Y curve."""
        marker = self._get_echem_marker_position(i)
        if marker is not None:
            ax_echem.plot(marker[0], marker[1], marker="o", markersize=8, markerfacecolor=BLUE,
                          markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    # ---- cumulative animation: patterns pile up, waterfall style ----
    # ---- dragging the electrochemistry marker to scrub through the patterns ----
    def _marker_hit(self, event):
        """True if the mouse is over the round marker of the animation currently drawn."""
        ax = self._anim_echem_ax
        if ax is None or event.x is None:
            return False
        pos = self._get_echem_marker_position(self._anim_drawn_frame)
        if pos is None:
            return False
        px, py = ax.transData.transform(pos)
        return (event.x - px) ** 2 + (event.y - py) ** 2 <= 14 ** 2

    def _marker_interaction_allowed(self):
        return (self.view_mode.get() in ANIM_MODES and self.series.n_frames > 0
                and self.toolbar.mode == "")  # the toolbar's zoom/pan tools keep priority

    def _set_hand_cursor(self, hand):
        if hand != self._cursor_is_hand:
            self._cursor_is_hand = hand
            self.canvas.get_tk_widget().config(cursor="hand2" if hand else "arrow")

    def _on_marker_press(self, event):
        if event.button != 1 or not self._marker_interaction_allowed() or not self._marker_hit(event):
            return
        self.pause_animation()
        self.current_frame.set(self._anim_drawn_frame)
        self._marker_drag = True

    def _on_marker_motion(self, event):
        if not self._marker_drag:
            self._set_hand_cursor(self._marker_interaction_allowed() and self._marker_hit(event))
            return
        ax = self._anim_echem_ax
        if ax is None or event.x is None:
            return
        # only the vertical mouse position matters: it picks the frame, and the marker
        # is then placed on the curve for that frame, so it can never leave the curve
        y_mouse = ax.transData.inverted().transform((event.x, event.y))[1]
        i = int(np.nanargmin(np.abs(self.series.y_values - y_mouse)))
        if i == self._anim_drawn_frame:
            return
        self.current_frame.set(i)
        mode = self.view_mode.get()
        self._slider_updating = True
        self.anim_widgets[mode]["slider"].set(i)
        self._slider_updating = False
        self._draw_anim_frame(i)

    def _on_marker_release(self, event):
        self._marker_drag = False

    def _cum_params(self):
        """Snapshot of every setting the cumulative drawing needs, so the same
        rendering code can also run in the video-export thread without touching Tk."""
        X = self.series.two_theta
        Z = self.series.intensities
        n = self.series.n_frames
        zmax = float(np.nanmax(Z))
        zmin = min(0.0, float(np.nanmin(Z)))
        offset = self.wf_offset.get() * zmax * 0.02
        xshift = self.wf_xshift.get()
        ymax = zmax + (n - 1) * offset
        pad = 0.02 * (ymax - zmin)
        highlights = {i: self._hl_color_for(i) for i in self._parse_highlight_frames(n)}
        return dict(
            highlights=highlights, hl_lw=self.wf_hl_lw.get(),
            offset=offset, xshift=xshift, lw=self.wf_lw.get(), cmap=plt.get_cmap(self.wf_cmap.get()),
            xlim=(float(X.min()), float(X.max()) + (n - 1) * xshift),
            ylim=(zmin, ymax + pad),
            base_title=self.title_var.get() or "Diffraction pattern",
            xlabel=self.xlabel_var.get(), ylabel=self.cbarlabel_var.get(),
            fs_title=self.fs_title.get(), fs_labels=self.fs_labels.get(), fs_ticks=self.fs_ticks.get())

    def _draw_cumulative_curves(self, ax, i, p):
        """Draws patterns 0..i stacked on an empty axes. Colors are tied to the frame
        index (not to how many are shown), so a curve keeps its color as others are added."""
        from matplotlib.collections import LineCollection
        X = self.series.two_theta
        n = self.series.n_frames
        Z = np.nan_to_num(self.series.intensities[:i + 1], nan=0.0)
        hl = p["highlights"]
        for group, lw, zorder in (([k for k in range(i + 1) if k not in hl], p["lw"], 2),
                                  ([k for k in range(i + 1) if k in hl], p["hl_lw"], 3)):
            if not group:
                continue
            segs = [np.column_stack((X + k * p["xshift"], Z[k] + k * p["offset"])) for k in group]
            colors = [hl[k] if k in hl else p["cmap"](k / max(1, n - 1)) for k in group]
            ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, zorder=zorder))
        ax.set_xlim(*p["xlim"])
        ax.set_ylim(*p["ylim"])
        yval = self.series.y_values[i]
        ax.set_title(f"{p['base_title']} — frame {i+1}/{n} (Y={yval:.3g})", fontsize=p["fs_title"])
        ax.set_xlabel(p["xlabel"], fontsize=p["fs_labels"])
        ax.set_ylabel(p["ylabel"], fontsize=p["fs_labels"])
        ax.tick_params(labelsize=p["fs_ticks"])

    def draw_cumulative_frame(self, i):
        if self.series.n_frames == 0:
            return
        i = max(0, min(i, self.series.n_frames - 1))
        self.figure.clear()
        ax, ax_echem = self._make_anim_axes(self.figure)

        self._draw_cumulative_curves(ax, i, self._cum_params())
        if self._cum_xlim is not None:
            ax.set_xlim(self._cum_xlim)
        if self._cum_ylim is not None:
            ax.set_ylim(self._cum_ylim)

        if ax_echem is not None:
            self._style_echem_panel(ax_echem)
            self._plot_echem_marker(ax_echem, i)
            self.figure.subplots_adjust(left=0.1, right=0.90, top=0.90, bottom=0.13, wspace=ECHEM_WSPACE)
        else:
            self.figure.tight_layout()

        self.ax = ax
        self._anim_echem_ax = ax_echem
        self._anim_drawn_frame = i
        ax.callbacks.connect("xlim_changed", self._on_cum_xlim_changed)
        ax.callbacks.connect("ylim_changed", self._on_cum_ylim_changed)
        self.canvas.draw()

    def _on_cum_xlim_changed(self, ax):
        self._cum_xlim = ax.get_xlim()

    def _on_cum_ylim_changed(self, ax):
        self._cum_ylim = ax.get_ylim()

    def draw_animation_frame(self, i):
        if self.series.n_frames == 0:
            return
        i = max(0, min(i, self.series.n_frames - 1))
        self.figure.clear()
        ax, ax_echem = self._make_anim_axes(self.figure)

        X = self.series.two_theta
        y = np.nan_to_num(self.series.intensities[i], nan=0.0)
        ax.plot(X, y, color=BLUE, linewidth=1.3)
        if self._anim_ylim is not None:
            ax.set_ylim(self._anim_ylim)
        else:
            ax.set_ylim(0, np.nanmax(self.series.intensities) * 1.05)
        if self._anim_xlim is not None:
            ax.set_xlim(self._anim_xlim)
        else:
            ax.set_xlim(X.min(), X.max())
        yval = self.series.y_values[i]
        base_title = self.title_var.get() or "Diffraction pattern"
        ax.set_title(f"{base_title} — frame {i+1}/{self.series.n_frames} (Y={yval:.3g})",
                     fontsize=self.fs_title.get())
        ax.set_xlabel(self.xlabel_var.get(), fontsize=self.fs_labels.get())
        ax.set_ylabel(self.cbarlabel_var.get(), fontsize=self.fs_labels.get())
        ax.tick_params(labelsize=self.fs_ticks.get())

        if ax_echem is not None:
            self._style_echem_panel(ax_echem)
            self._plot_echem_marker(ax_echem, i)
            self.figure.subplots_adjust(left=0.1, right=0.90, top=0.90, bottom=0.13, wspace=ECHEM_WSPACE)
        else:
            self.figure.tight_layout()

        self.ax = ax
        self._anim_echem_ax = ax_echem
        self._anim_drawn_frame = i
        ax.callbacks.connect("xlim_changed", self._on_anim_xlim_changed)
        ax.callbacks.connect("ylim_changed", self._on_anim_ylim_changed)
        self.canvas.draw()

    def _on_anim_xlim_changed(self, ax):
        self._anim_xlim = ax.get_xlim()

    def _on_anim_ylim_changed(self, ax):
        self._anim_ylim = ax.get_ylim()

    def reset_anim_zoom(self):
        self._anim_xlim = None
        self._anim_ylim = None
        self._cum_xlim = None
        self._cum_ylim = None
        if self.view_mode.get() in ANIM_MODES:
            self._draw_anim_frame(self.current_frame.get())

    # ---------------------------- Interactive animation ----------------------------
    def play_animation(self, loop=True, mode="anim"):
        if self.series.n_frames == 0:
            return
        if self.playing:
            return  # avoids stacking multiple loops on repeated clicks
        self.view_mode.set(mode)
        self._anim_loop = loop
        n = self.series.n_frames
        if not loop and self.current_frame.get() >= n - 1:
            self.current_frame.set(0)  # restart from the beginning for a fresh single pass
        self.playing = True
        self._animate_step()

    def pause_animation(self):
        self.playing = False

    def _animate_step(self):
        if not self.playing:
            return
        mode = self.view_mode.get()
        if mode not in ANIM_MODES:
            self.playing = False  # the user switched to another view while playing
            return
        i = self.current_frame.get()
        n = self.series.n_frames
        self._draw_anim_frame(i)
        self._slider_updating = True  # move the slider without redrawing the frame a second time
        self.anim_widgets[mode]["slider"].set(i)
        self._slider_updating = False
        if i >= n - 1 and not getattr(self, "_anim_loop", True):
            self.playing = False  # play-once mode: stop after the last frame
            return
        nxt = (i + 1) % n
        self.current_frame.set(nxt)
        self.after(int(self.anim_speed.get()), self._animate_step)

    # ---------------------------- Video export ----------------------------
    def export_mp4(self, mode="anim"):
        if self.series.n_frames == 0:
            messagebox.showinfo("Info", "Load a series of diffraction patterns first.")
            return
        if getattr(self, "_mp4_exporting", False):
            messagebox.showinfo("Info", "A video export is already in progress.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
        if not path:
            return

        # stop interactive playback and lock the controls during the export,
        # to avoid two threads drawing at the same time
        self.pause_animation()
        cum_params = self._cum_params() if mode == "anim_cum" else None  # read Tk settings on the main thread
        self._mp4_exporting = True
        self._mp4_done_path = None
        self._set_anim_controls_state("disabled")
        for w in self.anim_widgets.values():
            w["export"].config(state="disabled")

        self.status_var.set("Video export in progress…")
        self.progress["maximum"] = self.series.n_frames
        self.progress["value"] = 0
        thread = threading.Thread(target=self._export_mp4_worker, args=(path, cum_params), daemon=True)
        thread.start()

    def _set_anim_controls_state(self, state):
        """Only enables the controls of the animation tab matching the current mode."""
        active = self.view_mode.get()
        for mode, w in self.anim_widgets.items():
            st = state if (state == "normal" and mode == active) else "disabled"
            for key in ("loop", "once", "pause", "slider", "speed"):
                w[key].config(state=st)

    def _export_mp4_worker(self, path, cum_params=None):
        # This thread must NEVER touch a Tkinter widget directly:
        # any UI update is delegated to the main thread via self.after(0, ...).
        try:
            show_echem = self._echem_panel_ready()
            fig = plt.figure(figsize=(10, 5) if show_echem else (7, 5), dpi=self.anim_dpi.get())
            ax, ax_echem = self._make_anim_axes(fig)

            X = self.series.two_theta
            ymax = np.nanmax(self.series.intensities) * 1.05
            n = self.series.n_frames
            base_title = self.title_var.get() or "Diffraction pattern"

            marker_artist = None
            if ax_echem is not None:
                # the echem curve itself is static: draw it once, outside the
                # per-frame loop, and only move the marker each frame (much
                # faster than redrawing a multi-thousand-point curve every frame)
                self._style_echem_panel(ax_echem)
                marker_artist, = ax_echem.plot([], [], marker="o", markersize=8, markerfacecolor=BLUE,
                                                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
                fig.subplots_adjust(left=0.09, right=0.92, top=0.90, bottom=0.13, wspace=ECHEM_WSPACE)

            def update(i):
                ax.clear()
                if cum_params is not None:
                    self._draw_cumulative_curves(ax, i, cum_params)
                else:
                    y = np.nan_to_num(self.series.intensities[i], nan=0.0)
                    ax.plot(X, y, color=BLUE, linewidth=1.3)
                    ax.set_ylim(0, ymax)
                    ax.set_xlim(X.min(), X.max())
                    yval = self.series.y_values[i]
                    ax.set_title(f"{base_title} — frame {i+1}/{n} (Y={yval:.3g})", fontsize=self.fs_title.get())
                    ax.set_xlabel(self.xlabel_var.get(), fontsize=self.fs_labels.get())
                    ax.set_ylabel(self.cbarlabel_var.get(), fontsize=self.fs_labels.get())
                    ax.tick_params(labelsize=self.fs_ticks.get())
                if marker_artist is not None:
                    marker = self._get_echem_marker_position(i)
                    if marker is not None:
                        marker_artist.set_data([marker[0]], [marker[1]])
                if ax_echem is None:
                    fig.tight_layout()
                self.after(0, lambda v=i + 1: self.progress.config(value=v))
                return ax.lines

            fps = max(1, round(1000 / max(10, self.anim_speed.get())))
            anim = FuncAnimation(fig, update, frames=n, blit=False)
            writer = FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(path, writer=writer, dpi=self.anim_dpi.get())
            plt.close(fig)
            self.after(0, lambda: self.status_var.set(f"Video exported: {path}"))
            self._mp4_done_path = path  # announced by _finish_mp4_export, once the controls are re-enabled
        except FileNotFoundError:
            self.after(0, lambda: messagebox.showerror(
                "ffmpeg not found",
                "MP4 export requires ffmpeg installed and available in the PATH.\n\n"
                "Windows: winget install ffmpeg (then restart the terminal)\n"
                "macOS: brew install ffmpeg\n"
                "Linux: sudo apt install ffmpeg"))
            self.after(0, lambda: self.status_var.set("Export failed: ffmpeg not found."))
        except Exception as e:
            msg = str(e)
            self.after(0, lambda: messagebox.showerror("Video export error", msg))
            self.after(0, lambda: self.status_var.set("Video export failed."))
        finally:
            self.after(0, self._finish_mp4_export)

    def _finish_mp4_export(self):
        """Properly re-enables the controls once the export is done (called on the main thread)."""
        self.progress["value"] = 0
        self._mp4_exporting = False
        for w in self.anim_widgets.values():
            w["export"].config(state="normal")
        self._set_anim_controls_state("normal")
        done_path = getattr(self, "_mp4_done_path", None)
        if done_path:
            self._mp4_done_path = None
            messagebox.showinfo("Export complete", f"The video was exported successfully:\n{done_path}")


if __name__ == "__main__":
    app = OperandoViewerApp()
    app.mainloop()
