"""
Draw publication-ready figures for the satellite hybrid experiment.

This script only reads CSV files from satellite_hybrid_results and writes new
SVG figures to satellite_hybrid_figures. It does not modify the original
experiment code or the original result files.
"""

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# ----------------------------- User settings -----------------------------
# The paper figures are drawn from one iteration directory. The current paper
# target is iter_08, but this can be changed to another existing iteration.
TARGET_ITERATION = 8

# Error panels are exported for all iterations in this inclusive range.
ERROR_ITERATIONS = range(1, 9)

# Root paths. Keep the script self-contained inside Experiment_Code.
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "satellite_hybrid_results"
OUTPUT_DIR = SCRIPT_DIR / "satellite_hybrid_figures"

# 16 cm is a common full-column maximum width. Matplotlib uses inches.
CM_TO_INCH = 1.0 / 2.54
MAX_WIDTH_IN = 16.0 * CM_TO_INCH

FONT_SIZE = 10
PLOT_DPI = 300
TIMES_NEW_ROMAN_PATH = Path(r"C:\Windows\Fonts\times.ttf")
# Original blue/red pair used in the previous figures.
RAW_COLOR = "#1f77b4"
FILTERED_COLOR = "#d62728"
NUM_COLOR = "#1f77b4"
EXP_COLOR = "#d62728"
ERROR_COLOR_dis = "#1f77b4"
ERROR_COLOR_rot = "#d62728"
ERROR_PANEL_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.linewidth": 1,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 0.8,
    "figure.dpi": PLOT_DPI,
}


def setup_style():
    """Apply one consistent SCI-style matplotlib theme.

    Times New Roman is loaded from the Windows font file directly. This avoids
    matplotlib falling back to another serif font if its font cache is stale.
    """
    if TIMES_NEW_ROMAN_PATH.exists():
        font_manager.fontManager.addfont(str(TIMES_NEW_ROMAN_PATH))
    times_font_name = font_manager.FontProperties(
        fname=str(TIMES_NEW_ROMAN_PATH) if TIMES_NEW_ROMAN_PATH.exists() else None
    ).get_name()

    plt.rcParams.update({
        "font.family": times_font_name,
        "font.serif": [times_font_name],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Word may substitute fonts when SVG text remains editable. Convert all
        # text to vector paths so the Times New Roman appearance is preserved
        # after inserting or pasting the SVG into Word.
        "svg.fonttype": "path",
    })


def read_csv_columns(path):
    """Read a CSV file into a dict of numpy arrays keyed by column name."""
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data in {path}")

    data = {}
    for key in rows[0].keys():
        data[key] = np.array([float(row[key]) for row in rows], dtype=float)
    return data


def finish_figure(fig, path_stem, spine_width=0.8):
    """
    Save only SVG output and apply final axis styling.

    All legends are forced to the upper-right corner. For twin-y axes this may
    move both legends to the same corner; those legends are combined manually in
    the plotting function before this helper is called.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    # Force every text artist again at save time. The generated SVG should
    # explicitly contain Times New Roman rather than a generic serif fallback.
    for text in fig.findobj(match=plt.Text):
        text.set_fontname("Times New Roman")
    for ax in fig.axes:
        ax.grid(False)
        ax.tick_params(which="both", top=False)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            legend.set_bbox_to_anchor((1.0, 1.0))
            legend._loc = 1  # upper right
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
    fig.tight_layout()
    svg_path = path_stem.with_suffix(".svg")
    if svg_path.exists():
        svg_path.unlink()
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def normalize_error_svg_font(svg_path):
    """
    Make error-panel SVG font declarations explicit.

    Some editors do not reliably interpret matplotlib's compact SVG shorthand
    such as ``font: 10px 'Times New Roman'``. The error panels are small and
    text-heavy, so rewrite the shorthand into explicit font-size/font-family
    declarations while keeping all text editable.
    """
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(
        r"font:\s*([0-9.]+)px 'Times New Roman'",
        r"font-size: \1px; font-family: 'Times New Roman'",
        text,
    )
    svg_path.write_text(text, encoding="utf-8")


def add_panel_labels(axes):
    """Add (a), (b), ... labels inside the top-left of each subplot."""
    labels = [f"({chr(ord('a') + i)})" for i in range(len(axes))]
    for ax, label in zip(axes, labels):
        ax.text(
            0.02, 0.96, label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE,
            fontweight="normal",
        )


def get_iteration_dir(iteration):
    """Return the requested iteration directory."""
    iter_dir = RESULTS_DIR / "iteration_results" / f"iter_{iteration:02d}"
    if not iter_dir.exists():
        raise FileNotFoundError(f"Missing iteration result directory: {iter_dir}")
    return iter_dir


def single_sided_amplitude_spectrum(t, y):
    """Compute a simple single-sided FFT amplitude spectrum."""
    dt = float(np.mean(np.diff(t)))
    y_zero_mean = np.asarray(y, dtype=float) - np.mean(y)
    freq = np.fft.rfftfreq(y_zero_mean.size, dt)
    amp = 2.0 * np.abs(np.fft.rfft(y_zero_mean)) / y_zero_mean.size
    if amp.size:
        amp[0] *= 0.5
    return freq, amp


def expand_ylim(values, margin=0.24):
    """Return y-limits with extra headroom for upper-right legends."""
    ymin = min(float(np.min(v)) for v in values)
    ymax = max(float(np.max(v)) for v in values)
    if np.isclose(ymin, ymax):
        pad = 1.0 if np.isclose(ymax, 0.0) else abs(ymax) * margin
    else:
        pad = (ymax - ymin) * margin
    return ymin - 0.10 * pad, ymax + pad


def expand_ylim_with_small_lower_pad(values, margin=0.45):
    """Return y-limits with legend headroom but only a small lower-side pad."""
    ymin = min(float(np.min(v)) for v in values)
    ymax = max(float(np.max(v)) for v in values)
    if np.isclose(ymin, ymax):
        pad = 1.0 if np.isclose(ymax, 0.0) else abs(ymax) * margin
    else:
        pad = (ymax - ymin) * margin
    return ymin - 0.02 * pad, ymax + pad


def expand_positive_log_ylim(values, upper_factor=3.0):
    """Return semilog y-limits with extra upper room for a top-right legend."""
    positive = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return 1e-12, 1.0
    ymin = float(np.min(positive))
    ymax = float(np.max(positive))
    return ymin / 1.6, ymax * upper_factor


def symmetric_ylim(values, margin=1.12):
    """Return symmetric y-limits around zero."""
    max_abs = max(float(np.max(np.abs(v))) for v in values)
    if np.isclose(max_abs, 0.0):
        max_abs = 1.0
    return -margin * max_abs, margin * max_abs


def add_frequency_zoom_inset(ax, freq, raw_amp, filtered_amp):
    """Add a 0-3 Hz zoom inset to an FFT subplot.

    The inset keeps the same y-axis range as the parent FFT axis, so the local
    peak amplitudes can be compared directly with the full-spectrum panel.
    """
    # Put the inset at center-right so it does not collide with the legend that
    # is forced to the upper-right corner.
    axins = inset_axes(ax, width="50%", height="46%", loc="center right", borderpad=0.75)
    # Raw data are drawn as dashed lines; filtered data are drawn as solid
    # lines. This follows the manuscript convention for before/after filtering.
    axins.plot(freq, raw_amp, color=FILTERED_COLOR,linewidth=0.9, zorder=3)
    axins.plot(freq, filtered_amp, color=RAW_COLOR,  linewidth=1.1, linestyle="--", zorder=4)
    axins.set_xlim(0.0, 3.0)
    axins.set_ylim(*ax.get_ylim())
    axins.set_xticks([0, 1, 2, 3])
    axins.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axins.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    axins.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    axins.tick_params(labelsize=8, direction="in", top=False, right=False)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)


def plot_convergence(conv, out_dir):
    """Draw 1x2 convergence and relaxation-factor figure."""
    iteration = conv["index"]
    right_rmse = conv["right_interface_rmse"]
    left_rmse = conv["left_interface_rmse"]

    # The first plotted relaxation point is the initial fixed-point iteration.
    # It is defined as 1.0, then the adaptive factors from the CSV are shifted
    # one step to the right.
    right_relax = np.r_[1.0, conv["right_interface_relaxation"]]
    left_relax = np.r_[1.0, conv["left_interface_relaxation"]]
    relax_iteration = np.arange(1, right_relax.size + 1)

    fig, axes = plt.subplots(1, 2, figsize=(MAX_WIDTH_IN, 2.55))

    axes[0].semilogy(
        iteration, right_rmse,
        marker="o", markersize=3.0, markeredgewidth=0.6, linewidth=1,
        color=RAW_COLOR, label="Right interface",
    )
    axes[0].semilogy(
        iteration, left_rmse,
        marker="o", markersize=3.0, markeredgewidth=0.6, linewidth=1,
        linestyle="--", zorder=4,
        color=FILTERED_COLOR, label="Left interface",
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("RMSE (m)")
    axes[0].set_ylim(1e-3, 1.0)
    axes[0].set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(
        relax_iteration, right_relax,
        marker="o", markersize=3.0, markeredgewidth=0.6, linewidth=1,
        color=RAW_COLOR, label="Right interface",
    )
    axes[1].plot(
        relax_iteration, left_relax,
        marker="o", markersize=3.0, markeredgewidth=0.6, linewidth=1,
        linestyle="--", zorder=4,
        color=FILTERED_COLOR, label="Left interface",
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Relaxation factor")
    axes[1].set_ylim(0.0, 1.4)
    axes[1].set_yticks([0.0, 0.3, 0.6, 0.9, 1.2])
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend(loc="upper right", frameon=False)

    add_panel_labels(axes.ravel())
    finish_figure(fig, out_dir / "01_convergence_history")


def plot_one_interface_force_moment(ts, out_dir, side):
    """
    Draw a 2x2 force/moment time-frequency figure for one interface.

    Raw signals are dashed and filtered signals are solid. This makes the
    before/after filtering comparison visible even when the two curves overlap.
    """
    t = ts["Time(s)"]
    side_title = "Left" if side == "left" else "Right"
    prefix = "Left" if side == "left" else "Right"
    file_tag = "02l" if side == "left" else "02r"
    if side == "left":
        force_ylim = (-5.0, 7.0)
        force_ticks = np.linspace(-5.0, 7.0, 5)
        force_fft_ylim = (0.0, 1.5)
        moment_ylim = (-4.0, 6.0)
        moment_ticks = np.linspace(-4.0, 6.0, 6)
        moment_fft_ylim = (0.0, 1.0)
    else:
        force_ylim = (-9.0, 11.0)
        force_ticks = np.linspace(-9.0, 11.0, 5)
        force_fft_ylim = (0.0, 2.5)
        moment_ylim = (-4.0, 8.0)
        moment_ticks = np.linspace(-4.0, 8.0, 5)
        moment_fft_ylim = (0.0, 1.5)

    force_raw = ts[f"Fb_{prefix}_Raw(N)"]
    force_filtered = ts[f"Fb_{prefix}_Filtered(N)"]
    moment_raw = ts[f"Mb_{prefix}_Raw(Nm)"]
    moment_filtered = ts[f"Mb_{prefix}_Filtered(Nm)"]

    fig, axes = plt.subplots(2, 2, figsize=(MAX_WIDTH_IN, 4.85))

    filtered_force_line, = axes[0, 0].plot(
        t, force_filtered, color=RAW_COLOR, label="Filtered",
        linewidth=1.1, linestyle="--",zorder=4,
    )
    raw_force_line, = axes[0, 0].plot(
        t, force_raw, color=FILTERED_COLOR, label="Raw",
        linewidth=0.9,  zorder=3,
    )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Force (N)")
    axes[0, 0].set_title(f"{side_title} force time response")
    axes[0, 0].set_ylim(*force_ylim)
    axes[0, 0].set_yticks(force_ticks)
    axes[0, 0].legend([raw_force_line, filtered_force_line], ["Raw", "Filtered"], loc="upper right", frameon=False)

    filtered_moment_line, = axes[1, 0].plot(
        t, moment_filtered, color=RAW_COLOR, label="Filtered",
        linewidth=1.1, linestyle="--",zorder=4,
    )
    raw_moment_line, = axes[1, 0].plot(
        t, moment_raw, color=FILTERED_COLOR, label="Raw",
        linewidth=0.9,  zorder=3,
    )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Moment (N·m)")
    axes[1, 0].set_title(f"{side_title} moment time response")
    axes[1, 0].set_ylim(*moment_ylim)
    axes[1, 0].set_yticks(moment_ticks)
    axes[1, 0].legend([raw_moment_line, filtered_moment_line], ["Raw", "Filtered"], loc="upper right", frameon=False)

    freq, force_raw_amp = single_sided_amplitude_spectrum(t, force_raw)
    freq, force_filtered_amp = single_sided_amplitude_spectrum(t, force_filtered)
    filtered_force_amp_line, = axes[0, 1].plot(
        freq, force_filtered_amp, color=RAW_COLOR, label="Filtered",
        linewidth=1.1, linestyle="--",zorder=4,
    )
    raw_force_amp_line, = axes[0, 1].plot(
        freq, force_raw_amp, color=FILTERED_COLOR, label="Raw",
        linewidth=0.9,  zorder=3,
    )
    axes[0, 1].set_xlim(0, 20)
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Amplitude (N)")
    axes[0, 1].set_title(f"{side_title} force spectrum")
    axes[0, 1].set_ylim(*force_fft_ylim)
    add_frequency_zoom_inset(axes[0, 1], freq, force_raw_amp, force_filtered_amp)
    axes[0, 1].legend([raw_force_amp_line, filtered_force_amp_line], ["Raw", "Filtered"], loc="upper right", frameon=False)

    freq, moment_raw_amp = single_sided_amplitude_spectrum(t, moment_raw)
    freq, moment_filtered_amp = single_sided_amplitude_spectrum(t, moment_filtered)
    filtered_moment_amp_line, = axes[1, 1].plot(
        freq, moment_filtered_amp, color=RAW_COLOR, label="Filtered",
        linewidth=1.1, linestyle="--", zorder=4,
    )
    raw_moment_amp_line, = axes[1, 1].plot(
        freq, moment_raw_amp, color=FILTERED_COLOR, label="Raw",
        linewidth=0.9, zorder=3,
    )
    axes[1, 1].set_xlim(0, 20)
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Amplitude (N·m)")
    axes[1, 1].set_title(f"{side_title} moment spectrum")
    axes[1, 1].set_ylim(*moment_fft_ylim)
    add_frequency_zoom_inset(axes[1, 1], freq, moment_raw_amp, moment_filtered_amp)
    axes[1, 1].legend([raw_moment_amp_line, filtered_moment_amp_line], ["Raw", "Filtered"], loc="upper right", frameon=False)

    add_panel_labels(axes.ravel())
    finish_figure(fig, out_dir / f"{file_tag}_interface_force_moment")


def plot_external_excitation(ts, out_dir):
    """Draw external force and moment in one dual-y-axis figure."""
    t = ts["Time(s)"]
    fig, ax_force = plt.subplots(figsize=(MAX_WIDTH_IN * 0.72, 2.65))
    ax_moment = ax_force.twinx()

    line_force, = ax_force.plot(
        t, ts["Ext_Force_Body(N)"],
        color=RAW_COLOR,
        linewidth=1,
        label="Force",
    )
    line_moment, = ax_moment.plot(
        t, ts["Ext_Moment_Body(Nm)"],
        color=FILTERED_COLOR,
        linestyle="--",
        linewidth=1,
        label="Moment",
    )

    ax_force.set_xlabel("Time (s)")
    ax_force.set_ylabel("Force (N)")
    ax_moment.set_ylabel("Moment (N·m)")
    ax_force.set_ylim(-800.0, 1000.0)
    ax_moment.set_ylim(-120.0, 160.0)
    ax_force.set_yticks([-800.0, -400.0, 0.0, 400.0, 800.0])
    ax_moment.set_yticks([-120.0, -60.0, 0.0, 60.0, 120.0])
    # The curves keep their colors, but both y-axis tick labels stay black.
    # Axis tick styling for publication-quality hybrid-simulation figures.
    ax_force.tick_params(
        axis="y",
        direction="in",    # Robotic-arm-based offline RTHS workflow step.
        top=False,
        right=False,
        colors="black",
        labelcolor="black"
    )
    # Robotic-arm-based offline RTHS workflow step.
    ax_moment.tick_params(
        axis="y",
        direction="in",    # Robotic-arm-based offline RTHS workflow step.
        left=False,
        top=False,
        right=True,        # Robotic-arm-based offline RTHS workflow step.
        colors="black",
        labelcolor="black"
    )
    ax_force.spines["left"].set_color("black")
    ax_moment.spines["right"].set_color("black")

    ax_force.legend(
        [line_force, line_moment],
        ["Force", "Moment"],
        loc="upper right",
        frameon=False,
    )
    finish_figure(fig, out_dir / "02b_external_excitation")


def plot_interface_response(ts, out_dir):
    """
    Draw numerical and experimental interface responses.

    Rows are left and right interfaces. Columns are displacement and rotation.
    Command curves are intentionally omitted for the paper figure.
    """
    t = ts["Time(s)"]
    fig, axes = plt.subplots(2, 2, figsize=(MAX_WIDTH_IN*0.92, 4.3), sharex=True)

    rows = [
        ("Left", "Left", 0),
        ("Right", "Right", 1),
    ]
    response_ylims = {
        (0, 0): (-0.2, 0.3),
        (0, 1): (-0.1, 0.5),
        (1, 0): (-0.1, 0.5),
        (1, 1): (-0.1, 0.5),
    }
    for title, prefix, row in rows:
        x_num = ts[f"X_Num_{prefix}_Interface(m)"]
        x_exp = ts[f"X_Exp_{prefix}_Interface(m)"]
        rz_num = ts[f"Rz_Num_{prefix}_Interface(rad)"]
        rz_exp = ts[f"Rz_Exp_{prefix}_Interface(rad)"]

        exp_x_line, = axes[row, 0].plot(
        t, x_exp, color=EXP_COLOR, label="Experimental",
            linewidth=1.1, linestyle="--", zorder=4,
        )
        num_x_line, = axes[row, 0].plot(
            t, x_num, color=NUM_COLOR, label="Numerical",
            linewidth=0.9, zorder=3,
        )
        axes[row, 0].set_ylabel("Displacement (m)")
        axes[row, 0].set_title(f"{title} interface")
        axes[row, 0].set_ylim(*response_ylims[(row, 0)])
        axes[row, 0].set_yticks(np.arange(response_ylims[(row, 0)][0], response_ylims[(row, 0)][1] + 0.05, 0.1))
        axes[row, 0].legend([num_x_line, exp_x_line], ["Numerical", "Experimental"], loc="upper right", frameon=False)

        exp_rz_line, = axes[row, 1].plot(
            t, rz_exp, color=EXP_COLOR, label="Experimental",
            linewidth=1.1, linestyle="--", zorder=4,
        )
        num_rz_line, = axes[row, 1].plot(
            t, rz_num, color=NUM_COLOR, label="Numerical",
            linewidth=0.9, zorder=3,
        )
        axes[row, 1].set_ylabel("Rotation (rad)")
        axes[row, 1].set_title(f"{title} interface")
        axes[row, 1].set_ylim(*response_ylims[(row, 1)])
        axes[row, 1].set_yticks(np.arange(response_ylims[(row, 1)][0], response_ylims[(row, 1)][1] + 0.05, 0.1))
        axes[row, 1].legend([num_rz_line, exp_rz_line], ["Numerical", "Experimental"], loc="upper right", frameon=False)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")

    add_panel_labels(axes.ravel())
    finish_figure(fig, out_dir / "03_interface_response")


def first_iteration_error_limits(delta):
    """Use iteration-1 errors to define all later error-panel y-limits."""
    disp = [
        delta["Iter1_Right_X_Displacement_Delta(m)"],
        delta["Iter1_Left_X_Displacement_Delta(m)"],
    ]
    rot = [
        1.3*delta["Iter1_Right_Rotation_Delta(rad)"],
        1.3*delta["Iter1_Left_Rotation_Delta(rad)"],
    ]
    return symmetric_ylim(disp), symmetric_ylim(rot)


def plot_interface_error_panels(delta, out_dir):
    """
    Draw small error panels for all requested iterations.

    The y-limits are fixed from iteration 1, so the error reduction across
    iterations is visually comparable. Both interfaces use the same blue curve.
    """
    t = delta["Time(s)"]
    disp_ylim, rot_ylim = first_iteration_error_limits(delta)

    def finish_error_panel(fig, path_stem):
        """Save every error panel with the same canvas and axes dimensions."""
        path_stem.parent.mkdir(parents=True, exist_ok=True)
        for text in fig.findobj(match=plt.Text):
            text.set_fontname("Times New Roman")
        for panel_ax in fig.axes:
            panel_ax.grid(False)
            panel_ax.tick_params(which="both", top=False)
            for spine in panel_ax.spines.values():
                spine.set_linewidth(1)
        fig.subplots_adjust(left=0.32, right=0.91, bottom=0.33, top=0.79)
        svg_path = path_stem.with_suffix(".svg")
        if svg_path.exists():
            svg_path.unlink()
        fig.savefig(svg_path)
        plt.close(fig)

    for iteration in ERROR_ITERATIONS:
        panels = [
            ("right", f"Iter{iteration}_Right_X_Displacement_Delta(m)",
             f"Iter{iteration}_Right_Rotation_Delta(rad)"),
            ("left", f"Iter{iteration}_Left_X_Displacement_Delta(m)",
             f"Iter{iteration}_Left_Rotation_Delta(rad)"),
        ]

        for side, disp_key, rot_key in panels:
            if disp_key not in delta or rot_key not in delta:
                continue

            # Export displacement and rotation errors separately. These small
            # files are easier to assemble manually into a manuscript figure.
            with plt.rc_context(ERROR_PANEL_STYLE):
                fig, ax = plt.subplots(figsize= (4.29 / 2.54, 3.28 / 2.54))
                ax.plot(t, delta[disp_key], color=ERROR_COLOR_dis, linewidth=0.8)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Error (m)")
                ax.set_title(f"Iteration {iteration}")
                ax.set_xlim(t[0], t[-1])
                # Only the left-interface displacement series is tightened to
                # +/-0.1 m; rotation and all right-interface error limits remain
                # unchanged from the iteration-1 reference limits.
                ax.set_ylim(-0.1, 0.1) if side == "left" else ax.set_ylim(*disp_ylim)
                ax.set_xticks([t[0], 0.5 * (t[0] + t[-1]), t[-1]])
                disp_stem = out_dir / f"{side}_iter_{iteration:02d}_displacement_error"
                finish_error_panel(fig, disp_stem)
                normalize_error_svg_font(disp_stem.with_suffix(".svg"))

                fig, ax = plt.subplots(figsize= (4.29 / 2.54, 3.28 / 2.54))
                ax.plot(t, delta[rot_key], color=ERROR_COLOR_rot, linewidth=0.8)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Error (rad)")
                ax.set_title(f"Iteration {iteration}")
                ax.set_xlim(t[0], t[-1])
                ax.set_ylim(-0.2, 0.2) if side == "left" else ax.set_ylim(*rot_ylim)
                ax.set_xticks([t[0], 0.5 * (t[0] + t[-1]), t[-1]])
                rot_stem = out_dir / f"{side}_iter_{iteration:02d}_rotation_error"
                finish_error_panel(fig, rot_stem)
                normalize_error_svg_font(rot_stem.with_suffix(".svg"))


def main():
    """Entry point: read one iteration directory and export all paper figures."""
    setup_style()
    iter_dir = get_iteration_dir(TARGET_ITERATION)
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale outputs from older versions of this script. This keeps the
    # folder consistent with the current SVG-only and split-error-file rules.
    for pattern in (
        "*.png",
        "02_interface_force_moment.svg",
        "left_iter_*_interface_error.svg",
        "right_iter_*_interface_error.svg",
    ):
        for old_file in out_dir.glob(pattern):
            old_file.unlink()

    conv = read_csv_columns(iter_dir / "convergence_history.csv")
    ts = read_csv_columns(iter_dir / "data_time_series.csv")
    delta = read_csv_columns(iter_dir / "data_iteration_delta.csv")

    plot_convergence(conv, out_dir)
    plot_one_interface_force_moment(ts, out_dir, "left")
    plot_one_interface_force_moment(ts, out_dir, "right")
    plot_external_excitation(ts, out_dir)
    plot_interface_response(ts, out_dir)
    plot_interface_error_panels(delta, out_dir)

    print(f"SCI SVG figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
