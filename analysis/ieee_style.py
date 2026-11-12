"""
Shared plot style for the journal figures (IEEE two-column layout).

Times-compatible text (STIX, shipped with matplotlib, so the output is the same on every machine),
closed axes box, inward ticks at the labeled values only (no minor or mirrored ticks), thin lines, framed legends inside the axes.
Series are separated by marker shape and line style as well as by color, so the figures
remain readable in grayscale.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

COL_W, TEXT_W = 3.5, 7.16           # inches: one column, full text width
BLACK, BLUE, RED, GREY = "#000000", "#0072BD", "#D95319", "#7f7f7f"
LW = 0.6                            # axes, ticks, error bars


def apply():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "axes.titlesize": 8,
        "axes.linewidth": LW, "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "text.color": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": False, "ytick.right": False,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.major.width": LW, "ytick.major.width": LW, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "lines.linewidth": 1.0, "lines.markersize": 4, "hatch.linewidth": 0.45,
        "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": BLACK, "legend.framealpha": 1.0,
        "legend.borderpad": 0.35, "legend.labelspacing": 0.3, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.5,
        "grid.color": "#b0b0b0", "grid.linewidth": 0.35, "grid.linestyle": (0, (1, 2)),
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def legend(ax, *args, **kw):
    """Framed legend with the same line weight as the axes."""
    lg = ax.legend(*args, **kw)
    lg.get_frame().set_linewidth(0.5)
    return lg


def log_minor(axis):
    """No minor ticks on a log axis: only the labeled decades carry a tick."""
    axis.set_minor_locator(NullLocator())


def save(fig, outdir, name, dpi=1000):
    import os
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)
