import matplotlib as mpl

SERIES_KEYS = (
    "blue",
    "orange",
    "olive",
    "rose",
    "purple",
    "umber",
    "deep_cyan",
    "slate",
)

PAPER_TEX_PREAMBLE = (
    r"\usepackage[T1]{fontenc}"
    r"\renewcommand{\rmdefault}{ptm}"
    r"\renewcommand{\sfdefault}{phv}"
)
PAPER_SERIF_FONTS = ["Times", "Nimbus Roman No9 L", "Times New Roman"]
PAPER_SANS_FONTS = ["Helvetica", "Nimbus Sans", "Arial"]
PAPER_MONO_FONTS = ["Courier", "Nimbus Mono PS", "Courier New"]

STYLE_COLORS = {
    "ink": "#0E1417",
    "muted": "#36484D",
    "panel": "#F1F7F8",
    "rule": "#7EAABB",
    "white": "#FFFFFF",
    "link": "#0072F5",
    "url": "#D4145A",
    "takeaway_bg": "#D9F1F7",
    "takeaway_rule": "#0077B6",
    "highlight_soft": "#F2A900",
    "highlight_mineral": "#91D9F2",
    "macro": "#0072F5",

    # series colors
    "blue": "#0072F5",
    "orange": "#C94F12",
    "olive": "#4F7F19",
    "rose": "#D4145A",
    "purple": "#7B3FA1",
    "umber": "#8C4A2F",
    "deep_cyan": "#005F73",
    "slate": "#5A6772",
}

INK = STYLE_COLORS["ink"]
MUTED = STYLE_COLORS["muted"]
RULE = STYLE_COLORS["rule"]
WHITE = STYLE_COLORS["white"]
PANEL = WHITE


# Common conference widths (e.g., NeurIPS is ~ 5.5206)
TEXT_WIDTH_IN = 5.50
HALF_TEXT_WIDTH_IN = 2.64
COMPACT_WIDTH_IN = 2.50
DOUBLE_TEXT_WIDTH_IN = 2 * TEXT_WIDTH_IN

# Default heights adjust for readability
FULL_HEIGHT_IN = 2.55
TWO_PANEL_HEIGHT_IN = 2.42
HALF_PANEL_HEIGHT_IN = 1.95
COMPACT_HEIGHT_IN = 2.15
TWO_ROW_FOUR_PANEL_HEIGHT_IN = 2 * FULL_HEIGHT_IN

PALETTE = [STYLE_COLORS[key] for key in SERIES_KEYS]
def _make_rc_params() -> dict[str, object]:
    return {
        "text.usetex": True,
        "text.latex.preamble": PAPER_TEX_PREAMBLE,
        "font.family": "serif",
        "font.serif": PAPER_SERIF_FONTS,
        "font.sans-serif": PAPER_SANS_FONTS,
        "font.monospace": PAPER_MONO_FONTS,
        "mathtext.fontset": "cm",
        "font.size": 9.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.4,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "xtick.major.pad": 2.4,
        "ytick.major.pad": 2.4,
        "xtick.minor.pad": 2.2,
        "ytick.minor.pad": 2.2,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.55,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.facecolor": PANEL,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "grid.color": WHITE,
        "grid.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.86,
        #"legend.edgecolor": RULE, # Different for QuBD paper
        "legend.edgecolor": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }


RC_PARAMS = _make_rc_params()


def series_palette() -> list[str]:
    return list(PALETTE)


def use_pedram_style() -> None:
    mpl.rcParams.update(RC_PARAMS)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)


# apply style to axes
def apply_panel_style(ax: mpl.axes.Axes, *, background: str | None = None) -> None:
    if background is None:
        background = PANEL
    ax.set_facecolor(background)
    ax.grid(True, color=WHITE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.8)


# double column
def full_width_size(height: float = FULL_HEIGHT_IN, width: float = TEXT_WIDTH_IN) -> tuple[float, float]:
    return (width, height)

# for subplots
def two_panel_size(height: float = TWO_PANEL_HEIGHT_IN, width: float = TEXT_WIDTH_IN) -> tuple[float, float]:
    return (width, height)

def two_rows_size(height: float = 2*HALF_PANEL_HEIGHT_IN, width: float = HALF_TEXT_WIDTH_IN) -> tuple[float, float]:
    return (width, height)

def four_panel_row_wide_size(height: float = FULL_HEIGHT_IN, width: float = DOUBLE_TEXT_WIDTH_IN) -> tuple[float, float]:
    return (width, height)

def two_row_four_panel_wide_size(height: float = TWO_ROW_FOUR_PANEL_HEIGHT_IN, width: float = DOUBLE_TEXT_WIDTH_IN) -> tuple[float, float]:
    return (width, height)

# normal size plots (single coloumn)
def half_width_size(height: float = HALF_PANEL_HEIGHT_IN, width: float = HALF_TEXT_WIDTH_IN) -> tuple[float, float]:
    return (width, height)

# small compact (wrap figure)
def compact_size(height: float = COMPACT_HEIGHT_IN, width: float = COMPACT_WIDTH_IN) -> tuple[float, float]:
    return (width, height)

# scales with text if needed
def scaled_text_width(scale: float, height: float | None = None) -> tuple[float, float]:
    width = TEXT_WIDTH_IN * scale
    if height is None:
        height = width * 0.46
    return (width, height)




