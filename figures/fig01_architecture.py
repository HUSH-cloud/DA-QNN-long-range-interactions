"""Fig. 1: DA-QNN architecture schematic.

Shows the pipeline from one MNIST digit through PCA to per-qubit encoding angles, the N-qubit
circuit (encoding gate U, digital rotation layer R, analog block e^{-iH tau} with power-law
couplings J_ij ~ |i-j|^{-alpha} and transverse field h sigma^x, repeated ell times) and the
<sigma^z_1> readout.

Input:  figures/assets/mnist_3_01.png (a 28x28 grayscale MNIST "3").
Output: OUT/fig1_architecture.pdf (OUT from paths.py).

All coordinates are canvas inches on a FULL_W-wide canvas.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from PIL import Image

import style as S
from paths import OUT

DIGIT = Path(__file__).resolve().parent / "assets" / "mnist_3_01.png"

# Colors
C_DIG = "#3D7FA8"    # single-qubit gates: encoding U and rotation layers R
C_PCA = "#55555B"    # classical preprocessing (dark gray): input image frame and PCA box
C_ANA = "#C2482A"    # analog block: couplings and transverse field
C_WIRE = "#8C8C92"
Y_NOTE = 0.36        # baseline of the two notes under the circuit ("data encoding", J_ij / h sigma^x)
C_TXT = "#2B2B2E"
FC_NODE = "#F5DDD4"  # lattice-site fill

# Type sizes
FS_BIG = S.FS_LABEL
FS_MAIN = S.FS_TICK
FS_BOX = S.FS_TICK
FS_SMALL = S.FS_ANNOT
FS_NODE = 10.5       # glyph size of the site indices, scaled to the circle

# Line weights
LW_BOX = S.LW_AXES
LW_BIG = S.LW_AXES * 1.35    # analog block frame
LW_WIRE = S.LW_AXES * 0.80
LW_ARROW = S.LW_AXES * 0.90

W, H = S.FULL_W, 4.95

# Circuit rows: qubits 1..4 and N. The drawing illustrates N = 8, so the last row is lattice
# site 8; bond widths follow the true separation |i-j|.
ROWS = [4.00, 3.35, 2.70, 2.05, 1.10]
ROW_SUB = ["1", "2", "3", "4", "N"]
ROW_SITE = [1, 2, 3, 4, 8]
Y_GAP = 1.57         # height of the vertical ellipsis between row 4 and row N

X_BUS = 2.10
X_QLAB = 2.56
X_W0, X_W1 = 2.70, 9.98
X_KET = 2.84
BOX_H = 0.46

XE = (3.14, 4.54)    # encoding gate U(theta_q, phi_q)
XR0 = (4.76, 5.64)   # first rotation layer R(theta_{q,0})
XA = (5.78, 8.38)    # analog block
XR = (8.58, 9.40)    # rotation layer R(theta_{q,j})
XM = (9.54, 9.92)    # meter
YA0, YA1 = 0.66, 4.40

R_NODE = 0.105

LW_BOND1 = 2.40      # nearest-neighbor bond width
P_BOND = 0.45        # schematic decay exponent of the drawn bond width, chosen so distant bonds stay visible
LW_BOND_MIN = 0.75


def bond_lw(d):
    """Line width of the coupling between sites a lattice distance d apart (schematic power law)."""
    return max(LW_BOND_MIN, LW_BOND1 * d ** -P_BOND)


def rbox(ax, x0, x1, y0, y1, color, lw=LW_BOX, r=0.055, fc="white", z=3):
    """Rounded rectangle from (x0, y0) to (x1, y1)."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=color, facecolor=fc, zorder=z))


def gate(ax, xspan, y, label):
    """Single-qubit gate box on the wire at height y, label centered inside."""
    rbox(ax, xspan[0], xspan[1], y - BOX_H / 2, y + BOX_H / 2, C_DIG)
    ax.text(np.mean(xspan), y, label, ha="center", va="center", fontsize=FS_BOX, color=C_DIG, zorder=4)


def arrow(ax, p0, p1, color=C_TXT, lw=LW_ARROW, ms=6.0, z=4):
    """Straight arrow from p0 to p1 with a filled head."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms * 1.6, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter", capstyle="butt"))


def dots(ax, x, y, color=C_WIRE, dy=0.082, s=1.5, z=4):
    """Vertical ellipsis of three dots centered at (x, y)."""
    for k in (-1, 0, 1):
        ax.plot([x], [y + k * dy], marker="o", ms=s, color=color, zorder=z, markeredgewidth=0)


def bracket(ax, x0, x1, y, drop=0.10, lw=1.0):
    """Square bracket over [x0, x1] at height y, ends pointing down."""
    ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop], color=C_TXT, lw=lw, solid_joinstyle="miter", zorder=4)


def site(ax, x, y, sub):
    """Lattice site: circle with its index drawn as a path centered on the glyph outline.

    ax.text centers on the advance box, which leaves a narrow numeral such as "1" visibly off-center
    inside a small circle, so the glyph is converted to a path and centered on its own extents.
    """
    ax.add_patch(Circle((x, y), R_NODE, facecolor=FC_NODE, edgecolor=C_ANA, linewidth=1.0, zorder=8))
    fp = FontProperties(family="serif", size=FS_NODE, style="italic" if sub.isalpha() else "normal")
    tp = TextPath((0, 0), sub, prop=fp)
    bb = tp.get_extents()
    tr = (Affine2D()
          .translate(-(bb.x0 + bb.width / 2), -(bb.y0 + bb.height / 2))
          .scale(1 / 72)          # TextPath is in points, the axes in inches
          .translate(x, y))
    ax.add_patch(PathPatch(tr.transform_path(tp), facecolor=C_ANA, edgecolor="none", zorder=9))


def color_line(ax, xc, y, parts, fs=FS_SMALL):
    """One centered line of text whose segments have different colors; parts = [(text, color), ...].

    matplotlib has no rich text, so each segment is measured, then drawn left to right.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width(txt):
        t = ax.text(0, 0, txt, fontsize=fs)
        bb = t.get_window_extent(r)
        (x0, _), (x1, _) = inv.transform([(bb.x0, 0), (bb.x1, 0)])
        t.remove()
        return x1 - x0

    # A math-mode space has zero ink width, so the gap is taken from a real glyph.
    gap = 0.9 * width("m")
    widths = [width(t) for t, _ in parts]
    total = sum(widths) + gap * (len(parts) - 1)
    x = xc - total / 2
    for (txt, col), w in zip(parts, widths):
        ax.text(x, y, txt, ha="left", va="baseline", fontsize=fs, color=col)
        x += w + gap


def draw_analog(ax):
    """Analog block contents: sites on their own wires, every coupling J_ij as an arc to the right.

    Site i sits at the height of wire q_i, so vertical separation equals lattice separation. The
    arc for (i, j) rises in proportion to |y_i - y_j| and its width follows bond_lw(|i-j|).
    """
    xc = 6.50
    # Neutral backbone carrying the sites, broken where sites 5..N-1 are omitted.
    ymid = 0.5 * (ROWS[3] + ROWS[4])
    ax.plot([xc, xc], [ROWS[0], ymid + 0.17], color=C_WIRE, lw=0.9, zorder=5.6)
    ax.plot([xc, xc], [ymid - 0.17, ROWS[4]], color=C_WIRE, lw=0.9, zorder=5.6)
    dots(ax, xc, ymid, C_WIRE, dy=0.072, s=1.8, z=7)

    for i in range(len(ROWS)):
        for j in range(i + 1, len(ROWS)):
            d = ROW_SITE[j] - ROW_SITE[i]
            dy = ROWS[i] - ROWS[j]
            ax.add_patch(Arc((xc, 0.5 * (ROWS[i] + ROWS[j])), 2 * 0.55 * dy, dy, theta1=-90, theta2=90,
                             lw=bond_lw(d), edgecolor=C_ANA, zorder=6))

    for y, sub in zip(ROWS, ROW_SUB):
        site(ax, xc, y, sub)

    # Transverse field: a horizontal arrow, perpendicular to the vertical (Ising-axis) chain.
    x0, y = XA[0] + 0.10, 0.5 * (YA0 + YA1)
    arrow(ax, (x0, y), (x0 + 0.42, y), color=C_ANA, lw=1.4, ms=6.5, z=7)
    ax.text(XA[1] - 0.14, 4.20, r"$e^{-iH\tau}$", ha="right", va="center", fontsize=FS_BIG, color=C_ANA, zorder=6)


def draw_meter(ax, y):
    """Measurement symbol on wire q_1 with the <sigma^z_1> label above it."""
    rbox(ax, XM[0], XM[1], y - BOX_H / 2, y + BOX_H / 2, C_TXT, lw=LW_BOX, r=0.03)
    xm = np.mean(XM)
    ax.add_patch(Arc((xm, y - 0.11), 0.25, 0.25, theta1=20, theta2=160, lw=0.95, edgecolor=C_TXT, zorder=5))
    ax.plot([xm, xm + 0.10], [y - 0.11, y + 0.09], lw=0.95, color=C_TXT, zorder=5, solid_capstyle="round")
    ax.plot([xm], [y - 0.11], marker="o", ms=2.0, color=C_TXT, zorder=6, markeredgewidth=0)
    ax.text(xm, y + 0.36, r"$\langle \sigma^z_1\rangle$", ha="center", va="bottom", fontsize=FS_MAIN, color=C_TXT)


def main():
    # Only the font settings of style.use(): its savefig.dpi=600 would resample the embedded digit image.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input digit and arrow up to PCA
    img = np.array(Image.open(DIGIT).convert("L"))
    ax.imshow(img, cmap="gray", extent=(0.68, 1.52, 0.94, 1.78), interpolation="nearest", zorder=3)
    # Frame drawn exactly on the image edge (same extent as imshow), square corners.
    rbox(ax, 0.68, 1.52, 0.94, 1.78, C_PCA, lw=LW_BOX, r=0.0, fc="none", z=4)
    arrow(ax, (1.10, 1.92), (1.10, 2.36))

    # PCA box and the bus distributing (theta_q, phi_q) to the wires
    rbox(ax, 0.30, 1.90, 2.46, 3.58, C_PCA, lw=LW_BOX, r=0.09)
    ax.text(1.10, 3.24, "PCA", ha="center", va="center", fontsize=FS_BIG, color=C_PCA)
    ax.text(1.10, 2.78, r"$(\theta_1,\phi_1),\ \dots,\ (\theta_N,\phi_N)$",
            ha="center", va="center", fontsize=FS_SMALL, color=C_TXT)
    ax.plot([1.90, X_BUS], [3.02, 3.02], color=C_TXT, lw=LW_ARROW, zorder=2)
    ax.plot([X_BUS, X_BUS], [ROWS[-1], ROWS[0]], color=C_TXT, lw=LW_ARROW, zorder=2)

    # Wires, |0> kets and the single-qubit gates. Bold theta_{q,j} is qubit q's Euler triple in
    # layer j (Eq. 2); upright theta_q, phi_q are its encoding angles from PCA.
    for y, sub in zip(ROWS, ROW_SUB):
        arrow(ax, (X_BUS, y), (X_QLAB - 0.22, y), lw=0.95, ms=5.5)
        ax.text(X_QLAB, y, rf"$q_{{{sub}}}$", ha="right", va="center", fontsize=FS_MAIN, color=C_TXT)
        ax.plot([X_W0, X_W1], [y, y], color=C_WIRE, lw=LW_WIRE, zorder=1)
        ax.text(X_KET, y, r"$|0\rangle$", ha="left", va="center", fontsize=FS_MAIN, color=C_TXT, zorder=2,
                bbox=dict(fc="white", ec="none", pad=0.8))
        gate(ax, XE, y, rf"$U(\theta_{{{sub}}},\phi_{{{sub}}})$")
        gate(ax, XR0, y, rf"$R(\boldsymbol{{\theta}}_{{{sub},0}})$")
        gate(ax, XR, y, rf"$R(\boldsymbol{{\theta}}_{{{sub},j}})$")

    for x in (X_QLAB - 0.09, np.mean(XE), np.mean(XR0), np.mean(XR)):
        dots(ax, x, Y_GAP)

    # Analog block
    rbox(ax, XA[0], XA[1], YA0, YA1, C_ANA, lw=LW_BIG, r=0.11, fc="white", z=5)
    draw_analog(ax)
    color_line(ax, np.mean(XA), Y_NOTE,
                [(r"$J_{ij}\propto|i-j|^{-\alpha},$", C_ANA),
                 (r"$h\,\sigma^x$", C_ANA)])

    # Readout, the x ell bracket over the repeated [analog + digital] unit, encoding-order note
    draw_meter(ax, ROWS[0])
    bracket(ax, XA[0], XR[1], 4.62, drop=0.11)
    ax.text(0.5 * (XA[0] + XR[1]), 4.70, r"$\times\ \ell$", ha="center", va="bottom", fontsize=FS_MAIN, color=C_TXT)
    ax.text(np.mean(XE), Y_NOTE, "data encoding", ha="center", va="baseline",
            fontsize=FS_SMALL, color=C_DIG)

    S.save(fig, OUT, "fig1_architecture")
    plt.close(fig)


if __name__ == "__main__":
    main()
