import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from OPTICAL_FLOW.farneback.src.subjects import SubjectPair
from OPTICAL_FLOW.farneback.src.zone_masks import RetinalZoneMasks


def save_quiver_figure(subject: SubjectPair, u: np.ndarray, v: np.ndarray, zones: RetinalZoneMasks,
                        inner_radius_x: float, inner_radius_y: float, outer_radius_x: float, outer_radius_y: float,
                        out_path: str) -> None:
    """Salva una figura in stile paper con il campo vettoriale sovrapposto all'immagine PRE del soggetto."""

    cx, cy = subject.fovea_center
    h, w = u.shape

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    ax.imshow(subject.draw_image)

    grid_color = 'white'
    thickness = 0.5
    linestyle = '--'

    ax.add_patch(patches.Ellipse((cx, cy), 2 * inner_radius_x, 2 * inner_radius_y, fill=False, edgecolor=grid_color, linewidth=thickness, linestyle=linestyle))
    ax.add_patch(patches.Ellipse((cx, cy), 2 * outer_radius_x, 2 * outer_radius_y, fill=False, edgecolor=grid_color, linewidth=thickness, linestyle=linestyle))
    offset_x = outer_radius_x * 0.707
    offset_y = outer_radius_y * 0.707
    ax.plot([cx - offset_x, cx + offset_x], [cy - offset_y, cy + offset_y], color=grid_color, linewidth=thickness, linestyle=linestyle)
    ax.plot([cx - offset_x, cx + offset_x], [cy + offset_y, cy - offset_y], color=grid_color, linewidth=thickness, linestyle=linestyle)
    ax.plot(cx, cy, marker='+', color=grid_color, markersize=5, markeredgewidth=thickness)

    retina_mask = subject.img_pre < 250
    u_plot = np.where(retina_mask, u, np.nan)
    v_plot = np.where(retina_mask, v, np.nan)

    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    step = 30
    ax.quiver(xx[::step, ::step], yy[::step, ::step],
              u_plot[::step, ::step], v_plot[::step, ::step],
              angles='xy', color='lime', alpha=0.8, scale=500,
              width=0.0015, headwidth=3, headlength=4, headaxislength=3.5)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
