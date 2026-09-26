"""
Place short text labels next to a set of scatter points, each joined to its
point by a line.

Based on ggrepel's force simulation (labels repel one another, their own
points, and the background scatter, and spring back toward their point when
they touch nothing), followed by a deterministic clean-up that guarantees no
two boxes overlap and no two connecting lines cross. Both the simulation and
the clean-up run in display pixels, where the axes are isotropic, so the forces
push equally hard in every direction regardless of how the data axes are
scaled. Only the final text and line positions are converted back to data
coordinates. Deterministic: the only randomness is a fixed-seed jitter.
"""
import numpy as np
cimport numpy as np
from cpython.exc cimport PyErr_CheckSignals
from libcpp.algorithm cimport lower_bound
from libcpp.cmath cimport abs, sqrt

cdef extern from "<utility>" namespace "std":
    void swap[T](T&, T&) noexcept nogil

cdef extern from *:
    """
    struct Point2D { float x, y; };
    inline bool operator<(const Point2D& a,
                          const Point2D& b) { return a.x < b.x; }
    """
    cdef cppclass Point2D:
        float x
        float y

# ggrepel constants: MIN_SEP2 is the smallest squared separation used in the
# inverse-square repulsion, so coincident boxes/points get a large-but-finite
# push instead of dividing by zero (0.0004 = 0.02**2 in data-normalized units).
# The two decays anneal the forces a little each iteration, so the system cools
# and settles.
cdef float MIN_SEP2 = <float> 0.0004
cdef float REPULSION_DECAY = <float> 0.99999
cdef float ATTRACTION_DECAY = <float> 0.9999


cdef inline float clip(float v, const float lo, const float hi) noexcept nogil:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


cdef inline void line_end(const float ax, const float ay,
                          const float cx, const float cy,
                          const float hw, const float hh,
                          float& ex, float& ey) noexcept nogil:
    # Where the line from the anchor meets the box: the anchor clamped to it
    ex = clip(ax, cx - hw, cx + hw)
    ey = clip(ay, cy - hh, cy + hh)


cdef inline bint cross(float ax, float ay, float bx, float by,
                       float cx, float cy, float dx, float dy) noexcept nogil:
    # Whether segments a,b and c,d cross (shared endpoints don't)
    cdef float s1 = (ay - cy) * (dx - cx) - (dy - cy) * (ax - cx)  # side c,d,a
    cdef float s2 = (by - cy) * (dx - cx) - (dy - cy) * (bx - cx)  # side c,d,b
    if s1 * s2 >= <float> 0:
        return False
    cdef float s3 = (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)  # side a,b,c
    cdef float s4 = (dy - ay) * (bx - ax) - (by - ay) * (dx - ax)  # side a,b,d
    return s3 * s4 < <float> 0


cdef inline bint clip_edge(float p, float q, float& enter, float& leave) \
        noexcept nogil:
    """
    Clip one Liang-Barsky edge against the running [enter, leave] span: given
    the edge as p*t <= q, tighten enter (on entering edges) or leave (on
    leaving ones), and return False if that empties the span, i.e. if the
    segment is wholly outside this edge and so cannot hit the box.
    """
    cdef float t
    if abs(p) < <float> 1e-12:
        return q >= <float> 0
    t = q / p
    if p < <float> 0:
        if t > leave:
            return False
        if t > enter:
            enter = t
    else:
        if t < enter:
            return False
        if t < leave:
            leave = t
    return True


cdef inline bint segment_hits_box(const float ax, const float ay,
                                  const float bx, const float by,
                                  const float x0, const float y0,
                                  const float x1, const float y1,
                                  const float margin) noexcept nogil:
    # Whether segment a,b hits the box (x0, y0, x1, y1) grown by `margin`
    cdef float dx = bx - ax, dy = by - ay, enter = 0, leave = 1
    if not clip_edge(-dx, ax - (x0 - margin), enter, leave):
        return False
    if not clip_edge(dx, (x1 + margin) - ax, enter, leave):
        return False
    if not clip_edge(-dy, ay - (y0 - margin), enter, leave):
        return False
    if not clip_edge(dy, (y1 + margin) - ay, enter, leave):
        return False
    return enter < leave


cdef inline void repel(const float dx, const float dy, const float repulsion,
                       const float min_sep2, float& fx, float& fy) \
        noexcept nogil:
    # Add one inverse-square repulsion, doubled on the smaller-gap axis
    cdef float sq = dx * dx + dy * dy
    cdef float scale, gx, gy
    if sq < min_sep2:
        sq = min_sep2
    scale = repulsion / (sq * sqrt(sq))
    gx = dx * scale
    gy = dy * scale
    if abs(dx) > abs(dy):
        fx += gx
        fy += gy * <float> 2
    else:
        fx += gx * <float> 2
        fy += gy


cdef void repel_boxes(float[:, ::1] centers,
                      const float[::1] half_width,
                      const float[::1] half_height,
                      const float[:, ::1] anchors,
                      const float[:, ::1] scatter,
                      const float padding,
                      const float[::1] width,
                      float[:, ::1] velocity,
                      const float x0,
                      const float y0,
                      const float x1,
                      const float y1,
                      float attraction,
                      float repulsion,
                      const float min_sep2,
                      const unsigned max_iter) noexcept nogil:
    """
    Relax `centers` in-place with ggrepel's force model. Each box repels every
    other box and every point in `points` (kept `padding` clear; the first
    `n` points are the labels' own points, which double as the spring targets),
    and springs toward its point when it overlaps nothing. All coordinates are
    display pixels, so the force field is isotropic.
    """
    cdef unsigned i, j, k, hits, start_k, it = 0, overlaps = 1, \
        n = centers.shape[0], m = scatter.shape[0]
    cdef float cx, cy, fx, fy, dx, dy, gapx, gapy, momentum, point_coeff, \
        max_dx, max_dy, d, max_line, padding2 = padding * padding
    cdef Point2D target
    cdef Point2D* pts
    cdef Point2D* res

    while overlaps and it < max_iter:
        it += 1
        overlaps = 0
        attraction *= ATTRACTION_DECAY
        repulsion *= REPULSION_DECAY
        point_coeff = <float> 100 * padding * repulsion
        for i in range(n):
            cx = centers[i, 0]
            cy = centers[i, 1]
            fx = 0
            fy = 0
            hits = 0
            for j in range(n):
                # box-box repulsion
                if j == i:
                    continue
                dx = cx - centers[j, 0]
                dy = cy - centers[j, 1]
                if abs(dx) < half_width[i] + half_width[j] and \
                        abs(dy) < half_height[i] + half_height[j]:
                    repel(dx, dy, repulsion, min_sep2, fx, fy)
                    hits += 1
            for k in range(n):
                # Point repulsion (own points)
                dx = cx - anchors[k, 0]
                if abs(dx) > half_width[i] + padding:
                    continue
                dy = cy - anchors[k, 1]
                if abs(dy) > half_height[i] + padding:
                    continue
                gapx = abs(dx) - half_width[i]
                gapy = abs(dy) - half_height[i]
                if gapx < <float> 0:
                    gapx = 0
                if gapy < <float> 0:
                    gapy = 0
                if gapx * gapx + gapy * gapy <= padding2:
                    repel(dx, dy, point_coeff, min_sep2, fx, fy)
                    hits += 1
            if m > 0:
                # Point repulsion (background scatter)
                pts = <Point2D*> &scatter[0, 0]
                max_dx = half_width[i] + padding
                max_dy = half_height[i] + padding
                target.x = cx - max_dx
                res = lower_bound(pts, pts + m, target)
                start_k = res - pts
                for k in range(start_k, m):
                    dx = cx - scatter[k, 0]
                    if dx < -max_dx:
                        break
                    dy = cy - scatter[k, 1]
                    if dy < -max_dy or dy > max_dy:
                        continue
                    gapx = abs(dx) - half_width[i]
                    gapy = abs(dy) - half_height[i]
                    if gapx < <float> 0:
                        gapx = 0
                    if gapy < <float> 0:
                        gapy = 0
                    if gapx * gapx + gapy * gapy <= padding2:
                        repel(dx, dy, point_coeff, min_sep2, fx, fy)
                        hits += 1
            overlaps += hits
            if hits == 0:
                # Spring back to own point
                fx += attraction * (anchors[i, 0] - cx)
                fy += attraction * (anchors[i, 1] - cy)
            momentum = (<float> 1 + (<float> 0.5 if hits > 10 else
                                     <float> 0.05 * hits)) * \
                (width[i] + <float> 1e-6) * <float> 0.7
            velocity[i, 0] = velocity[i, 0] * momentum + fx
            velocity[i, 1] = velocity[i, 1] * momentum + fy
            cx = clip(cx + velocity[i, 0], x0 + half_width[i],
                      x1 - half_width[i])
            cy = clip(cy + velocity[i, 1], y0 + half_height[i],
                      y1 - half_height[i])
            # Leash the box to within a width-scaled distance of its point, so
            # it settles in nearby whitespace instead of drifting far off; a
            # wider label gets proportionally more room to clear its neighbours
            dx = cx - anchors[i, 0]
            dy = cy - anchors[i, 1]
            d = sqrt(dx * dx + dy * dy)
            max_line = <float> 2 * half_width[i]
            if d > max_line:
                cx = anchors[i, 0] + dx / d * max_line
                cy = anchors[i, 1] + dy / d * max_line
            centers[i, 0] = cx
            centers[i, 1] = cy
            if overlaps == 0 or it % 5 == 0:
                # Swap to uncross lines
                for j in range(n):
                    if j != i and cross(cx, cy, anchors[i, 0], anchors[i, 1],
                                        centers[j, 0], centers[j, 1],
                                        anchors[j, 0], anchors[j, 1]):
                        overlaps += 1
                        swap(centers[i, 0], centers[j, 0])
                        swap(centers[i, 1], centers[j, 1])
                        cx = centers[i, 0]
                        cy = centers[i, 1]


cdef bint separate(float[:, ::1] centers,
                   const float[::1] half_width,
                   const float[::1] half_height,
                   const float x0,
                   const float y0,
                   const float x1,
                   const float y1,
                   const unsigned max_iter) noexcept nogil:
    """
    Push overlapping boxes apart along their smaller-overlap axis, in-place,
    until none overlap, keeping them within bounds. Returns whether any box
    was moved to resolve an overlap.
    """
    cdef unsigned i, j, it, n = centers.shape[0]
    cdef float dx, dy, ox, oy, s
    cdef bint moved, any_moved = False
    for it in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                ox = half_width[i] + half_width[j] - abs(dx)
                oy = half_height[i] + half_height[j] - abs(dy)
                if ox > <float> 0 and oy > <float> 0:
                    if ox <= oy:
                        s = ox * <float> 0.5
                        if dx < <float> 0:
                            s = -s
                        centers[i, 0] += s
                        centers[j, 0] -= s
                    else:
                        s = oy * <float> 0.5
                        if dy < <float> 0:
                            s = -s
                        centers[i, 1] += s
                        centers[j, 1] -= s
                    moved = True
        for i in range(n):
            centers[i, 0] = clip(centers[i, 0], x0 + half_width[i],
                                 x1 - half_width[i])
            centers[i, 1] = clip(centers[i, 1], y0 + half_height[i],
                                 y1 - half_height[i])
        if moved:
            any_moved = True
        else:
            break
    return any_moved


cdef void declutter(float[:, ::1] centers,
                    const float[:, ::1] anchors,
                    const float[::1] half_width,
                    const float[::1] half_height,
                    const float x0,
                    const float y0,
                    const float x1,
                    const float y1,
                    const float[:, ::1] scatter,
                    const float min_line,
                    float[::1] ex_px,
                    float[::1] ey_px) noexcept nogil:
    """
    Turn a relaxed layout into a clean one (display pixels), in place. Each
    round swaps the centers of any two labels whose lines cross (a swap
    shortens total line length, so it terminates); nudges boxes clear of any
    other label's line passing through them, any plotted point they cover
    (`avoid`, whose first `n` rows are the anchors, so `i == j` skips a label's
    own point), and their own point when closer than `min_line`; then separates
    overlapping boxes. Separation runs last, so on exit no boxes overlap, no
    lines cross, and no line pierces a box.
    """
    cdef unsigned i, j, r, start_k, n = centers.shape[0], \
        m = scatter.shape[0], rounds = 6 * n + 6
    cdef bint changed, have_best
    cdef float sx, sy, length, perpx, perpy, min_x, max_x, min_y, max_y, \
        side, repulsion, px, py, dx, dy, ox, oy, ax, ay, hw, hh, \
        cx, cy, tx, ty, d, bestd, bestx, besty
    cdef Point2D* pts
    cdef Point2D target
    cdef Point2D* res

    for r in range(rounds):
        changed = False

        # Uncross lines by swapping the two labels' centers
        for i in range(n):
            line_end(anchors[i, 0], anchors[i, 1], centers[i, 0],
                     centers[i, 1], half_width[i], half_height[i],
                     ex_px[i], ey_px[i])
        for i in range(n):
            for j in range(i + 1, n):
                if cross(anchors[i, 0], anchors[i, 1], ex_px[i], ey_px[i],
                         anchors[j, 0], anchors[j, 1], ex_px[j], ey_px[j]):
                    swap(centers[i, 0], centers[j, 0])
                    swap(centers[i, 1], centers[j, 1])
                    changed = True
                    line_end(anchors[i, 0], anchors[i, 1], centers[i, 0],
                             centers[i, 1], half_width[i], half_height[i],
                             ex_px[i], ey_px[i])
                    line_end(anchors[j, 0], anchors[j, 1], centers[j, 0],
                             centers[j, 1], half_width[j], half_height[j],
                             ex_px[j], ey_px[j])

        # Lines through boxes
        for i in range(n):
            sx = ex_px[i] - anchors[i, 0]
            sy = ey_px[i] - anchors[i, 1]
            length = sqrt(sx * sx + sy * sy)
            if length < <float> 1e-6:
                continue
            perpx = -sy / length
            perpy = sx / length
            min_x = anchors[i, 0] if anchors[i, 0] < ex_px[i] else ex_px[i]
            max_x = anchors[i, 0] if anchors[i, 0] > ex_px[i] else ex_px[i]
            min_y = anchors[i, 1] if anchors[i, 1] < ey_px[i] else ey_px[i]
            max_y = anchors[i, 1] if anchors[i, 1] > ey_px[i] else ey_px[i]
            for j in range(n):
                if i == j:
                    continue
                if max_x < centers[j, 0] - half_width[j] - <float> 0.5 or \
                        min_x > centers[j, 0] + half_width[j] + <float> 0.5:
                    continue
                if max_y < centers[j, 1] - half_height[j] - <float> 0.5 or \
                        min_y > centers[j, 1] + half_height[j] + <float> 0.5:
                    continue
                if not segment_hits_box(anchors[i, 0], anchors[i, 1],
                                        ex_px[i], ey_px[i],
                                        centers[j, 0] - half_width[j],
                                        centers[j, 1] - half_height[j],
                                        centers[j, 0] + half_width[j],
                                        centers[j, 1] + half_height[j], 0.5):
                    continue
                side = (centers[j, 0] - anchors[i, 0]) * perpx + \
                       (centers[j, 1] - anchors[i, 1]) * perpy
                repulsion = abs(half_width[j] * perpx) + \
                    abs(half_height[j] * perpy) + <float> 1.5 - abs(side)
                if repulsion > <float> 0:
                    if side < <float> 0:
                        repulsion = -repulsion
                    centers[j, 0] += perpx * repulsion
                    centers[j, 1] += perpy * repulsion
                    changed = True

        # Boxes over points
        for i in range(n):
            hw = half_width[i]
            hh = half_height[i]

            for j in range(n):
                if i == j:
                    continue
                px = anchors[j, 0]
                py = anchors[j, 1]
                dx = centers[i, 0] - px
                dy = centers[i, 1] - py
                if abs(dx) < hw and abs(dy) < hh:
                    ox = hw - abs(dx)
                    oy = hh - abs(dy)
                    if ox <= oy:
                        centers[i, 0] += ox if dx >= <float> 0 else -ox
                    else:
                        centers[i, 1] += oy if dy >= <float> 0 else -oy
                    changed = True

            if m > 0:
                pts = <Point2D*> &scatter[0, 0]
                target.x = centers[i, 0] - hw
                res = lower_bound(pts, pts + m, target)
                start_k = res - pts
                for j in range(start_k, m):
                    px = scatter[j, 0]
                    dx = centers[i, 0] - px
                    if dx < -hw:
                        break
                    py = scatter[j, 1]
                    dy = centers[i, 1] - py
                    if abs(dy) < hh:
                        ox = hw - abs(dx)
                        oy = hh - abs(dy)
                        if ox <= oy:
                            centers[i, 0] += ox if dx >= <float> 0 else -ox
                        else:
                            centers[i, 1] += oy if dy >= <float> 0 else -oy
                        changed = True

            # Keep the box `min_line` off its own point, moving the smallest
            # amount that still fits in bounds, so every label gets a line
            ax = anchors[i, 0]
            ay = anchors[i, 1]
            if abs(centers[i, 0] - ax) < hw + min_line and \
                    abs(centers[i, 1] - ay) < hh + min_line:
                cx = centers[i, 0]
                cy = centers[i, 1]
                have_best = False
                bestd = 0
                bestx = 0
                besty = 0
                for j in range(4):
                    if j == 0:
                        tx = ax + hw + min_line
                        ty = cy
                    elif j == 1:
                        tx = ax - hw - min_line
                        ty = cy
                    elif j == 2:
                        tx = cx
                        ty = ay + hh + min_line
                    else:
                        tx = cx
                        ty = ay - hh - min_line
                    if x0 <= tx - hw and tx + hw <= x1 and \
                            y0 <= ty - hh and ty + hh <= y1:
                        d = abs(tx - cx) + abs(ty - cy)
                        if not have_best or d < bestd:
                            have_best = True
                            bestd = d
                            bestx = tx
                            besty = ty
                if have_best:
                    centers[i, 0] = bestx
                    centers[i, 1] = besty
                    changed = True

        if separate(centers, half_width, half_height, x0, y0, x1, y1, 200):
            changed = True
        if not changed:
            break


def label(ax,
          x,
          y,
          texts,
          x_scatter=None,
          y_scatter=None,
          *,
          attraction: float = 5e-2,
          repulsion: float = 1e-6,
          max_iter: int = 4000,
          seed: int = 0,
          padding: float = 4.0,
          box_padding: float = 4.0,
          fontsize: float = 10,
          linewidth: float = 0.6,
          linecolor='0.45') -> None:
    """
    Place text labels next to a set of points and connect each label to its
    point with a line, keeping the labels from overlapping one another, their
    lines from crossing, and the labels off the plotted points.

    The scatter must already be drawn on `ax` before this is called, so the
    axes limits and size are final. Labels are positioned by ggrepel's force
    simulation and a deterministic clean-up pass, both run in display pixels so
    the layout is unaffected by how differently the two data axes are scaled;
    the text and line artists are added directly to `ax`, the text drawn above
    its line above the scatter by Matplotlib's default z-ordering.

    Args:
        ax: the Matplotlib axes to draw the labels and lines on. Its scatter
            must already be drawn, so that the axes limits and pixel size are
            final; the layout is computed against those and does not update if
            the axes are subsequently rescaled or resized.
        x: the x data coordinates of the points to label, one per label, in the
           same order as `texts`.
        y: the y data coordinates of the points to label, one per label, in the
           same order as `texts` and `x`.
        texts: the label strings, one per point, in the same order as `x`/`y`.
        x_scatter: the x data coordinates of the background scatter to keep
                   labels off, or `None` to repel labels only from the points
                   being labeled. If given, `y_scatter` must be given too.
                   Only points near the labeled region are considered.
        y_scatter: the y data coordinates of the background scatter, in the
                   same order as `x_scatter`. Must be given together with
                   `x_scatter`, and be the same length.
        attraction: the strength of the spring that pulls each label back
                    toward its own point when it touches nothing; larger values
                    keep labels nearer their points.
        repulsion: the strength of the repulsion that pushes labels away from
                   one another and from points; larger values spread the labels
                   wider before they settle. Also the standard deviation, in
                   pixels, of the small initial jitter of the labels.
        max_iter: the maximum number of force-simulation iterations to run. The
                  simulation also stops early once no box overlaps anything, so
                  this is only reached for crowded layouts.
        seed: the seed for the small initial jitter that breaks ties between
              symmetric labels. Fixed, so a given input always yields the same
              layout.
        padding: the clearance kept, in pixels, between a label box and any
                 point (its own, another label's, or a background point). It is
                 also the shortest line drawn, so no label sits on its own
                 point without a visible connecting line.
        box_padding: the padding, in pixels, added around each label's text
                     box. It sets the minimum gap kept between any two labels.
        fontsize: the label font size, in points.
        linewidth: the width of the connecting lines, in points.
        linecolor: the color of the connecting lines. Can be any valid
                   Matplotlib color, like a hex string (e.g. `'#FF0000'`), a
                   named color (e.g. 'red'), a 3- or 4-element RGB/RGBA tuple
                   of integers 0-255 or floats 0-1, or a single float 0-1 for
                   grayscale.
    """
    cdef unsigned i, n = len(texts)
    cdef float bx0, by0, bw, bh, S, min_sep2, mx, my, ptp
    cdef np.ndarray half_width, half_height, xy, xy_px, sizes, scatter, \
        labeled, scatter_px, width, centers_px, velocity, centers_data, \
        ends_data, ex_px_buffer = np.empty(n, dtype=np.float32), \
        ey_px_buffer = np.empty(n, dtype=np.float32)
    cdef float[::1] ex_px = ex_px_buffer, ey_px = ey_px_buffer

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    to_pixels = ax.transData.transform
    to_data = ax.transData.inverted().transform

    # The whole layout runs in display pixels, where the axes are isotropic, so
    # the forces push equally hard in x and y no matter how the data axes are
    # scaled. Convert the labeled points to pixels up front.
    xy = np.c_[x, y]
    xy_px = to_pixels(xy).astype(np.float32)

    # Measure each label box in pixels (half-extents, including box padding)
    sizes = np.empty((n, 2), dtype=np.float32)
    artist = ax.text(0, 0, '', fontsize=fontsize)
    for i in range(n):
        artist.set_text(texts[i])
        sizes[i] = artist.get_window_extent(renderer).size
    artist.remove()
    half_width = sizes[:, 0] * 0.5 + box_padding
    half_height = sizes[:, 1] * 0.5 + box_padding

    # Get the bounding box of the axis in pixels
    bx0, by0, bw, bh = ax.bbox.bounds

    # Obstacle points, in pixels: the labels' own points, then the other points
    # in the scatter plot (if any)
    if x_scatter is not None:
        scatter = np.c_[x_scatter, y_scatter]
        # Drop the labeled points so they aren't counted twice
        labeled = (scatter[:, None, 0] == x) & (scatter[:, None, 1] == y)
        scatter = scatter[~labeled.any(axis=1)]
        if len(scatter):
            scatter_px = to_pixels(scatter).astype(np.float32)
            scatter_px = scatter_px[np.argsort(scatter_px[:, 0])]
        else:
            scatter_px = np.empty((0, 2), dtype=np.float32)
    else:
        scatter_px = np.empty((0, 2), dtype=np.float32)

    # ggrepel scales momentum by box width; normalize widths to ~[0, 1]
    width = 2 * half_width
    ptp = np.ptp(width)
    if ptp > <float> 0:
        width -= width.min()
        width /= ptp
    else:
        width[:] = 0.5

    # Relax the label centers in pixel space, starting from their points plus a
    # tiny (sub-pixel) jitter to break ties between coincident labels
    centers_px = xy_px + np.random.default_rng(seed).normal(
        0, repulsion, (n, 2)).astype(np.float32)
    velocity = np.zeros((n, 2), dtype=np.float32)

    # Calculate geometric scale mapping from normalized device coordinates to
    # pixels
    S = sqrt(bw * bh)

    # Dynamically scale ggrepel parameters to the current pixel space
    repulsion *= <double> (S * S * S)
    min_sep2 = MIN_SEP2 * S * S

    # Perform the label repulsion
    repel_boxes(centers_px, half_width, half_height, xy_px, scatter_px,
                <float> padding, width, velocity, bx0, by0, bx0 + bw, by0 + bh,
                <float> attraction, <float> repulsion, min_sep2, max_iter)

    PyErr_CheckSignals()

    # Clean up, still in pixels
    declutter(centers_px, xy_px, half_width, half_height, bx0, by0,
              bx0 + bw, by0 + bh, scatter_px, <float> padding, ex_px, ey_px)

    PyErr_CheckSignals()

    # Recompute final line ends into the pre-allocated buffers `ex_px` and
    # `ey_px`. Stop the drawn line at the text glyphs, not the padded box, so
    # it visibly meets its own label instead of ending in the gap between
    # labels.
    for i in range(n):
        line_end(xy_px[i, 0], xy_px[i, 1], centers_px[i, 0],
                 centers_px[i, 1], half_width[i] - box_padding,
                 half_height[i] - box_padding, ex_px[i], ey_px[i])

    # Convert the final label centers and line ends back to data coordinates,
    # batched for speed
    centers_data = to_data(centers_px)
    ends_data = to_data(np.c_[ex_px_buffer, ey_px_buffer])

    # Plot texts and lines. Line starts are the original points, in data
    # coordinates.
    for i in range(n):
        ax.text(centers_data[i, 0], centers_data[i, 1], texts[i],
                fontsize=fontsize, ha='center', va='center')
        ax.plot([xy[i, 0], ends_data[i, 0]], [xy[i, 1], ends_data[i, 1]],
                lw=linewidth, color=linecolor)