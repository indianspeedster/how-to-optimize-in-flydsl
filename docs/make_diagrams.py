# SPDX-License-Identifier: Apache-2.0
"""Generate one access-pattern diagram per rung into ``figure/access/``.

    python docs/make_diagrams.py

Every rung in this repo changes *which thread touches which element, when*.
That is the one thing prose is worst at and a picture is best at, so each rung
gets a diagram of exactly that -- and nothing else.

A single visual language across all 33, so the diagrams are diffable by eye the
same way the source files are:

    a square            one element (global memory, LDS slot, or register)
    filled blue         touched in this step
    filled orange       what CHANGED versus the previous rung
    hollow              present but not touched in this step
    a bracket above     one memory transaction, labelled with its width
    an arrow            data moving between slots

Counts are scaled down so a diagram fits on a page: threads-per-block is drawn
as 16 rather than 256, wavefronts as 8 lanes rather than 64. Every caption says
so. The shapes of the patterns are exact; only the counts are reduced.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "figure" / "access"

LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
             grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834",
             hollow="#fcfcfb")
DARK = dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
            grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926",
            hollow="#1a1a19")
FONT = "system-ui,-apple-system,'Segoe UI',sans-serif"

CW = 26      # cell width
CH = 20      # cell height
CG = 4       # gap between cells (>= the 2px surface-gap minimum)
LEFT = 104   # room for row labels


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Fig:
    """A tiny SVG canvas: cells, brackets, arrows, labels."""

    def __init__(self, th, title, caption):
        self.th, self.o = th, []
        self.title, self.caption = title, caption
        self.w, self.h = 0, 0
        self.y = 54 if caption else 36

    def _grow(self, x, y):
        self.w, self.h = max(self.w, x), max(self.h, y)

    def row_label(self, text, y, sub=None):
        t = self.th
        self.o.append(f'<text x="{LEFT - 10}" y="{y + CH / 2 + 4:.0f}" font-size="10.5" '
                      f'text-anchor="end" fill="{t["ink2"]}">{esc(text)}</text>')
        if sub:
            self.o.append(f'<text x="{LEFT - 10}" y="{y + CH / 2 + 15:.0f}" '
                          f'font-size="9" text-anchor="end" fill="{t["muted"]}">'
                          f'{esc(sub)}</text>')

    def cells(self, y, n, state, labels=None, start=0):
        """``state[i]`` is 0 hollow, 1 touched (blue), 2 changed (orange)."""
        t = self.th
        for i in range(n):
            x = LEFT + (start + i) * (CW + CG)
            s = state[i]
            fill = (t["hollow"], t["s1"], t["s2"])[s]
            stroke = t["axis"] if s == 0 else "none"
            self.o.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="3" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
            if labels and labels[i] is not None:
                col = t["surface"] if s else t["muted"]
                self.o.append(f'<text x="{x + CW / 2}" y="{y + CH / 2 + 3.5:.0f}" '
                              f'font-size="9.5" text-anchor="middle" fill="{col}">'
                              f'{esc(str(labels[i]))}</text>')
            self._grow(x + CW, y + CH)

    def bracket(self, y, i0, i1, text):
        """A transaction bracket spanning cells i0..i1 (inclusive), above ``y``."""
        t = self.th
        x0 = LEFT + i0 * (CW + CG) + 1
        x1 = LEFT + i1 * (CW + CG) + CW - 1
        self.o.append(f'<path d="M{x0} {y + 6} V{y + 1} H{x1} V{y + 6}" fill="none" '
                      f'stroke="{t["ink2"]}" stroke-width="1.2"/>')
        self.o.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y - 3}" font-size="9" '
                      f'text-anchor="middle" fill="{t["ink2"]}">{esc(text)}</text>')
        self._grow(x1, y)

    def arrow(self, i_from, i_to, y0, y1):
        """A curved arrow from cell i_from at y0 down into cell i_to at y1."""
        t = self.th
        xa = LEFT + i_from * (CW + CG) + CW / 2
        xb = LEFT + i_to * (CW + CG) + CW / 2
        self.o.append(f'<path d="M{xa:.0f} {y0} C{xa:.0f} {(y0 + y1) / 2:.0f} '
                      f'{xb:.0f} {(y0 + y1) / 2:.0f} {xb:.0f} {y1 - 5:.0f}" '
                      f'fill="none" stroke="{t["muted"]}" stroke-width="1.2"/>')
        self.o.append(f'<path d="M{xb - 3:.0f} {y1 - 6:.0f} L{xb:.0f} {y1 - 1:.0f} '
                      f'L{xb + 3:.0f} {y1 - 6:.0f} Z" fill="{t["muted"]}"/>')

    def grid(self, y, rows, cols, owner, cw=None, ch=None, rowlab=None):
        """A rows x cols matrix; ``owner(r, c)`` -> (state, label or None)."""
        t = self.th
        cw = cw or 22
        ch = ch or 15
        for r in range(rows):
            if rowlab:
                self.o.append(f'<text x="{LEFT - 8}" y="{y + r * (ch + 2) + ch - 3:.0f}" '
                              f'font-size="9" text-anchor="end" fill="{t["muted"]}">'
                              f'{esc(rowlab(r))}</text>')
            for c in range(cols):
                st, lab = owner(r, c)
                x = LEFT + c * (cw + 2)
                yy = y + r * (ch + 2)
                fill = (t["hollow"], t["s1"], t["s2"])[st]
                stroke = t["axis"] if st == 0 else "none"
                self.o.append(f'<rect x="{x}" y="{yy}" width="{cw}" height="{ch}" rx="2" '
                              f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
                if lab:
                    self.o.append(f'<text x="{x + cw / 2}" y="{yy + ch - 4:.0f}" '
                                  f'font-size="8.5" text-anchor="middle" '
                                  f'fill="{t["surface"] if st else t["muted"]}">'
                                  f'{esc(lab)}</text>')
                self._grow(x + cw, yy + ch)
        return y + rows * (ch + 2)

    def note(self, y, text):
        self.o.append(f'<text x="{LEFT}" y="{y}" font-size="10" '
                      f'fill="{self.th["muted"]}">{esc(text)}</text>')
        self._grow(LEFT + 8 * len(text), y)

    def legend(self, y, items):
        t, x = self.th, LEFT
        for col, name in items:
            # A hollow swatch needs its outline, or it is surface-on-surface.
            edge = f' stroke="{t["axis"]}" stroke-width="1"' if col == t["hollow"] else ""
            self.o.append(f'<rect x="{x}" y="{y - 8}" width="11" height="11" rx="2.5" '
                          f'fill="{col}"{edge}/>')
            self.o.append(f'<text x="{x + 16}" y="{y + 1}" font-size="9.5" '
                          f'fill="{t["ink2"]}">{esc(name)}</text>')
            x += 22 + 5.5 * len(name)
        self._grow(x, y)

    def render(self):
        t = self.th
        W, H = self.w + 20, self.h + 16
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
                f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>',
                f'<text x="0" y="17" font-size="13" font-weight="600" '
                f'fill="{t["ink"]}">{esc(self.title)}</text>']
        if self.caption:
            head.append(f'<text x="0" y="34" font-size="10.5" fill="{t["ink2"]}">'
                        f'{esc(self.caption)}</text>')
        return "\n".join(head + self.o + ["</svg>"])


def emit(name, build):
    for tag, th in (("light", LIGHT), ("dark", DARK)):
        f = build(th)
        (OUT / f"{name}-{tag}.svg").write_text(f.render())
    return name


# ===================================================================== ops ===

N = 8            # lanes drawn per wavefront
TB = 16          # threads drawn per block (the kernels use 256)


def elementwise(vec, per_thread, title, caption, changed_lane=None):
    def build(th):
        f = Fig(th, title, caption)
        y = f.y
        total = N * vec * per_thread
        for p in range(per_thread):
            f.row_label("A / B / C" if p == 0 else "", y,
                        f"pass {p}" if per_thread > 1 else None)
            state, labels = [], []
            for i in range(N * vec):
                lane = i // vec
                state.append(2 if (changed_lane is not None and lane == 0) else 1)
                labels.append(f"t{lane}" if i % vec == 0 else None)
            f.cells(y, N * vec, state, labels)
            for lane in range(N):
                f.bracket(y, lane * vec, lane * vec + vec - 1,
                          {1: "dword", 2: "x2", 4: "x4"}[vec] if lane == 0 else "")
            y += CH + 30
        f.note(y, f"{N} lanes shown of a 64-lane wavefront; each lane moves "
                  f"{vec} f32 = {vec * 32} bits per transaction"
                  + (f", x{per_thread} passes" if per_thread > 1 else ""))
        y += 22
        f.legend(y, [(th["s1"], "elements this lane touches"),
                     (th["s2"], "lane 0, highlighted")])
        f.y = y + 8
        f._grow(0, y + 8)
        return f
    return build


def reduce_tree(scheme, title, caption):
    """v0/v1/v2 -- the same tree, three choices of which thread does what."""
    def build(th):
        f = Fig(th, title, caption)
        y = f.y
        f.row_label("global", y)
        f.cells(y, TB, [1] * TB, [f"t{i}" for i in range(TB)])
        f.bracket(y, 0, TB - 1, "one element per thread, straight into LDS")
        y += CH + 34

        s = 1 if scheme != "sequential" else TB // 2
        level = 0
        while (s < TB) if scheme != "sequential" else (s >= 1):
            if scheme == "interleaved":
                act = [i for i in range(TB) if i % (2 * s) == 0]
                slots = act
            elif scheme == "contiguous":
                act = [i for i in range(TB) if 2 * s * i < TB]
                slots = [2 * s * i for i in act]
            else:
                act = [i for i in range(TB) if i < s]
                slots = act

            f.row_label(f"step {level}", y, f"stride {s}")
            st = [2 if i in act else 0 for i in range(TB)]
            f.cells(y, TB, st, [f"t{i}" for i in range(TB)])
            y += CH + 4
            ly = y + 16
            for a, sl in zip(act, slots):
                f.arrow(sl + s, sl, y + 2, ly)
            st2 = [1 if i in slots else (0 if i not in
                   [x + s for x in slots] else 1) for i in range(TB)]
            f.cells(ly, TB, st2, [None] * TB)
            f.row_label("LDS", ly)
            y = ly + CH + 28
            level += 1
            s = s * 2 if scheme != "sequential" else s // 2

        f.note(y, f"{TB} threads shown of 256. Top row of each step: which "
                  f"threads are ACTIVE. Bottom: the LDS slots they touch.")
        y += 22
        f.legend(y, [(th["s2"], "active thread"), (th["s1"], "LDS slot read/written"),
                     (th["hollow"], "idle / untouched")])
        f.y = y + 8
        f._grow(0, y + 8)
        return f
    return build


def reduce_folded(tail, unroll, title, caption):
    """v3/v4/v5 -- two globals folded per thread, then a shortened tree."""
    def build(th):
        f = Fig(th, title, caption)
        y = f.y
        f.row_label("global", y, "2N elements")
        f.cells(y, TB, [1] * TB, [f"t{i}" for i in range(TB)])
        f.cells(y, TB, [2] * TB, [f"t{i}" for i in range(TB)], start=TB)
        f.bracket(y, 0, 2 * TB - 1, "each thread adds TWO elements on the way in")
        y += CH + 30
        f.row_label("LDS", y, "N slots")
        f.cells(y, TB, [1] * TB, [None] * TB)
        y += CH + 26

        floor = TB // 4 if tail == "wave" else 1     # 'wavefront' scaled to 4
        st, level = TB // 2, 0
        while st >= floor:
            f.row_label(f"step {level}", y, f"stride {st}")
            f.cells(y, TB, [1 if i < st else 0 for i in range(TB)], [None] * TB)
            f.bracket(y, 0, st - 1, "barrier" + (" (unrolled)" if unroll else ""))
            y += CH + 26
            st //= 2
            level += 1

        if tail == "wave":
            f.row_label("wavefront", y, "registers")
            f.cells(y, floor, [2] * floor, [f"l{i}" for i in range(floor)])
            f.bracket(y, 0, floor - 1, "shuffle ladder -- no barrier, no LDS")
            y += CH + 26

        f.note(y, "16 threads shown of 256; the wavefront tail is 64 lanes, "
                  "drawn as 4. Blue = LDS, orange = the change from the rung before.")
        y += 22
        f.legend(y, [(th["s1"], "LDS slot"), (th["s2"], "what changed"),
                     (th["hollow"], "idle")])
        f.y = y + 8
        f._grow(0, y + 8)
        return f
    return build


def reduce_serial(vec, wide, finish, title, caption):
    """v6-v9 -- serial accumulation, then a cheap finish."""
    def build(th):
        f = Fig(th, title, caption)
        y = f.y
        steps = 3
        for k in range(steps):
            f.row_label("global" if k == 0 else "", y, f"step {k}")
            st = [2 if vec == 4 else 1] * (N * vec)
            lab = [f"t{i // vec}" if i % vec == 0 else None for i in range(N * vec)]
            f.cells(y, N * vec, st, lab)
            if k == 0:
                for lane in range(N):
                    f.bracket(y, lane * vec, lane * vec + vec - 1,
                              ("dwordx4" if vec == 4 else "dword") if lane == 0 else "")
            y += CH + 24
        f.note(y, "... each step advances by a whole block, so every wavefront "
                  "issues one fully coalesced request per step")
        y += 24
        f.row_label("register", y)
        f.cells(y, N, [1] * N, [f"t{i}" for i in range(N)])
        f.bracket(y, 0, N - 1, "one accumulator per thread -- the tree is now almost irrelevant")
        y += CH + 28
        if finish == "lds":
            f.row_label("LDS tree", y)
            f.cells(y, N, [1 if i < N // 2 else 0 for i in range(N)], [None] * N)
            f.bracket(y, 0, N // 2 - 1, "the v5 tree, on one value per thread")
        else:
            f.row_label("shuffle", y)
            f.cells(y, N, [2] * N, [f"l{i}" for i in range(N)])
            f.bracket(y, 0, N - 1, "wave shuffle, then 4 LDS slots -- no tree at all")
        y += CH + 26
        f.note(y, ("grid widened to 32 blocks/CU" if wide else
                   "grid fixed at 1024 blocks, as in the CUDA original")
                  + "; 8 lanes shown of 64")
        y += 22
        f.legend(y, [(th["s1"], "touched"), (th["s2"], "what changed"),
                     (th["hollow"], "idle")])
        f.y = y + 8
        f._grow(0, y + 8)
        return f
    return build


def sgemv_map(mode, title, caption):
    """v0-v3 -- how rows of A are mapped onto lanes."""
    ROWS, COLS = 8, 16

    def build(th):
        f = Fig(th, title, caption)
        y = f.y

        def owner(r, c):
            if mode == "wave":                       # one wavefront per row
                return (1 if r == 0 else 0, f"l{c}" if r == 0 else None)
            if mode == "wave4":                      # one wavefront, float4 each
                return (1 if r == 0 else 0,
                        f"l{c // 4}" if r == 0 and c % 4 == 0 else None)
            if mode == "subwave":
                # N=16 wide, so a 64-lane wavefront splits into FOUR 16-lane
                # segments; each segment owns a whole row at once.
                segs = 4
                return (1 if r < segs else 0,
                        f"l{r * COLS + c}" if r < segs else None)
            return (1 if r == 0 else 0,              # whole workgroup per row
                    f"t{c}" if r == 0 else None)

        y = f.grid(y, ROWS, COLS, owner, rowlab=lambda r: f"row {r}")
        y += 20
        f.note(y, {"wave": "one 64-lane wavefront owns one row; lane l takes "
                           "column l, then strides by 64",
                   "wave4": "same mapping, but each lane takes 4 consecutive "
                            "columns as one dwordx4",
                   "subwave": "the matrix is 16 wide, so one 64-lane wavefront "
                              "splits into four 16-lane segments -- four rows at "
                              "once, no idle lanes",
                   "block": "the whole 256-thread workgroup takes one row, "
                            "finishing with an LDS block reduction"}[mode])
        y += 18
        f.note(y, "A drawn as 8x16; lanes drawn as 8 of 64")
        y += 22
        f.legend(y, [(th["s1"], "covered this step"), (th["hollow"], "not yet")])
        f.y = y + 8
        f._grow(0, y + 8)
        return f
    return build


def sgemm_block(mode, title, caption):
    """v0-v5 -- what one thread owns, and where its operands come from."""
    def build(th):
        f = Fig(th, title, caption)
        y = f.y
        if mode == "naive":
            f.row_label("C", y)
            y = f.grid(y, 4, 8, lambda r, c: (2 if (r, c) == (0, 0) else 0,
                                              "t0" if (r, c) == (0, 0) else None))
            y += 18
            f.note(y, "one thread owns ONE element of C and streams K elements "
                      "of A and K of B from global for it -- no reuse at all")
        elif mode == "lds":
            f.row_label("C tile", y)
            y = f.grid(y, 4, 8, lambda r, c: (1, "t" + str(r * 8 + c)
                                              if r * 8 + c < 10 else None))
            y += 22
            f.row_label("from LDS", y)
            y = f.grid(y, 2, 8, lambda r, c: (2, None))
            y += 18
            f.note(y, "a 16x16x16 tile is staged in LDS once, then every element "
                      "is used 16 times instead of once")
        elif mode == "tuned":
            f.row_label("grid @1024^3", y, "128x128 tile")
            y = f.grid(y, 4, 8, lambda r, c: (0, None), cw=16, ch=12)
            y += 12
            f.note(y, "64 blocks on a 256-CU part: three quarters of the machine idle")
            y += 26
            f.row_label("grid @1024^3", y, "64x64 tile")
            y = f.grid(y, 8, 16, lambda r, c: (2, None), cw=16, ch=12)
            y += 12
            f.note(y, "256 blocks: the tile is picked from the shape, which is "
                      "the whole 3.1x over v5 at this size")
            y += 26
            f.row_label("K-loop", y)
            f.cells(y, 4, [1, 2, 1, 2],
                    ["barrier", "load k+1", "mfma(p)", "->LDS(1-p)"])
            y += CH + 18
            f.note(y, "LDS ping-pong + register prefetch + sched hints "
                      "(the hints measured flat -- kept, labelled)")
        elif mode in ("tile", "prefetch", "double"):
            f.row_label("C block tile", y)
            y = f.grid(y, 4, 8, lambda r, c: (2 if (r < 2 and c < 2) else 1,
                                              "t0" if (r, c) == (0, 0) else None))
            y += 18
            f.note(y, "128x128 per workgroup; each thread owns an 8x8 REGISTER "
                      "tile (orange) -- 64 accumulators, 162 VGPRs, no spill")
            y += 26
            lanes = {"tile": ["load", "->LDS", "barrier", "math", "barrier"],
                     "prefetch": ["->LDS", "barrier", "load k+1", "math", "barrier"],
                     "double": ["barrier", "load k+1", "math(p)", "->LDS(1-p)"]}[mode]
            f.row_label("K-loop", y)
            st = [2 if ("load" in s or "LDS" in s) else 1 for s in lanes]
            f.cells(y, len(lanes), st, lanes)
            y += CH + 18
            f.note(y, {"tile": "global load latency fully exposed; two barriers per tile",
                       "prefetch": "the next tile's loads issue BEFORE this tile's "
                                   "math, so VMEM retires under the FMAs",
                       "double": "LDS ping-pong: write buffer 1-p while reading p, "
                                 "so one barrier per tile instead of two"}[mode])
        else:   # mfma
            f.row_label("A (16x4)", y)
            y = f.grid(y, 4, 8, lambda r, c: (1, f"l{r * 16 + c}"), cw=30)
            y += 20
            f.row_label("D (16x16)", y)
            y = f.grid(y, 4, 8, lambda r, c: (2, f"l{c}"), cw=30)
            y += 18
            f.note(y, "v_mfma_f32_16x16x4_f32: lane l holds A[l%16][l/16], "
                      "B[l/16][l%16], and D[4*(l/16)+r][l%16] for r in 0..3")
            y += 18
            f.note(y, "confirmed against A @ B on hardware before any GEMM was "
                      "built on it -- see docs/porting-notes.md Sec. 2.8")
        y += 22
        f.legend(y, [(th["s1"], "touched"), (th["s2"], "the step's subject"),
                     (th["hollow"], "other threads")])
        f.y = y + 8
        f._grow(0, y + 8)
        return f
    return build


def spmv_lanes(lanes, title, caption):
    """v0-v4 -- how many lanes walk one CSR row."""
    NNZ = 16

    def build(th):
        f = Fig(th, title, caption)
        y = f.y
        f.row_label("row 0 nnz", y, f"{NNZ} entries")
        f.cells(y, NNZ, [2] * NNZ,
                [f"l{i % lanes}" if lanes <= 8 else f"l{i}" for i in range(NNZ)])
        f.bracket(y, 0, min(lanes, NNZ) - 1,
                  f"{lanes} lane(s) in step 0, then stride by {lanes}")
        y += CH + 30
        f.row_label("row 1 nnz", y)
        f.cells(y, NNZ, [1] * NNZ, [None] * NNZ)
        y += CH + 26
        f.note(y, f"{lanes} lanes per row. Their gathers into x land in the SAME "
                  f"row, so they coalesce; the cost is a "
                  f"{max(1, lanes.bit_length() - 1)}-step shuffle reduction.")
        y += 18
        f.note(y, "16 non-zeros drawn per row; the reduction runs at segment width")
        y += 22
        f.legend(y, [(th["s2"], "this row's entries"), (th["s1"], "the next row")])
        f.y = y + 8
        f._grow(0, y + 8)
        return f
    return build


def spmm_map(vec, staged, title, caption):
    def build(th):
        f = Fig(th, title, caption)
        y = f.y
        if staged:
            f.row_label("A row -> LDS", y)
            f.cells(y, N, [2] * N, [f"t{i}" for i in range(N)])
            f.bracket(y, 0, N - 1, "CHUNK (col,val) pairs staged cooperatively")
            y += CH + 30
        f.row_label("A row nnz", y)
        f.cells(y, 4, [1] * 4, ["nz0", "nz1", "nz2", "nz3"])
        f.bracket(y, 0, 3, "walked serially -- same row for every thread in the block")
        y += CH + 30
        f.row_label("C columns", y)
        st = [2] * (N * vec)
        lab = [f"t{i // vec}" if i % vec == 0 else None for i in range(N * vec)]
        f.cells(y, N * vec, st, lab)
        for lane in range(N):
            f.bracket(y, lane * vec, lane * vec + vec - 1,
                      ("dwordx4" if vec == 4 else "dword") if lane == 0 else "")
        y += CH + 28
        f.note(y, f"each thread owns {vec} output column(s); the dense side is "
                  f"where essentially all the traffic is")
        if staged:
            y += 18
            f.note(y, "the LDS staging also forces a CHUNK-sized outer loop -- "
                      "which is what wins 4.6x on a power-law matrix")
        y += 22
        f.legend(y, [(th["s1"], "sparse side"), (th["s2"], "dense side / staged")])
        f.y = y + 8
        f._grow(0, y + 8)
        return f
    return build


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, b in [
        ("elementwise-v0", elementwise(1, 1, "elementwise v0 -- one f32 per lane",
         "lane i owns element i; one buffer_load_dword each", 0)),
        ("elementwise-v1", elementwise(2, 1, "elementwise v1 -- two f32 per lane",
         "lane i owns 2i, 2i+1; half the instructions for the same bytes", 0)),
        ("elementwise-v2", elementwise(4, 1, "elementwise v2 -- four f32 per lane",
         "lane i owns 4i..4i+3; one dwordx4 = 128 B per wavefront request", 0)),
        ("elementwise-v3", elementwise(4, 2, "elementwise v3 -- 4x float4 per lane",
         "same width, more loads in flight. Measured: slower than v2.", 0)),
        ("reduce-v0", reduce_tree("interleaved", "reduce v0 -- tid % (2s) == 0",
         "active threads are STRIDED: every wavefront runs half-masked")),
        ("reduce-v1", reduce_tree("contiguous", "reduce v1 -- index = 2*s*tid",
         "active threads packed at the bottom; LDS slots still strided")),
        ("reduce-v2", reduce_tree("sequential", "reduce v2 -- s halves, tid < s",
         "threads packed AND slots unit-stride: conflict-free")),
        ("reduce-v3", reduce_folded("lds", False, "reduce v3 -- add during load",
         "two globals folded per thread: half the blocks, one level less of tree")),
        ("reduce-v4", reduce_folded("wave", False, "reduce v4 -- wavefront tail",
         "the last wavefront finishes in registers: no barrier, no LDS")),
        ("reduce-v5", reduce_folded("wave", True, "reduce v5 -- full unroll",
         "same tail, LDS levels emitted straight-line at compile time")),
        ("reduce-v6", reduce_serial(1, False, "lds", "reduce v6 -- serial accumulate",
         "each thread streams N/(1024*256) elements into a register first")),
        ("reduce-v7", reduce_serial(1, False, "wave", "reduce v7 -- wave shuffle",
         "the tree is replaced by one shuffle per wavefront. Fastest rung.")),
        ("reduce-v8", reduce_serial(4, False, "wave", "reduce v8 -- dwordx4 loads",
         "wider transaction on the same pattern. Measured: no gain.")),
        ("reduce-v9", reduce_serial(4, True, "wave", "reduce v9 -- wider grid",
         "32 blocks per CU instead of 1024 total. Measured: no gain.")),
        ("sgemv-v0", sgemv_map("wave", "sgemv v0 -- one wavefront per row",
         "the CUDA N==32 case becomes N==64 here")),
        ("sgemv-v1", sgemv_map("wave4", "sgemv v1 -- float4 per lane",
         "same mapping, wider transaction. Does not beat v0.")),
        ("sgemv-v2", sgemv_map("subwave", "sgemv v2 -- rows share a wavefront",
         "N lanes per row, 64/N rows at once: no idle lanes on short rows")),
        ("sgemv-v3", sgemv_map("block", "sgemv v3 -- one workgroup per row",
         "for long rows: keeps every CU busy when M is small")),
        ("sgemm-v0", sgemm_block("naive", "sgemm v0 -- no blocking",
         "one thread, one C element, everything from global")),
        ("sgemm-v1", sgemm_block("lds", "sgemm v1 -- global to LDS",
         "blocking level 1: each loaded element is used 16 times")),
        ("sgemm-v2", sgemm_block("tile", "sgemm v2 -- LDS to registers",
         "blocking level 2: an 8x8 register tile per thread")),
        ("sgemm-v3", sgemm_block("prefetch", "sgemm v3 -- prefetch",
         "the next tile's global loads issue before this tile's math")),
        ("sgemm-v4", sgemm_block("double", "sgemm v4 -- LDS ping-pong",
         "one barrier per K-tile instead of two")),
        ("sgemm-v6", sgemm_block("tuned", "sgemm v6 -- tile picked from the shape",
         "the same MFMA kernel, sized so the grid fills the machine")),
        ("sgemm-v5", sgemm_block("mfma", "sgemm v5 -- the matrix cores",
         "same blocking, v_mfma_f32_16x16x4_f32 instead of vector FMA")),
        ("spmv-v0", spmv_lanes(1, "spmv v0 -- one lane per row",
         "64 lanes walk 64 DIFFERENT rows: uncorrelated gathers")),
        ("spmv-v1", spmv_lanes(4, "spmv v1 -- four lanes per row",
         "the four gathers now land in the same row and coalesce")),
        ("spmv-v2", spmv_lanes(8, "spmv v2 -- eight lanes per row",
         "best setting when rows are short (8 nnz/row)")),
        ("spmv-v3", spmv_lanes(16, "spmv v3 -- sixteen lanes per row",
         "best setting at 32 nnz/row: about two non-zeros per lane")),
        ("spmv-v4", spmv_lanes(8, "spmv v4 -- a whole wavefront per row",
         "past the optimum on a uniform matrix; 13.8x on a power-law one")),
        ("spmm-v0", spmm_map(1, False, "spmm v0 -- one output column per thread",
         "the whole block walks the same sparse row, broadcast out of L1")),
        ("spmm-v1", spmm_map(1, True, "spmm v1 -- stage the row in LDS",
         "the kernel the CUDA source calls 'useless optimize'")),
        ("spmm-v2", spmm_map(4, False, "spmm v2 -- four output columns per thread",
         "the sparse walk is unchanged; only the dense side gets wider")),
    ]:
        emit(name, b)
        n += 1
    print(f"{n} diagrams x 2 themes -> figure/access/")


if __name__ == "__main__":
    main()
