# SPDX-License-Identifier: Apache-2.0
"""Generate the figures in ``figure/`` from ``results/bench.json``.

    python docs/make_figures.py          # after `make bench-json`

The charts are drawn from the measured JSON, never typed by hand, so a figure
cannot drift from the numbers in the tables beside it. Output is plain SVG with
no dependencies -- the repo stays installable without a plotting stack.

Every chart is emitted twice, ``-light`` and ``-dark``, and embedded in the
markdown with ``<picture>`` so it reads correctly in either GitHub theme. The
dark variant is a *selected* set of steps for the dark surface, not an inverted
copy of the light one.

Palette: slots 1 and 2 (blue / orange) of the reference categorical palette,
used unmodified. Bars are coloured by *provenance* -- ported from the CUDA
original, or added for CDNA -- which is identity, not magnitude; bar length
already carries magnitude and colouring by value would double-encode it.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "bench.json"
OUT = ROOT / "figure"

# ---------------------------------------------------------------- palette ---

LIGHT = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834",
)
DARK = dict(
    surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926",
)
FONT = "system-ui,-apple-system,'Segoe UI',sans-serif"

BAR_H = 18          # <= 24px: never fill the slot, leave the band's air
BAR_GAP = 10        # well clear of the 2px minimum surface gap
PANEL_PAD = 34
LABEL_W = 178
PLOT_W = 430
RIGHT_PAD = 74      # room for a value label past the bar end

# Panels: which shapes tell each op's story, in order.
PANELS = {
    "elementwise": ["N=32M"],
    "reduce": ["N=32M"],
    "sgemv": ["M=16384,N=16", "M=16384,N=256", "M=16384,N=4096"],
    "sgemm": ["1024^3", "2048^3", "4096^3"],
    "spmv": ["1M rows, 32 nnz/row", "1M rows, 8 nnz/row", "1M rows, skewed"],
    "spmm": ["4096x4096, 32nnz, n=256", "4096x4096, 32nnz, n=1024",
             "4096x4096, skewed, n=256"],
}
TITLES = {
    "elementwise": ("elementwise: C = A + B",
                    "achieved bandwidth per rung -- wider lane transactions"),
    "reduce": ("reduce: block-wise sum",
               "achieved bandwidth per rung -- the ten-step ladder"),
    "sgemv": ("sgemv: y = A x",
              "achieved bandwidth per rung, by matrix shape"),
    "sgemm": ("sgemm: C = A B (f32)",
              "throughput per rung, by problem size"),
    "spmv": ("spmv: y = A_csr x",
             "throughput as the lanes-per-row knob is swept"),
    "spmm": ("spmm: C = A_csr B",
             "throughput per rung, by dense width and sparsity pattern"),
}


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def nice_ticks(vmax, n=4):
    """Round tick steps (1/2/5 x 10^k) covering ``vmax``."""
    if vmax <= 0:
        return [0], 1.0
    raw = vmax / n
    mag = 10 ** int(pathlib.Path and __import__("math").floor(
        __import__("math").log10(raw)))
    for m in (1, 2, 2.5, 5, 10):
        if mag * m >= raw:
            step = mag * m
            break
    top = step * (int(vmax / step) + 1)
    ticks = []
    t = 0.0
    while t <= top + 1e-9:
        ticks.append(t)
        t += step
    return ticks, top


def fmt(v):
    """Value labels: enough precision to be useful, never more."""
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_tick(v, step):
    """Axis ticks: one consistent format across the axis, driven by the step.

    Mixing `0.00`, `50.0` and `100` down one axis reads as three different
    quantities. The step decides the decimals, so every tick matches.
    """
    dp = 0 if step >= 1 else (1 if step >= 0.1 else 2)
    return f"{v:,.{dp}f}"


# ------------------------------------------------------------------ chart ---


def bar_panels(rows_by_panel, title, subtitle, unit, th):
    """Small multiples: one horizontal bar panel per shape, each own x scale."""
    panels = []
    y = 0
    head = 72 if not subtitle else 88
    y = head
    body = []

    for label, rows, ref in rows_by_panel:
        n = len(rows)
        ph = n * (BAR_H + BAR_GAP) - BAR_GAP
        vmax = max([r["value"] for r in rows] + ([ref["value"]] if ref else []))
        ticks, top = nice_ticks(vmax)
        step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
        sx = (lambda v, top=top: LABEL_W + PLOT_W * (v / top if top else 0))

        body.append(f'<text x="0" y="{y - 10}" font-size="12.5" font-weight="600" '
                    f'fill="{th["ink2"]}">{esc(label)}</text>')

        # gridlines + x ticks (solid hairlines, one step off the surface)
        for t in ticks:
            x = sx(t)
            body.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + ph}" '
                        f'stroke="{th["grid"]}" stroke-width="1"/>')
            body.append(f'<text x="{x:.1f}" y="{y + ph + 15}" font-size="10.5" '
                        f'text-anchor="middle" fill="{th["muted"]}" '
                        f'style="font-variant-numeric:tabular-nums">{fmt_tick(t, step)}</text>')
        body.append(f'<line x1="{LABEL_W}" y1="{y}" x2="{LABEL_W}" y2="{y + ph}" '
                    f'stroke="{th["axis"]}" stroke-width="1"/>')

        best = max(rows, key=lambda r: r["value"])
        for i, r in enumerate(rows):
            by = y + i * (BAR_H + BAR_GAP)
            w = max(sx(r["value"]) - LABEL_W, 0.6)
            col = th["s2"] if r["new"] else th["s1"]
            rr = min(4.0, w)     # 4px rounded data-end, square at the baseline
            body.append(
                f'<path d="M{LABEL_W} {by} H{LABEL_W + w - rr:.1f} '
                f'a{rr:.1f} {rr:.1f} 0 0 1 {rr:.1f} {rr:.1f} '
                f'V{by + BAR_H - rr:.1f} a{rr:.1f} {rr:.1f} 0 0 1 -{rr:.1f} {rr:.1f} '
                f'H{LABEL_W} Z" fill="{col}"/>')
            body.append(f'<text x="{LABEL_W - 9}" y="{by + BAR_H / 2 + 4:.1f}" '
                        f'font-size="11" text-anchor="end" fill="{th["ink2"]}">'
                        f'{esc(r["name"])}</text>')
            # Label selectively: the baseline and the winner carry the story.
            if r is best or r["baseline"]:
                body.append(
                    f'<text x="{LABEL_W + w + 7:.1f}" y="{by + BAR_H / 2 + 4:.1f}" '
                    f'font-size="11" font-weight="{"600" if r is best else "400"}" '
                    f'fill="{th["ink"] if r is best else th["ink2"]}" '
                    f'style="font-variant-numeric:tabular-nums">{fmt(r["value"])}</text>')

        if ref:
            x = sx(ref["value"])
            body.append(f'<line x1="{x:.1f}" y1="{y - 5}" x2="{x:.1f}" y2="{y + ph + 3}" '
                        f'stroke="{th["ink2"]}" stroke-width="1.5" '
                        f'stroke-dasharray="5 3"/>')
            body.append(f'<text x="{x + 5:.1f}" y="{y - 9}" font-size="10.5" '
                        f'fill="{th["ink2"]}">{esc(ref["label"])}</text>')

        y += ph + PANEL_PAD + 20

    W = LABEL_W + PLOT_W + RIGHT_PAD
    H = y - PANEL_PAD + 34

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
           f'<rect width="{W}" height="{H}" fill="{th["surface"]}"/>',
           f'<text x="0" y="20" font-size="15" font-weight="600" '
           f'fill="{th["ink"]}">{esc(title)}</text>',
           f'<text x="0" y="38" font-size="11.5" fill="{th["ink2"]}">'
           f'{esc(subtitle)} ({esc(unit)})</text>']
    # Legend: two series, so a legend is always present.
    lx = 0
    for col, name in ((th["s1"], "ported from the CUDA original"),
                      (th["s2"], "added for CDNA")):
        out.append(f'<circle cx="{lx + 4}" cy="{head - 34}" r="4" fill="{col}"/>')
        out.append(f'<text x="{lx + 13}" y="{head - 30}" font-size="10.5" '
                   f'fill="{th["ink2"]}">{esc(name)}</text>')
        lx += 16 + 6.05 * len(name)
    out += body
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------- roofline ---


def roofline(points, th):
    """Log-log roofline: where each ladder's best rung sits against the roofs.

    ``points`` is a list of (name, arithmetic_intensity, achieved_gflops).
    """
    import math

    W, H = 700, 420
    L, R, T, B = 62, 150, 64, 46
    HBM = 8000.0          # GB/s  [CDNA4 whitepaper]
    VEC = 72_100.0        # GFLOP/s, FP32 vector at MI350X clocks
    MTX = 144_200.0       # GFLOP/s, FP32 matrix (MFMA)

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    x0, x1 = 10 ** math.floor(math.log10(min(xs) * 0.5)), 10 ** math.ceil(
        math.log10(max(xs) * 2))
    y0, y1 = 10 ** math.floor(math.log10(min(ys) * 0.5)), MTX * 1.6

    def px(v):
        return L + (W - L - R) * (math.log10(v) - math.log10(x0)) / (
            math.log10(x1) - math.log10(x0))

    def py(v):
        return H - B - (H - T - B) * (math.log10(v) - math.log10(y0)) / (
            math.log10(y1) - math.log10(y0))

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{th["surface"]}"/>',
         f'<text x="0" y="20" font-size="15" font-weight="600" fill="{th["ink"]}">'
         f'Roofline: where each ladder tops out</text>',
         f'<text x="0" y="38" font-size="11.5" fill="{th["ink2"]}">'
         f"best rung per ladder, MI350X. On a roof = at that resource's limit; "
         f"above the HBM roof = served from cache.</text>"]

    for d in range(int(math.log10(x0)), int(math.log10(x1)) + 1):
        x = px(10.0 ** d)
        o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" '
                 f'stroke="{th["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{H - B + 16}" font-size="10.5" '
                 f'text-anchor="middle" fill="{th["muted"]}">'
                 f'{("%g" % (10.0 ** d))}</text>')
    for d in range(int(math.log10(y0)), int(math.log10(y1)) + 1):
        yv = py(10.0 ** d)
        o.append(f'<line x1="{L}" y1="{yv:.1f}" x2="{W - R}" y2="{yv:.1f}" '
                 f'stroke="{th["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{L - 8}" y="{yv + 4:.1f}" font-size="10.5" '
                 f'text-anchor="end" fill="{th["muted"]}">{("%g" % (10.0 ** d))}</text>')

    # roofs
    for peak, name in ((MTX, "FP32 matrix core roof"), (VEC, "FP32 vector roof")):
        knee = peak / HBM
        pts = [(x0, HBM * x0), (knee, peak), (x1, peak)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(a):.1f} {py(b):.1f}"
                     for i, (a, b) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{th["ink2"]}" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
        o.append(f'<text x="{W - R + 6}" y="{py(peak) + 4:.1f}" font-size="10.5" '
                 f'fill="{th["ink2"]}">{esc(name)}</text>')
    # Park the roof label high on the diagonal, clear of the low-intensity
    # points that cluster near its foot.
    lx, ly = px(x0 * 60), py(HBM * x0 * 60) - 9
    o.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10.5" '
             f'fill="{th["ink2"]}" transform="rotate(-31 {lx:.1f} {ly:.1f})">'
             f'HBM 8 TB/s</text>')

    for name, ai, gf in points:
        cx, cy = px(ai), py(gf)
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" '
                 f'fill="{th["surface"]}"/>')          # 2px surface ring
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{th["s1"]}"/>')
        o.append(f'<text x="{cx + 10:.1f}" y="{cy + 4:.1f}" font-size="11" '
                 f'fill="{th["ink"]}">{esc(name)}</text>')

    o.append(f'<text x="{L + (W - L - R) / 2:.0f}" y="{H - 8}" font-size="11" '
             f'text-anchor="middle" fill="{th["ink2"]}">'
             f'arithmetic intensity (FLOP / byte of HBM traffic)</text>')
    o.append(f'<text x="14" y="{T + (H - T - B) / 2:.0f}" font-size="11" '
             f'text-anchor="middle" fill="{th["ink2"]}" '
             f'transform="rotate(-90 14 {T + (H - T - B) / 2:.0f})">'
             f'achieved GFLOP/s</text>')
    o.append("</svg>")
    return "\n".join(o)


# ------------------------------------------------------------------- main ---


def main():
    rows = json.loads(BENCH.read_text())["rows"]
    OUT.mkdir(exist_ok=True)
    ok = [r for r in rows if r.get("status") == "OK"]

    for op, shapes in PANELS.items():
        title, subtitle = TITLES[op]
        panels, unit = [], None
        for shape in shapes:
            sel = [r for r in ok if r["op"] == op and r["shape"] == shape]
            if not sel:
                continue
            unit = next(iter(sel[0]["metrics"]))
            bars = [dict(name=r["variant"], value=r["metrics"][unit],
                         # Any rung whose origin cites CDNA is an addition, not a port --
                         # including sgemm v5, whose origin names the chapter it
                         # replaces rather than saying "addition".
                         new="CDNA4" in r.get("origin", ""),
                         baseline=r["variant"].startswith("v0")) for r in sel]
            ref = None
            with_t = [r for r in sel if r.get("torch_seconds")]
            if with_t:
                # metrics() is monotone in time, so recover the vendor value by
                # scaling the rung's own metric by the time ratio.
                r = with_t[0]
                ref = dict(value=r["metrics"][unit] * r["seconds"] / r["torch_seconds"],
                           label="torch / rocBLAS")
            panels.append((shape, bars, ref))
        for tag, th in (("light", LIGHT), ("dark", DARK)):
            (OUT / f"{op}-{tag}.svg").write_text(
                bar_panels(panels, title, subtitle, unit, th))
        print(f"figure/{op}-{{light,dark}}.svg  ({len(panels)} panel(s))")

    # Roofline: arithmetic intensity = FLOP / byte, both already in metrics.
    pts = []
    for op, shape in (("elementwise", "N=32M"), ("reduce", "N=32M"),
                      ("sgemv", "M=16384,N=4096"), ("sgemm", "4096^3"),
                      ("spmv", "1M rows, 32 nnz/row"),
                      ("spmm", "4096x4096, 32nnz, n=1024")):
        sel = [r for r in ok if r["op"] == op and r["shape"] == shape]
        if not sel:
            continue
        best = max(sel, key=lambda r: next(iter(r["metrics"].values())))
        m = best["metrics"]
        gb = m.get("GB/s")
        gf = m.get("GFLOP/s") or (m.get("TFLOP/s", 0) * 1000)
        if not gb or not gf:
            # elementwise/reduce report bandwidth only: one add per element.
            p = best["params"]
            n = p.get("N", 0)
            gf = {"elementwise": n, "reduce": n}[op] / best["seconds"] / 1e9
            gb = m["GB/s"]
        pts.append((op, gf / gb, gf))
    for tag, th in (("light", LIGHT), ("dark", DARK)):
        (OUT / f"roofline-{tag}.svg").write_text(roofline(pts, th))
    print("figure/roofline-{light,dark}.svg")


if __name__ == "__main__":
    main()
