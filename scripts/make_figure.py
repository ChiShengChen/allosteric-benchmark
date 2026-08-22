#!/usr/bin/env python
"""Render the method-comparison figure (light + dark SVG) from the held-out results.

The figure's job is not "which method scores highest" — it is **which methods beat
the trivial controls**, so the controls are a distinct colour and the random control
is drawn as a reference line on the two rate panels. Rows are shared across three
panels and sorted by top-5 hit rate, the metric that separates the methods.

Usage: python3 scripts/make_figure.py [--results data/results_tierB.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8981",
                  grid="#e6e5e0", accent="#2a78d6", control="#eb6834", muted="#6f6e69"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8981",
                 grid="#33322f", accent="#3987e5", control="#d95926", muted="#93928a"),
}
CONTROLS = {"ctrl_dist", "ctrl_burial", "ctrl_random"}
ACCENT = {"ALPS"}
PRETTY = {"ctrl_dist": "control: distance", "ctrl_burial": "control: burial",
          "ctrl_random": "control: random", "qasc_baseline": "qasc (CTQW baseline)"}

W, LABEL_W, ROW_H, PANEL_GAP = 1000, 168, 20, 34
PANELS = [("top-5 hit rate", "hit5", 100.0, "%"),
          ("permutation significance", "sig", 100.0, "%"),
          ("ROC-AUC vs chance", "auc", 1.0, "")]


def load(path):
    R = json.load(open(path))
    rows = {}
    for m in sorted({k for r in R for k in r["rows"]}):
        v = [r["rows"][m] for r in R if m in r["rows"]]
        p = np.array([x["perm_p"] for x in v], float)
        rows[m] = dict(
            sig=float(np.nanmean(p < 0.05) * 100),
            auc=float(np.nanmean([x["auc"] for x in v])),
            hit5=float(np.nanmean([x["hit5"] for x in v]) * 100),
            n=len(v))
    return rows, len(R)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(rows, n_targets, mode):
    c = THEME[mode]
    order = sorted(rows, key=lambda m: -rows[m]["hit5"])
    top = 96
    H = top + len(order) * ROW_H + 54
    panel_w = (W - LABEL_W - 24 - PANEL_GAP * (len(PANELS) - 1)) / len(PANELS)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,BlinkMacSystemFont,'
         f'\'Segoe UI\',Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>']

    o.append(f'<text x="16" y="30" font-size="16" font-weight="600" fill="{c["ink"]}">'
             f'Allosteric site prediction from C&#946; coordinates + active site</text>')
    o.append(f'<text x="16" y="50" font-size="12" fill="{c["ink2"]}">'
             f'Held-out benchmark, {n_targets} family-declustered targets. '
             f'Sorted by top-5 hit rate. Dashed line = random control.</text>')

    # legend — identity is never colour-alone: each swatch is labelled
    lg = [(c["accent"], "ALPS (this work)"), (c["muted"], "other methods"),
          (c["control"], "trivial controls")]
    x = 16
    for col, lab in lg:
        o.append(f'<rect x="{x}" y="64" width="10" height="10" rx="2" fill="{col}"/>')
        o.append(f'<text x="{x+15}" y="73" font-size="11.5" fill="{c["ink2"]}">'
                 f'{esc(lab)}</text>')
        x += 22 + len(lab) * 6.4

    rnd = rows.get("ctrl_random", {})
    for pi, (title, key, scale, unit) in enumerate(PANELS):
        px = LABEL_W + pi * (panel_w + PANEL_GAP)
        o.append(f'<text x="{px}" y="{top-12}" font-size="12" font-weight="600" '
                 f'fill="{c["ink"]}">{esc(title)}</text>')
        y0, y1 = top - 4, top + len(order) * ROW_H

        if key == "auc":
            # AUC 0.5 is chance, so bars diverge from 0.5. Drawing them from zero
            # would make 0.39 look substantial and would hide the finding that
            # several methods rank true sites BELOW background.
            dev = max(abs(r["auc"] - 0.5) for r in rows.values()) * 1.12
            mid = px + panel_w / 2
            half = panel_w / 2
            for t in (-1, -0.5, 0.5, 1):
                gx = mid + t * half
                o.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" '
                         f'stroke="{c["grid"]}" stroke-width="1"/>')
            o.append(f'<line x1="{mid:.1f}" y1="{y0}" x2="{mid:.1f}" y2="{y1}" '
                     f'stroke="{c["ink3"]}" stroke-width="1.25"/>')
            o.append(f'<text x="{mid:.1f}" y="{y1+15}" font-size="10" '
                     f'text-anchor="middle" fill="{c["ink3"]}">0.5 = chance</text>')
            o.append(f'<text x="{mid-half:.1f}" y="{y1+15}" font-size="10" '
                     f'fill="{c["ink3"]}">worse</text>')
            o.append(f'<text x="{mid+half:.1f}" y="{y1+15}" font-size="10" '
                     f'text-anchor="end" fill="{c["ink3"]}">better</text>')
        else:
            vmax = scale
            for t in (0, .25, .5, .75, 1):
                gx = px + t * panel_w
                o.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" '
                         f'stroke="{c["grid"]}" stroke-width="1"/>')
            if rnd:
                gx = px + (rnd[key] / vmax) * panel_w
                o.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" '
                         f'stroke="{c["control"]}" stroke-width="1.5" '
                         f'stroke-dasharray="3 3" opacity="0.85"/>')

        for ri, m in enumerate(order):
            v = rows[m][key]
            y = top + ri * ROW_H + 4
            col = c["accent"] if m in ACCENT else (c["control"] if m in CONTROLS
                                                  else c["muted"])
            if key == "auc":
                d = (v - 0.5) / dev * (panel_w / 2)
                bx = mid + min(d, 0)
                bw = max(1.5, abs(d))
                tx = mid + d + (5 if d >= 0 else -5)
                anchor_ = "start" if d >= 0 else "end"
                txt = f'{v:.3f}'
            else:
                bw = max(1.5, (v / vmax) * panel_w)
                bx, tx, anchor_ = px, px + bw + 5, "start"
                txt = f'{v:.1f}{unit}'
            o.append(f'<rect x="{bx:.1f}" y="{y}" width="{bw:.1f}" height="11" '
                     f'rx="4" fill="{col}"/>')
            # halo drawn as a separate element beneath the text: paint-order is
            # not honoured by every SVG renderer and silently hides the label
            for stroke, fill in ((c["surface"], "none"), ("none", c["ink3"])):
                o.append(f'<text x="{tx:.1f}" y="{y+9}" font-size="10.5" '
                         f'text-anchor="{anchor_}" fill="{fill}" '
                         f'stroke="{stroke}" stroke-width="2.5" '
                         f'stroke-linejoin="round">{txt}</text>')

    for ri, m in enumerate(order):
        y = top + ri * ROW_H + 13
        lab = PRETTY.get(m, m)
        weight = "600" if m in ACCENT else "400"
        fill = c["ink"] if m in ACCENT or m in CONTROLS else c["ink2"]
        o.append(f'<text x="{LABEL_W-10}" y="{y}" font-size="11" text-anchor="end" '
                 f'font-weight="{weight}" fill="{fill}" '
                 f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace">'
                 f'{esc(lab)}</text>')

    o.append(f'<text x="16" y="{H-16}" font-size="10.5" fill="{c["ink3"]}">'
             f'Labels are proxy annotations; the benchmark carries a distance bias '
             f'(see README &#167;9). ALPS hyperparameters were tuned on a separate '
             f'11-target set.</text>')
    o.append("</svg>")
    return "\n".join(o)


def _fp_size_matched(cache, method, stat="top5", tol=0.25):
    """Protein-level AUC, positives vs negatives, matched on log size."""
    pos = [v for v in cache.values() if v["label"] == "pos" and method in v]
    neg = [v for v in cache.values() if v["label"] == "neg" and method in v]
    if len(pos) < 5 or len(neg) < 5:
        return None
    pv = np.array([v[method][stat] for v in pos], float)
    pn = np.log(np.array([v["n"] for v in pos], float))
    nv = np.array([v[method][stat] for v in neg], float)
    nn = np.log(np.array([v["n"] for v in neg], float))
    close = np.abs(pn[:, None] - nn[None, :]) <= tol
    if close.sum() < 50:
        return None
    d = pv[:, None] - nv[None, :]
    w = (d > 0).astype(float) + 0.5 * (d == 0)
    return float(w[close].sum() / close.sum())


def render_two_panel(resid, fp, floor, floor_sd, n_res, n_pos, n_neg, mode):
    """The two questions, side by side, on shared rows.

    Left: can a method rank the allosteric site highly *inside* a protein known to
    have one. Right: can it tell a protein that has a site from one that does not.
    ALPS wins the left panel and fails the right one exactly like everything else,
    and putting them on shared rows is the only honest way to show that.

    Bars run from the chance line, not from zero, so direction carries meaning.
    """
    c = THEME[mode]
    order = sorted(resid, key=lambda m: -resid[m])
    top, row_h = 150, 26
    W2, lab_w, pw, gap = 940, 196, 300, 46
    H = top + len(order) * row_h + 78

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W2}" height="{H}" '
         f'viewBox="0 0 {W2} {H}" font-family="-apple-system,BlinkMacSystemFont,'
         f'\'Segoe UI\',Helvetica,Arial,sans-serif">',
         f'<rect width="{W2}" height="{H}" fill="{c["surface"]}"/>',
         f'<text x="16" y="30" font-size="16" font-weight="600" fill="{c["ink"]}">'
         f'Two questions. One method answers the first; nothing answers the second.</text>',
         f'<text x="16" y="51" font-size="12" fill="{c["ink2"]}">'
         f'Curated allosteric annotations. Both panels remove the confound that '
         f'dominates the raw numbers &#8212; proximity on the left,</text>',
         f'<text x="16" y="67" font-size="12" fill="{c["ink2"]}">'
         f'structure size on the right &#8212; by comparing only matched pairs. '
         f'Bars are drawn from chance.</text>',
         f'<text x="16" y="105" font-size="11.5" fill="{c["control"]}">'
         f'Right panel caveat: ctrl_dist still reaches 0.752, so matching on residue '
         f'count does not remove every difference between the two sets.</text>']
    lg = [(c["accent"], "ALPS"), (c["muted"], "other methods"),
          (c["control"], "controls / quantum readouts")]
    x = 16
    for col, lab in lg:
        o.append(f'<rect x="{x}" y="82" width="10" height="10" rx="2" fill="{col}"/>')
        o.append(f'<text x="{x+15}" y="91" font-size="11.5" fill="{c["ink2"]}">{lab}</text>')
        x += 22 + len(lab) * 6.4

    QUANT = {"qfi", "ctqw_only", "qasc_baseline"}
    CTRL = {"ctrl_random", "ctrl_dist", "ctrl_burial", "ctrl_closeness"}
    y1 = top + len(order) * row_h
    panels = [(lab_w, resid, floor, "within a protein that has a site",
               f"residue ranking, {n_res} targets", (0.44, 0.62)),
              (lab_w + pw + gap, fp, 0.5, "does this protein have a site at all?",
               f"protein level, {n_pos} vs {n_neg}, size-matched", (0.35, 0.72))]

    for px, data, chance, title, sub, (lo, hi) in panels:
        o.append(f'<text x="{px}" y="{top-30}" font-size="12.5" font-weight="600" '
                 f'fill="{c["ink"]}">{esc(title)}</text>')
        o.append(f'<text x="{px}" y="{top-14}" font-size="11" fill="{c["ink3"]}">'
                 f'{esc(sub)}</text>')

        def X(v):
            return px + (min(max(v, lo), hi) - lo) / (hi - lo) * pw
        for t in np.linspace(lo, hi, 4):
            o.append(f'<line x1="{X(t):.1f}" y1="{top-6}" x2="{X(t):.1f}" y2="{y1}" '
                     f'stroke="{c["grid"]}" stroke-width="1"/>')
            o.append(f'<text x="{X(t):.1f}" y="{y1+16}" font-size="10" '
                     f'text-anchor="middle" fill="{c["ink3"]}">{t:.2f}</text>')
        if floor_sd and chance != 0.5:
            o.append(f'<rect x="{X(chance-floor_sd):.1f}" y="{top-6}" '
                     f'width="{X(chance+floor_sd)-X(chance-floor_sd):.1f}" '
                     f'height="{y1-top+6}" fill="{c["control"]}" opacity="0.13"/>')
        o.append(f'<line x1="{X(chance):.1f}" y1="{top-6}" x2="{X(chance):.1f}" '
                 f'y2="{y1}" stroke="{c["control"]}" stroke-width="1.5" '
                 f'stroke-dasharray="3 3"/>')
        lbl = (f"floor {chance:.3f} \u00b1 {floor_sd:.3f}" if chance != 0.5
               else "chance 0.50")
        o.append(f'<text x="{X(chance):.1f}" y="{y1+32}" font-size="10.5" '
                 f'text-anchor="middle" fill="{c["control"]}">{lbl}</text>')

        for ri, m in enumerate(order):
            v = data.get(m)
            y = top + ri * row_h + 6
            if v is None:
                o.append(f'<text x="{X(chance):.1f}" y="{y+10}" font-size="10.5" '
                         f'text-anchor="middle" fill="{c["ink3"]}">not run</text>')
                continue
            col = (c["accent"] if m.startswith("ALPS")
                   else c["control"] if (m in CTRL or m in QUANT) else c["muted"])
            x0, x1 = X(min(v, chance)), X(max(v, chance))
            o.append(f'<rect x="{x0:.1f}" y="{y}" width="{max(1.5, x1-x0):.1f}" '
                     f'height="12" rx="4" fill="{col}"/>')
            tx = X(v) + (6 if v >= chance else -6)
            anc = "start" if v >= chance else "end"
            for stroke, fill in ((c["surface"], "none"), ("none", c["ink3"])):
                o.append(f'<text x="{tx:.1f}" y="{y+10}" font-size="10.5" '
                         f'text-anchor="{anc}" fill="{fill}" stroke="{stroke}" '
                         f'stroke-width="2.5" stroke-linejoin="round">{v:.3f}</text>')

    for ri, m in enumerate(order):
        y = top + ri * row_h + 16
        note = " (quantum)" if m in QUANT else (" (control)" if m in CTRL else "")
        o.append(f'<text x="{lab_w-12}" y="{y}" font-size="11.5" text-anchor="end" '
                 f'fill="{c["ink"] if m.startswith("ALPS") else c["ink2"]}" '
                 f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace">'
                 f'{esc(m)}{note}</text>')

    o.append(f'<text x="16" y="{H-14}" font-size="10.5" fill="{c["ink3"]}">'
             f'Left: only ALPS clears the floor (p = 0.0030, survives Bonferroni). '
             f'Right: no method beats 0.57, and a distance-only control reaches 0.752 '
             f'&#8212; the negative set still differs geometrically.</text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "data", "results_tierB.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "docs"))
    a = ap.parse_args()
    rows, n = load(a.results)
    os.makedirs(a.out, exist_ok=True)
    for mode in ("light", "dark"):
        p = os.path.join(a.out, f"methods-{mode}.svg")
        open(p, "w").write(render(rows, n, mode))
        print(f"wrote {p}  ({len(rows)} methods, {n} targets)")

    ep = os.path.join(HERE, "data", "results_expanded.json")
    fpp = os.path.join(HERE, "data", "results_false_positive.json")
    if os.path.exists(ep) and os.path.exists(fpp):
        E = {k: v for k, v in json.load(open(ep)).items() if k != "_params"}
        resid = {}
        for m in sorted({k for v in E.values() for k in v if k != "n"}):
            vals = [v[m] for v in E.values() if v.get(m) is not None]
            if vals:
                resid[m] = float(np.mean(vals))
        cache = json.load(open(fpp))
        alias = {"ALPS": "ALPS_raw", "ALPS_noresid": "ALPS_raw"}
        # ctrl_random anchors the right panel; without it there is no reference
        fp = {}
        for m in resid:
            got = _fp_size_matched(cache, alias.get(m, m))
            if got is not None:
                fp[m] = got
        npos = sum(1 for v in cache.values() if v["label"] == "pos")
        nneg = sum(1 for v in cache.values() if v["label"] == "neg")
        resid.pop("ALPS_noresid", None)          # identical to ALPS, one row is enough
        for mode in ("light", "dark"):
            p = os.path.join(a.out, f"stratified-{mode}.svg")
            open(p, "w").write(render_two_panel(resid, fp, 0.4963, 0.0157,
                                                len(E), npos, nneg, mode))
            print(f"wrote {p}  ({len(resid)} methods, {len(E)} targets)")


if __name__ == "__main__":
    main()
