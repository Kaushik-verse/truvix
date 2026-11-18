"""Draw docs/results.svg from results/metrics.json in the README's palette."""
import json
from pathlib import Path

BG, INK, MUTED, BLUE, RISK, GRID = "#F7F6F2", "#2B2B2B", "#6B6860", "#1E4D8C", "#B4552D", "#DDD9CE"
NAMES = {"ela_lr": "ELA + logistic", "srm_lr": "SRM noise + logistic", "patchnet": "EfficientNet-B0 patch net"}
EDITS = [("paint", "painted cracks, stains", "per_family"), ("splice", "pasted region", "per_family"),
         ("diffusion", "SD inpainting", "per_family"), ("telea_unseen", "OpenCV inpaint (unseen)", "unseen_tools"),
         ("fullgen", "fully generated (unseen)", "unseen_tools")]
W, LW, X0 = 880, 210, 24
BAR_W = W - X0 - LW - 70


def bars(rows, title, y0):
    """rows: [(label, auc, colour)], AUC drawn on a 0..1 axis with a dashed chance line at 0.5."""
    out = [f'<text x="{X0}" y="{y0}" font-size="15" fill="{INK}">{title}</text>']
    top = y0 + 16
    for i, (label, v, col) in enumerate(rows):
        y = top + i * 30
        w = BAR_W * v
        out.append(f'<text x="{X0}" y="{y + 14}" font-size="12.5" fill="{MUTED}">{label}</text>')
        out.append(f'<rect x="{X0 + LW}" y="{y}" width="{BAR_W}" height="18" fill="{GRID}" opacity="0.45"/>')
        out.append(f'<rect x="{X0 + LW}" y="{y}" width="{w:.1f}" height="18" fill="{col}"/>')
        out.append(f'<text x="{X0 + LW + w + 6:.1f}" y="{y + 14}" font-size="12.5" fill="{INK}">{v:.2f}</text>')
    bottom = top + len(rows) * 30 - 8
    cx = X0 + LW + BAR_W * 0.5
    out.append(f'<line x1="{cx:.1f}" y1="{top - 4}" x2="{cx:.1f}" y2="{bottom + 4}" stroke="{INK}" stroke-width="1" stroke-dasharray="3 3" opacity="0.55"/>')
    out.append(f'<text x="{cx + 4:.1f}" y="{bottom + 18}" font-size="11" fill="{MUTED}">chance</text>')
    return out, bottom + 30


def main() -> None:
    m = json.load(open("results/metrics.json"))
    det = m["detectors"]
    svg = []
    rows = [(NAMES[k], det[k]["test"]["auc"], BLUE if k == "patchnet" else MUTED) for k in NAMES]
    part, y = bars(rows, "Held-out test ROC-AUC, all edit types vs honest photos", 36)
    svg += part
    pn = det["patchnet"]
    rows = [(label, pn[sect][key]["auc"], RISK if key in ("diffusion", "fullgen") else BLUE)
            for key, label, sect in EDITS if key in pn.get(sect, {})]
    part, y = bars(rows, "Patch net by edit type (terracotta = generative AI)", y + 18)
    svg += part
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {y}" font-family="Georgia, \'Times New Roman\', serif">',
            f'<rect width="{W}" height="{y}" fill="{BG}"/>']
    Path("docs/results.svg").write_text("\n".join(head + svg + ["</svg>"]))


if __name__ == "__main__":
    main()
