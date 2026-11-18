"""Draw docs/results.svg from results/metrics.json in the README's palette."""
import json
from pathlib import Path

BG, INK, MUTED, BLUE, RISK, GRID = "#F7F6F2", "#2B2B2B", "#6B6860", "#1E4D8C", "#B4552D", "#DDD9CE"
NAMES = {"ela_lr": "ELA + logistic", "srm_lr": "SRM noise + logistic", "patchnet": "ResNet-18 patch net"}


def bars(rows, title, x0, y0, width):
    """rows: [(label, value, colour)] value in 0..1, drawn as horizontal bars."""
    out = [f'<text x="{x0}" y="{y0}" font-size="15" fill="{INK}">{title}</text>']
    lw = 190
    for i, (label, v, col) in enumerate(rows):
        y = y0 + 22 + i * 30
        w = (width - lw - 60) * v
        out.append(f'<text x="{x0}" y="{y + 14}" font-size="12.5" fill="{MUTED}">{label}</text>')
        out.append(f'<rect x="{x0 + lw}" y="{y}" width="{width - lw - 60}" height="18" fill="{GRID}" opacity="0.45"/>')
        out.append(f'<rect x="{x0 + lw}" y="{y}" width="{w:.1f}" height="18" fill="{col}"/>')
        out.append(f'<text x="{x0 + lw + w + 6:.1f}" y="{y + 14}" font-size="12.5" fill="{INK}">{v:.2f}</text>')
    return out, y0 + 22 + len(rows) * 30


def main() -> None:
    m = json.load(open("results/metrics.json"))
    det = m["detectors"]
    W = 880
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} H" font-family="Georgia, \'Times New Roman\', serif">',
           f'<rect width="{W}" height="H" fill="{BG}"/>']
    y = 36
    rows = [(f"{NAMES[k]}", det[k]["test"]["auc"], BLUE if k == "patchnet" else MUTED) for k in NAMES]
    part, y = bars(rows, "Held-out test ROC-AUC (paint + splice edits vs honest photos)", 24, y, W - 24)
    svg += part
    rows = []
    for k in NAMES:
        for f, s in det[k].get("unseen_tools", {}).items():
            rows.append((f"{NAMES[k]}", s["auc"], RISK))
    if rows:
        part, y = bars(rows, "Unseen editing tool (OpenCV inpainting, never in training): ROC-AUC", 24, y + 26, W - 24)
        svg += part
    svg.append("</svg>")
    s = "\n".join(svg).replace('"0 0 880 H"', f'"0 0 {W} {y + 20}"').replace('height="H"', f'height="{y + 20}"')
    Path("docs/results.svg").write_text(s)


if __name__ == "__main__":
    main()
