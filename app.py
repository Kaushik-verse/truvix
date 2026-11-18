"""Truvix in the browser: check photos, browse the results, run the tests.

    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from truvix.cli import load_model, overlay
from truvix.patchnet import heatmap

ROOT = Path(__file__).resolve().parent
EX = ROOT / "docs" / "examples"
LOGO = (ROOT / "docs" / "logo.svg").read_text().replace('width="64" height="64"', 'width="100%" height="100%"')
BLUE, BLUE_SOFT, RISK, RISK_SOFT, MUTED, INK, PAPER = "#1E4D8C", "#DCE5F2", "#B4552D", "#F3E1D8", "#6B6860", "#2B2B2B", "#F7F6F2"
KIND = {"authentic": "Honest photo", "paint": "Painted crack or stain", "splice": "Pasted region",
        "diffusion": "AI inpainting", "telea_unseen": "OpenCV inpaint (unseen)", "fullgen": "Fully AI-generated (unseen)"}
GALLERY = ["splice_1", "paint_3", "fullgen_2", "authentic_1", "diffusion_2", "telea_unseen_1"]
SHORT = {"authentic": "Honest photo", "paint": "Painted stain", "splice": "Pasted region", "diffusion": "AI inpainting",
         "telea_unseen": "OpenCV fill", "fullgen": "AI-made room"}
TAG = {"authentic": "honest", "paint": "edited", "splice": "edited", "diffusion": "AI edit",
       "telea_unseen": "unseen tool", "fullgen": "AI-made"}
MAX_SIDE = 1024

st.set_page_config(page_title="Truvix · doctored photo detector", page_icon=str(ROOT / "docs" / "logo.svg"),
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown(f"""
<style>
  #MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] {{ display: none !important; }}
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: {PAPER} !important; color: {INK}; }}
  .stApp p, .stApp li, .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"] {{ color: {INK}; }}
  .stButton > button, .stDownloadButton > button, [data-testid="stBaseLinkButton-secondary"] {{
     background: #FFFFFF !important; color: {INK} !important; }}
  .stButton > button p, .stDownloadButton > button p {{ color: inherit !important; }}
  .stButton > button[kind="primary"] {{ background: {BLUE} !important; color: #FFFFFF !important; border-color: {BLUE} !important; }}
  .stButton > button[kind="primary"] * {{ color: #FFFFFF !important; }}
  [data-testid="stExpander"] details {{ background: #FFFFFF; border-radius: 12px; border-color: #E4E0D6; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}
  html, body, [class*="css"], .stMarkdown, .stText, p, li, label {{
    font-family: "Avenir Next", "Helvetica Neue", "Segoe UI", sans-serif; }}
  h1, h2, h3, h4 {{ font-family: Georgia, "Times New Roman", serif !important; font-weight: 400 !important; color: {BLUE}; letter-spacing: -0.01em; }}
  [data-testid="stSidebar"] {{ background: #EFECE4; border-right: 1px solid #E2DED3; }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

  .tv-hero {{ display: flex; align-items: center; gap: 22px; padding: 26px 30px; border-radius: 18px;
     background: #FFFFFF; border: 1px solid #E4E0D6; box-shadow: 0 1px 2px rgba(43,43,43,.04), 0 8px 24px rgba(43,43,43,.05); margin-bottom: 18px; }}
  .tv-chip {{ display: inline-block; font-size: .78rem; padding: 3px 11px; margin: 0 6px 4px 0; border-radius: 999px;
     background: {PAPER}; border: 1px solid #E4E0D6; color: {MUTED}; }}
  .tv-chip b {{ color: {BLUE}; font-weight: 600; }}

  .tv-card {{ background: #FFFFFF; border: 1px solid #E4E0D6; border-radius: 16px; padding: 18px 20px;
     box-shadow: 0 1px 2px rgba(43,43,43,.04), 0 6px 18px rgba(43,43,43,.04); height: 100%; }}
  .tv-kpi {{ font-family: Georgia, serif; font-size: 2.2rem; color: {BLUE}; line-height: 1.05; }}
  .tv-kpi.risk {{ color: {RISK}; }}
  .tv-kpi-label {{ color: {MUTED}; font-size: .84rem; margin-top: 6px; line-height: 1.35; }}
  .tv-eyebrow {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: {RISK}; margin-bottom: 2px;
     white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .tv-muted {{ color: {MUTED}; font-size: .92rem; }}
  .tv-verdict {{ border-radius: 12px; padding: 12px 16px; font-family: Georgia, serif; font-size: 1.2rem; margin-top: 6px; }}
  .tv-verdict small {{ display: block; font-family: "Avenir Next", sans-serif; font-size: .84rem; margin-top: 3px; opacity: .85; }}
  .tv-truth {{ border-left: 3px solid {BLUE}; padding: 6px 12px; background: {PAPER}; border-radius: 0 8px 8px 0; font-size: .92rem; margin-top: 10px; }}

  /* only the bordered containers that start with an eyebrow label become white cards */
  [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child .tv-eyebrow) {{
     background: #FFFFFF; border-radius: 16px !important; border: 1px solid #E4E0D6 !important;
     box-shadow: 0 1px 2px rgba(43,43,43,.04), 0 6px 18px rgba(43,43,43,.04); }}
  [data-testid="stImage"] img {{ border-radius: 10px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #E2DED3; }}
  .stTabs [data-baseweb="tab"] {{ font-family: Georgia, serif; font-size: 1.02rem; padding: 10px 16px; border-radius: 10px 10px 0 0; }}
  .stTabs [aria-selected="true"] {{ background: #FFFFFF; color: {BLUE} !important; }}
  .stButton > button, .stDownloadButton > button, .stLinkButton > a {{ border-radius: 10px !important; border: 1px solid #D6D2C6;
     font-size: .85rem; white-space: nowrap; min-height: 2.3rem; }}
  .tv-gtitle {{ font-family: Georgia, serif; font-size: 1rem; color: {INK}; margin: 2px 0 6px; white-space: nowrap;
     overflow: hidden; text-overflow: ellipsis; }}
  .stButton > button[kind="primary"] {{ background: {BLUE}; border-color: {BLUE}; }}
  [data-testid="stFileUploaderDropzone"] {{ background: #FFFFFF; border: 1.5px dashed #C9C6BC; border-radius: 14px; }}
  [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading the detector…")
def get_model():
    return load_model()


@st.cache_data
def get_metrics() -> dict:
    return json.loads((ROOT / "results" / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def analyse(image_bytes: bytes):
    """(rgb image, heatmap, score). Large photos are shrunk the same way the CLI does."""
    img = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    if max(img.shape[:2]) > MAX_SIDE:
        s = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    model, _ = get_model()
    hm, score = heatmap(model, img)
    return img, hm, float(score)


def all_examples() -> dict[str, Path]:
    out = {}
    for p in sorted(EX.glob("*_[0-9].png")):
        if not any(s in p.stem for s in ("_mask", "_original", "_heatmap")):
            out[p.stem] = p
    return out


def label(stem: str) -> str:
    kind, n = stem.rsplit("_", 1)
    return f"{KIND.get(kind, kind)} #{n}"


def gauge(score: float, thr: float) -> str:
    """Semicircle gauge (SVG) with the threshold marked."""
    import math
    flagged = score > thr
    col = RISK if flagged else BLUE
    r, cx, cy = 90, 110, 110

    def pt(v):
        a = math.pi * (1 - v)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    x0, y0 = pt(0)
    x1, y1 = pt(max(min(score, 0.999), 0.001))
    tx, ty = pt(thr)
    tx2, ty2 = cx + (r + 12) * math.cos(math.pi * (1 - thr)), cy - (r + 12) * math.sin(math.pi * (1 - thr))
    return f"""
<svg viewBox="0 0 220 135" width="100%" style="max-width:300px; display:block; margin:auto;">
  <path d="M{x0},{y0} A{r},{r} 0 0 1 {cx + r},{cy}" fill="none" stroke="#ECE9E1" stroke-width="16" stroke-linecap="round"/>
  <path d="M{x0},{y0} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}" fill="none" stroke="{col}" stroke-width="16" stroke-linecap="round"/>
  <line x1="{tx:.1f}" y1="{ty:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="{INK}" stroke-width="2"/>
  <text x="{cx}" y="{cy - 14}" text-anchor="middle" font-family="Georgia" font-size="38" fill="{col}">{score:.2f}</text>
  <text x="{cx}" y="{cy + 8}" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="10.5" fill="{MUTED}">edit score · tick = threshold</text>
  <text x="{x0}" y="{cy + 22}" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="10" fill="{MUTED}">0</text>
  <text x="{cx + r}" y="{cy + 22}" text-anchor="middle" font-family="Avenir Next, sans-serif" font-size="10" fill="{MUTED}">1</text>
</svg>"""


def verdict(score: float, thr: float) -> str:
    if score > thr:
        return (f'<div class="tv-verdict" style="background:{RISK_SOFT}; color:{RISK}">Send to a human reviewer'
                f'<small style="color:{INK}">The most suspicious window scored above the threshold ({thr:.3f}).</small></div>')
    return (f'<div class="tv-verdict" style="background:{BLUE_SOFT}; color:{BLUE}">No edit found'
            f'<small style="color:{INK}">Every window scored below the threshold ({thr:.3f}).</small></div>')


def kpi(value: str, text: str, risk: bool = False) -> str:
    return f'<div class="tv-card"><div class="tv-kpi{" risk" if risk else ""}">{value}</div><div class="tv-kpi-label">{text}</div></div>'


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def auc_chart(df: pd.DataFrame, y: str, height: int = 240):
    base = alt.Chart(df).encode(
        x=alt.X("AUC:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(grid=False, tickCount=5, title="AUC")),
        y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        tooltip=list(df.columns))
    bars = base.mark_bar(size=18, cornerRadiusEnd=4).encode(
        color=alt.Color("group:N", scale=alt.Scale(domain=["classic edit", "AI-made", "baseline"],
                                                   range=[BLUE, RISK, MUTED]), legend=None))
    text = base.mark_text(align="left", dx=5, color=INK, font="Georgia").encode(text=alt.Text("AUC:Q", format=".2f"))
    chance = alt.Chart(pd.DataFrame({"v": [0.5]})).mark_rule(strokeDash=[4, 4], color=INK, opacity=0.45).encode(x="v:Q")
    return (bars + text + chance).properties(height=height).configure_view(strokeWidth=0).configure_axis(
        labelFont="Avenir Next", titleFont="Avenir Next", labelColor=INK, titleColor=MUTED, domainColor="#D6D2C6")


# ------------------------------------------------------------------ shared sidebar
model, THRESHOLD = get_model()
if "thr" not in st.session_state:
    st.session_state.thr = float(round(THRESHOLD, 3))
try:
    st.logo(str(ROOT / "docs" / "logo.svg"))
except Exception:
    pass


def sidebar_footer():
    with st.sidebar:
        st.markdown(f'<div style="font-family:Georgia;font-size:1.35rem;color:{BLUE};margin-top:4px">Truvix</div>'
                    f'<div class="tv-muted">Doctored-photo detector for rental damage claims.</div>', unsafe_allow_html=True)
        st.write("")
        guide = ROOT / "docs" / "Truvix_Guide.pdf"
        if guide.exists():
            st.download_button("Illustrated guide (PDF)", guide.read_bytes(), "Truvix_Guide.pdf", "application/pdf",
                               use_container_width=True)
        st.link_button("GitHub repository", "https://github.com/Kaushik-verse/truvix", use_container_width=True)
        st.caption("A score is a reason to look closer, never a verdict on its own. Example photos: MIT Indoor-67, "
                   "research use only.")


# ------------------------------------------------------------------ page 1: for everyone
def home():
    thr = st.session_state.thr
    # ------------------------------------------------------------------ hero
    m_all = get_metrics()
    fg = m_all["detectors"]["patchnet"]["unseen_tools"]["fullgen"]["auc"]
    st.markdown(f"""
    <div class="tv-hero">
      <div style="width:72px;height:72px;flex:none">{LOGO}</div>
      <div>
        <div style="font-family:Georgia,serif;font-size:2.7rem;line-height:1;color:{BLUE}">Truvix</div>
        <div style="margin:8px 0 12px;color:{INK};font-size:1.02rem;max-width:760px">Was the damage in this claim photo
          really there? Truvix scores the photo for local edits and shades exactly where it thinks the edit is, so a
          reviewer looks at the right spot before any money moves.</div>
        <span class="tv-chip"><b>3,716</b> photos built for training</span>
        <span class="tv-chip"><b>{fg:.2f}</b> AUC on AI-generated rooms it never saw</span>
        <span class="tv-chip"><b>EfficientNet-B0</b> patch detector</span>
        <span class="tv-chip">runs on a laptop</span>
      </div>
    </div>""", unsafe_allow_html=True)

    t_try, t_how, t_results = st.tabs(["Try it", "How it works", "Results"])

    # ---------------------------------------------------------- try it
    with t_try:
        ex = all_examples()
        if "pick" not in st.session_state:
            st.session_state.pick = "splice_1"
        src = st.radio("Source", ["Example gallery", "Upload my own photo"], horizontal=True, label_visibility="collapsed")
        data, stem = None, None
        if src == "Example gallery":
            st.markdown('<div class="tv-muted" style="margin-bottom:8px">Test photos the model never saw during training. '
                        'Pick one to analyse.</div>', unsafe_allow_html=True)
            cols = st.columns(len(GALLERY), gap="small")
            for c, g in zip(cols, GALLERY):
                kind = g.rsplit("_", 1)[0]
                chosen = st.session_state.pick == g
                with c:
                    with st.container(border=True):
                        st.markdown(f'<div class="tv-eyebrow" style="color:{RISK if kind != "authentic" else BLUE}">'
                                    f'{TAG[kind]}</div>', unsafe_allow_html=True)
                        st.image(str(ex[g]), use_container_width=True)
                        st.markdown(f'<div class="tv-gtitle">{SHORT[kind]}</div>', unsafe_allow_html=True)
                        if st.button("Selected" if chosen else "Analyse", key=f"g_{g}", use_container_width=True,
                                     type="primary" if chosen else "secondary"):
                            st.session_state.pick = g
                            st.rerun()
            with st.expander("More examples (18 in total)"):
                others = list(ex)
                choice = st.selectbox("Example", others, index=others.index(st.session_state.pick), format_func=label,
                                      label_visibility="collapsed")
                if choice != st.session_state.pick:
                    st.session_state.pick = choice
                    st.rerun()
            stem = st.session_state.pick
            data = ex[stem].read_bytes()
        else:
            up = st.file_uploader("Drop a room photo (JPG, PNG or WEBP)", type=["jpg", "jpeg", "png", "webp"])
            if up is not None:
                data = up.getvalue()
            else:
                st.info("Tip: take a photo of a room, paste a patch onto it in Preview or Photos, and compare the "
                        "original with the edited version.")

        if data is not None:
            with st.spinner("Scanning the photo window by window…"):
                img, hm, score = analyse(data)
            st.write("")
            left, mid, right = st.columns([1.15, 1.15, 0.9], gap="medium")
            with left:
                with st.container(border=True):
                    st.markdown('<div class="tv-eyebrow">The photo</div>', unsafe_allow_html=True)
                    st.image(img, use_container_width=True)
            with mid:
                with st.container(border=True):
                    st.markdown('<div class="tv-eyebrow">Where Truvix looks</div>', unsafe_allow_html=True)
                    st.image(overlay(img, hm), use_container_width=True)
            with right:
                with st.container(border=True):
                    st.markdown('<div class="tv-eyebrow">Verdict</div>', unsafe_allow_html=True)
                    st.markdown(gauge(score, thr), unsafe_allow_html=True)
                    st.markdown(verdict(score, thr), unsafe_allow_html=True)
                    st.markdown(f'<div class="tv-muted" style="margin-top:10px">Area marked: <b>{(hm > 0.5).mean():.1%}</b> '
                                f'of the photo</div>', unsafe_allow_html=True)
                    if stem:
                        kind = stem.rsplit("_", 1)[0]
                        right_call = (score > thr) == (kind != "authentic")
                        st.markdown(f'<div class="tv-truth"><b>Ground truth:</b> {KIND[kind]}.<br>'
                                    f'{"✓ Truvix got this one right." if right_call else "✗ Truvix got this one wrong."}</div>',
                                    unsafe_allow_html=True)
                    st.download_button("Download heatmap", png_bytes(overlay(img, hm)), "truvix_heatmap.png", "image/png",
                                       use_container_width=True)
            if stem and ((EX / f"{stem}_original.png").exists() or (EX / f"{stem}_mask.png").exists()):
                with st.expander("Compare with the honest original and the real edit"):
                    a, b = st.columns(2)
                    if (EX / f"{stem}_original.png").exists():
                        a.image(str(EX / f"{stem}_original.png"), caption="Honest original", use_container_width=True)
                    if (EX / f"{stem}_mask.png").exists():
                        b.image(str(EX / f"{stem}_mask.png"), caption="Pixels that were really edited", use_container_width=True)
            with st.expander("How to read this"):
                st.markdown(f"""
    - **Edit score** is the highest score of any 128-pixel window, from 0 to 1. One odd-looking patch can raise it,
      which is why honest photos are sometimes flagged.
    - **Threshold {thr:.3f}**: above it, the photo goes to a person. At the default, about half of the edited test
      photos are caught and about 8.5% of honest ones are flagged.
    - Truvix is strongest on pasted regions and fully AI-generated rooms, and weakest on thin painted cracks and small
      AI edits. It was trained on indoor room photos and works best on them.
    """)


    # ---------------------------------------------------------- how it works
    with t_how:
        st.markdown("### Why Truvix exists")
        st.markdown("""
    In 2025 and 2026, holiday-rental guests were billed for "damage" shown in photos that had been doctored, in one
    case apparently generated by AI. The people deciding had a few minutes per claim and nothing but their eyes.
    Truvix gives them a score and a map of where to look.
    """)
        st.image(str(ROOT / "docs" / "pipeline.svg"), use_container_width=True)
        a, b = st.columns([1.1, 1], gap="large")
        with a:
            with st.container(border=True):
                st.image(str(ROOT / "docs" / "dataset_examples.jpg"), use_container_width=True,
                         caption="Honest original · edited copy · pixels that changed")
        with b:
            st.markdown("""
    #### The dataset: 3,716 photos
    - 1,653 real room photos (MIT Indoor-67), each used twice: honest, and with fake damage.
    - Three kinds of fake: painted cracks and stains, pasted regions, Stable Diffusion inpainting.
    - Two more appear only at test time: OpenCV inpainting and fully AI-generated rooms.
    - Same "delivery" for every photo: shrink, JPEG, sometimes a second chat-app JPEG.
    - Split by room, so nothing seen in training appears in the test.

    #### The detector
    - EfficientNet-B0, pre-trained on ImageNet, fine-tuned on 128-pixel windows.
    - Every training photo also gets a laundered copy (re-saved, shrunk or screenshotted).
    - Chosen from three backbones on validation data only; the test set was used once, at the end.
    """)
        st.info("The sidebar has a 14-page illustrated guide with key terms, results, limits and a showcase script.")


    # ---------------------------------------------------------- results
    with t_results:
        m = m_all
        pn = m["detectors"]["patchnet"]
        t = pn["test"]
        st.markdown("### Results on 660 photos the model never saw")
        a, b, c, d = st.columns(4)
        a.markdown(kpi(f"{t['auc']:.2f}", f"overall AUC<br>95% interval {t['auc_95ci'][0]:.2f}–{t['auc_95ci'][1]:.2f}"),
                   unsafe_allow_html=True)
        b.markdown(kpi(f"{t['tpr']:.0%}", "of edited photos caught<br>at the default threshold"), unsafe_allow_html=True)
        c.markdown(kpi(f"{t['fpr']:.1%}", "of honest photos flagged<br>(target was 5%)", risk=True), unsafe_allow_html=True)
        d.markdown(kpi(f"{m['detectors']['ela_lr']['test']['auc']:.2f}", "classic ELA forensics<br>(0.50 = coin flip)"),
                   unsafe_allow_html=True)
        st.caption("AUC: pick one edited and one honest photo at random; AUC is the chance the edited one scores higher.")

        names = {"paint": "Painted cracks, stains", "splice": "Pasted region", "diffusion": "AI inpainting",
                 "telea_unseen": "OpenCV inpaint (unseen)", "fullgen": "Fully AI-generated (unseen)"}
        rows = [{"edit type": names[k], "AUC": v["auc"], "caught": f"{v['tpr']:.0%}",
                 "group": "AI-made" if k in ("diffusion", "fullgen") else "classic edit"}
                for sect in ("per_family", "unseen_tools") for k, v in pn[sect].items()]
        with st.container(border=True):
            st.markdown('<div class="tv-eyebrow">By kind of edit</div>', unsafe_allow_html=True)
            st.altair_chart(auc_chart(pd.DataFrame(rows), "edit type"), use_container_width=True)
            st.caption("Dashed line = coin flip. Terracotta = AI-made fakes.")

        l, r = st.columns(2, gap="medium")
        with l:
            with st.container(border=True):
                st.markdown('<div class="tv-eyebrow">Truvix vs classic forensics</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{"detector": n, "AUC": f"{m['detectors'][k]['test']['auc']:.2f}",
                                            "caught": f"{m['detectors'][k]['test']['tpr']:.1%}",
                                            "honest flagged": f"{m['detectors'][k]['test']['fpr']:.1%}"}
                                           for k, n in [("patchnet", "Truvix"), ("srm_lr", "SRM noise"),
                                                        ("ela_lr", "ELA")]]), hide_index=True, use_container_width=True)
            with st.container(border=True):
                st.markdown('<div class="tv-eyebrow">Stress test: one more trip through an app</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{"extra step": {"jpeg50": "re-saved at JPEG 50", "half": "shrunk to half size",
                                                           "screenshot": "screenshot"}[k], "AUC": f"{v['auc']:.2f}",
                                            "caught": f"{v['tpr']:.1%}", "honest flagged": f"{v['fpr']:.1%}"}
                                           for k, v in pn["robustness"].items()]), hide_index=True, use_container_width=True)
        with r:
            with st.container(border=True):
                st.markdown('<div class="tv-eyebrow">Can it catch a tool it never saw?</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{"left out of training": names[k], "AUC on that type": f"{v['patchnet']['auc']:.2f}"}
                                           for k, v in m["lofo"].items()]), hide_index=True, use_container_width=True)
                st.caption("Near 0.5 means no: it learns the tools it trained on, not editing in general.")
            with st.container(border=True):
                st.markdown('<div class="tv-eyebrow">Versions</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    "": ["overall AUC", "AI inpainting", "fully generated", "screenshot false flags"],
                    "v1": ["0.82*", "–", "–", "12.8%"], "v2": ["0.73", "0.59", "0.81", "29.3%"],
                    "v3 now": ["0.76", "0.68", "0.91", "14.8%"]}), hide_index=True, use_container_width=True)
                st.caption("(*) v1 had no AI edits, an easier test.")
        with st.container(border=True):
            st.markdown('<div class="tv-eyebrow">Twelve random test photos, mistakes included</div>', unsafe_allow_html=True)
            st.image(str(ROOT / "docs" / "heatmaps_random.jpg"), use_container_width=True)



# ------------------------------------------------------------------ page 2: developer tools
def developer():
    st.markdown("## Developer tools")
    st.markdown('<div class="tv-muted">For people working on Truvix: change the review threshold, score many photos '
                'at once, and run the automated tests.</div>', unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        st.markdown('<div class="tv-eyebrow">Review threshold</div>', unsafe_allow_html=True)
        st.session_state.thr = st.slider(
            "Photos scoring above this go to a person", 0.50, 0.999, st.session_state.thr, 0.001,
            help="The default was set on validation data so that about 5% of honest photos get flagged. Lower it "
                 "to catch more fakes, at the price of more false alarms. Applies to the Try it page too.")
        if abs(st.session_state.thr - THRESHOLD) > 1e-6:
            st.caption(f"Default: {THRESHOLD:.3f}. Published results always use the default.")
            if st.button("Reset to default"):
                st.session_state.thr = float(round(THRESHOLD, 3))
                st.rerun()
    thr = st.session_state.thr
    d_batch, d_tests = st.tabs(["Check many photos", "Run the tests"])
    with d_batch:
        st.markdown("### Check a batch of photos")
        files = st.file_uploader("Drop several room photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                                 key="batch")
        use_examples = st.toggle("Use the 18 example test photos", value=not files)
        items = ([(label(p.stem), p.read_bytes()) for p in all_examples().values()] if use_examples
                 else [(f.name, f.getvalue()) for f in files] if files else [])
        if items:
            rows, tiles = [], []
            bar = st.progress(0.0, text="Scanning…")
            for i, (name, data) in enumerate(items):
                img, hm, score = analyse(data)
                rows.append({"photo": name, "score": round(score, 3), "verdict": "review" if score > thr else "ok",
                             "area marked": round(float((hm > 0.5).mean()), 3)})
                tiles.append((name, overlay(img, hm), score))
                bar.progress((i + 1) / len(items), text=f"Scanned {i + 1} of {len(items)}")
            bar.empty()
            df = pd.DataFrame(rows).sort_values("score", ascending=False)
            n_flag = int((df["verdict"] == "review").sum())
            k1, k2, k3 = st.columns(3)
            k1.markdown(kpi(str(len(df)), "photos scanned"), unsafe_allow_html=True)
            k2.markdown(kpi(str(n_flag), f"sent to review at threshold {thr:.3f}", risk=n_flag > 0), unsafe_allow_html=True)
            k3.markdown(kpi(f"{df['score'].median():.2f}", "median edit score"), unsafe_allow_html=True)
            st.write("")
            df["area marked"] = (df["area marked"] * 100).round(1)
            st.dataframe(df, use_container_width=True, hide_index=True, column_config={
                "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.3f"),
                "area marked": st.column_config.NumberColumn("area marked", format="%.1f%%")})
            st.download_button("Download table (CSV)", df.to_csv(index=False).encode(), "truvix_scores.csv", "text/csv")
            st.write("")
            cols = st.columns(4)
            for i, (name, ov, score) in enumerate(sorted(tiles, key=lambda t: -t[2])):
                cols[i % 4].image(ov, caption=f"{name} · {score:.2f} · {'REVIEW' if score > thr else 'ok'}",
                                  use_container_width=True)

    with d_tests:
        st.markdown("### Run the automated tests")
        st.markdown('<div class="tv-muted">Thirteen checks: the editing code, the delivery pipeline, the forensic features, '
                    'the patch grid, the dataset split, the trained model on the shipped examples, and this app.</div>',
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Run tests", type="primary"):
            with st.spinner("Running pytest…"):
                res = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider",
                                      "--color=no", "-W", "ignore::DeprecationWarning", "-rA"],
                                     cwd=ROOT, capture_output=True, text=True)
            last = (res.stdout.strip().splitlines() or ["(no output)"])[-1]
            (st.success if res.returncode == 0 else st.error)(last)
            passed = [ln.split("::", 1)[1] for ln in res.stdout.splitlines() if ln.startswith("PASSED ")]
            if passed:
                st.markdown("\n".join(f"- ✓ `{t.replace('test_', '', 1).replace('_', ' ')}`" for t in passed))
            with st.expander("Full pytest output"):
                st.code(res.stdout[-6000:] + res.stderr[-2000:], language="text")



nav = st.navigation([st.Page(home, title="Try Truvix", icon=":material/image_search:", default=True),
                     st.Page(developer, title="Developer tools", icon=":material/build:", url_path="developer")])
sidebar_footer()
nav.run()
