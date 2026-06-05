# ============================================================
# STEREONET GEOLOGI STRUKTUR - STREAMLIT APP
# Jalankan: streamlit run app.py
#
# Install dependencies:
#   pip install streamlit mplstereonet matplotlib numpy
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import mplstereonet

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Stereonet Geologi Struktur",
    page_icon="🪨",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}
.main { background-color: #F7F5F0; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    font-size: 13px;
    color: #555;
}
.stButton button {
    background: #1a1a1a;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 6px 16px;
}
.stButton button:hover { background: #333; }
.metric-box {
    background: #fff;
    border: 1px solid #e0ddd6;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
.metric-value { font-size: 22px; font-weight: 600; color: #1a1a1a; font-family: 'IBM Plex Mono', monospace; }
.info-box {
    background: #EEF4FB;
    border-left: 3px solid #185FA5;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
    color: #1a3a5c;
    margin-top: 10px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ── MATH UTILITIES ───────────────────────────────────────────

def to_rad(d):
    return np.radians(d)

def to_deg(r):
    return np.degrees(r)

def mean_angle(angles):
    r = np.radians(angles)
    return float((np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) + 360) % 360)

def angular_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)

def plane_to_vec(strike, dip):
    s, d = to_rad(strike), to_rad(dip)
    v = np.array([np.sin(d)*np.sin(s), np.sin(d)*np.cos(s), np.cos(d)])
    if v[2] < 0:
        v = -v
    return v / np.linalg.norm(v)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def cross(a, b):
    return np.cross(a, b)

def vec_to_trend_plunge(v):
    x, y, z = v
    trend  = float((to_deg(np.arctan2(x, y)) + 360) % 360)
    plunge = float(to_deg(np.arcsin(np.clip(abs(z), -1, 1))))
    if z < 0:
        trend = (trend + 180) % 360
    return trend, plunge

def classify_fold(plunge):
    if plunge < 10:   return "Horizontal fold (< 10°)"
    elif plunge < 30: return "Gentle plunging fold (10°–30°)"
    elif plunge < 60: return "Moderate plunging fold (30°–60°)"
    else:             return "Steeply plunging fold (≥ 60°)"

def classify_fault(dips):
    arr = np.array(dips)
    n   = len(arr)
    if np.sum(arr > 60) / n > 0.5:  return "Normal fault dominant"
    if np.sum(arr < 30) / n > 0.5:  return "Strike-slip dominant"
    if np.sum((arr >= 30) & (arr <= 60)) / n > 0.5: return "Reverse / thrust dominant"
    return "Oblique slip / mixed"


# ── PLOT: STEREONET SESAR ────────────────────────────────────

COLORS = ['#185FA5', '#D85A30', '#1D9E75', '#D4537E', '#BA7517',
          '#7F77DD', '#0F6E56', '#993C1D']

def plot_fault_stereonet(fault_sets):
    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw=dict(projection='stereonet'))
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')

    legend_handles = []
    for i, fs in enumerate(fault_sets):
        col = COLORS[i % len(COLORS)]
        ax.plane(fs['strike'], fs['dip'], color=col, linewidth=1.5, alpha=0.7)
        ax.pole(fs['strike'],  fs['dip'], color=col, marker='o', markersize=7)
        legend_handles.append(
            Line2D([0],[0], color=col, linewidth=2,
                   label=f"Set {i+1}  {fs['strike']}°/{fs['dip']}°")
        )

    ax.grid(True, alpha=0.3)
    ax.set_title("Stereonet Proyeksi\n(Great Circle & Pole)",
                 fontsize=11, fontfamily='monospace', pad=12)
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
              framealpha=0.85, edgecolor='#ccc')
    plt.tight_layout()
    return fig


def plot_rose(strikes):
    fig = plt.figure(figsize=(4, 4))
    ax  = fig.add_subplot(111, polar=True)
    fig.patch.set_facecolor('#FAFAF8')
    ax.set_facecolor('#FAFAF8')

    all_s = np.concatenate([strikes, (strikes + 180) % 360])
    bins  = np.linspace(0, 2*np.pi, 37)
    n, _  = np.histogram(np.radians(all_s), bins=bins)
    width = 2*np.pi / 36

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.bar(bins[:-1], n, width=width, bottom=0,
           color='#185FA5', edgecolor='white', alpha=0.8, linewidth=0.5)

    ax.set_yticks([])
    ax.set_title("Rose Diagram\n(Orientasi Strike)",
                 fontsize=11, fontfamily='monospace', pad=14)
    plt.tight_layout()
    return fig


def plot_stress_stereonet(sigma1, sigma2, sigma3):
    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw=dict(projection='stereonet'))
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')

    # bidang tegasan (vertikal, strike = sigma - 90)
    for trend, col, lbl, ls in [
        (sigma1, '#1D9E75', 'σ1 plane', '-'),
        (sigma2, '#BA7517', 'σ2 plane', '--'),
        (sigma3, '#D85A30', 'σ3 plane', ':'),
    ]:
        strike_p = (trend - 90) % 360
        ax.plane(strike_p, 90, color=col, linewidth=2, linestyle=ls)
        ax.line(0, trend, marker='^' if lbl=='σ1 plane' else ('s' if lbl=='σ2 plane' else 'v'),
                color=col, markersize=11, label=lbl)

    ax.legend(loc='lower right', fontsize=8, framealpha=0.85, edgecolor='#ccc')
    ax.grid(True, alpha=0.3)
    ax.set_title("Stress Axes & Planes\n(σ1, σ2, σ3)",
                 fontsize=11, fontfamily='monospace', pad=12)
    plt.tight_layout()
    return fig


# ── PLOT: STEREONET LIPATAN ──────────────────────────────────

def plot_fold_stereonet(fold_left, fold_right):
    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw=dict(projection='stereonet'))
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')

    for fs in fold_left:
        ax.plane(fs['strike'], fs['dip'], color='#185FA5', linewidth=1.5, alpha=0.6)
        ax.pole(fs['strike'],  fs['dip'], color='#185FA5', marker='o', markersize=6)

    for fs in fold_right:
        ax.plane(fs['strike'], fs['dip'], color='#D85A30', linewidth=1.5,
                 linestyle='--', alpha=0.6)
        ax.pole(fs['strike'],  fs['dip'], color='#D85A30', marker='o', markersize=6)

    # Mean poles & fold axis
    def mean_vec(sets):
        vecs = [plane_to_vec(s['strike'], s['dip']) for s in sets]
        return normalize(np.mean(vecs, axis=0))

    mL = mean_vec(fold_left)
    mR = mean_vec(fold_right)
    fv = normalize(cross(mL, mR))
    if fv[2] < 0: fv = -fv
    if np.isnan(fv[0]): fv = normalize((mL + mR) / 2)

    tL, pL = vec_to_trend_plunge(mL)
    tR, pR = vec_to_trend_plunge(mR)
    tA, pA = vec_to_trend_plunge(fv)

    ax.line(pL, tL, marker='*', color='#185FA5', markersize=16, label=f'Mean pole L ({tL:.0f}°/{pL:.0f}°)')
    ax.line(pR, tR, marker='*', color='#D85A30', markersize=16, label=f'Mean pole R ({tR:.0f}°/{pR:.0f}°)')
    ax.line(pA, tA, marker='*', color='#1D9E75', markersize=18, label=f'Fold axis ({tA:.0f}°/{pA:.0f}°)')

    handles = [
        mpatches.Patch(color='#185FA5', label='Left limb'),
        mpatches.Patch(color='#D85A30', label='Right limb'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='#1D9E75', markersize=12, label='Fold axis'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.85, edgecolor='#ccc')
    ax.grid(True, alpha=0.3)
    ax.set_title("Stereonet Lipatan\n(Great Circle & Fold Axis)",
                 fontsize=11, fontfamily='monospace', pad=12)
    plt.tight_layout()
    return fig, tA, pA


# ── SESSION STATE INIT ───────────────────────────────────────

if 'fault_sets' not in st.session_state:
    st.session_state.fault_sets = [
        {'strike': 120, 'dip': 45},
        {'strike': 130, 'dip': 50},
        {'strike': 140, 'dip': 60},
        {'strike': 300, 'dip': 30},
        {'strike': 310, 'dip': 35},
        {'strike': 320, 'dip': 40},
    ]

if 'fold_left' not in st.session_state:
    st.session_state.fold_left = [
        {'strike': 120, 'dip': 40},
        {'strike': 125, 'dip': 45},
        {'strike': 130, 'dip': 50},
    ]

if 'fold_right' not in st.session_state:
    st.session_state.fold_right = [
        {'strike': 300, 'dip': 35},
        {'strike': 305, 'dip': 40},
        {'strike': 310, 'dip': 45},
    ]


# ── HEADER ───────────────────────────────────────────────────

st.title("🪨 Stereonet Geologi Struktur")
st.caption("Proyeksi stereonet interaktif untuk analisis sesar dan lipatan")

st.divider()

# ── MODE SELECTOR ────────────────────────────────────────────

mode = st.radio(
    "Pilih tipe analisis",
    ["⚡ Analisis Sesar (Fault)", "〰️ Analisis Lipatan (Fold)"],
    horizontal=True,
)
is_fault = mode.startswith("⚡")

st.divider()


# ════════════════════════════════════════════════════════════
#  MODE: SESAR
# ════════════════════════════════════════════════════════════

if is_fault:
    col_input, col_plot = st.columns([1, 2], gap="large")

    # ── INPUT PANEL ──────────────────────────────────────────
    with col_input:
        st.subheader("Input Data Sesar")

        fs = st.session_state.fault_sets
        to_delete = None

        for i, s in enumerate(fs):
            col_a, col_b, col_c = st.columns([2, 2, 1])
            with col_a:
                fs[i]['strike'] = st.number_input(
                    f"Strike {i+1} (°)", 0, 360, s['strike'],
                    key=f"fs_{i}_str")
            with col_b:
                fs[i]['dip'] = st.number_input(
                    f"Dip {i+1} (°)", 0, 90, s['dip'],
                    key=f"fs_{i}_dip")
            with col_c:
                st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
                if len(fs) > 1:
                    if st.button("✕", key=f"del_fs_{i}", help="Hapus set ini"):
                        to_delete = i
                st.markdown("</div>", unsafe_allow_html=True)

        if to_delete is not None:
            st.session_state.fault_sets.pop(to_delete)
            st.rerun()

        if st.button("＋ Tambah set data", use_container_width=True):
            st.session_state.fault_sets.append({'strike': 0, 'dip': 45})
            st.rerun()

        st.divider()
        shear_sense = st.selectbox(
            "Shear sense",
            ["Dextral (kanan)", "Sinistral (kiri)"]
        )

    # ── PLOTS ────────────────────────────────────────────────
    with col_plot:
        strikes = np.array([s['strike'] for s in fs])
        dips    = np.array([s['dip']    for s in fs])

        tab1, tab2, tab3 = st.tabs(["Stereonet", "Rose Diagram", "Stress Axes"])

        with tab1:
            fig = plot_fault_stereonet(fs)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            fig = plot_rose(strikes)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab3:
            sigma1 = (mean_angle(list(strikes)) + 90) % 360
            sigma2 = (sigma1 + 90) % 360
            sigma3 = (sigma1 + 180) % 360
            fig    = plot_stress_stereonet(sigma1, sigma2, sigma3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.divider()
    st.subheader("Hasil Analisis")

    # Fault type
    fault_type = classify_fault(list(dips))

    # Stress
    sigma1 = (mean_angle(list(strikes)) + 90) % 360
    sigma3 = (sigma1 + 180) % 360

    # Riedel
    hist = np.zeros(18)
    for s in strikes:
        hist[int(((s % 360) + 360) % 360 // 20) % 18] += 1
    shear_zone = (np.argmax(hist) * 20 + 10)
    riedel_types = []
    for s in strikes:
        a = angular_diff(float(s), float(shear_zone))
        if a <= 10:          riedel_types.append('Y shear')
        elif a <= 20:        riedel_types.append('R shear')
        elif a <= 40:        riedel_types.append('P shear')
        elif 70 <= a <= 80:  riedel_types.append("R' shear")
        else:                riedel_types.append('Unclassified')
    from collections import Counter
    rcount   = Counter(riedel_types)
    dominant = rcount.most_common(1)[0][0]

    # Rake estimate
    mean_dip = float(np.mean(dips))
    if mean_dip > 60:   rake = -90
    elif mean_dip < 30: rake = 0
    else:               rake = 30 if 'Dextral' in shear_sense else -30

    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val in [
        (m1, "Fault type",       fault_type),
        (m2, "σ1 trend",         f"{sigma1:.1f}°"),
        (m3, "σ3 trend",         f"{sigma3:.1f}°"),
        (m4, "Rake estimasi",    f"{rake}°"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
      <b>Riedel shear zone:</b> {shear_zone:.0f}° &nbsp;|&nbsp;
      <b>Dominant structure:</b> {dominant} &nbsp;|&nbsp;
      <b>Shear sense:</b> {shear_sense}<br>
      <b>Distribusi Riedel:</b> {' · '.join(f"{k}: {v}" for k, v in rcount.items())}
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  MODE: LIPATAN
# ════════════════════════════════════════════════════════════

else:
    col_input, col_plot = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("Input Data Lipatan")

        # LEFT LIMB
        st.markdown("**Limb kiri (left limb)**")
        fl   = st.session_state.fold_left
        del_l = None
        for i, s in enumerate(fl):
            ca, cb, cc = st.columns([2, 2, 1])
            with ca:
                fl[i]['strike'] = st.number_input(
                    f"Strike L{i+1}", 0, 360, s['strike'], key=f"fl_{i}_str")
            with cb:
                fl[i]['dip'] = st.number_input(
                    f"Dip L{i+1}", 0, 90, s['dip'], key=f"fl_{i}_dip")
            with cc:
                st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
                if len(fl) > 1:
                    if st.button("✕", key=f"del_fl_{i}"):
                        del_l = i
                st.markdown("</div>", unsafe_allow_html=True)

        if del_l is not None:
            st.session_state.fold_left.pop(del_l)
            st.rerun()

        if st.button("＋ Tambah limb kiri", use_container_width=True):
            st.session_state.fold_left.append({'strike': 0, 'dip': 45})
            st.rerun()

        st.divider()

        # RIGHT LIMB
        st.markdown("**Limb kanan (right limb)**")
        fr    = st.session_state.fold_right
        del_r = None
        for i, s in enumerate(fr):
            ca, cb, cc = st.columns([2, 2, 1])
            with ca:
                fr[i]['strike'] = st.number_input(
                    f"Strike R{i+1}", 0, 360, s['strike'], key=f"fr_{i}_str")
            with cb:
                fr[i]['dip'] = st.number_input(
                    f"Dip R{i+1}", 0, 90, s['dip'], key=f"fr_{i}_dip")
            with cc:
                st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
                if len(fr) > 1:
                    if st.button("✕", key=f"del_fr_{i}"):
                        del_r = i
                st.markdown("</div>", unsafe_allow_html=True)

        if del_r is not None:
            st.session_state.fold_right.pop(del_r)
            st.rerun()

        if st.button("＋ Tambah limb kanan", use_container_width=True):
            st.session_state.fold_right.append({'strike': 0, 'dip': 45})
            st.rerun()

    with col_plot:
        fig_fold, tA, pA = plot_fold_stereonet(fl, fr)
        st.pyplot(fig_fold, use_container_width=True)
        plt.close(fig_fold)

    st.divider()
    st.subheader("Hasil Analisis Lipatan")

    fold_type = classify_fold(pA)

    # Mean poles
    def mean_vec(sets):
        vecs = [plane_to_vec(s['strike'], s['dip']) for s in sets]
        return normalize(np.mean(vecs, axis=0))

    mL = mean_vec(fl); mR = mean_vec(fr)
    tL, pL = vec_to_trend_plunge(mL)
    tR, pR = vec_to_trend_plunge(mR)

    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val in [
        (m1, "Fold axis trend",    f"{tA:.1f}°"),
        (m2, "Fold axis plunge",   f"{pA:.1f}°"),
        (m3, "Mean pole – kiri",   f"{tL:.1f}° / {pL:.1f}°"),
        (m4, "Mean pole – kanan",  f"{tR:.1f}° / {pR:.1f}°"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
      <b>Tipe lipatan:</b> {fold_type}<br>
      <b>Interpretasi:</b> Sumbu lipatan diperoleh dari cross product mean pole kedua limb,
      menunjukkan deformasi kompresional dengan plunge <b>{pA:.1f}°</b>
      ke arah <b>{tA:.0f}°</b>.
    </div>""", unsafe_allow_html=True)


# ── FOOTER ───────────────────────────────────────────────────
st.divider()
st.caption("Stereonet Geologi Struktur · Proyeksi Wulff (Equal-Angle) · Lower Hemisphere")
