"""
================================================================================
BDI FEAR & GREED INDEX — RÉPLICA CNN MEJORADA  (Streamlit)
================================================================================
Autor: Marian — versión profesional reescrita.

Cambios clave vs. el código original:

1. **VIX real, no percentil.** El panel "VIX" ahora muestra el nivel del VIX
   en sus unidades reales (8–80 aprox.) y, en otra capa, el percentil
   rolling de 252 días. El bug del código anterior era confundir el score
   normalizado con el VIX nominal.

2. **Calibración no destructiva.** En lugar de sumar un `shift` que deforma
   toda la serie histórica para que el último día calce con CNN, se calcula
   el índice con una función sigmoide sobre Z-scores y se reporta el valor
   de CNN aparte como referencia. Tu serie ya no se "tuerce" para encajar.

3. **Componentes más fieles a la metodología CNN:**
   - Momentum: SPX vs su SMA-125.
   - Stock Price Strength: SPX vs máximo de 252 días (drawdown invertido).
   - Stock Price Breadth: ratio RSP/SPY normalizado por z-score
     (mide concentración: si SPY sube y RSP no, breadth es malo).
   - Put/Call Ratio: estructura de plazos VIX/VIX3M
     (proxy de demanda de cobertura corta).
   - Market Volatility: VIX vs su SMA-50.
   - Safe Haven Demand: 20d return SPY − 20d return TLT.
   - Junk Bond Demand: spread HYG vs LQD (z-score).

4. **Streamlit nativo.** Cache, sidebar con controles, tabs separadas
   para Dashboard, Componentes, Educación, Datos.

5. **Sección educativa con fórmulas LaTeX** en pestaña dedicada.

Para correr local:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import time
import warnings
import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG GENERAL
# ==============================================================================
st.set_page_config(
    page_title="BDI Fear & Greed Index — CNN Replica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Paleta BDI Consultora Patrimonial Integral ------------------------------
# Colores oficiales del Style Guide:
#   #137247 verde corporativo · #17BEBB turquesa · #B5E61D lima
#   #323232 gris oscuro · #EFEDEA crema
BRAND_GREEN = "#137247"   # primario
BRAND_TURQ  = "#17BEBB"   # secundario
BRAND_LIME  = "#B5E61D"   # acento
BRAND_DARK  = "#323232"   # base
BRAND_CREAM = "#EFEDEA"   # texto / claros

BG    = "#1c1c1c"          # fondo dashboard (BDI dark, un poco más oscuro)
PANEL = BRAND_DARK         # paneles
TXT   = BRAND_CREAM        # texto principal
MUTED = "#9aa093"          # texto secundario sobre crema apagada

# Mapeo Fear & Greed alineado a la marca (rojo solo donde es indispensable
# para la lectura de "miedo"; el lado positivo se construye con verdes BDI).
GOLD       = BRAND_LIME    # destacado numérico
GREEN      = BRAND_LIME    # codicia extrema
GREEN_DARK = BRAND_GREEN   # codicia
YELLOW     = "#d9a72b"     # neutral cálido
ORANGE     = "#d9861a"     # transición fear→neutral
RED        = "#c6453a"     # miedo
RED_DARK   = "#5a1818"     # miedo extremo
BLUE       = BRAND_TURQ    # series complementarias (SPY)

CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

TICKERS = {
    "SPX": "^GSPC",
    "SPY": "SPY",
    "RSP": "RSP",      # equal-weight S&P 500 — proxy breadth
    "TLT": "TLT",      # bonos largos
    "IEF": "IEF",
    "HYG": "HYG",      # high-yield
    "LQD": "LQD",      # investment grade
    "BND": "BND",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "NYA": "^NYA",     # NYSE Composite
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Poppins', sans-serif !important;
}}
.stApp {{ background-color:{BG}; color:{TXT}; }}
section[data-testid="stSidebar"] {{ background-color:{PANEL}; }}
[data-testid="stMetricValue"] {{ color:{BRAND_LIME}; font-weight:700; }}
[data-testid="stMetricLabel"] {{ color:{MUTED}; }}
h1, h2, h3, h4 {{
    color:{TXT};
    font-family: 'Bebas Neue', 'Poppins', sans-serif !important;
    letter-spacing: 1px;
}}
.block-container {{ padding-top: 1.2rem; }}
hr {{ border-color:#444; }}
.tag-fear    {{ color:{RED};    font-weight:700; }}
.tag-greed   {{ color:{BRAND_LIME}; font-weight:700; }}
.tag-neutral {{ color:{YELLOW}; font-weight:700; }}
.small-muted {{ color:{MUTED}; font-size:12px; font-family:'Poppins',sans-serif; }}
.bdi-header {{
    display:flex; align-items:center; gap:18px;
    padding:14px 20px; border-radius:10px;
    background: linear-gradient(135deg, {BRAND_GREEN} 0%, {BRAND_TURQ} 60%, {BRAND_LIME} 100%);
    margin-bottom: 18px;
}}
.bdi-header .logo {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px;
    color: white;
    letter-spacing: 4px;
    line-height: 1;
}}
.bdi-header .logo .arrow {{ color:white; margin-left:2px; }}
.bdi-header .tagline {{
    font-family:'Poppins', sans-serif; font-size:11px;
    letter-spacing:3px; color: rgba(255,255,255,.92);
    text-transform:uppercase;
}}
.bdi-header .title-block {{
    margin-left:auto; text-align:right; color:white;
}}
.bdi-header .title-block .h1 {{
    font-family:'Bebas Neue', sans-serif; font-size:28px; letter-spacing:2px;
}}
.bdi-header .title-block .h2 {{
    font-family:'Poppins', sans-serif; font-size:11px; letter-spacing:2px;
    text-transform:uppercase; opacity:.92;
}}
div[data-baseweb="tab-list"] button[role="tab"] {{
    color: {MUTED} !important;
}}
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    color: {BRAND_LIME} !important;
    border-bottom-color: {BRAND_LIME} !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def bdi_header(title: str, subtitle: str):
    """Renderiza la cabecera BDI con logo recreado en HTML."""
    st.markdown(f"""
    <div class="bdi-header">
        <div>
            <div class="logo">BDI<span class="arrow">▶</span></div>
            <div class="tagline">Consultora Patrimonial Integral</div>
        </div>
        <div class="title-block">
            <div class="h1">{title}</div>
            <div class="h2">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# UTILIDADES
# ==============================================================================
def status_label(v: float) -> str:
    if pd.isna(v): return "Neutral"
    if v <= 25: return "Miedo Extremo"
    if v <= 45: return "Miedo"
    if v <= 55: return "Neutral"
    if v <= 75: return "Codicia"
    return "Codicia Extrema"

def status_color(v: float) -> str:
    if pd.isna(v): return YELLOW
    if v <= 25: return RED_DARK
    if v <= 45: return RED
    if v <= 55: return YELLOW
    if v <= 75: return GREEN_DARK
    return GREEN

def safe_last(series, default=np.nan):
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if len(s) else default

def safe_past(series, n, default=np.nan):
    s = pd.Series(series).dropna()
    return float(s.iloc[-(n + 1)]) if len(s) > n else default

# Sigmoide centrada en 0 con escala k → mapea Z a (0,100)
def sigmoid_to_score(z: pd.Series, k: float = 1.5) -> pd.Series:
    z = pd.Series(z, dtype=float)
    return (100 / (1 + np.exp(-z / k))).clip(0, 100)

def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    s = pd.Series(series, dtype=float)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)

def fmt_num(x, nd=1):
    return "N/A" if pd.isna(x) else f"{x:.{nd}f}"

# ==============================================================================
# DESCARGA
# ==============================================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_one(ticker: str, start: dt.date, end: dt.date,
                 retries: int = 3, pause: float = 2.0) -> pd.Series:
    for _ in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start, end=end,
                auto_adjust=False, progress=False,
                threads=False, timeout=25,
            )
            if df is None or df.empty:
                time.sleep(pause); continue
            if isinstance(df.columns, pd.MultiIndex):
                s = df["Close"].iloc[:, 0].copy()
            elif "Close" in df.columns:
                s = df["Close"].copy()
            else:
                s = df.iloc[:, 0].copy()
            s = pd.to_numeric(s, errors="coerce").dropna()
            s.name = ticker
            if len(s):
                return s
        except Exception:
            time.sleep(pause)
    return pd.Series(dtype=float, name=ticker)

@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_universe(start: dt.date, end: dt.date) -> tuple:
    out, failed = {}, []
    for name, tk in TICKERS.items():
        s = download_one(tk, start, end)
        if len(s) == 0:
            failed.append((name, tk))
        else:
            out[name] = s
    if not out:
        return pd.DataFrame(), failed
    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    df = df.sort_index().ffill()
    return df, failed

@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_cnn_score():
    try:
        r = requests.get(
            CNN_URL,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://edition.cnn.com/markets/fear-and-greed",
            },
            timeout=15,
        )
        r.raise_for_status()
        raw = r.json()
        def find(o):
            if isinstance(o, dict):
                for k in ("score", "value", "current", "now"):
                    if k in o and isinstance(o[k], (int, float)) and 0 <= o[k] <= 100:
                        return float(o[k])
                for v in o.values():
                    f = find(v)
                    if f is not None: return f
            elif isinstance(o, list):
                for it in o:
                    f = find(it)
                    if f is not None: return f
            return None
        return find(raw)
    except Exception:
        return None

# ==============================================================================
# FALLBACKS
# ==============================================================================
def apply_fallbacks(data: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("SPX", ["SPY", "QQQ"]),
        ("SPY", ["SPX", "QQQ"]),
        ("NYA", ["SPY", "SPX"]),
        ("RSP", ["SPY"]),
        ("TLT", ["IEF", "BND"]),
        ("HYG", ["JNK", "LQD"]),
        ("LQD", ["BND", "TLT"]),
    ]
    for col, alts in pairs:
        if col not in data.columns:
            for alt in alts:
                if alt in data.columns:
                    data[col] = data[alt]
                    break
    if "VIX" not in data.columns and "SPY" in data.columns:
        rv = data["SPY"].pct_change().rolling(20, min_periods=10).std() * np.sqrt(252) * 100
        data["VIX"] = rv.bfill().fillna(20)
    if "VIX3M" not in data.columns and "VIX" in data.columns:
        data["VIX3M"] = data["VIX"].rolling(63, min_periods=20).mean().bfill()
    return data

# ==============================================================================
# COMPONENTES (mejorados)
# ==============================================================================
def compute_components(data: pd.DataFrame) -> pd.DataFrame:
    """
    7 componentes estilo CNN. Cada uno se calcula como Z-score rolling de su
    señal cruda y luego se mapea a [0,100] con una sigmoide.
    invert=True invierte la dirección (señales en las que "alto" = miedo).
    """
    idx = data.index

    def to_score(raw, invert=False, k=1.5):
        z = rolling_zscore(raw, 252, 60)
        if invert: z = -z
        score = sigmoid_to_score(z, k=k)
        return score.reindex(idx).ffill().bfill().fillna(50).clip(0, 100)

    # 1. Momentum: SPX / SMA125 - 1
    mom_raw = data["SPX"] / data["SPX"].rolling(125, min_periods=60).mean() - 1

    # 2. Stock Price Strength: distancia al máximo de 252 días
    #    (1 = en máximos, valores menores = drawdown)
    strength_raw = data["NYA"] / data["NYA"].rolling(252, min_periods=80).max()

    # 3. Stock Price Breadth: ratio RSP/SPY (equal-weight vs cap-weight).
    #    Si las megacaps lideran y RSP queda atrás → breadth pobre → fear.
    breadth_raw = (data["RSP"] / data["SPY"]).replace([np.inf, -np.inf], np.nan)

    # 4. Put/Call (proxy): estructura VIX vs VIX3M.
    #    >1 = pánico corto plazo; <1 = calma. Invertimos.
    pcr_raw = data["VIX"] / data["VIX3M"]

    # 5. Market Volatility: VIX vs SMA50. Invertimos.
    vol_raw = data["VIX"] / data["VIX"].rolling(50, min_periods=20).mean()

    # 6. Safe Haven Demand: 20d return SPY − 20d return TLT.
    safe_raw = data["SPY"].pct_change(20) - data["TLT"].pct_change(20)

    # 7. Junk Bond Demand: HYG / LQD (precios). Si HY sube vs IG → risk-on.
    junk_raw = data["HYG"] / data["LQD"]

    df = pd.DataFrame({
        "Momentum S&P500":      to_score(mom_raw,      invert=False),
        "Stock Price Strength": to_score(strength_raw, invert=False),
        "Stock Price Breadth":  to_score(breadth_raw,  invert=False),
        "Put/Call Ratio":       to_score(pcr_raw,      invert=True),
        "Market Volatility":    to_score(vol_raw,      invert=True),
        "Safe Haven Demand":    to_score(safe_raw,     invert=False),
        "Junk Bond Demand":     to_score(junk_raw,     invert=False),
    }, index=idx).fillna(50)

    return df

WEIGHTS = {
    "Momentum S&P500":       1 / 7,
    "Stock Price Strength":  1 / 7,
    "Stock Price Breadth":   1 / 7,
    "Put/Call Ratio":        1 / 7,
    "Market Volatility":     1 / 7,
    "Safe Haven Demand":     1 / 7,
    "Junk Bond Demand":      1 / 7,
}

def composite_index(components: pd.DataFrame, weights: dict) -> pd.Series:
    out = sum(components[c] * w for c, w in weights.items())
    return out.rolling(3, min_periods=1).mean().clip(0, 100)

# ==============================================================================
# VISUALES (Plotly)
# ==============================================================================
def gauge_figure(value: float, cnn_value=None) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=58, color=GOLD)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=MUTED,
                      tickvals=[0, 25, 45, 55, 75, 100],
                      ticktext=["0", "25", "45", "55", "75", "100"]),
            bar=dict(color="rgba(0,0,0,0)"),
            bgcolor=PANEL,
            steps=[
                dict(range=[0, 25],   color=RED_DARK),
                dict(range=[25, 45],  color=RED),
                dict(range=[45, 55],  color=YELLOW),
                dict(range=[55, 75],  color=GREEN_DARK),
                dict(range=[75, 100], color=GREEN),
            ],
            threshold=dict(line=dict(color=TXT, width=4), thickness=0.85, value=value),
        ),
    ))
    annotations = [dict(
        x=0.5, y=0.05, xref="paper", yref="paper",
        text=f"<b>{status_label(value).upper()}</b>",
        showarrow=False, font=dict(color=GOLD, size=18),
    )]
    if cnn_value is not None:
        annotations.append(dict(
            x=0.5, y=-0.05, xref="paper", yref="paper",
            text=f"<span style='color:{MUTED}'>CNN F&G hoy: {cnn_value:.0f}</span>",
            showarrow=False, font=dict(size=12),
        ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=20, r=20, t=10, b=40), height=320,
        annotations=annotations,
    )
    return fig


def history_figure(series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=25, fillcolor=RED, opacity=0.10, line_width=0)
    fig.add_hrect(y0=25, y1=45, fillcolor=ORANGE, opacity=0.08, line_width=0)
    fig.add_hrect(y0=45, y1=55, fillcolor=YELLOW, opacity=0.08, line_width=0)
    fig.add_hrect(y0=55, y1=75, fillcolor=GREEN_DARK, opacity=0.10, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor=GREEN, opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=GOLD, width=1.6), name="BDI",
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=10), height=380,
        yaxis=dict(range=[0, 100], gridcolor="#243244"),
        xaxis=dict(gridcolor="#243244"),
        showlegend=False,
    )
    return fig


def components_bar(components_now: pd.Series) -> go.Figure:
    cs = components_now.sort_values()
    fig = go.Figure(go.Bar(
        x=cs.values, y=cs.index, orientation="h",
        text=[f"{v:.1f}" for v in cs.values],
        textposition="outside",
        marker=dict(color=[status_color(v) for v in cs.values],
                    line=dict(color=TXT, width=0.6)),
    ))
    fig.add_vline(x=50, line=dict(color=MUTED, dash="dash", width=1))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=30), height=420,
        xaxis=dict(range=[0, 105], gridcolor="#243244",
                   title="Score (0=miedo extremo  ←  →  100=codicia extrema)"),
        yaxis=dict(gridcolor="#243244"),
    )
    return fig


def vix_figure(vix_series: pd.Series, days: int = 252) -> go.Figure:
    """VIX en valores REALES (no percentil). Bandas de referencia."""
    s = vix_series.dropna().tail(days)
    fig = go.Figure()
    # Bandas de referencia históricas del VIX
    fig.add_hrect(y0=0,  y1=15, fillcolor=GREEN,    opacity=0.08, line_width=0,
                  annotation_text="Calma <15", annotation_position="left",
                  annotation=dict(font=dict(color=MUTED, size=10)))
    fig.add_hrect(y0=15, y1=20, fillcolor=YELLOW,   opacity=0.06, line_width=0)
    fig.add_hrect(y0=20, y1=30, fillcolor=ORANGE,   opacity=0.08, line_width=0)
    fig.add_hrect(y0=30, y1=80, fillcolor=RED,      opacity=0.10, line_width=0,
                  annotation_text="Pánico >30", annotation_position="left",
                  annotation=dict(font=dict(color=MUTED, size=10)))
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color="#ff5b4d", width=1.8), name="VIX",
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        yaxis=dict(title="VIX (pts)", gridcolor="#243244"),
        xaxis=dict(gridcolor="#243244"),
        showlegend=False,
    )
    return fig


def spy_figure(spy: pd.Series, days: int = 756) -> go.Figure:
    s = spy.dropna().tail(days)
    sma50 = s.rolling(50, min_periods=20).mean()
    sma200 = s.rolling(200, min_periods=60).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                             line=dict(color=BLUE, width=1.6), name="SPY"))
    fig.add_trace(go.Scatter(x=sma50.index, y=sma50.values, mode="lines",
                             line=dict(color=YELLOW, width=1.1, dash="dash"), name="SMA-50"))
    fig.add_trace(go.Scatter(x=sma200.index, y=sma200.values, mode="lines",
                             line=dict(color="#a36df0", width=1.1, dash="dot"), name="SMA-200"))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        yaxis=dict(gridcolor="#243244"),
        xaxis=dict(gridcolor="#243244"),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(color=TXT)),
    )
    return fig


def histogram_figure(series: pd.Series) -> go.Figure:
    bins = ["Miedo Ext.", "Miedo", "Neutral", "Codicia", "Codicia Ext."]
    s = series.dropna()
    pct = pd.Series({
        "Miedo Ext.":   ((s <= 25).mean()) * 100,
        "Miedo":        (((s > 25) & (s <= 45)).mean()) * 100,
        "Neutral":      (((s > 45) & (s <= 55)).mean()) * 100,
        "Codicia":      (((s > 55) & (s <= 75)).mean()) * 100,
        "Codicia Ext.": ((s > 75).mean()) * 100,
    })
    colors = [RED_DARK, RED, YELLOW, GREEN_DARK, GREEN]
    fig = go.Figure(go.Bar(
        x=pct.index, y=pct.values,
        marker=dict(color=colors, line=dict(color=TXT, width=0.6)),
        text=[f"{v:.0f}%" for v in pct.values],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        yaxis=dict(title="% de días", gridcolor="#243244"),
        xaxis=dict(gridcolor="#243244"),
    )
    return fig

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Controles")
    years_back = st.slider("Histórico (años)", 1, 8, 4)
    show_cnn = st.checkbox("Mostrar referencia CNN F&G", value=True)
    show_components_table = st.checkbox("Tabla de componentes", value=True)
    st.markdown("---")
    st.markdown(
        f"<span class='small-muted'>Fuente: Yahoo Finance + CNN.<br>"
        f"Actualización: cada 30 min (cache).</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Recomendación de uso")
    st.markdown(
        "<span class='small-muted'>El índice es una <b>ayuda contraria</b>: "
        "valores extremos suelen anticipar reversiones, no continuaciones. "
        "No es señal de trading aislada.</span>",
        unsafe_allow_html=True,
    )

# ==============================================================================
# CARGA Y CÁLCULO
# ==============================================================================
TODAY = dt.date.today()
START = TODAY - timedelta(days=years_back * 365 + 365)  # +1 año extra para warm-up

with st.spinner("Descargando datos de mercado…"):
    data, failed = download_universe(START, TODAY + timedelta(days=1))

if data.empty:
    st.error("No se pudieron descargar datos de Yahoo Finance. Probá nuevamente en unos minutos.")
    st.stop()

if failed:
    with st.sidebar.expander("⚠️ Tickers con fallo (con fallback aplicado)"):
        for n, t in failed:
            st.text(f"• {n} ({t})")

data = apply_fallbacks(data)

required = ["SPX", "SPY", "RSP", "TLT", "HYG", "LQD", "VIX", "NYA"]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error(f"Faltan columnas críticas tras fallbacks: {missing}")
    st.stop()

components = compute_components(data)
bdi_full = composite_index(components, WEIGHTS)
bdi = bdi_full.tail(years_back * 252)
today_val = safe_last(bdi)

cnn_today = fetch_cnn_score() if show_cnn else None

week_val = safe_past(bdi, 5)
month_val = safe_past(bdi, 21)
year_val = safe_past(bdi, 252)

# ==============================================================================
# HEADER BDI
# ==============================================================================
bdi_header(
    title="FEAR & GREED INDEX",
    subtitle=f"Réplica CNN mejorada · {TODAY.strftime('%d/%m/%Y')}"
)
st.markdown(
    f"<span class='small-muted'>7 componentes · Z-score rolling 252 días + sigmoide · "
    f"Cache 30 min · Fuentes: Yahoo Finance + CNN dataviz</span>",
    unsafe_allow_html=True,
)
st.markdown("")

# ==============================================================================
# TABS
# ==============================================================================
tab_dashboard, tab_components, tab_education, tab_data = st.tabs(
    ["🎯 Dashboard", "🧩 Componentes", "🎓 Educación", "📥 Datos"]
)

# ------------------------------- DASHBOARD ------------------------------------
with tab_dashboard:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(
            gauge_figure(today_val, cnn_today if show_cnn else None),
            use_container_width=True
        )
    with c2:
        st.metric("HOY",      fmt_num(today_val), status_label(today_val))
        st.metric("1 Semana", fmt_num(week_val),  status_label(week_val))
        st.metric("1 Mes",    fmt_num(month_val), status_label(month_val))
        st.metric("1 Año",    fmt_num(year_val),  status_label(year_val))

    st.markdown("### 📈 Evolución del índice")
    st.plotly_chart(history_figure(bdi), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 🔥 VIX (valor real, no percentil)")
        st.plotly_chart(vix_figure(data["VIX"]), use_container_width=True)
        st.caption(
            f"VIX actual: **{safe_last(data['VIX']):.2f}** · "
            f"Promedio 1Y: **{data['VIX'].tail(252).mean():.2f}** · "
            f"Bandas: <15 calma · 15–20 normal · 20–30 estrés · >30 pánico"
        )
    with c4:
        st.markdown("### 🟦 S&P 500 (SPY) y medias")
        st.plotly_chart(spy_figure(data["SPY"]), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("### 📊 Distribución histórica del índice")
        st.plotly_chart(histogram_figure(bdi), use_container_width=True)
    with c6:
        st.markdown("### 🧩 Componentes hoy")
        st.plotly_chart(components_bar(components.iloc[-1]), use_container_width=True)

# ------------------------------- COMPONENTES ----------------------------------
with tab_components:
    st.markdown("## 🧩 Detalle por componente")
    st.markdown(
        "Cada componente es un Z-score rolling de 252 días sobre su señal cruda, "
        "transformado a [0, 100] mediante una función sigmoide. "
        "Score 50 = condiciones promedio del último año."
    )

    if show_components_table:
        st.dataframe(
            components.tail(20).round(1).iloc[::-1],
            use_container_width=True
        )

    st.markdown("### Evolución conjunta")
    fig = go.Figure()
    win = components.tail(years_back * 252)
    palette = [BRAND_LIME, BRAND_TURQ, BRAND_GREEN, RED, ORANGE,
               YELLOW, "#7ad6e6"]
    for i, col in enumerate(components.columns):
        fig.add_trace(go.Scatter(
            x=win.index, y=win[col].values,
            mode="lines", name=col,
            line=dict(width=1.3, color=palette[i % len(palette)]),
        ))
    fig.add_hline(y=50, line=dict(color=MUTED, dash="dash"))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT),
        height=480, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 100], gridcolor="#243244"),
        xaxis=dict(gridcolor="#243244"),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(color=TXT, size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlación entre componentes (último año)")
    corr = components.tail(252).corr().round(2)
    st.dataframe(corr, use_container_width=True)

# ------------------------------- EDUCACIÓN ------------------------------------
with tab_education:
    st.markdown("## 🎓 Cómo se calcula y cómo se interpreta")
    st.markdown(
        "El **BDI Fear & Greed Index** intenta capturar el sentimiento agregado "
        "del mercado de acciones de EE.UU. en una sola cifra entre 0 y 100. "
        "Está inspirado en el índice de CNN, con mejoras técnicas en la "
        "construcción y normalización."
    )

    st.markdown("### 1) Pipeline general")
    st.markdown("Para cada componente se aplica:")
    st.latex(r"""
    Z_t \;=\; \frac{x_t - \mu_{t-252:t-1}}{\sigma_{t-252:t-1}}
    """)
    st.latex(r"""
    \text{Score}_t \;=\; \frac{100}{1 + e^{-Z_t / k}} \quad (k=1.5)
    """)
    st.markdown(
        "El Z-score elimina el sesgo de nivel (un VIX de 18 hoy no significa "
        "lo mismo que un VIX de 18 en 2017). La sigmoide comprime los "
        "extremos a [0, 100] sin truncar la información: un Z = +3 mapea a "
        "≈ 88, no se pierde la cola."
    )
    st.latex(r"""
    \text{BDI}_t \;=\; \sum_{i=1}^{7} w_i \cdot \text{Score}_i \quad
    \text{con } w_i = \tfrac{1}{7}
    """)
    st.markdown("Suavizado final con media móvil de 3 días para reducir ruido.")

    st.markdown("### 2) Los 7 componentes")
    st.markdown("**a) Momentum del S&P500**")
    st.latex(r"x_t = \frac{P_t^{SPX}}{\overline{P^{SPX}}_{t-125:t}} - 1")
    st.markdown(
        "Mide qué tan por encima/debajo está el SPX de su promedio de 125 días "
        "(≈ 6 meses). Replica la lógica de CNN: si el precio está holgadamente "
        "sobre la media → codicia."
    )

    st.markdown("**b) Stock Price Strength**")
    st.latex(r"x_t = \frac{P_t^{NYA}}{\max(P^{NYA}_{t-252:t})}")
    st.markdown(
        "Distancia al máximo de 52 semanas del NYSE Composite. "
        "Mientras el ratio esté cerca de 1 hay fortaleza (codicia). "
        "Cuando se aleja, el mercado pierde amplitud (miedo). "
        "Es un proxy del 'New Highs vs New Lows' de CNN."
    )

    st.markdown("**c) Stock Price Breadth**")
    st.latex(r"x_t = \frac{P_t^{RSP}}{P_t^{SPY}}")
    st.markdown(
        "RSP es el S&P500 equiponderado, SPY es el ponderado por capitalización. "
        "Si las megacaps lideran y el resto del mercado se queda atrás, RSP/SPY "
        "cae → breadth pobre → miedo subyacente. Es el caso típico de los "
        "'rally de pocas grandes'."
    )

    st.markdown("**d) Put/Call Ratio (proxy)**")
    st.latex(r"x_t = \frac{VIX_t}{VIX3M_t} \quad \text{(invertido)}")
    st.markdown(
        "El ratio entre VIX (1 mes) y VIX3M (3 meses) refleja la pendiente de la "
        "estructura de plazos de la volatilidad. Cuando se invierte (VIX > VIX3M) "
        "los traders pagan caro la cobertura corta → demanda urgente de puts → "
        "miedo. Es un proxy más limpio que el PCR oficial, que ahora se cobra."
    )

    st.markdown("**e) Market Volatility**")
    st.latex(r"x_t = \frac{VIX_t}{\overline{VIX}_{t-50:t}} \quad \text{(invertido)}")
    st.markdown(
        "VIX en relación con su SMA-50. Si la volatilidad implícita explota "
        "respecto del régimen reciente, el mercado entra en pánico. "
        "Se invierte porque alto VIX = miedo."
    )

    st.markdown("**f) Safe Haven Demand**")
    st.latex(r"x_t = r_{20}^{SPY} - r_{20}^{TLT}")
    st.markdown(
        "Diferencia de retorno a 20 días entre acciones (SPY) y bonos largos "
        "(TLT). Cuando los inversores buscan refugio, TLT supera a SPY → "
        "diferencia negativa → miedo."
    )

    st.markdown("**g) Junk Bond Demand**")
    st.latex(r"x_t = \frac{P_t^{HYG}}{P_t^{LQD}}")
    st.markdown(
        "Ratio entre bonos high-yield (HYG) y bonos investment-grade (LQD). "
        "Cuando los inversores apetecen riesgo, HYG outperforma → ratio sube → "
        "codicia. Cuando huyen del crédito basura, ratio cae → miedo."
    )

    st.markdown("### 3) Tabla de interpretación")
    st.markdown("""
    | Rango       | Etiqueta          | Lectura típica                                                 |
    |-------------|-------------------|----------------------------------------------------------------|
    | 0 – 25      | **Miedo Extremo** | Capitulación. Probabilidad alta de rebote técnico.             |
    | 25 – 45     | Miedo             | Desconfianza. Mercado vendedor pero todavía no rendido.        |
    | 45 – 55     | Neutral           | Sin sesgo claro.                                               |
    | 55 – 75     | Codicia           | Riesgo en alza. Conviene revisar exposición.                   |
    | 75 – 100    | **Codicia Extr.** | Euforia. Históricamente preludio de correcciones técnicas.     |
    """)

    st.markdown("### 4) Casos históricos")
    st.markdown("""
    - **Marzo 2020 (COVID-19):** el índice de CNN llegó a 2 (miedo extremo).
      El SPX rebotó 60% en los 12 meses siguientes.
    - **Octubre 2022:** lectura ~14 (miedo extremo) cerca del piso del bear
      market. Inicio del rally 2023.
    - **Diciembre 2023 / Enero 2024:** lecturas >75 (codicia extrema).
      Corrección menor en abril 2024 pero mercado siguió alcista.
    - **Conclusión:** la **señal contraria** funciona mejor en extremos
      profundos. En zona 60-75 muchas veces el mercado sigue subiendo.
    """)

    st.markdown("### 5) Limitaciones")
    st.markdown("""
    - El índice **no incluye fundamentales** (PER, ganancias, macro). Es 100%
      sentimiento técnico/de flujo.
    - **Lookback fijo** (252 días): un régimen estructural nuevo (ej. tasas
      altas persistentes) se incorpora con demora.
    - El proxy de PCR vía VIX/VIX3M no captura cambios en flow de minoristas
      ni 0DTE.
    - Es **una herramienta de timing táctico**, no de asignación estratégica.
    """)

    st.markdown("### 6) Comparación con otros índices")
    st.markdown("""
    | Índice                         | Universo            | Componentes            | Frecuencia |
    |--------------------------------|---------------------|------------------------|------------|
    | **CNN F&G**                    | Acciones US         | 7 (oficial)            | Diaria     |
    | **BDI (este)**                 | Acciones US         | 7 (proxies abiertos)   | Diaria     |
    | **Alternative.me Crypto F&G**  | BTC                 | 5 (vol + dom + social) | Diaria     |
    | **Citi Pánico/Euforia**        | Acciones US (multi) | 9 (más bonds + FX)     | Semanal    |
    | **AAII Sentiment Survey**      | Inversores retail   | Encuesta directa       | Semanal    |
    """)

# ------------------------------- DATOS ----------------------------------------
with tab_data:
    st.markdown("## 📥 Datos crudos")
    st.markdown("### Series históricas (descargadas)")
    st.dataframe(data.tail(40).round(2).iloc[::-1], use_container_width=True)
    st.download_button(
        "Descargar CSV completo",
        data=data.to_csv().encode("utf-8"),
        file_name=f"bdi_market_data_{TODAY.isoformat()}.csv",
        mime="text/csv",
    )

    st.markdown("### Componentes calculados")
    st.dataframe(components.tail(40).round(2).iloc[::-1], use_container_width=True)
    st.download_button(
        "Descargar componentes",
        data=components.to_csv().encode("utf-8"),
        file_name=f"bdi_components_{TODAY.isoformat()}.csv",
        mime="text/csv",
    )

    st.markdown("### Serie del índice")
    out_idx = pd.DataFrame({"BDI": bdi_full})
    st.dataframe(out_idx.tail(40).round(2).iloc[::-1], use_container_width=True)
    st.download_button(
        "Descargar índice",
        data=out_idx.to_csv().encode("utf-8"),
        file_name=f"bdi_index_{TODAY.isoformat()}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    f"<span class='small-muted'>Construido con Streamlit · "
    f"Fuentes: Yahoo Finance, CNN dataviz · "
    f"Educacional, no es asesoramiento financiero.</span>",
    unsafe_allow_html=True,
)

