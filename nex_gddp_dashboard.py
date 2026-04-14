"""
NEX-GDDP CMIP6 · Dashboard Interactivo de Temperatura
======================================================
IEEE SciVis Contest 2026

Carga datos reales desde el servidor remoto OpenVisus (atlantis.sci.utah.edu)
usando la biblioteca OpenVisus. Reproduce y extiende las interacciones del
HTML original:
  - Toggle de escenarios (Histórico / SSP2-4.5 / SSP5-8.5)
  - Toggle de modelos climáticos
  - Bandas de incertidumbre inter-modelos
  - Líneas de umbral París (+1.5 °C / +2 °C)
  - Tooltip interactivo al pasar el cursor
  - Estadísticas resumen (tarjetas)
  - Cache local para no re-descargar datos

Instalación de dependencias:
    pip install openvisuspy OpenVisus plotly dash dash-bootstrap-components numpy tqdm

Ejecutar:
    python nex_gddp_dashboard.py
Luego abrir http://127.0.0.1:8050 en el navegador.
"""

# ──────────────────────────────────────────────────────────────────
# 0.  IMPORTACIONES
# ──────────────────────────────────────────────────────────────────
import os, json, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

import OpenVisus as ov

# ──────────────────────────────────────────────────────────────────
# 1.  CONFIGURACIÓN GENERAL
# ──────────────────────────────────────────────────────────────────
CACHE_DIR = Path("./nex_gddp_cache")
CACHE_DIR.mkdir(exist_ok=True)
os.environ["VISUS_CACHE"] = str(CACHE_DIR / "visus_arco")

REMOTE_URL = "http://atlantis.sci.utah.edu/mod_visus?dataset=nex-gddp-cmip6&cached=arco"

# Modelos disponibles (mismos del HTML original + extras)
MODELS = {
    "ACCESS-CM2":   "r1i1p1f1",
    "GFDL-ESM4":    "r1i1p1f1",
    "MRI-ESM2-0":   "r1i1p1f1",
    "IPSL-CM6A-LR": "r1i1p1f1",
}

SCENARIOS = {
    "historical": dict(years=range(1950, 2015), color="#6db3ff", label="Histórico (1950–2014)"),
    "ssp245":     dict(years=range(2015, 2101), color="#38c7a0", label="SSP2-4.5 (2015–2100)"),
    "ssp585":     dict(years=range(2015, 2101), color="#f4714a", label="SSP5-8.5 (2015–2100)"),
}

# Calidad de resolución: 0 = completa, -2 = 4× menos datos (más rápido)
# Para exploración rápida usar quality=-4; para publicación usar quality=0
QUALITY = -4  # cambia a 0 para máxima resolución (más lento)

# Muestreo anual: día del año a usar como representativo
DOY = 182  # ~1 julio, día 182

PREINDUSTRIAL_BASE = 13.82  # °C, referencia 1850-1900

# ──────────────────────────────────────────────────────────────────
# 2.  FUNCIONES DE CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────
def get_field_name(variable: str, model: str, scenario: str, run: str) -> str:
    return f"{variable}_day_{model}_{scenario}_{run}_gn"


def timestep_for(year: int, doy: int) -> int:
    """Convierte año + día del año a índice de tiempo que usa OpenVisus."""
    return year * 365 + doy


def load_global_mean_temperature(
    db, field: str, years: list[int], doy: int, quality: int
) -> list[float]:
    """
    Carga la temperatura media global para cada año dado.
    Usa concurrencia para acelerar las descargas.

    Returns lista de temperaturas en °C (NaN si falla).
    """
    # Primero obtenemos máscara de celdas válidas (no NaN) con un timestep de referencia
    try:
        ref_data = db.read(field=field, time=timestep_for(years[0], doy), quality=quality)
        num_valid = ref_data.size - np.isnan(ref_data).sum()
        if num_valid == 0:
            num_valid = ref_data.size
    except Exception:
        num_valid = 600 * 1440  # fallback

    def fetch_year(year: int) -> float:
        try:
            data = db.read(time=timestep_for(year, doy), field=field, quality=quality)
            val = np.nansum(data) / num_valid - 273.15
            return float(val)
        except Exception:
            return float("nan")

    temps = thread_map(fetch_year, years, desc=f"  {field[:40]}", max_workers=8, leave=False)
    return list(temps)


def cache_path(model: str, scenario: str) -> Path:
    return CACHE_DIR / f"{model}_{scenario}_q{QUALITY}_doy{DOY}.json"


def load_or_fetch(model: str, scenario: str, db) -> dict:
    """
    Carga desde cache JSON si existe, de lo contrario descarga del servidor remoto.

    Returns dict: {"years": [...], "temps": [...]}
    """
    cp = cache_path(model, scenario)
    if cp.exists():
        with open(cp) as f:
            return json.load(f)

    run = MODELS[model]
    field = get_field_name("tas", model, scenario, run)
    years = list(SCENARIOS[scenario]["years"])

    print(f"\n⬇  Descargando {model} / {scenario} ({len(years)} años) ...")
    temps = load_global_mean_temperature(db, field, years, DOY, QUALITY)

    result = {"years": years, "temps": temps}
    with open(cp, "w") as f:
        json.dump(result, f)

    return result


def build_dataset() -> dict:
    """
    Construye el dataset completo para todos los modelos y escenarios.
    Returns nested dict: data[model][scenario] = {"years": [...], "temps": [...]}
    """
    print("Conectando a OpenVisus remoto...")
    db = ov.LoadDataset(REMOTE_URL)
    print(f"✓ Conectado: {db.getUrl()}")

    data = {}
    for model in MODELS:
        data[model] = {}
        for scenario in SCENARIOS:
            data[model][scenario] = load_or_fetch(model, scenario, db)

    return data


# ──────────────────────────────────────────────────────────────────
# 3.  FUNCIONES DE ESTADÍSTICAS Y ENSAMBLE
# ──────────────────────────────────────────────────────────────────
def ensemble_stats(data: dict, active_models: list[str], scenario: str) -> dict:
    """Calcula media, mín y máx del ensamble inter-modelos para un escenario."""
    all_temps = []
    years = None
    for m in active_models:
        if m in data and scenario in data[m]:
            d = data[m][scenario]
            all_temps.append(d["temps"])
            years = d["years"]

    if not all_temps or years is None:
        return {"years": [], "mean": [], "min": [], "max": []}

    arr = np.array(all_temps, dtype=float)
    return {
        "years": years,
        "mean": np.nanmean(arr, axis=0).tolist(),
        "min":  np.nanmin(arr, axis=0).tolist(),
        "max":  np.nanmax(arr, axis=0).tolist(),
    }


def compute_stats_cards(data: dict, active_models: list[str]) -> dict:
    """Calcula las 4 métricas resumen."""
    if not active_models:
        return {"base": None, "hist_delta": None, "ssp245_2100": None, "ssp585_2100": None}

    def mean_at_year(scenario, year):
        vals = []
        for m in active_models:
            d = data.get(m, {}).get(scenario, {})
            years = d.get("years", [])
            temps = d.get("temps", [])
            if year in years:
                idx = years.index(year)
                v = temps[idx]
                if not np.isnan(v):
                    vals.append(v)
        return np.mean(vals) if vals else None

    base = mean_at_year("historical", 1950)
    hist2014 = mean_at_year("historical", 2014)
    s245_2100 = mean_at_year("ssp245", 2100)
    s585_2100 = mean_at_year("ssp585", 2100)

    return {
        "base":       base,
        "hist_delta": (hist2014 - base) if (base and hist2014) else None,
        "ssp245_2100": s245_2100,
        "ssp585_2100": s585_2100,
    }


# ──────────────────────────────────────────────────────────────────
# 3b. FUNCIONES DE MAPA MUNDIAL DE HUMEDAD (hurs)
# ──────────────────────────────────────────────────────────────────
# El grid NEX-GDDP tiene forma (600, 1440):
#   600 filas  → latitudes de -90° a +90°  (paso 0.3°)
#   1440 cols  → longitudes de 0° a 360°   (paso 0.25°)
# Se pre-calculan coordenadas geográficas para el mapa Plotly.

_HURS_LATS = np.linspace(-90.0, 90.0,  600)   # centro de cada celda
_HURS_LONS = np.linspace(  0.0, 359.75, 1440)  # centro de cada celda (0–360)
# Para Plotly usamos longitudes en -180…180
_HURS_LONS_180 = np.where(_HURS_LONS > 180, _HURS_LONS - 360, _HURS_LONS)

# Factor de submuestreo para el mapa: cada N filas/cols → punto en el mapa.
# Con quality=-4 la resolución ya es reducida; tomamos cada 4 celdas para
# mantener el mapa fluido sin perder legibilidad.
_MAP_STEP = 4
_MAP_LATS = _HURS_LATS[::_MAP_STEP]
_MAP_LONS = _HURS_LONS_180[::_MAP_STEP]


def hurs_map_cache_path(model: str, scenario: str, year: int) -> Path:
    return CACHE_DIR / f"hurs_map_{model}_{scenario}_y{year}_q{QUALITY}_doy{DOY}.json"


def load_hurs_map(model: str, scenario: str, year: int) -> np.ndarray:
    """
    Descarga o lee de caché la cuadrícula 2-D de humedad (float32, NaN donde
    sin dato) para un modelo, escenario y año dados.

    El grid NEX-GDDP viene con longitudes 0→360. Lo reorganizamos a -180→180
    con np.roll para que el mapa quede centrado en el meridiano 0°.

    Retorna ndarray shape (n_lat, n_lon) con latitudes S→N y lons -180→180.
    Los valores ya están en % (hurs viene en %).
    """
    cp = hurs_map_cache_path(model, scenario, year)
    if cp.exists():
        with open(cp) as f:
            raw = json.load(f)
        # Reconstruir ndarray; None → NaN
        arr = np.array(
            [[np.nan if v is None else v for v in row] for row in raw],
            dtype=np.float32,
        )
        return arr

    db    = ov.LoadDataset(REMOTE_URL)
    run   = MODELS[model]
    field = get_field_name("hurs", model, scenario, run)
    try:
        data = db.read(field=field, time=timestep_for(year, DOY), quality=QUALITY)
        # data shape: (600, 1440), lons 0→360
        # Submuestrear cada _MAP_STEP celdas
        sub = data[::_MAP_STEP, ::_MAP_STEP].astype(np.float32)
        # Roll: desplazar columnas n_lon//2 posiciones → lons -180→180
        n_lon = sub.shape[1]
        sub   = np.roll(sub, n_lon // 2, axis=1)
    except Exception as exc:
        print(f"  [hurs_map] error {model}/{scenario}/{year}: {exc}")
        sub = np.full((len(_MAP_LATS), len(_MAP_LONS)), np.nan, dtype=np.float32)

    # Serializar: NaN → None para JSON
    result = [
        [None if np.isnan(v) else round(float(v), 2) for v in row]
        for row in sub
    ]
    with open(cp, "w") as f:
        json.dump(result, f)
    return sub

def load_global_mean_hurs(model: str, scenario: str) -> dict:
    """Carga o calcula la humedad relativa media global por año."""
    cache_key = CACHE_DIR / f"hurs_global_{model}_{scenario}_q{QUALITY}_doy{DOY}.json"
    if cache_key.exists():
        with open(cache_key) as f:
            return json.load(f)

    db = ov.LoadDataset(REMOTE_URL)
    run = MODELS[model]
    field = get_field_name("hurs", model, scenario, run)
    years = list(SCENARIOS[scenario]["years"])

    try:
        ref = db.read(field=field, time=timestep_for(years[0], DOY), quality=QUALITY)
        num_valid = ref.size - np.isnan(ref).sum()
        if num_valid == 0:
            num_valid = ref.size
    except Exception:
        num_valid = 600 * 1440

    def fetch_year_hurs(year: int) -> float:
        try:
            data = db.read(time=timestep_for(year, DOY), field=field, quality=QUALITY)
            return float(np.nansum(data) / num_valid)
        except Exception:
            return float("nan")

    print(f"\n⬇ Humedad global {model} / {scenario} ({len(years)} años) ...")
    vals = thread_map(fetch_year_hurs, years, max_workers=8, leave=False)
    result = {"years": years, "hurs": vals}
    with open(cache_key, "w") as f:
        json.dump(result, f)
    return result


def ensemble_hurs_stats(active_models: list[str], scenario: str) -> dict:
    """Calcula media, mín y máx del ensamble de humedad para un escenario."""
    all_hurs = []
    years = None
    for m in active_models:
        try:
            d = load_global_mean_hurs(m, scenario)
            all_hurs.append(d["hurs"])
            years = d["years"]
        except Exception as e:
            print(f"[hurs_trend] ERROR {m}/{scenario}: {e}")
            continue
    if not all_hurs or years is None:
        return {"years": [], "mean": [], "min": [], "max": []}
    arr = np.array(all_hurs, dtype=float)
    return {
        "years": years,
        "mean": np.nanmean(arr, axis=0).tolist(),
        "min":  np.nanmin(arr,  axis=0).tolist(),
        "max":  np.nanmax(arr,  axis=0).tolist(),
    }


def build_hurs_trend_figure(
    active_models: list[str],
    active_scens: list[str],
    show_bands: bool,
) -> go.Figure:
    """Líneas de evolución de humedad relativa global promedio 1950–2100."""
    fig = go.Figure()

    scen_cfg = {
        "historical": ("Humedad histórica",  "#c084fc", "solid"),
        "ssp245":     ("Humedad SSP2-4.5",   "#a855f7", "solid"),
        "ssp585":     ("Humedad SSP5-8.5",   "#7e22ce", "solid"),
    }

    # Divisor histórico / proyección
    fig.add_vline(
        x=2015, line_dash="dot", line_color="rgba(255,255,255,0.15)",
        annotation_text="◀ HISTÓRICO | PROYECCIÓN ▶",
        annotation_font_size=9, annotation_font_color="rgba(255,255,255,0.3)",
        annotation_position="top",
    )

    # Bandas de incertidumbre
    if show_bands and len(active_models) > 1:
        for scen, fill_color in [
            ("ssp245", "rgba(168,85,247,0.10)"),
            ("ssp585", "rgba(126,34,206,0.10)"),
        ]:
            if scen not in active_scens:
                continue
            ens = ensemble_hurs_stats(active_models, scen)
            if not ens["years"]:
                continue
            fig.add_trace(go.Scatter(
                x=ens["years"] + ens["years"][::-1],
                y=ens["max"] + ens["min"][::-1],
                fill="toself", fillcolor=fill_color,
                line=dict(width=0),
                name=f"Rango inter-modelos {scen}",
                hoverinfo="skip", showlegend=True,
            ))

    # Líneas de ensamble
    for scen in active_scens:
        ens = ensemble_hurs_stats(active_models, scen)
        if not ens["years"]:
            continue
        label, color, dash = scen_cfg[scen]
        fig.add_trace(go.Scatter(
            x=ens["years"], y=ens["mean"],
            mode="lines",
            line=dict(color=color, width=2.8, dash=dash),
            name=label,
            hovertemplate=f"{label}<br>Año: %{{x}}<br>Humedad: %{{y:.2f}}%<extra></extra>",
        ))

    # Franja zona histórica
    fig.add_vrect(
        x0=1950, x1=2015,
        fillcolor="rgba(192,132,252,0.03)",
        line_width=0, layer="below",
    )

    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0d1526",
        font=dict(family="Space Mono, monospace", color="#c8d8f0", size=11),
        xaxis=dict(
            title="Año", range=[1950, 2100],
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#1a2a4a", tickfont=dict(size=10), dtick=10,
        ),
        yaxis=dict(
            title="Humedad relativa media global (%)",
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#1a2a4a", tickfont=dict(size=10), ticksuffix="%",
        ),
        legend=dict(
            bgcolor="rgba(13,21,38,0.85)", bordercolor="#1a2a4a", borderwidth=1,
            font=dict(size=10), orientation="h",
            yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        hovermode="x unified",
        margin=dict(l=70, r=80, t=60, b=60),
        height=460,
        dragmode="pan",
    )
    return fig

def build_hurs_map_figure(model: str, scenario: str, year: int) -> go.Figure:
    """
    Mapa mundial de humedad relativa.

    • go.Heatmap con zsmooth="best" rellena el grid de forma continua.
    • Se aplica un filtro gaussiano para suavizar los bordes pixelados,
      preservando los NaN del océano (máscara tierra/océano intacta).
    • Sin fronteras ni líneas de países — solo la paleta de colores.
    """
    arr = load_hurs_map(model, scenario, year)  # ndarray (n_lat, n_lon)

    # ── Antialiasing de bordes tierra/océano ─────────────────────
    # Técnica "normalized convolution" (feathering):
    #   1. Máscara binaria tierra (1) / océano-NaN (0).
    #   2. Se suavizan TANTO los valores como la máscara con el mismo kernel.
    #   3. Dividir valores_suavizados / máscara_suavizada → promedio ponderado
    #      que solo usa vecinos de tierra: el interior no cambia, los bordes
    #      se difuminan gradualmente hacia NaN en lugar de cortar en duro.
    #   4. Celdas con peso < umbral → NaN (océano limpio sin artefactos).
    try:
        from scipy.ndimage import gaussian_filter

        mask_land  = (~np.isnan(arr)).astype(np.float64)      # 1=tierra, 0=océano
        arr_f64    = np.where(mask_land.astype(bool), arr.astype(np.float64), 0.0)

        # ── Paso 1: suavizado leve del interior ──────────────────
        sigma_int  = 0.9
        sv_int     = gaussian_filter(arr_f64,   sigma=sigma_int)
        sm_int     = gaussian_filter(mask_land, sigma=sigma_int)
        with np.errstate(invalid="ignore", divide="ignore"):
            arr_int = np.where(sm_int > 0.5,
                               sv_int / sm_int, np.nan).astype(np.float32)

        # ── Paso 2: feathering fuerte en los bordes ───────────────
        sigma_edge = 2.0
        sv_edge    = gaussian_filter(arr_f64,   sigma=sigma_edge)
        sm_edge    = gaussian_filter(mask_land, sigma=sigma_edge)
        with np.errstate(invalid="ignore", divide="ignore"):
            arr_edge = np.where(sm_edge > 0.05,
                                sv_edge / sm_edge, np.nan).astype(np.float32)

        # ── Combinar: interior usa suavizado leve, borde usa feathering ──
        interior = mask_land.astype(bool)
        arr = np.where(interior, arr_int, arr_edge)

    except ImportError:
        pass  # scipy no disponible → array original sin suavizado

    # ── Coordenadas derivadas del tamaño real del array ───────────
    n_lat, n_lon = arr.shape
    lats_arr = np.linspace(-90.0,  90.0, n_lat)
    lons_180 = np.linspace(-180.0, 180.0, n_lon, endpoint=False)

    colorscale = [
        [0.00, "#f4714a"],
        [0.25, "#f4d44a"],
        [0.50, "#38c7a0"],
        [0.75, "#6db3ff"],
        [1.00, "#c084fc"],
    ]

    scen_label = SCENARIOS[scenario]["label"]

    # ── Dimensiones fijas del figure ──────────────────────────────
    FIG_H = 520
    ML, MR, MT, MB = 55, 90, 75, 40

    fig = go.Figure()

    # ── Heatmap continuo sin fronteras ────────────────────────────
    fig.add_trace(go.Heatmap(
        z=arr.tolist(),
        x=lons_180.tolist(),
        y=lats_arr.tolist(),
        zmin=0,
        zmax=100,
        colorscale=colorscale,
        zsmooth="best",
        connectgaps=False,
        colorbar=dict(
            title=dict(
                text="HR (%)",
                font=dict(family="Space Mono", color="#c8d8f0", size=10),
            ),
            tickfont=dict(family="Space Mono", color="#c8d8f0", size=9),
            ticksuffix="%",
            thickness=14,
            len=0.85,
            bgcolor="rgba(13,21,38,0.75)",
            bordercolor="#1a2a4a",
            borderwidth=1,
            outlinewidth=0,
            x=1.01,
        ),
        hovertemplate="Lat: %{y:.1f}°  Lon: %{x:.1f}°<br>Humedad: %{z:.1f}%<extra></extra>",
        name="",
    ))

    fig.update_layout(
        # ── Eje X (longitud) ──────────────────────────────────────
        xaxis=dict(
            range=[-180, 180],
            showgrid=False,
            zeroline=False,
            tickfont=dict(family="Space Mono", size=8, color="#5a7099"),
            tickvals=list(range(-150, 181, 30)),
            ticktext=[
                f"{abs(v)}°{'O' if v < 0 else ('E' if v > 0 else '')}"
                for v in range(-150, 181, 30)
            ],
            linecolor="#1a2a4a",
            domain=[0, 1],
        ),
        # ── Eje Y (latitud) ───────────────────────────────────────
        yaxis=dict(
            range=[-90, 90],
            showgrid=False,
            zeroline=False,
            tickfont=dict(family="Space Mono", size=8, color="#5a7099"),
            tickvals=list(range(-90, 91, 30)),
            ticktext=[
                f"{abs(v)}°{'S' if v < 0 else ('N' if v > 0 else '')}"
                for v in range(-90, 91, 30)
            ],
            linecolor="#1a2a4a",
        ),
        # ── Estilo general ────────────────────────────────────────
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        font=dict(family="Space Mono, monospace", color="#c8d8f0", size=11),
        
        margin=dict(l=ML, r=MR, t=20, b=MB),  # MT=75 → t=20
        height=FIG_H,
    )

    return fig


# ──────────────────────────────────────────────────────────────────
# 4.  FUNCIÓN DE CONSTRUCCIÓN DEL GRÁFICO PLOTLY
# ──────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "ACCESS-CM2":   {"historical": "#5b9cf6", "ssp245": "#2dd4a8", "ssp585": "#fb7a58"},
    "GFDL-ESM4":    {"historical": "#3b7bd4", "ssp245": "#20b890", "ssp585": "#e05a3a"},
    "MRI-ESM2-0":   {"historical": "#7ab2ff", "ssp245": "#5de8c4", "ssp585": "#ff9878"},
    "IPSL-CM6A-LR": {"historical": "#2a60b0", "ssp245": "#17a07e", "ssp585": "#c94422"},
}


def build_figure(
    data: dict,
    active_models: list[str],
    active_scens: list[str],
    show_bands: bool,
    show_thresh: bool,
) -> go.Figure:
    fig = go.Figure()

    # ── Divisor histórico / proyección ──────────────────────────
    fig.add_vline(
        x=2015,
        line_dash="dot",
        line_color="rgba(255,255,255,0.15)",
        annotation_text="◀ HISTÓRICO | PROYECCIÓN ▶",
        annotation_font_size=9,
        annotation_font_color="rgba(255,255,255,0.3)",
        annotation_position="top",
    )

    # ── Bandas de incertidumbre ──────────────────────────────────
    if show_bands and len(active_models) > 1:
        for scen, fill_color, name_band in [
            ("ssp245", "rgba(56,199,160,0.12)",  "Rango inter-modelos SSP2-4.5"),
            ("ssp585", "rgba(244,113,74,0.12)",   "Rango inter-modelos SSP5-8.5"),
        ]:
            if scen not in active_scens:
                continue
            ens = ensemble_stats(data, active_models, scen)
            if not ens["years"]:
                continue
            # Banda (área rellena)
            fig.add_trace(go.Scatter(
                x=ens["years"] + ens["years"][::-1],
                y=ens["max"] + ens["min"][::-1],
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                name=name_band,
                hoverinfo="skip",
                legendgroup=f"band_{scen}",
                showlegend=True,
            ))

    # ── Líneas individuales por modelo ───────────────────────────
    for model in active_models:
        if model not in data:
            continue
        mc = MODEL_COLORS.get(model, {})
        for scen in active_scens:
            if scen not in data[model]:
                continue
            d = data[model][scen]
            color = mc.get(scen, "#aaaaaa")
            fig.add_trace(go.Scatter(
                x=d["years"],
                y=d["temps"],
                mode="lines",
                line=dict(color=color, width=1.2),
                opacity=0.55,
                name=f"{model} / {scen}",
                legendgroup=f"model_{model}_{scen}",
                showlegend=False,
                hovertemplate=f"<b>{model}</b><br>Año: %{{x}}<br>T: %{{y:.2f}}°C<extra></extra>",
            ))

    # ── Líneas de ensamble (media) ───────────────────────────────
    scen_cfg = {
        "historical": ("Ensamble histórico",   "#6db3ff", "solid"),
        "ssp245":     ("Ensamble SSP2-4.5",    "#38c7a0", "solid"),
        "ssp585":     ("Ensamble SSP5-8.5",    "#f4714a", "solid"),
    }
    for scen in active_scens:
        ens = ensemble_stats(data, active_models, scen)
        if not ens["years"]:
            continue
        label, color, dash = scen_cfg[scen]
        fig.add_trace(go.Scatter(
            x=ens["years"],
            y=ens["mean"],
            mode="lines",
            line=dict(color=color, width=2.8, dash=dash),
            name=label,
            legendgroup=f"ens_{scen}",
            hovertemplate=f"<b>{label}</b><br>Año: %{{x}}<br>T media: %{{y:.2f}}°C<extra></extra>",
        ))

    # ── Líneas de umbral París ───────────────────────────────────
    if show_thresh:
        for delta, color, label in [
            (1.5, "rgba(255,220,80,0.7)",  "+1.5 °C (París)"),
            (2.0, "rgba(255,110,80,0.7)",  "+2.0 °C (París)"),
        ]:
            thresh = PREINDUSTRIAL_BASE + delta
            fig.add_hline(
                y=thresh,
                line_dash="dash",
                line_color=color,
                line_width=1.5,
                annotation_text=label,
                annotation_font_size=9,
                annotation_font_color=color,
                annotation_position="right",
            )

    # ── Diseño ───────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0d1526",
        font=dict(family="Space Mono, monospace", color="#c8d8f0", size=11),
        xaxis=dict(
            title="Año",
            range=[1950, 2100],
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#1a2a4a",
            tickfont=dict(size=10),
            dtick=10,
        ),
        yaxis=dict(
            title="Temperatura media global (°C)",
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#1a2a4a",
            tickfont=dict(size=10),
            ticksuffix="°C",
        ),
        legend=dict(
            bgcolor="rgba(13,21,38,0.85)",
            bordercolor="#1a2a4a",
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
        margin=dict(l=70, r=80, t=60, b=60),
        height=500,
        dragmode="pan",
    )

    # Franja de fondo para resaltar zona histórica
    fig.add_vrect(
        x0=1950, x1=2015,
        fillcolor="rgba(107,179,255,0.03)",
        line_width=0,
        layer="below",
    )

    return fig

#GRAFICA MAR PARTE 1: Tasa de calentamiento por década
def build_warming_rate_figure(
    data: dict,
    active_models: list[str],
    active_scens: list[str],
) -> go.Figure:
    """
    Gráfico de barras agrupadas: tasa de calentamiento por década (°C/10 años).
    Para cada escenario activo calcula la media del ensamble en cada década
    y computa el delta respecto a la década anterior.
    """
    DECADE_SCEN_CFG = {
        "historical": ("#6db3ff", "Histórico"),
        "ssp245":     ("#38c7a0", "SSP2-4.5"),
        "ssp585":     ("#f4714a", "SSP5-8.5"),
    }

    fig = go.Figure()

    for scen in active_scens:
        if scen not in DECADE_SCEN_CFG:
            continue
        ens = ensemble_stats(data, active_models, scen)
        if not ens["years"]:
            continue

        years = np.array(ens["years"])
        temps = np.array(ens["mean"])

        # Agrupar en décadas y calcular media por década
        year_min = int(years[0])
        year_max = int(years[-1])
        decade_starts = list(range((year_min // 10) * 10, year_max, 10))

        dec_labels = []
        dec_means  = []
        for d in decade_starts:
            mask = (years >= d) & (years < d + 10)
            if mask.sum() > 0:
                dec_labels.append(f"{d}s")
                dec_means.append(float(np.nanmean(temps[mask])))

        if len(dec_means) < 2:
            continue

        # Tasa = diferencia entre décadas consecutivas
        rates = [dec_means[i] - dec_means[i - 1] for i in range(1, len(dec_means))]
        labels = dec_labels[1:]  # la etiqueta de la década destino

        color_base, scen_name = DECADE_SCEN_CFG[scen]

        # Color por intensidad: positivo usa el color del escenario, negativo gris
        bar_colors = [
            color_base if r >= 0 else "#5a7099"
            for r in rates
        ]
        # Opacidad proporcional a la magnitud
        max_abs = max(abs(r) for r in rates) if rates else 1.0

        fig.add_trace(go.Bar(
            name=scen_name,
            x=labels,
            y=rates,
            marker=dict(
                color=bar_colors,
                opacity=0.85,
                line=dict(width=0),
            ),
            hovertemplate=(
                f"<b>{scen_name}</b><br>"
                "Década: %{x}<br>"
                "Calentamiento: %{y:+.3f} °C<extra></extra>"
            ),
            legendgroup=f"rate_{scen}",
        ))

    # Línea de cero
    fig.add_hline(
        y=0,
        line_color="rgba(255,255,255,0.15)",
        line_width=1,
    )

    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0d1526",
        font=dict(family="Space Mono, monospace", color="#c8d8f0", size=11),
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        xaxis=dict(
            title="Década",
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#1a2a4a",
            tickfont=dict(size=10),
            tickangle=-35,
        ),
        yaxis=dict(
            title="Calentamiento por década (°C)",
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#1a2a4a",
            tickfont=dict(size=10),
            ticksuffix="°C",
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(13,21,38,0.85)",
            bordercolor="#1a2a4a",
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
        margin=dict(l=70, r=80, t=60, b=70),
        height=420,
    )

    return fig
def build_anomaly_figure(
    data: dict,
    active_models: list[str],
    active_scens: list[str],
) -> go.Figure:
    """Heatmap de anomalía de temperatura por año y escenario."""
    BASELINE_START, BASELINE_END = 1950, 1980

    scen_labels = {
        "historical": "Histórico",
        "ssp245":     "SSP2-4.5",
        "ssp585":     "SSP5-8.5",
    }

    hist_ens = ensemble_stats(data, active_models, "historical")
    if hist_ens["years"]:
        hist_years = np.array(hist_ens["years"])
        hist_means = np.array(hist_ens["mean"])
        mask = (hist_years >= BASELINE_START) & (hist_years <= BASELINE_END)
        baseline = float(np.nanmean(hist_means[mask])) if mask.any() else 0.0
    else:
        baseline = 0.0

    z_matrix = []
    y_labels  = []

    for scen in ["historical", "ssp245", "ssp585"]:
        if scen not in active_scens:
            continue
        ens = ensemble_stats(data, active_models, scen)
        if not ens["years"]:
            continue

        years   = np.array(ens["years"])
        anomaly = np.array(ens["mean"]) - baseline

        full_years   = np.arange(1950, 2101)
        full_anomaly = np.full(len(full_years), np.nan)
        for i, y in enumerate(years):
            if 1950 <= y <= 2100:
                full_anomaly[y - 1950] = anomaly[i]

        z_matrix.append(full_anomaly.tolist())
        y_labels.append(scen_labels[scen])

    if not z_matrix:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=list(range(1950, 2101)),
        y=y_labels,
        colorscale=[
            [0.00, "#0d47a1"],
            [0.25, "#42a5f5"],
            [0.45, "#e3f2fd"],
            [0.5,  "#ffffff"],
            [0.55, "#fff9c4"],
            [0.75, "#ff7043"],
            [1.00, "#b71c1c"],
        ],
        zmid=0,
        zmin=-1,
        zmax=6,
        colorbar=dict(
            title=dict(text="Anomalía (°C)", font=dict(size=11, color="#c8d8f0")),
            ticksuffix="°C",
            tickfont=dict(color="#c8d8f0", size=10),
            outlinecolor="#1a2a4a",
            outlinewidth=1,
            len=0.8,
        ),
        hovertemplate="Año: %{x}<br>Escenario: %{y}<br>Anomalía: %{z:+.2f}°C<extra></extra>",
        xgap=0.5,
        ygap=3,
    ))

    fig.add_vline(
        x=2015, line_dash="dot", line_color="rgba(255,255,255,0.4)", line_width=1.5,
        annotation_text="◀ HISTÓRICO | PROYECCIÓN ▶",
        annotation_font_size=9, annotation_font_color="rgba(255,255,255,0.4)",
        annotation_position="top",
    )

    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0d1526",
        font=dict(family="Space Mono, monospace", color="#c8d8f0", size=11),
        title=dict(
            text="Mapa de Calor · Anomalía de Temperatura 1950–2100 (base: 1950–1980)",
            font=dict(size=13, color="#e8f2ff", family="Syne, sans-serif"),
            x=0.01, y=0.97, yanchor="top",
        ),
        xaxis=dict(
            title="Año",
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#1a2a4a",
            tickfont=dict(size=10),
            dtick=10,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#c8d8f0"),
            linecolor="#1a2a4a",
        ),
        margin=dict(l=90, r=100, t=80, b=60),
        height=320,
    )
    return fig
# ──────────────────────────────────────────────────────────────────
# 5.  APP DASH
# ──────────────────────────────────────────────────────────────────
DARK = "#080c14"
SURFACE = "#0d1526"
BORDER = "#1a2a4a"
TEXT = "#c8d8f0"
MUTED = "#5a7099"
ACCENT = "#6db3ff"

def make_toggle_btn(label: str, id: str, color: str, active: bool = True) -> dbc.Button:
    return dbc.Button(
        label,
        id=id,
        n_clicks=0,
        size="sm",
        style={
            "fontFamily": "Space Mono, monospace",
            "fontSize": "10px",
            "borderRadius": "2px",
            "padding": "5px 12px",
            "background": color if active else "transparent",
            "borderColor": color,
            "color": "#000" if active else MUTED,
            "marginRight": "6px",
            "transition": "all 0.15s",
        },
    )


def stat_card(title: str, value_id: str, sub: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, style={"fontSize": "8px", "letterSpacing": "0.15em",
                                 "textTransform": "uppercase", "color": MUTED, "marginBottom": "4px"}),
            html.H3("–", id=value_id, style={"fontFamily": "Syne, sans-serif",
                                              "fontWeight": "800", "color": color, "marginBottom": "2px"}),
            html.P(sub, style={"fontSize": "9px", "color": MUTED}),
        ], style={"padding": "14px 18px"}),
        style={"background": SURFACE, "border": f"1px solid {BORDER}", "borderRadius": "3px"},
    )


# ── Carga de datos al iniciar ────────────────────────────────────
print("\n" + "═"*60)
print("  NEX-GDDP CMIP6 · Dashboard · IEEE SciVis 2026")
print("═"*60)
DATA = build_dataset()
print("\n✓ Dataset listo.\n")

# Pre-cargar el mapa de humedad del año inicial para el primer render
print("Pre-cargando mapa de humedad inicial (1980, ACCESS-CM2, historical)...")
_HURS_INITIAL_FIG = build_hurs_map_figure("ACCESS-CM2", "historical", 1980)
print("✓ Mapa de humedad inicial listo.\n")

# ── Layout ───────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap",
    ],
    title="NEX-GDDP CMIP6 · Temperatura Global",
)

model_buttons = dbc.ButtonGroup(
    [
        dbc.Button(
            m,
            id=f"btn-model-{m}",
            n_clicks=1,  # 1 = activo
            size="sm",
            color="primary",
            outline=False,
            style={
                "fontFamily": "Space Mono, monospace",
                "fontSize": "10px",
                "borderRadius": "2px",
                "padding": "5px 12px",
                "marginRight": "4px",
                "background": ACCENT,
                "borderColor": ACCENT,
                "color": DARK,
            },
        )
        for m in MODELS
    ],
    style={"flexWrap": "wrap", "gap": "4px"},
)

app.layout = dbc.Container(
    [
        # ── Encabezado ──────────────────────────────────────────
        html.Div(
            [
                html.P(
                    "IEEE SciVis Contest 2026 · NEX-GDDP CMIP6",
                    style={"fontFamily": "Space Mono", "fontSize": "10px", "letterSpacing": "0.2em",
                           "color": ACCENT, "textTransform": "uppercase", "marginBottom": "6px"},
                ),
                html.H1(
                    ["Temperatura del Aire ", html.Span("Superficial Global", style={"color": ACCENT})],
                    style={"fontFamily": "Syne, sans-serif", "fontWeight": "800",
                           "fontSize": "clamp(22px,4vw,38px)", "color": "#e8f2ff",
                           "letterSpacing": "-0.02em", "lineHeight": "1.1"},
                ),
                html.P(
                    "Temperatura media near-surface (tas) en °C · Modelos CMIP6 · "
                    "Período histórico 1950–2014 y proyecciones 2015–2100 bajo SSPs. "
                    "Datos reales: NASA NEX-GDDP-CMIP6 vía OpenVisus / atlantis.sci.utah.edu.",
                    style={"fontSize": "11px", "color": MUTED, "lineHeight": "1.7",
                           "maxWidth": "680px", "marginTop": "6px"},
                ),
            ],
            style={"marginBottom": "28px"},
        ),

        # ── Controles ───────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Span("Escenario", style={"fontSize": "9px", "color": MUTED,
                                                       "letterSpacing": "0.15em", "textTransform": "uppercase",
                                                       "marginRight": "8px"}),
                        dbc.Checklist(
                            options=[
                                {"label": "Histórico",  "value": "historical"},
                                {"label": "SSP2-4.5",   "value": "ssp245"},
                                {"label": "SSP5-8.5",   "value": "ssp585"},
                            ],
                            value=["historical", "ssp245", "ssp585"],
                            id="scen-checklist",
                            inline=True,
                            switch=False,
                            style={"fontFamily": "Space Mono", "fontSize": "11px", "color": TEXT},
                            input_checked_style={"backgroundColor": ACCENT, "borderColor": ACCENT},
                            label_checked_style={"color": ACCENT},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.Span("Modelos", style={"fontSize": "9px", "color": MUTED,
                                                      "letterSpacing": "0.15em", "textTransform": "uppercase",
                                                      "marginRight": "8px"}),
                        dbc.Checklist(
                            options=[{"label": m, "value": m} for m in MODELS],
                            value=list(MODELS.keys()),
                            id="model-checklist",
                            inline=True,
                            style={"fontFamily": "Space Mono", "fontSize": "11px", "color": TEXT},
                            input_checked_style={"backgroundColor": ACCENT, "borderColor": ACCENT},
                            label_checked_style={"color": ACCENT},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        dbc.Checklist(
                            options=[
                                {"label": "Bandas de incertidumbre", "value": "bands"},
                                {"label": "Umbrales +1.5°C / +2°C",  "value": "thresh"},
                            ],
                            value=["bands", "thresh"],
                            id="opts-checklist",
                            inline=True,
                            style={"fontFamily": "Space Mono", "fontSize": "11px", "color": TEXT},
                            input_checked_style={"backgroundColor": ACCENT, "borderColor": ACCENT},
                            label_checked_style={"color": ACCENT},
                        ),
                    ],
                    width="auto",
                ),
            ],
            align="center",
            style={"marginBottom": "16px", "gap": "16px", "rowGap": "10px",
                   "flexWrap": "wrap", "background": SURFACE,
                   "border": f"1px solid {BORDER}", "borderRadius": "4px",
                   "padding": "12px 16px"},
            class_name="g-2",
        ),
        
            # ── Gráfico principal ────────────────────────────────────
            dbc.Card(
                dcc.Graph(
                    id="main-chart",
                    config={"scrollZoom": True, "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "displaylogo": False},
                    style={"height": "500px"},
                ),
                style={
                    "background": SURFACE,
                    "border": f"1px solid {BORDER}",
                    "borderRadius": "4px",
                    "overflow": "hidden",
                    "borderTop": f"2px solid {ACCENT}",
                },
            ),  

        # ── Tarjetas de estadísticas ─────────────────────────────
        dbc.Row(
            [
                dbc.Col(stat_card("Temperatura base (1950)", "stat-base",
                                  "Media multi-modelo global", "#6db3ff"), md=3),
                dbc.Col(stat_card("Calentamiento hist. (2014)", "stat-hist",
                                  "Respecto a 1950", TEXT), md=3),
                dbc.Col(stat_card("Proyección SSP2-4.5 (2100)", "stat-245",
                                  "Escenario moderado", "#38c7a0"), md=3),
                dbc.Col(stat_card("Proyección SSP5-8.5 (2100)", "stat-585",
                                  "Escenario de altas emisiones", "#f4714a"), md=3),
            ],
            style={"marginTop": "16px"},
            class_name="g-2",
        ),

        # ── Nota de calidad y fuente ─────────────────────────────
        html.P(
            [
                f"Variable: tas (Near-Surface Air Temperature, K→°C) · "
                f"Resolución descargada: quality={QUALITY} · "
                f"Día del año muestreado: {DOY} (~1 julio) · ",
                html.Br(),
                "Fuente: NASA NEX-GDDP-CMIP6 · Acceso: OpenVisus atlantis.sci.utah.edu · "
                "SciVis Contest 2026",
            ],
            style={"marginTop": "20px", "fontSize": "9px", "color": MUTED,
                   "textAlign": "center", "letterSpacing": "0.06em"},
        ),
        
        # ── Separador visual ─────────────────────────────────────
        html.Hr(style={"borderColor": BORDER, "marginTop": "40px", "marginBottom": "36px"}),

        # ── Encabezado sección Tendencia de Humedad ──────────────
        html.Div([
            html.P(
                "IEEE SciVis Contest 2026 · NEX-GDDP CMIP6 · Variable: hurs",
                style={"fontFamily": "Space Mono", "fontSize": "10px",
                    "letterSpacing": "0.2em", "color": "#c084fc",
                    "textTransform": "uppercase", "marginBottom": "6px"},
            ),
            html.H2(
                ["Tendencia de Humedad ", html.Span("Relativa Global 1950–2100", style={"color": "#c084fc"})],
                style={"fontFamily": "Syne, sans-serif", "fontWeight": "800",
                    "fontSize": "clamp(18px,3vw,30px)", "color": "#e8f2ff",
                    "letterSpacing": "-0.02em", "lineHeight": "1.1"},
            ),
            html.P(
                "Evolución de la humedad relativa near-surface (hurs, %) promediada globalmente · "
                "Compara el período histórico con las proyecciones SSP2-4.5 y SSP5-8.5 · "
                "Las bandas muestran el rango de incertidumbre entre modelos.",
                style={"fontSize": "11px", "color": MUTED, "lineHeight": "1.7",
                    "maxWidth": "680px", "marginTop": "6px"},
            ),
        ], style={"marginBottom": "16px"}),

        # ── Gráfico de tendencia de humedad ──────────────────────
        dbc.Card(
            dcc.Graph(
                id="hurs-trend-chart",
                config={"scrollZoom": True, "displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "displaylogo": False},
                style={"height": "460px"},
            ),
            style={
                "background": SURFACE,
                "border": f"1px solid {BORDER}",
                "borderRadius": "4px",
                "overflow": "hidden",
                "borderTop": "2px solid #c084fc",
            },
        ),
        html.P(
            [
                f"Variable: hurs (Near-Surface Relative Humidity, %) · "
                f"Media global del ensamble de modelos activos · ",
                html.Br(),
                "Fuente: NASA NEX-GDDP-CMIP6 · SciVis Contest 2026",
            ],
            style={"marginTop": "12px", "fontSize": "9px", "color": MUTED,
                "textAlign": "center", "letterSpacing": "0.06em",
                "marginBottom": "8px"},
        ),

        # ── Separador visual ────────────────────────────────────────
        html.Hr(style={"borderColor": BORDER, "marginTop": "36px", "marginBottom": "32px"}),

        # ── Encabezado sección Anomalía ──────────────────────────────
        html.Div([
            html.P(
                "IEEE SciVis Contest 2026 · NEX-GDDP CMIP6 · Anomalía de Temperatura",
                style={"fontFamily": "Space Mono", "fontSize": "10px",
                    "letterSpacing": "0.2em", "color": "#f4714a",
                    "textTransform": "uppercase", "marginBottom": "6px"},
            ),
            html.H2(
                ["Mapa de Calor · ", html.Span("Anomalía Térmica 1950–2100", style={"color": "#f4714a"})],
                style={"fontFamily": "Syne, sans-serif", "fontWeight": "800",
                    "fontSize": "clamp(18px,3vw,30px)", "color": "#e8f2ff",
                    "letterSpacing": "-0.02em", "lineHeight": "1.1"},
            ),
            html.P(
                "Diferencia respecto a la temperatura media del período 1950–1980 · "
                "Azul = más frío que la referencia · Blanco = igual a la referencia · "
                "Rojo = más caliente · Las líneas de umbral del Acuerdo de París son +1.5°C y +2.0°C.",
                style={"fontSize": "11px", "color": MUTED, "lineHeight": "1.7",
                    "maxWidth": "680px", "marginTop": "6px"},
            ),
        ], style={"marginBottom": "16px"}),

        # ── Gráfico de anomalía (heatmap) ────────────────────────────
        dbc.Card(
            dcc.Graph(
                id="anomaly-chart",
                config={"displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "displaylogo": False},
                style={"height": "320px"},
            ),
            style={
                "background": SURFACE,
                "border": f"1px solid {BORDER}",
                "borderRadius": "4px",
                "overflow": "hidden",
                "borderTop": "2px solid #f4714a",
            },
        ),
        html.P(
            "Anomalía calculada como T(año) − T̄(1950–1980) · Ensamble multi-modelo activo · "
            "Fuente: NASA NEX-GDDP-CMIP6 · SciVis Contest 2026",
            style={"marginTop": "12px", "fontSize": "9px", "color": MUTED,
                "textAlign": "center", "letterSpacing": "0.06em",
                "marginBottom": "8px"},
        ),
        #GRAFICA MAR PARTE 2: Mapa de humedad relativa near-surface
        # ── Separador visual (temperatura / tasa de calentamiento) ──
        html.Hr(style={"borderColor": BORDER, "marginTop": "36px", "marginBottom": "32px"}),
        # ── Encabezado sección Tasa de Calentamiento ─────────────────
        html.Div(
            [
                html.P(
                    "IEEE SciVis Contest 2026 · NEX-GDDP CMIP6 · Análisis temporal",
                    style={"fontFamily": "Space Mono", "fontSize": "10px",
                           "letterSpacing": "0.2em", "color": "#38c7a0",
                           "textTransform": "uppercase", "marginBottom": "6px"},
                ),
                html.H2(
                    ["Tasa de Calentamiento ", html.Span("por Década", style={"color": "#38c7a0"})],
                    style={"fontFamily": "Syne, sans-serif", "fontWeight": "800",
                           "fontSize": "clamp(18px,3vw,30px)", "color": "#e8f2ff",
                           "letterSpacing": "-0.02em", "lineHeight": "1.1"},
                ),
                html.P(
                    "Cambio promedio de temperatura por cada período de 10 años (°C/década) · "
                    "Cada barra representa cuánto se calentó el planeta en esa década respecto "
                    "a la anterior · Responde directamente a los modelos y escenarios activos.",
                    style={"fontSize": "11px", "color": MUTED, "lineHeight": "1.7",
                           "maxWidth": "680px", "marginTop": "6px"},
                ),
            ],
            style={"marginBottom": "16px"},
        ),
        # ── Gráfico de tasa de calentamiento por década ───────────────
        dbc.Card(
            dcc.Graph(
                id="rate-chart",
                config={"displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "displaylogo": False},
                style={"height": "420px"},
            ),
            style={
                "background": SURFACE,
                "border": f"1px solid {BORDER}",
                "borderRadius": "4px",
                "overflow": "hidden",
                "borderTop": "2px solid #38c7a0",
            },
        ),
        html.P(
            [
                "Tasa de calentamiento = diferencia de temperatura media entre décadas consecutivas · "
                "Calculada sobre el ensamble de modelos activos · ",
                html.Br(),
                "Fuente: NASA NEX-GDDP-CMIP6 · SciVis Contest 2026",
            ],
            style={"marginTop": "12px", "fontSize": "9px", "color": MUTED,
                   "textAlign": "center", "letterSpacing": "0.06em",
                   "marginBottom": "8px"},
        ),






        # ══════════════════════════════════════════════════════════
        # ── Separador visual ─────────────────────────────────────
        html.Hr(style={"borderColor": BORDER, "marginTop": "40px", "marginBottom": "36px"}),

        # ── Encabezado sección Humedad ───────────────────────────
        html.Div(
            [
                html.P(
                    "IEEE SciVis Contest 2026 · NEX-GDDP CMIP6 · Variable: hurs",
                    style={"fontFamily": "Space Mono", "fontSize": "10px",
                           "letterSpacing": "0.2em", "color": "#c084fc",
                           "textTransform": "uppercase", "marginBottom": "6px"},
                ),
                html.H2(
                    ["Humedad Relativa ", html.Span("Near-Surface · Mapa Mundial", style={"color": "#c084fc"})],
                    style={"fontFamily": "Syne, sans-serif", "fontWeight": "800",
                           "fontSize": "clamp(18px,3vw,30px)", "color": "#e8f2ff",
                           "letterSpacing": "-0.02em", "lineHeight": "1.1"},
                ),
                html.P(
                    "Humedad relativa near-surface (hurs, %) a escala global · "
                    "Selecciona el año con el deslizador, el modelo climático y el escenario "
                    "para explorar cómo varía la humedad entre regiones. "
                    "Escala de color: naranja = seco · verde = moderado · violeta = muy húmedo.",
                    style={"fontSize": "11px", "color": MUTED, "lineHeight": "1.7",
                           "maxWidth": "720px", "marginTop": "6px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),

        # ── Controles del mapa de humedad ────────────────────────
        dbc.Row(
            [
                # — Selector de modelo —
                dbc.Col(
                    [
                        html.Span("Modelo", style={"fontSize": "9px", "color": MUTED,
                                                    "letterSpacing": "0.15em",
                                                    "textTransform": "uppercase",
                                                    "display": "block",
                                                    "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="hurs-model-dropdown",
                            options=[{"label": m, "value": m} for m in MODELS],
                            value="ACCESS-CM2",
                            clearable=False,
                            style={
                                "fontFamily": "Space Mono, monospace",
                                "fontSize": "11px",
                                "backgroundColor": "#0d1526",
                                "color": "#c8d8f0",
                                "border": f"1px solid #1a2a4a",
                                "borderRadius": "3px",
                                "minWidth": "180px",
                            },
                        ),
                    ],
                    width="auto",
                ),
                # — Selector de escenario —
                dbc.Col(
                    [
                        html.Span("Escenario", style={"fontSize": "9px", "color": MUTED,
                                                       "letterSpacing": "0.15em",
                                                       "textTransform": "uppercase",
                                                       "display": "block",
                                                       "marginBottom": "4px"}),
                        dbc.RadioItems(
                            id="hurs-scen-radio",
                            options=[
                                {"label": "Histórico (1950–2014)", "value": "historical"},
                                {"label": "SSP2-4.5 (2015–2100)",  "value": "ssp245"},
                                {"label": "SSP5-8.5 (2015–2100)",  "value": "ssp585"},
                            ],
                            value="historical",
                            inline=True,
                            style={"fontFamily": "Space Mono", "fontSize": "11px", "color": TEXT},
                            input_checked_style={"backgroundColor": "#c084fc", "borderColor": "#c084fc"},
                            label_checked_style={"color": "#c084fc"},
                        ),
                    ],
                    width="auto",
                ),
            ],
            align="center",
            style={"marginBottom": "12px", "gap": "24px", "rowGap": "10px",
                   "flexWrap": "wrap", "background": SURFACE,
                   "border": f"1px solid {BORDER}", "borderRadius": "4px",
                   "padding": "14px 18px"},
            class_name="g-2",
        ),

        # — Deslizador de año —
        html.Div(
            [
                html.Span("Año", style={"fontSize": "9px", "color": MUTED,
                                        "letterSpacing": "0.15em",
                                        "textTransform": "uppercase",
                                        "marginRight": "16px",
                                        "verticalAlign": "middle"}),
                dcc.Slider(
                    id="hurs-year-slider",
                    min=1950,
                    max=2014,    # se actualiza dinámicamente según escenario
                    step=1,
                    value=1980,
                    marks={y: {"label": str(y),
                                "style": {"fontFamily": "Space Mono", "fontSize": "9px",
                                          "color": MUTED}}
                           for y in range(1950, 2015, 10)},
                    tooltip={"always_visible": True, "placement": "bottom",
                             "style": {"fontFamily": "Space Mono", "fontSize": "10px",
                                       "color": "#c8d8f0", "backgroundColor": "#0d1526"}},
                    updatemode="mouseup",
                ),
            ],
            style={"background": SURFACE, "border": f"1px solid {BORDER}",
                   "borderRadius": "4px", "padding": "14px 18px",
                   "marginBottom": "16px"},
            id="hurs-slider-container",
        ),

        # ── Mapa mundial de humedad ──────────────────────────────
        dbc.Card(
            dcc.Graph(
                id="hurs-map",
                figure=_HURS_INITIAL_FIG,
                config={"scrollZoom": True, "displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "displaylogo": False},
                style={"height": "520px"},
            ),
            style={
                "background": SURFACE,
                "border": f"1px solid {BORDER}",
                "borderRadius": "4px",
                "overflow": "hidden",
                "borderTop": "2px solid #c084fc",
            },
        ),

        # ── Leyenda de escala de color ───────────────────────────
        html.Div(
            [
                html.Span("Escala de humedad:  ", style={"fontSize": "9px", "color": MUTED}),
                *[
                    html.Span(
                        f"{'▇' * 3} {label}",
                        style={"color": color, "fontFamily": "Space Mono",
                               "fontSize": "9px", "marginRight": "16px"},
                    )
                    for color, label in [
                        ("#f4714a", "Muy seco (0–20%)"),
                        ("#f4d44a", "Seco (20–40%)"),
                        ("#38c7a0", "Moderado (40–60%)"),
                        ("#6db3ff", "Húmedo (60–80%)"),
                        ("#c084fc", "Muy húmedo (80–100%)"),
                    ]
                ],
            ],
            style={"marginTop": "10px", "paddingLeft": "4px", "flexWrap": "wrap",
                   "display": "flex", "alignItems": "center"},
        ),

        # ── Nota de fuente humedad ───────────────────────────────
        html.P(
            [
                f"Variable: hurs (Near-Surface Relative Humidity, %) · "
                f"Resolución descargada: quality={QUALITY} · "
                f"Día del año muestreado: {DOY} (~1 julio) · ",
                html.Br(),
                f"Grid: 600×1440 (0.25°×0.25°), submuetreado ×{_MAP_STEP} para visualización · "
                "Fuente: NASA NEX-GDDP-CMIP6 · Mapa: Carto Dark Matter · SciVis Contest 2026",
            ],
            style={"marginTop": "16px", "fontSize": "9px", "color": MUTED,
                   "textAlign": "center", "letterSpacing": "0.06em",
                   "marginBottom": "20px"},
        ),
    ],
    fluid=True,
    style={"background": DARK, "minHeight": "100vh", "padding": "32px 24px 60px",
           "color": TEXT, "fontFamily": "Space Mono, monospace"},
)


# ──────────────────────────────────────────────────────────────────
# 6.  CALLBACKS
# ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("main-chart", "figure"),
    Output("stat-base",  "children"),
    Output("stat-hist",  "children"),
    Output("stat-245",   "children"),
    Output("stat-585",   "children"),
    Input("scen-checklist",  "value"),
    Input("model-checklist", "value"),
    Input("opts-checklist",  "value"),
)
def update_chart(active_scens, active_models, opts):
    show_bands  = "bands"  in (opts or [])
    show_thresh = "thresh" in (opts or [])

    fig = build_figure(DATA, active_models or [], active_scens or [],
                       show_bands, show_thresh)

    stats = compute_stats_cards(DATA, active_models or [])

    base_str  = f"{stats['base']:.2f}°C"         if stats["base"]        else "–"
    hist_str  = f"+{stats['hist_delta']:.2f}°C"  if stats["hist_delta"]  else "–"
    s245_str  = f"{stats['ssp245_2100']:.2f}°C"  if stats["ssp245_2100"] else "–"
    s585_str  = f"{stats['ssp585_2100']:.2f}°C"  if stats["ssp585_2100"] else "–"

    return fig, base_str, hist_str, s245_str, s585_str


@app.callback(
    Output("hurs-map", "figure"),
    Input("hurs-year-slider",   "value"),
    Input("hurs-model-dropdown","value"),
    Input("hurs-scen-radio",    "value"),
)
def update_hurs_map(year, model, scenario):
    model    = model    or "ACCESS-CM2"
    scenario = scenario or "historical"
    year     = year     or 1980
    return build_hurs_map_figure(model, scenario, year)

@app.callback(
    Output("hurs-trend-chart", "figure"),
    Input("scen-checklist", "value"),
    Input("model-checklist", "value"),
    Input("opts-checklist", "value"),
)
def update_hurs_trend(active_scens, active_models, opts):
    show_bands = "bands" in (opts or [])
    return build_hurs_trend_figure(active_models or [], active_scens or [], show_bands)
@app.callback(
    Output("hurs-year-slider", "min"),
    Output("hurs-year-slider", "max"),
    Output("hurs-year-slider", "value"),
    Output("hurs-year-slider", "marks"),
    Input("hurs-scen-radio", "value"),
    State("hurs-year-slider", "value"),
)
def update_hurs_slider_range(scenario, current_year):
    """Ajusta el rango del slider según el escenario seleccionado."""
    if scenario == "historical":
        min_y, max_y = 1950, 2014
    else:
        min_y, max_y = 2015, 2100

    # Mantener el año actual si es válido, si no usar el año central del rango
    if current_year and min_y <= current_year <= max_y:
        new_val = current_year
    else:
        new_val = (min_y + max_y) // 2

    marks = {y: {"label": str(y),
                 "style": {"fontFamily": "Space Mono", "fontSize": "9px", "color": MUTED}}
             for y in range(min_y, max_y + 1, 10)}

    return min_y, max_y, new_val, marks

# GRAFICA MAR PARTE X: Callback para actualizar el gráfico de tasa de calentamiento por década
@app.callback(
    Output("rate-chart", "figure"),
    Input("scen-checklist",  "value"),
    Input("model-checklist", "value"),
)
def update_rate_chart(active_scens, active_models):
    return build_warming_rate_figure(DATA, active_models or [], active_scens or [])
@app.callback(
    Output("anomaly-chart", "figure"),
    Input("scen-checklist", "value"),
    Input("model-checklist", "value"),
)
def update_anomaly_chart(active_scens, active_models):
    return build_anomaly_figure(DATA, active_models or [], active_scens or [])
# ──────────────────────────────────────────────────────────────────
# 7.  PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)