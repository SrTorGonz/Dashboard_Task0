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


# ──────────────────────────────────────────────────────────────────
# 7.  PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
