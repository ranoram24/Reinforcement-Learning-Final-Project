"""Rendering + charting helpers for the Escape Room UI.

    * HTML "game-board" renderer for the grid rooms (1-3).
    * Inline-SVG renderer for the continuous rooms (4-5), incl. the Room-5
      partial-observation vision cone and falling obstacles.
    * Plotly figures for every training-metric chart the spec asks for.

All renderers return self-contained HTML strings meant to be shown through
``streamlit.components.v1.html`` (isolated iframe => our <style>/SVG survive).
"""
from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go

from environments import ACTION_ARROWS

# ---- palette (consistent across every chart) ----------------------------- #
C_RAW = "#94a3b8"
C_MAIN = "#2563eb"
C_ACCENT = "#f59e0b"
C_GOOD = "#16a34a"
C_BAD = "#dc2626"

# ---- emoji vocabulary ---------------------------------------------------- #
E_AGENT = "🐕"          # Hezki
E_WALL = "🧱"
E_ICE = "🧊"
E_PIT = "🕳️"
E_CLONE = "🕶️"         # Matrix clones (cliff)
E_EXIT = "🚪"
E_START = "🐾"
E_CAR = "🏎️"           # Room 4 hovercar
E_SHIP = "🚀"          # Room 5 escape pod
E_DRONE = "🛸"          # Room 5 neuralyzer-drones


# ========================================================================== #
# Grid rooms (1-3)
# ========================================================================== #
def render_grid_html(meta, agent=None, policy=None, cell=44):
    """Return an HTML board.  `policy` (dict state->action) draws greedy arrows."""
    size = meta["size"]
    walls, ice = meta["walls"], meta["slippery"]
    pits, cliff = meta["traps"], meta.get("cliff", set())
    start, goal = meta["start"], meta["goal"]

    def bg(x, y):
        if (x, y) in walls:  return "#334155"
        if (x, y) in pits:   return "#7f1d1d"
        if (x, y) in cliff:  return "#4c1d95"
        if (x, y) == goal:   return "#166534"
        if (x, y) in ice:    return "#cbeafe"
        if (x, y) == start:  return "#1e3a2f"
        return "#0f172a" if (x + y) % 2 else "#111c31"

    def content(x, y):
        if agent is not None and (x, y) == tuple(agent):
            return E_AGENT
        if (x, y) in walls:  return E_WALL
        if (x, y) in pits:   return E_PIT
        if (x, y) in cliff:  return E_CLONE
        if (x, y) == goal:   return E_EXIT
        if (x, y) in ice:
            return (ARROW(policy, x, y) or E_ICE)
        arrow = ARROW(policy, x, y)
        if arrow:            return f"<span style='color:#e2e8f0;opacity:.85'>{arrow}</span>"
        if (x, y) == start:  return f"<span style='opacity:.5'>{E_START}</span>"
        return ""

    rows = []
    for y in range(size - 1, -1, -1):                        # y=0 at the bottom
        cells = "".join(
            f"<td style='width:{cell}px;height:{cell}px;background:{bg(x, y)};"
            f"text-align:center;font-size:{int(cell*0.55)}px;"
            f"border:1px solid #0b1220;'>{content(x, y)}</td>"
            for x in range(size))
        rows.append(f"<tr>{cells}</tr>")
    board = (f"<table style='border-collapse:collapse;margin:auto;"
             f"box-shadow:0 6px 24px rgba(0,0,0,.35);border-radius:8px;"
             f"overflow:hidden'>{''.join(rows)}</table>")
    return _wrap(board, height=size * cell + 24)


def ARROW(policy, x, y):
    if policy and (x, y) in policy:
        return ACTION_ARROWS[policy[(x, y)]]
    return ""


# ========================================================================== #
# Continuous rooms (4-5)
# ========================================================================== #
def render_space_svg(meta, agent=None, obstacles=None, vision=None,
                     trail=None, scale=44, pad=16):
    """Continuous 10x10 m arena.  `obstacles`:
        * Room 4 -> boxes (xmin,xmax,ymin,ymax)
        * Room 5 -> centres (x, y) drawn as drones (passed as `obstacles`)."""
    span = 10.0
    W = int(span * scale + 2 * pad)

    def px(x):  return pad + x * scale
    def py(y):  return pad + (span - y) * scale                # flip: up is up

    parts = [f"<svg width='{W}' height='{W}' viewBox='0 0 {W} {W}' "
             f"xmlns='http://www.w3.org/2000/svg'>"]
    parts.append(f"<rect x='{pad}' y='{pad}' width='{span*scale}' "
                 f"height='{span*scale}' rx='10' fill='#0b1220' stroke='#334155'/>")
    # grid lines
    for i in range(1, 10):
        parts.append(f"<line x1='{px(i)}' y1='{pad}' x2='{px(i)}' "
                     f"y2='{pad+span*scale}' stroke='#1e293b'/>")
        parts.append(f"<line x1='{pad}' y1='{py(i)}' x2='{pad+span*scale}' "
                     f"y2='{py(i)}' stroke='#1e293b'/>")

    kind = meta.get("kind")
    if kind == "garage":
        for (xn, xx, yn, yx) in meta["obstacles"]:            # parked cars
            parts.append(f"<rect x='{px(xn)}' y='{py(yx)}' "
                         f"width='{(xx-xn)*scale}' height='{(yx-yn)*scale}' "
                         f"rx='4' fill='#7f1d1d' stroke='#ef4444' opacity='.85'/>")
            parts.append(f"<text x='{px((xn+xx)/2)}' y='{py((yn+yx)/2)+6}' "
                         f"font-size='20' text-anchor='middle'>🚗</text>")
        ex, ey = meta["exit_region"]                          # exit corner
        parts.append(f"<rect x='{px(ex)}' y='{py(span)}' width='{(span-ex)*scale}' "
                     f"height='{(span-ey)*scale}' fill='#16a34a' opacity='.25'/>")
        parts.append(f"<text x='{px((ex+span)/2)}' y='{py((ey+span)/2)+6}' "
                     f"font-size='22' text-anchor='middle'>{E_EXIT}</text>")
        agent_emoji = E_CAR
    else:                                                     # space escape
        ay = meta.get("agent_y", 2.0)
        if vision:                                            # sensor cone
            ax = agent[0] if agent else 5.0
            w = meta.get("obs_width", 0.5)
            parts.append(f"<rect x='{px(ax-w)}' y='{py(ay+vision)}' "
                         f"width='{2*w*scale}' height='{vision*scale}' "
                         f"fill='#38bdf8' opacity='.14'/>")
        for (ox, oy) in (obstacles or []):                    # drones
            parts.append(f"<text x='{px(ox)}' y='{py(oy)+7}' font-size='20' "
                         f"text-anchor='middle'>{E_DRONE}</text>")
        agent_emoji = E_SHIP

    if trail:
        pts = " ".join(f"{px(x)},{py(y)}" for x, y in trail)
        parts.append(f"<polyline points='{pts}' fill='none' "
                     f"stroke='#38bdf8' stroke-width='2' opacity='.5'/>")
    if agent is not None:
        parts.append(f"<text x='{px(agent[0])}' y='{py(agent[1])+8}' font-size='24' "
                     f"text-anchor='middle'>{agent_emoji}</text>")
    parts.append("</svg>")
    return _wrap("".join(parts), height=W + 8)


def _wrap(inner, height):
    return (f"<div style='font-family:system-ui,Segoe UI,sans-serif;"
            f"display:flex;justify-content:center;padding:6px'>{inner}</div>")


# ========================================================================== #
# Client-side replay player (smooth, no Streamlit reruns)
# ========================================================================== #
def render_player(frames, delay_ms=90, autoplay=True, caption=""):
    """Embed the list of pre-rendered HTML `frames` with play/pause/scrub/speed
    controls that run entirely in the browser."""
    data = json.dumps(frames)
    auto = "true" if autoplay else "false"
    cap = f"<div style='color:#94a3b8;font-size:13px;margin-bottom:4px'>{caption}</div>" if caption else ""
    return f"""
<div style="font-family:system-ui,Segoe UI,sans-serif;text-align:center;color:#e2e8f0">
  {cap}
  <div id="stage"></div>
  <div style="margin-top:10px;display:flex;gap:10px;align-items:center;justify-content:center;flex-wrap:wrap">
    <button id="pp" style="padding:6px 14px;border-radius:8px;border:1px solid #334155;
      background:#1e293b;color:#e2e8f0;cursor:pointer;font-size:14px">⏸ Pause</button>
    <input id="sl" type="range" min="0" max="{len(frames)-1}" value="0" style="width:45%">
    <span id="lbl" style="font-variant-numeric:tabular-nums;color:#94a3b8">0/{len(frames)-1}</span>
    <label style="color:#94a3b8;font-size:13px">speed
      <input id="sp" type="range" min="20" max="400" value="{delay_ms}" style="width:110px"></label>
  </div>
</div>
<script>
  const F = {data};
  let i = 0, playing = {auto}, delay = {delay_ms}, timer = null;
  const stage=document.getElementById('stage'), sl=document.getElementById('sl'),
        lbl=document.getElementById('lbl'), pp=document.getElementById('pp'),
        sp=document.getElementById('sp');
  function show(k){{ i=k; stage.innerHTML=F[k]; sl.value=k; lbl.textContent=k+'/'+(F.length-1); }}
  function tick(){{ if(i>=F.length-1){{ playing=false; pp.textContent='▶ Play'; return; }} show(i+1); }}
  function loop(){{ if(timer) clearInterval(timer); timer=setInterval(()=>{{ if(playing) tick(); }}, delay); }}
  pp.onclick=()=>{{ if(i>=F.length-1) show(0); playing=!playing; pp.textContent=playing?'⏸ Pause':'▶ Play'; }};
  sl.oninput=()=>{{ playing=false; pp.textContent='▶ Play'; show(parseInt(sl.value)); }};
  sp.oninput=()=>{{ delay=parseInt(sp.value); loop(); }};
  show(0); loop();
</script>"""


# ========================================================================== #
# Plotly charts
# ========================================================================== #
def _moving_avg(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or w <= 1:
        return x
    w = int(min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _layout(fig, title, xlab, ylab, log_y=False):
    fig.update_layout(
        title=title, template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=45), height=340,
        xaxis_title=xlab, yaxis_title=ylab,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.5)")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def convergence_chart(deltas):
    fig = go.Figure(go.Scatter(y=deltas, mode="lines", line=dict(color=C_MAIN),
                               name="max |ΔV|"))
    return _layout(fig, "Value-Iteration convergence", "Sweep (iteration)",
                   "max |ΔV|  (log)", log_y=True)


def reward_curve(rewards, window=50):
    fig = go.Figure()
    fig.add_scatter(y=rewards, mode="lines", line=dict(color=C_RAW, width=1),
                    name="per episode", opacity=0.55)
    ma = _moving_avg(rewards, window)
    fig.add_scatter(x=np.arange(len(ma)) + (len(rewards) - len(ma)), y=ma,
                    mode="lines", line=dict(color=C_ACCENT, width=2.5),
                    name=f"{window}-ep moving avg")
    return _layout(fig, "Cumulative reward per episode", "Episode", "Return")


def epsilon_curve(eps):
    fig = go.Figure(go.Scatter(y=eps, mode="lines", line=dict(color=C_GOOD),
                               name="ε"))
    return _layout(fig, "Exploration (ε) decay", "Episode", "ε")


def length_curve(lengths, window=50, title="Episode length", ylab="Steps"):
    fig = go.Figure()
    fig.add_scatter(y=lengths, mode="lines", line=dict(color=C_RAW, width=1),
                    name="per episode", opacity=0.5)
    ma = _moving_avg(lengths, window)
    fig.add_scatter(x=np.arange(len(ma)) + (len(lengths) - len(ma)), y=ma,
                    mode="lines", line=dict(color=C_MAIN, width=2.5),
                    name=f"{window}-ep moving avg")
    return _layout(fig, title, "Episode", ylab)


def value_heatmap(V, meta):
    size = meta["size"]
    z = np.full((size, size), np.nan)
    for (x, y), v in V.items():
        z[y, x] = v
    for (x, y) in meta["walls"]:
        z[y, x] = np.nan
    fig = go.Figure(go.Heatmap(
        z=z, colorscale="Viridis", colorbar=dict(title="V(s)"),
        hovertemplate="x=%{x}, y=%{y}<br>V=%{z:.1f}<extra></extra>"))
    fig.update_yaxes(autorange=True)                          # y=0 at bottom
    return _layout(fig, "Learned state-value V(s)  (❄️ optimal cost-to-go)",
                   "x", "y")


def path_compare(meta, path):
    """Room 3: draw the learned greedy path over the cliff to show the
    aggressive cliff-hugging route."""
    size = meta["size"]
    fig = go.Figure()
    cx = [x for (x, y) in meta["cliff"]]
    cy = [y for (x, y) in meta["cliff"]]
    fig.add_scatter(x=cx, y=cy, mode="markers", marker=dict(
        symbol="square", size=18, color=C_BAD), name="cliff (clones)")
    fig.add_scatter(x=[meta["start"][0]], y=[meta["start"][1]], mode="markers",
                    marker=dict(size=14, color=C_GOOD), name="start")
    fig.add_scatter(x=[meta["goal"][0]], y=[meta["goal"][1]], mode="markers",
                    marker=dict(size=14, color=C_ACCENT, symbol="star"), name="exit")
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    fig.add_scatter(x=px, y=py, mode="lines+markers",
                    line=dict(color=C_MAIN, width=3), name="learned path")
    fig.update_xaxes(range=[-0.5, size - 0.5], dtick=1)
    fig.update_yaxes(range=[-0.5, size - 0.5], dtick=1)
    return _layout(fig, "Q-Learning greedy path (cliff-hugging)", "x", "y")
