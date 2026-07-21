"""Rendering + charting helpers for the Escape Room UI.

Each room is themed to its movie (Ice Age, Raiders of the Lost Ark, The Matrix,
The Fast and the Furious, Star Wars).  Grid cells carry their reward so the
learned policy is interpretable; a legend explains every tile.

Renderers return self-contained HTML meant for ``streamlit.components.v1.html`` /
``st.iframe`` (isolated iframe => our <style>/SVG survive).
"""
from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go

from environments import ACTION_ARROWS

# ---- chart palette (consistent across every chart) ----------------------- #
C_RAW = "#94a3b8"
C_MAIN = "#2563eb"
C_ACCENT = "#f59e0b"
C_GOOD = "#16a34a"
C_BAD = "#dc2626"


# ========================================================================== #
# Movie themes
# ========================================================================== #
THEMES = {
    "ice": dict(  # Ice Age
        movie="Ice Age", board="radial-gradient(circle at 50% 0%, #16436a, #0a1a2e 72%)",
        frame="#38bdf8", accent="#7dd3fc",
        even="#102a44", odd="#0c2338", start_c="#0e3a2e",
        wall_c="#1e4d63", pit_c="#7f1d1d", cliff_c="#4c1d95", goal_c="#14532d",
        slip_a="#0e4a63", slip_b="#1a6f8c",
        agent="🐕", wall="🧊", slip="❄️", slip_name="ice", pit="🕳️", cliff="🕶️",
        goal="🌄", start="🐾"),
    "temple": dict(  # Raiders of the Lost Ark
        movie="Raiders of the Lost Ark", board="radial-gradient(circle at 50% 0%, #3a2a17, #160f08 72%)",
        frame="#b45309", accent="#f59e0b",
        even="#2a2018", odd="#1f1712", start_c="#3f2d16",
        wall_c="#57534e", pit_c="#7f1d1d", cliff_c="#4c1d95", goal_c="#3f3417",
        slip_a="#4a3418", slip_b="#63481f",
        agent="🐕", wall="🗿", slip="💧", slip_name="mud", pit="🕳️", cliff="🕶️",
        goal="🏆", start="🐾"),
    "matrix": dict(  # The Matrix
        movie="The Matrix", board="radial-gradient(circle at 50% 0%, #012a12, #000600 72%)",
        frame="#22c55e", accent="#00ff41",
        even="#02180c", odd="#010f07", start_c="#043016",
        wall_c="#065f46", pit_c="#7f1d1d", cliff_c="#0b3d1e", goal_c="#064e2b",
        slip_a="#022c14", slip_b="#043a1b",
        agent="🐕", wall="🟩", slip="🟩", slip_name="code", pit="🕳️", cliff="🕶️",
        goal="☎️", start="🐾"),
    "garage": dict(  # The Fast and the Furious
        movie="The Fast and the Furious", board="#0a0a12", grid="#241b2e",
        accent="#ec4899", car="🚗", agent="🏎️", exit="🏁"),
    "space": dict(  # Star Wars
        movie="Star Wars", board="#02030a", grid="#0b1030",
        accent="#eab308", drone="🛸", agent="🚀"),
}

ROOM_THEME = {"room1": "ice", "room2": "temple", "room3": "matrix",
              "room4": "garage", "room5": "space"}


# ========================================================================== #
# Grid rooms (1-3)
# ========================================================================== #
_GRID_CSS = (
    "<style>"
    "html,body{margin:0;padding:0}"
    ".rlb{border-collapse:separate;border-spacing:0;margin:auto;border-radius:12px;"
    "box-sizing:border-box;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,.45)}"
    ".rlb td{position:relative;text-align:center;vertical-align:middle;"
    "box-sizing:border-box;border:1px solid rgba(0,0,0,.35)}"
    ".rlrw{position:absolute;bottom:1px;right:2px;font-size:9px;font-weight:700;"
    "line-height:1;padding:1px 2px;border-radius:3px;font-family:ui-monospace,monospace}"
    ".rlrw.pos{color:#bbf7d0;background:rgba(20,83,45,.9)}"
    ".rlrw.neg{color:#fecaca;background:rgba(127,29,29,.9)}"
    ".rlst{opacity:.5}"
    "</style>")


def ARROW(policy, x, y):
    return ACTION_ARROWS[policy[(x, y)]] if policy and (x, y) in policy else ""


def render_grid_html(meta, theme, agent=None, policy=None, cell=44, fill=False):
    T = THEMES[theme]
    size, walls, ice = meta["size"], meta["walls"], meta["slippery"]
    pits, cliff = meta["traps"], meta.get("cliff", set())
    start, goal = meta["start"], meta["goal"]
    fs = int(cell * 0.5)

    def bg(x, y):
        if (x, y) in walls:  return T["wall_c"]
        if (x, y) in pits:   return T["pit_c"]
        if (x, y) in cliff:  return T["cliff_c"]
        if (x, y) == goal:   return T["goal_c"]
        if (x, y) in ice:
            return (f"repeating-linear-gradient(45deg,{T['slip_a']},{T['slip_a']} 5px,"
                    f"{T['slip_b']} 5px,{T['slip_b']} 10px)")
        if (x, y) == start:  return T["start_c"]
        return T["even"] if (x + y) % 2 else T["odd"]

    def content(x, y):
        if agent is not None and (x, y) == tuple(agent):
            return f"<span style='filter:drop-shadow(0 0 6px {T['accent']})'>{T['agent']}</span>"
        if (x, y) in walls:  return T["wall"]
        if (x, y) in pits:   return f"{T['pit']}<span class='rlrw neg'>-100</span>"
        if (x, y) in cliff:  return f"{T['cliff']}<span class='rlrw neg'>-100</span>"
        if (x, y) == goal:   return f"{T['goal']}<span class='rlrw pos'>+100</span>"
        arrow = ARROW(policy, x, y)
        if (x, y) in ice:
            return (f"<span style='color:{T['accent']};font-weight:700'>{arrow}</span>"
                    if arrow else T["slip"])
        if arrow:
            return f"<span style='color:{T['accent']};font-weight:700'>{arrow}</span>"
        if (x, y) == start:  return f"<span class='rlst'>{T['start']}</span>"
        return ""

    rows = []
    for y in range(size - 1, -1, -1):                        # y=0 at the bottom
        tds = "".join(
            f"<td style='width:{cell}px;height:{cell}px;background:{bg(x, y)};"
            f"font-size:{fs}px'>{content(x, y)}</td>" for x in range(size))
        rows.append(f"<tr>{tds}</tr>")
    board = f"<table class='rlb' style='border:2px solid {T['frame']}'>{''.join(rows)}</table>"
    if fill:   # static view: fill + vertically centre the board in the themed frame
        return (_GRID_CSS + f"<div style='min-height:100vh;box-sizing:border-box;padding:12px;"
                f"background:{T['board']};display:flex;align-items:center;justify-content:center'>"
                f"{board}</div>")
    return (_GRID_CSS + f"<style>body{{background:{T['board']}}}</style>"
            f"<div style='padding:8px;display:flex;justify-content:center'>{board}</div>")


def render_legend(theme, meta):
    T = THEMES[theme]
    chips = [(T["agent"], "Hezki")]
    if theme in ("garage", "space"):
        if theme == "garage":
            chips += [(T["car"], "parked car · crash −100"), (T["exit"], "exit · +100")]
        else:
            chips += [(T["drone"], "drone · hit −1000"), ("👁️", "vision (sensor)"),
                      ("✅", "survive · +1 / step")]
    else:
        if meta.get("slippery"):
            chips.append((T["slip"], f"{T['slip_name']} · slippery 70/10/10/10"))
        if meta.get("walls"):
            chips.append((T["wall"], "wall"))
        if meta.get("traps"):
            chips.append((T["pit"], "pit · −100"))
        if meta.get("cliff"):
            chips.append((T["cliff"], "clones · −100 + reset"))
        chips.append((T["goal"], "exit · +100"))
        chips.append(("👣", "each step · −1"))
    inner = "".join(
        "<span style='display:inline-flex;align-items:center;gap:6px;"
        "background:rgba(148,163,184,.12);border:1px solid rgba(148,163,184,.25);"
        f"border-radius:999px;padding:4px 11px;margin:3px;font-size:13px'>"
        f"<span style='font-size:16px'>{e}</span>{t}</span>" for e, t in chips)
    return (f"<div style='font-family:system-ui,Segoe UI,sans-serif;color:#cbd5e1;"
            f"text-align:center;padding:2px'>{inner}</div>")


# ========================================================================== #
# Continuous rooms (4-5)
# ========================================================================== #
def render_space_svg(meta, theme, agent=None, obstacles=None, vision=None,
                     trail=None, scale=44, pad=14, fill=False):
    T = THEMES[theme]
    span = 10.0
    W = int(span * scale + 2 * pad)

    def px(x):  return pad + x * scale
    def py(y):  return pad + (span - y) * scale                # flip: up is up

    p = [f"<svg width='{W}' height='{W}' viewBox='0 0 {W} {W}' "
         f"xmlns='http://www.w3.org/2000/svg'>"]
    p.append(f"<rect x='0' y='0' width='{W}' height='{W}' rx='16' fill='{T['board']}'/>")

    if theme == "space":                                       # starfield
        rng = np.random.default_rng(7)
        for _ in range(80):
            sx, sy = pad + rng.random() * span * scale, pad + rng.random() * span * scale
            p.append(f"<circle cx='{sx:.0f}' cy='{sy:.0f}' r='{rng.random()*1.3+0.3:.1f}' "
                     f"fill='#e5e7eb' opacity='{0.25+rng.random()*0.6:.2f}'/>")
    else:                                                      # neon garage grid
        for i in range(1, 10):
            p.append(f"<line x1='{px(i)}' y1='{pad}' x2='{px(i)}' y2='{pad+span*scale}' "
                     f"stroke='{T['grid']}'/>")
            p.append(f"<line x1='{pad}' y1='{py(i)}' x2='{pad+span*scale}' y2='{py(i)}' "
                     f"stroke='{T['grid']}'/>")
    p.append(f"<rect x='{pad}' y='{pad}' width='{span*scale}' height='{span*scale}' rx='10' "
             f"fill='none' stroke='{T['accent']}' stroke-opacity='.55'/>")

    if meta.get("kind") == "garage":
        for (xn, xx, yn, yx) in meta["obstacles"]:
            p.append(f"<rect x='{px(xn)}' y='{py(yx)}' width='{(xx-xn)*scale}' "
                     f"height='{(yx-yn)*scale}' rx='5' fill='#7f1d1d' stroke='#ef4444' opacity='.85'/>")
            p.append(f"<text x='{px((xn+xx)/2)}' y='{py((yn+yx)/2)+8}' font-size='24' "
                     f"text-anchor='middle'>{T['car']}</text>")
        ex, ey = meta["exit_region"]
        p.append(f"<rect x='{px(ex)}' y='{py(span)}' width='{(span-ex)*scale}' "
                 f"height='{(span-ey)*scale}' fill='#22c55e' opacity='.22'/>")
        p.append(f"<text x='{px((ex+span)/2)}' y='{py((ey+span)/2)+8}' font-size='26' "
                 f"text-anchor='middle'>{T['exit']}</text>")
    else:  # space
        ay = meta.get("agent_y", 2.0)
        if vision:
            ax = agent[0] if agent else 5.0
            w = meta.get("obs_width", 0.5)
            p.append(f"<rect x='{px(ax-w)}' y='{py(ay+vision)}' width='{2*w*scale}' "
                     f"height='{vision*scale}' fill='{T['accent']}' opacity='.16'/>")
        for (ox, oy) in (obstacles or []):
            p.append(f"<text x='{px(ox)}' y='{py(oy)+8}' font-size='24' "
                     f"text-anchor='middle'>{T['drone']}</text>")

    if trail:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in trail)
        p.append(f"<polyline points='{pts}' fill='none' stroke='{T['accent']}' "
                 f"stroke-width='2.5' opacity='.6'/>")
    if agent is not None:
        p.append(f"<text x='{px(agent[0])}' y='{py(agent[1])+9}' font-size='28' "
                 f"text-anchor='middle' style='filter:drop-shadow(0 0 6px {T['accent']})'>"
                 f"{T['agent']}</text>")
    p.append("</svg>")
    if fill:
        return (f"<style>html,body{{margin:0;padding:0}}</style>"
                f"<div style='min-height:100vh;box-sizing:border-box;padding:10px;"
                f"background:{T['board']};display:flex;align-items:center;justify-content:center'>"
                f"{''.join(p)}</div>")
    return (f"<style>html,body{{margin:0;padding:0;background:{T['board']}}}</style>"
            f"<div style='padding:8px;display:flex;justify-content:center'>{''.join(p)}</div>")


# ========================================================================== #
# Client-side replay player (smooth, no Streamlit reruns)
# ========================================================================== #
def render_player(frames, delay_ms=90, autoplay=True, caption=""):
    data = json.dumps(frames)
    auto = "true" if autoplay else "false"
    cap = (f"<div style='color:#94a3b8;font-size:13px;margin-bottom:4px'>{caption}</div>"
           if caption else "")
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
    fig = go.Figure(go.Scatter(y=eps, mode="lines", line=dict(color=C_GOOD), name="ε"))
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
    fig.update_yaxes(autorange=True)
    return _layout(fig, "Learned state-value V(s)  (❄️ optimal cost-to-go)", "x", "y")


def path_compare(meta, path):
    size = meta["size"]
    fig = go.Figure()
    cx = [x for (x, y) in meta["cliff"]]
    cy = [y for (x, y) in meta["cliff"]]
    fig.add_scatter(x=cx, y=cy, mode="markers",
                    marker=dict(symbol="square", size=18, color=C_BAD), name="cliff (clones)")
    fig.add_scatter(x=[meta["start"][0]], y=[meta["start"][1]], mode="markers",
                    marker=dict(size=14, color=C_GOOD), name="start")
    fig.add_scatter(x=[meta["goal"][0]], y=[meta["goal"][1]], mode="markers",
                    marker=dict(size=14, color=C_ACCENT, symbol="star"), name="exit")
    fig.add_scatter(x=[p[0] for p in path], y=[p[1] for p in path], mode="lines+markers",
                    line=dict(color=C_MAIN, width=3), name="learned path")
    fig.update_xaxes(range=[-0.5, size - 0.5], dtick=1)
    fig.update_yaxes(range=[-0.5, size - 0.5], dtick=1)
    return _layout(fig, "Q-Learning greedy path (cliff-hugging)", "x", "y")
