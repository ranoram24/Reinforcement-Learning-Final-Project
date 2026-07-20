"""Hezki the Dog vs. Men in Black — an RL Escape Room.

Streamlit entry point.  Hezki (a dog) escapes Agent J through five Hollywood
movie sets of increasing difficulty, each solved by a different RL algorithm:

    Room 1  Frozen Archive  (Ice Age)          Value Iteration / DP
    Room 2  Dark Temple      (Indiana Jones)    SARSA
    Room 3  Cloning Lab      (The Matrix)       Q-Learning (Cliff Walking)
    Room 4  Hovercar Garage  (Fast & Furious)   Tile-coding Function Approximation
    Room 5  Space Escape     (Star Wars)        Q-Learning + partial observability

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import algorithms as A
import environments as E
import utils as U

st.set_page_config(page_title="Hezki vs. MIB — RL Escape Room",
                   page_icon="🐕", layout="wide")


def embed(html, height):
    """Render self-contained HTML (scripts included) in an isolated iframe.
    Uses the modern `st.iframe` when available, else the classic component."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                                    # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height)

# --------------------------------------------------------------------------- #
# Room catalogue (drives the difficulty progression selector)
# --------------------------------------------------------------------------- #
ROOMS = {
    "room1": dict(label="Frozen Archive", emoji="❄️", stars=1, kind="dp",
                  movie="Ice Age (2002)", algo="Value Iteration (Dynamic Programming)",
                  plot="The MIB archive is frozen solid. Agent J's patrols are known, "
                       "so Hezki plans the perfect escape with a full model of the world."),
    "room2": dict(label="Dark Temple", emoji="🏛️", stars=2, kind="grid",
                  movie="Raiders of the Lost Ark (1981)", algo="SARSA (on-policy TD)",
                  plot="Hezki falls into an ancient alien temple — unknown terrain, "
                       "slippery mud and deadly spike pits. He must learn by trial."),
    "room3": dict(label="Cloning Lab", emoji="🕶️", stars=3, kind="grid",
                  movie="The Matrix (1999)", algo="Q-Learning (off-policy TD)",
                  plot="Agent J clones himself like Agent Smith. Hezki must take a "
                       "dangerous, aggressive shortcut along the cliff of clones."),
    "room4": dict(label="Hovercar Garage", emoji="🏎️", stars=4, kind="fa",
                  movie="The Fast and the Furious (2001)",
                  algo="Tile-coding + semi-gradient Q-Learning (Function Approximation)",
                  plot="Hezki steals a hovercar. Continuous physics, momentum and two "
                       "parked cars force a high-speed weave to the exit."),
    "room5": dict(label="Space Escape", emoji="🚀", stars=5, kind="space",
                  movie="Star Wars (1977)", algo="Q-Learning over a partial-observation sensor",
                  plot="Hezki flies an escape pod. Agent J fires neuralyzer-drones. "
                       "Hezki only senses what's right in front of him — dodge or be erased."),
}
ORDER = ["room1", "room2", "room3", "room4", "room5"]

GRID_H, SPACE_H = 500, 540


def store():
    return st.session_state.setdefault("store", {})


# --------------------------------------------------------------------------- #
# Sidebar — room selector + ALL hyperparameters + train/reset
# --------------------------------------------------------------------------- #
def sidebar():
    st.sidebar.title("🐕 Escape Room")
    st.sidebar.caption("Hezki the Dog vs. Men in Black")

    key = st.sidebar.radio(
        "**Difficulty progression**", ORDER,
        format_func=lambda k: f"{ROOMS[k]['emoji']}  Room {k[-1]} · {ROOMS[k]['label']}  "
                              f"{'★' * ROOMS[k]['stars']}{'☆' * (5 - ROOMS[k]['stars'])}")
    r = ROOMS[key]
    st.sidebar.markdown(f"**🎬 {r['movie']}**")
    st.sidebar.markdown(f"**🧠 {r['algo']}**")
    st.sidebar.divider()

    st.sidebar.subheader("⚙️ Hyperparameters")
    p = {}
    if key == "room1":
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key="g1")
        p["theta"] = st.sidebar.number_input("θ  convergence threshold", 1e-6, 1e-1,
                                             1e-4, format="%.6f", key="t1")
    elif key in ("room2", "room3"):
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.01, 1.0, 0.10, 0.01, key=f"a{key}")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key=f"g{key}")
        p["epsilon"] = st.sidebar.slider("ε  initial exploration", 0.0, 1.0, 1.0, 0.01, key=f"e{key}")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.90, 1.0, 0.995, 0.001,
                                               format="%.3f", key=f"ed{key}")
        p["epsilon_min"] = st.sidebar.slider("ε minimum", 0.0, 0.5, 0.01, 0.01, key=f"em{key}")
        p["episodes"] = st.sidebar.number_input("episodes", 100, 20000,
                                               1500 if key == "room2" else 1000, 100, key=f"ep{key}")
        p["max_steps"] = st.sidebar.number_input("max steps / episode", 50, 2000, 400, 50, key=f"ms{key}")
    elif key == "room4":
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.05, 1.0, 0.50, 0.05, key="a4")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.99, 0.001, key="g4")
        p["epsilon"] = st.sidebar.slider("ε  exploration", 0.0, 1.0, 0.10, 0.01, key="e4")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.90, 1.0, 1.0, 0.001,
                                              format="%.3f", key="ed4")
        p["episodes"] = st.sidebar.number_input("episodes", 200, 20000, 2000, 100, key="ep4")
        with st.sidebar.expander("Function-approximation & physics"):
            p["optimistic_init"] = st.slider("optimistic init Q₀", 0.0, 200.0, 100.0, 10.0, key="oi4")
            p["n_tilings"] = st.slider("tilings", 4, 16, 8, 1, key="nt4")
            p["n_bins"] = st.slider("bins / dim", 4, 12, 8, 1, key="nb4")
            p["v_max"] = st.slider("max speed (m/s)", 1.0, 6.0, 3.0, 0.5, key="vm4")
            p["max_steps"] = st.number_input("max steps / episode", 200, 2000, 800, 50, key="ms4")
            p["shaping"] = st.checkbox("reward shaping (wave-front potential)", True, key="sh4")
            p["shaping_coef"] = st.slider("shaping coefficient", 0.0, 10.0, 5.0, 0.5, key="sc4")
            p["hard_walls"] = st.checkbox("hard walls (−100 terminal on boundary)", False, key="hw4")
        p["epsilon_min"] = 0.0
        st.sidebar.caption("⏱️ ~1 min to train at default settings.")
    else:  # room5
        p["vision"] = st.sidebar.slider("👁️ vision range X_obs (m)", 0.5, 8.0, 3.0, 0.5, key="v5")
        p["spawn_every"] = st.sidebar.slider("drone spawn every N steps", 5, 60, 15, 1, key="sp5")
        p["alpha"] = st.sidebar.slider("α  learning rate", 0.01, 1.0, 0.10, 0.01, key="a5")
        p["gamma"] = st.sidebar.slider("γ  discount", 0.50, 0.999, 0.95, 0.001, key="g5")
        p["epsilon"] = st.sidebar.slider("ε  initial exploration", 0.0, 1.0, 1.0, 0.01, key="e5")
        p["epsilon_decay"] = st.sidebar.slider("ε decay / episode", 0.90, 1.0, 0.999, 0.001,
                                              format="%.3f", key="ed5")
        p["epsilon_min"] = st.sidebar.slider("ε minimum", 0.0, 0.5, 0.01, 0.01, key="em5")
        p["episodes"] = st.sidebar.number_input("episodes", 200, 30000, 3000, 100, key="ep5")
        with st.sidebar.expander("Physics"):
            p["v_max"] = st.slider("max speed (m/s)", 1.0, 6.0, 5.0, 0.5, key="vm5")
            p["max_steps"] = st.number_input("max steps (survival cap)", 100, 3000, 500, 50, key="ms5")

    st.sidebar.divider()
    c1, c2 = st.sidebar.columns(2)
    train_clicked = c1.button("🚀 Train", use_container_width=True, type="primary")
    if c2.button("🗑️ Reset", use_container_width=True):
        store().pop(key, None)
        st.rerun()

    random_clicked = False
    if key == "room5":
        random_clicked = st.sidebar.button("🎲 Generate Random Room & Test Policy",
                                           use_container_width=True,
                                           disabled=key not in store())
    return key, p, train_clicked, random_clicked


# --------------------------------------------------------------------------- #
# Env / agent construction
# --------------------------------------------------------------------------- #
def build_env(key, p, seed=None):
    if key == "room1":
        return E.Room1FrozenArchive()
    if key == "room2":
        return E.Room2DarkTemple()
    if key == "room3":
        return E.Room3CloningLab()
    if key == "room4":
        return E.Room4Garage(v_max=p["v_max"], max_steps=p["max_steps"],
                             shaping=p["shaping"], shaping_coef=p["shaping_coef"],
                             hard_walls=p["hard_walls"])
    return E.Room5SpaceEscape(vision=p["vision"], spawn_every=p["spawn_every"],
                              v_max=p["v_max"], max_steps=p["max_steps"], seed=seed)


def train(key, p):
    env = build_env(key, p, seed=0)
    bar = st.progress(0.0, text="Training…")

    def cb(i, n):
        if i % max(1, n // 100) == 0 or i == n:
            bar.progress(i / n, text=f"Training… {i}/{n} episodes")

    entry: dict = dict(params=dict(p), meta=env.render_meta(), kind=ROOMS[key]["kind"])
    if key == "room1":
        vi = A.ValueIteration(env, gamma=p["gamma"], theta=p["theta"])
        res = vi.run()
        entry.update(res=res, policy=res["policy"], final_policy=res["final_policy"])
    else:
        common = dict(alpha=p["alpha"], gamma=p["gamma"], epsilon=p["epsilon"],
                      epsilon_decay=p["epsilon_decay"], epsilon_min=p["epsilon_min"],
                      episodes=p["episodes"], seed=0)
        if key == "room4":
            agent = A.LinearFAAgent(env, optimistic_init=p["optimistic_init"],
                                    n_tilings=p["n_tilings"], n_bins=p["n_bins"],
                                    max_steps=p["max_steps"], **common)
        elif key == "room2":
            agent = A.Sarsa(env, max_steps=p["max_steps"], **common)
        else:  # room3 & room5 -> Q-Learning
            agent = A.QLearning(env, max_steps=p["max_steps"], **common)
        res = agent.train(snapshots=6, progress=cb)
        entry.update(res=res, final_policy=res["final_policy"],
                     policy=getattr(agent, "Q", None) and
                            {s: int(a.argmax()) for s, a in agent.Q.items()})
    bar.empty()
    store()[key] = entry


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def board_html(key, meta, agent=None, policy=None, obstacles=None, vision=None, trail=None):
    if ROOMS[key]["kind"] in ("dp", "grid"):
        return U.render_grid_html(meta, agent=agent, policy=policy)
    return U.render_space_svg(meta, agent=agent, obstacles=obstacles,
                              vision=vision, trail=trail)


def eval_roll(key, entry, policy, seed=1):
    """Run one greedy episode with `policy` (no rendering)."""
    env = build_env(key, entry["params"], seed=seed)
    return A.rollout(env, policy, max_steps=entry["params"].get("max_steps", 600) + 50)


def frames_for(key, entry, policy, seed=1):
    """Greedy episode rendered frame-by-frame to HTML for the replay player."""
    roll = eval_roll(key, entry, policy, seed)
    meta, vision = entry["meta"], (entry["params"].get("vision") if key == "room5" else None)
    htmls = [board_html(key, meta, agent=f["agent"],
                        obstacles=f.get("obstacles"), vision=vision) for f in roll["frames"]]
    return htmls, roll


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_simulation(key, entry):
    r = ROOMS[key]
    st.markdown(f"### {r['emoji']} Room {key[-1]} · {r['label']}")
    st.caption(r["plot"])
    meta = entry["meta"] if entry else build_env(key, default_params(key)).render_meta()

    if not entry:
        st.info("Configure the hyperparameters on the left and press **🚀 Train** "
                "to teach Hezki this room. The static layout is shown below.")
        embed(board_html(key, meta), height=GRID_H if r["kind"] in ("dp", "grid") else SPACE_H)
        return

    policy = entry.get("policy") if r["kind"] in ("dp", "grid") else None
    if r["kind"] in ("dp", "grid"):
        st.caption("Arrows show the learned greedy policy · 🐕 start · 🚪 exit "
                   "· 🧊 ice · 🧱 wall · 🕳️ pit · 🕶️ clones")
        embed(board_html(key, meta, agent=meta["start"], policy=policy), height=GRID_H)
    else:
        roll = eval_roll(key, entry, entry["final_policy"], seed=1)
        trail = [f["agent"] for f in roll["frames"]]
        st.caption(("🏎️ Hovercar · 🚗 parked cars · green = exit corner"
                    if key == "room4" else
                    "🚀 escape pod · 🛸 neuralyzer-drones · blue band = vision cone"))
        # show final greedy trajectory as a static trail + end position
        last = roll["frames"][-1]
        embed(board_html(key, meta, agent=last["agent"],
                                   obstacles=last.get("obstacles"),
                                   vision=entry["params"].get("vision") if key == "room5" else None,
                                   trail=trail if key == "room4" else None),
                        height=SPACE_H)
        ok = "✅ escaped" if roll["success"] else "❌ did not finish"
        st.metric("Greedy evaluation", f"{ok} · {roll['steps']} steps · return {roll['reward']:.1f}")


def tab_charts(key, entry):
    if not entry:
        st.info("Train the room to see learning-progress charts.")
        return
    res = entry["res"]
    if key == "room1":
        c1, c2 = st.columns(2)
        c1.plotly_chart(U.convergence_chart(res["deltas"]), use_container_width=True)
        c2.plotly_chart(U.value_heatmap(res["V"], entry["meta"]), use_container_width=True)
        st.success(f"Converged in **{res['iterations']}** sweeps "
                   f"(θ = {entry['params']['theta']:g}). "
                   f"V(start) = {res['V'][entry['meta']['start']]:.2f}")
    elif key in ("room2", "room3"):
        c1, c2 = st.columns(2)
        c1.plotly_chart(U.reward_curve(res["rewards"]), use_container_width=True)
        c2.plotly_chart(U.epsilon_curve(res["epsilons"]), use_container_width=True)
        if key == "room3":
            roll = eval_roll(key, entry, entry["final_policy"], seed=1)
            path = [f["agent"] for f in roll["frames"]]
            st.plotly_chart(U.path_compare(entry["meta"], path), use_container_width=True)
    elif key == "room4":
        c1, c2 = st.columns(2)
        c1.plotly_chart(U.reward_curve(res["rewards"], window=50), use_container_width=True)
        c2.plotly_chart(U.length_curve(res["lengths"], window=50,
                        title="Episode duration (moving avg)"), use_container_width=True)
    else:  # room5
        st.plotly_chart(U.length_curve(res["lengths"], window=50,
                        title="Survival time per episode", ylab="Steps survived"),
                        use_container_width=True)
        st.plotly_chart(U.reward_curve(res["rewards"], window=50), use_container_width=True)


def tab_replay(key, entry, random_clicked):
    if not entry:
        st.info("Train the room, then replay what Hezki learned at each stage here.")
        return
    r = ROOMS[key]

    if r["kind"] == "dp":
        st.caption("Dynamic Programming computes the optimal policy directly — there "
                   "are no training episodes. Watch Hezki execute the optimal plan:")
        htmls, roll = frames_for(key, entry, entry["final_policy"], seed=1)
        st.caption(f"Optimal greedy run · {'escaped ✅' if roll['success'] else 'failed ❌'} "
                   f"· {roll['steps']} steps · return {roll['reward']:.1f}")
        embed(U.render_player(htmls, delay_ms=220), height=GRID_H + 110)
        return

    if key == "room5" and random_clicked:
        seed = int(st.session_state.get("rand_seed", 0)) + 1
        st.session_state["rand_seed"] = seed
        htmls, roll = frames_for(key, entry, entry["final_policy"], seed=1000 + seed)
        st.caption(f"🎲 Random test room (seed {seed}) · final policy · "
                   f"{'survived ✅' if roll['success'] else 'hit a drone ❌'} "
                   f"— {roll['steps']} steps")
        embed(U.render_player(htmls, delay_ms=60), height=SPACE_H + 110)
        return

    snaps = entry["res"]["snapshots"]
    labels = [f"episode {ep}" if ep < snaps[-1][0] else f"episode {ep} (final)"
              for ep, _ in snaps]
    st.caption("Replay a greedy episode using the policy **as it was at a given "
               "training stage** — watch Hezki improve.")
    choice = st.select_slider("Training stage", options=list(range(len(snaps))),
                              value=len(snaps) - 1, format_func=lambda i: labels[i],
                              key=f"stage_{key}")
    ep, policy = snaps[choice]
    htmls, roll = frames_for(key, entry, policy, seed=1)
    ok = "escaped ✅" if roll["success"] else "failed ❌"
    st.caption(f"Stage: **{labels[choice]}** · {ok} · {roll['steps']} steps · return {roll['reward']:.1f}")
    h = (GRID_H if r["kind"] in ("dp", "grid") else SPACE_H) + 110
    embed(U.render_player(htmls, delay_ms=90), height=h)


def default_params(key):
    """Minimal params so an untrained room can still render its static layout."""
    return dict(v_max=3.0, max_steps=800, shaping=True, shaping_coef=5.0,
                hard_walls=False, vision=3.0, spawn_every=15)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    key, p, train_clicked, random_clicked = sidebar()

    st.title("🐕 Hezki the Dog vs. the Men in Black")
    st.caption("A Reinforcement-Learning escape room — five movie sets, five algorithms, "
               "rising difficulty. Agent J wants to neuralyze Hezki; help him escape.")

    if train_clicked:
        with st.spinner("Hezki is learning…"):
            train(key, p)

    entry = store().get(key)
    t1, t2, t3 = st.tabs(["🎬 Room Simulation (HTML)", "📈 Training Metrics", "⏪ Episode Replay"])
    with t1:
        tab_simulation(key, entry)
    with t2:
        tab_charts(key, entry)
    with t3:
        tab_replay(key, entry, random_clicked)


if __name__ == "__main__":
    main()
