# 🐕 Hezki the Dog vs. the Men in Black — a Reinforcement-Learning Escape Room

A Streamlit escape-room game built for the **Reinforcement-Learning final project**
(תשפ"ו). Hezki the dog is on the run from **Agent J**, who wants to *neuralyze* (erase) his
memory. Hezki escapes through **five Hollywood movie sets** of rising difficulty — each one a
different RL problem solved by a different algorithm.

| # | Room | Movie | Model | Algorithm | RL concept |
|---|------|-------|-------|-----------|------------|
| 1 | ❄️ Frozen Archive | *Ice Age* | **known** | **Value Iteration** | Dynamic Programming, slippery (stochastic) transitions |
| 2 | 🏛️ Dark Temple | *Raiders of the Lost Ark* | unknown | **SARSA** | on-policy TD control, ε-greedy |
| 3 | 🕶️ Cloning Lab | *The Matrix* | unknown | **Q-Learning** | off-policy TD control (Cliff Walking) |
| 4 | 🏎️ Hovercar Garage | *The Fast and the Furious* | unknown | **Tile-coding linear FA** | function approximation over a continuous 4-D state |
| 5 | 🚀 Space Escape | *Star Wars* | unknown | **Q-Learning + sensor** | dynamic obstacles & partial observability (POMDP) |

Every room lets you tune **all** algorithm hyperparameters in the sidebar, watch **live
training-progress charts**, and **replay single episodes** to see what Hezki learned at each
stage of training.

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### Deploy to Streamlit Community Cloud
1. Push this repo to GitHub.
2. On <https://share.streamlit.io> → **New app** → pick the repo, branch, and `app.py`.
3. `requirements.txt` in the repo root is picked up automatically. Deploy.

---

## Project structure

```
app.py            Streamlit UI: sidebar controls, 3 tabs, session state, replay
environments.py   The 5 room environments + shared grid / slip / physics logic
algorithms.py     ValueIteration, SARSA, QLearning, TileCoder + LinearFAAgent, rollout
utils.py          HTML grid renderer, SVG arena renderer, Plotly charts, JS replay player
tests/test_core.py  Unit tests for the environment dynamics and the learners
```

The three main tabs are: **Room Simulation (HTML)**, **Training Metrics (Charts)**, and
**Episode Replay** (a browser-side player with play / pause / scrub / speed).

---

## The rooms in detail

> **Coordinate convention (grid rooms):** state `(x, y)`, `x` = column, `y` = row, both in
> `0..9`. `Right`=x+1, `Left`=x−1, `Up`=y+1, `Down`=y−1. The board is drawn with `y=0` at the
> bottom. **Terminal states are absorbing** everywhere (the episode ends on entry — no further
> sliding), the single exception being the Room-3 cliff (see below).
>
> **Slip model (Rooms 1–2):** from a slippery tile an action lands on the *intended* cell with
> prob **0.70**, on the cell **90° left** with **0.10**, **90° right** with **0.10**, and
> **backwards** with **0.10**. Any outcome that would hit a wall or the boundary keeps the
> agent in place.

### Room 1 · The Frozen Archive — Value Iteration (DP)
* **State space** `S = {(x,y) : 0 ≤ x,y ≤ 9}`, minus wall cells. Start `(0,0)`, exit `(9,9)`.
* **Actions** Up / Down / Left / Right.
* **Model** *known* → Value Iteration:
  `V(s) ← maxₐ Σₛ′ P(s′|s,a)·[R + γ·V(s′)]`, repeated until `max|ΔV| < θ`.
* **Layout (Level 1 — easiest)** a **frozen lake** of slippery ice fills the middle
  (`x∈3..6, y∈3..5`) with three ice-boulder walls `(6,2),(2,6),(7,6)`. No deadly traps — the
  agent may go the safe way around or risk the slippery short-cut; DP weighs the slips exactly.
* **Rewards** step `−1`, reaching `(9,9)` `+100` (terminal).
* **Chart** max `|ΔV|` per sweep (convergence) + a heat-map of the learned `V(s)`.
* **Good hyperparameters** `γ = 0.99`, `θ = 1e-4` → converges in **~25 sweeps**.

### Room 2 · The Dark Temple — SARSA
* **State / actions** same grid; start `(0,0)`, exit `(9,9)`. Model *unknown* (learned by
  interaction). Update: `Q(s,a) ← Q(s,a) + α·[r + γ·Q(s′,a′) − Q(s,a)]` (**on-policy**).
* **Layout (Level 2 — moderate)** a booby-trapped temple: a gauntlet of spike-pit traps
  `(2,2),(5,2),(3,5),(6,4),(4,7),(7,7)` (terminal, `−100`, **not** slippery), slippery mud
  `(2,3),(4,5),(5,5),(3,6),(6,6)` beside the pits, and stone walls `(1,7),(7,3),(5,8),(8,5)`.
  SARSA must learn a *safe* route that keeps clear of the mud-next-to-pit cells.
* **Rewards** step `−1`, pit `−100`, exit `+100`.
* **Charts** ε-decay and cumulative reward per episode.
* **Good hyperparameters** `α = 0.1`, `γ = 0.99`, `ε₀ = 1.0`, `ε-decay = 0.995`, `ε_min = 0.01`,
  `episodes = 1500`, `max_steps = 400` → ~100 % escape rate.

### Room 3 · The Cloning Lab — Q-Learning (Cliff Walking)
* **State / actions** grid; start `(0,0)`, exit `(9,0)`; deterministic movement. Update uses the
  **greedy** bootstrap `Q(s,a) ← Q(s,a) + α·[r + γ·maxₐ′Q(s′,a′) − Q(s,a)]` (**off-policy**).
* **Layout (Level 3 — hard)** the bottom row `y=0`, `x = 1..8`, is the *cliff of clones*;
  two "firewall" pillars `(3,2),(6,2)` force the safe route up and over, sharpening the
  risk/reward of hugging the cliff.
* **Rewards** step `−1`, cliff `−100` **and reset to start** (episode continues — canonical Cliff
  Walking), exit `+100` (terminal).
* **Charts** reward per episode + a plot of the greedy path (the **aggressive cliff-hugging route**
  along `y=1`).
* **Good hyperparameters** `α = 0.1`, `γ = 0.99`, `ε₀ = 1.0`, `ε-decay = 0.995`,
  `episodes = 1000`, `max_steps = 400` → an **11-step** cliff-hugging solution.

### Room 4 · The Hovercar Garage — Function Approximation
* **State** continuous `[X, Y, Vₓ, V_y]`, arena `0 ≤ X,Y ≤ 10 m`. Start `(1,1)`, exit region
  `X > 8.5 ∧ Y > 8.5`.
* **Actions** 9 discrete accelerations `(aₓ, a_y) ∈ {−1,0,1}²`.
* **Physics** (`dt = 0.02 s`): `V ← clip(V + a, ±v_max)`, then `X ← X + Vₓ·dt`, `Y ← Y + V_y·dt`.
* **Obstacles** two parked cars `(X 3–4, Y 0–6)` and `(X 6–7, Y 4–10)` — these force a
  *serpentine*: you can only cross `X∈[3,4]` above `Y=6` and only cross `X∈[6,7]` below `Y=4`.
* **Learner** **tile coding** (8 tilings × 8 bins over the 4-D box) feeding a linear
  `Q(s,a) = wₐ·φ(s)`, trained by **semi-gradient Q-learning**. This is genuine function
  approximation — the table is replaced by weight vectors over overlapping tilings.
* **Rewards** step `−0.1`, crash `−100` (terminal), exit `+100` (terminal).
* **Charts** moving average of episode duration + reward progression.
* **Good hyperparameters** `α = 0.5`, `γ = 0.99`, `ε = 0.1`, `optimistic init Q₀ = 100`,
  `tilings = 8`, `bins = 8`, `v_max = 3`, shaping on (`coef = 5`), `episodes = 2000`
  → **100 % greedy success**, ~347-step weave (≈ 1 min to train).

### Room 5 · The Space Escape — dynamic obstacles + partial observability
* **State the agent sees** only `[Vₓ, d]` — its own x-velocity and the distance `d` to the
  nearest drone **directly ahead** (aligned within the 0.5 m collision width and within the
  **vision range** `X_obs`); `d = −1` when nothing is in range. It does **not** know the full
  obstacle field — a deliberate POMDP handled with tabular Q-learning over the discretised sensor.
* **Agent** moves on the X axis only (`0 ≤ X ≤ 10`, `Y` fixed at `2.0`), `aₓ ∈ {−1,0,1}`,
  `X ← X + Vₓ·dt`.
* **Obstacles** neuralyzer-drones spawn at `Y = 10`, random `X ∈ [1,9]`, fall at `−5 m/s`, width
  `0.5 m`, one every `N` steps.
* **Rewards** `+1` per surviving step, collision `−1000` (terminal).
* **Chart** max survival time (steps) per episode.
* **Extras** the sidebar exposes the **vision range `X_obs`** and a **“Generate Random Room &
  Test Policy”** button that runs the learned policy on a fresh random drone stream.
* **Good hyperparameters** `vision = 3`, `spawn N = 15`, `α = 0.1`, `γ = 0.95`, `ε₀ = 1.0`,
  `ε-decay = 0.999`, `episodes = 3000`, `v_max = 5`, survival cap = 500.

---

## Design decisions & interpretations (for the defense)

The task brief left a few points under-specified or self-contradictory. The choices below are
deliberate; each keeps the pedagogical goal intact.

1. **Reward on the goal transition** *replaces* the step cost (entering the exit gives `+100`, not
   `+100 − 1`). Fewer steps ⇒ higher return, matching “faster escape = higher reward”.
2. **Room 3 cliff.** The brief calls it both *terminal* and *“sends the agent back to start.”* We
   implement the **canonical Cliff-Walking** rule — `−100` **and teleport to start, episode
   continues**; only the exit is terminal. This is exactly what produces the famous
   cliff-hugging optimal path.
3. **Room 4 needs help to be learnable.** The exact obstacle geometry makes a sparse-reward,
   momentum-controlled maze that random exploration almost never solves. Three standard,
   *policy-preserving* techniques make it reliable, all toggleable in the UI:
   * **Potential-based reward shaping** (Ng et al., 1999) using an **obstacle-aware wave-front
     distance-to-goal** as the potential Φ, with `γ_shaping = 1` (telescoping ⇒ no “camping”
     reward). Shaping cannot change the optimal policy; it only densifies the signal.
   * **Optimistic initialisation** (`Q₀ = 100`) to drive systematic exploration.
   * **Soft outer walls** — the parked *cars* are fatal `−100` crashes, but bumping the garage
     boundary just clamps position with a small `−1` nudge (set *hard walls* to restore the strict
     `−100`-terminal reading).
4. **Velocity is clipped** to `±v_max` in Rooms 4–5 (the brief’s `V ← V + a` is otherwise
   unbounded). The brief’s docx describes velocity as discrete; the MD spec describes
   *acceleration* actions with accumulating velocity — we follow the MD (richer, momentum-based).
5. **Room 5 has no fixed exit;** it is a survival task. Surviving to the step cap counts as
   “escaped”.
6. **Levels were redesigned as a difficulty ramp.** Rather than the brief's scattered
   coordinates, each grid is an intentional, movie-themed level whose hazards escalate
   (Room 1: ice only → Room 2: pits + mud → Room 3: cliff of clones). Each cell shows its
   reward so the learned policy is interpretable, and a legend explains every tile.
7. **Reproducible & robust evaluation.** Every environment uses a **seeded RNG**, so training
   and replay are reproducible. Because slippery rooms are stochastic, the replay reports a
   **success rate over many runs** (not one lucky/unlucky episode) and plays a representative
   run — so “more episodes” never *looks* worse due to a single random slip.
