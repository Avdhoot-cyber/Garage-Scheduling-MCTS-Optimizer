import streamlit as st
import random
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
from copy import deepcopy

# =========================
# CRITICAL PATH
# =========================
def compute_critical_paths(graphs):
    cp = {}
    for gid, G in enumerate(graphs):
        order = list(nx.topological_sort(G))[::-1]
        dist = {node: 0 for node in G.nodes()}
        for u in order:
            for _, v in G.out_edges(u):
                dist[u] = max(dist[u], 1 + dist[v])
        for node in G.nodes():
            cp[(gid, node)] = dist[node]
    return cp

# =========================
# INPUT
# =========================
def read_input(file):
    graphs = []
    lines = [
        line.decode("utf-8").strip()
        for line in file.readlines()
        if line.strip() and not line.decode("utf-8").strip().startswith("#")
    ]

    idx = 0
    G_count = int(lines[idx]); idx += 1
    M = int(lines[idx]); idx += 1
    k = int(lines[idx]); idx += 1

    for _ in range(G_count):
        N = int(lines[idx]); idx += 1
        Gx = nx.DiGraph()
        Gx.add_nodes_from(range(N))

        while idx < len(lines):
            parts = lines[idx].split()
            if len(parts) != 3:
                break
            u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
            Gx.add_edge(u, v, prob=p)
            idx += 1

        graphs.append(Gx)

    return graphs, M, k

# =========================
# STATE
# =========================
class State:
    def __init__(self, graphs, M, k, cp):
        self.graphs = graphs
        self.M = M
        self.k = k
        self.cp = cp

        self.time = 0
        self.ready = []
        self.indegree = {}
        self.completed = set()
        # FIX: track which task each mechanic is currently working on
        self.in_progress = {}   # mechanic_id -> (gid, node)

        self.mechanics = {
            i: {"busy_until": 0, "count": 0, "idle": 0}
            for i in range(M)
        }

        for gid, G in enumerate(graphs):
            for node in G.nodes():
                self.indegree[(gid, node)] = G.in_degree(node)
                if self.indegree[(gid, node)] == 0:
                    self.ready.append((gid, node))

    def is_terminal(self):
        # FIX: must also check no tasks are currently in progress
        no_ready       = len(self.ready) == 0
        no_in_progress = len(self.in_progress) == 0
        all_free       = all(m["busy_until"] <= self.time for m in self.mechanics.values())
        return no_ready and no_in_progress and all_free

# =========================
# NEXT EVENT TIME
# =========================
def next_event_time(state):
    future = [
        m["busy_until"] for m in state.mechanics.values()
        if m["busy_until"] > state.time
    ]
    return min(future) if future else state.time + 1

# =========================
# ACTIONS
# =========================
def get_actions(state):
    free_mechanics = [
        m for m in range(state.M)
        if state.mechanics[m]["busy_until"] <= state.time
    ]

    if not free_mechanics or not state.ready:
        return [None]

    # Prioritise tasks with the longest remaining critical path
    tasks = sorted(state.ready, key=lambda x: -state.cp.get(x, 0))
    # Cap so the combinatorial space stays tractable
    tasks = tasks[:min(len(tasks), len(free_mechanics) + 2)]

    actions = []
    n_assign = min(len(free_mechanics), len(tasks))

    for k in range(1, n_assign + 1):
        for mech_combo in itertools.combinations(free_mechanics, k):
            for task_combo in itertools.permutations(tasks, k):
                actions.append(list(zip(mech_combo, task_combo)))
                if len(actions) >= 50:
                    return actions

    return actions if actions else [None]

# =========================
# STEP  (core simulation tick)
# =========================
def step(state, action, deterministic=False):
    """
    One simulation tick:
      1. Process completions — mechanics whose busy_until <= state.time
         finish their tasks and may spawn probabilistic follow-ups.
      2. Apply the chosen action — assign ready tasks to free mechanics.
      3. Mark un-assigned free mechanics as idle.
      4. Advance clock to the next event.

    deterministic=True is used by the initial planner: every edge is
    treated as certain (prob = 1) so the full expected graph is scheduled.
    """
    state = deepcopy(state)

    # ---- 1. Process completions ----------------------------------------
    newly_done = [
        m for m in list(state.in_progress.keys())
        if state.mechanics[m]["busy_until"] <= state.time
    ]
    for m in newly_done:
        gid, node = state.in_progress.pop(m)
        state.completed.add((gid, node))

        for _, v, data in state.graphs[gid].out_edges(node, data=True):
            # FIX: random() < prob  (higher prob -> more likely to spawn)
            # deterministic mode treats every edge as certain
            threshold = 1.0 if deterministic else data["prob"]
            if random.random() < threshold:
                state.indegree[(gid, v)] -= 1
                if state.indegree[(gid, v)] == 0:
                    state.ready.append((gid, v))

    # ---- 2. Apply action -----------------------------------------------
    assigned = set()
    if action is not None:
        for m, (gid, node) in action:
            # Guard: task might no longer be ready (stale MCTS action)
            if (gid, node) not in state.ready:
                continue

            assigned.add(m)
            state.ready.remove((gid, node))

            mech = state.mechanics[m]
            mech["count"] += 1

            finish = state.time + 1
            # Mandatory break after k consecutive tasks
            if mech["count"] >= state.k:
                finish += 1
                mech["count"] = 0

            mech["busy_until"] = finish
            state.in_progress[m] = (gid, node)

    # ---- 3. Count idle -------------------------------------------------
    for m in range(state.M):
        if state.mechanics[m]["busy_until"] <= state.time and m not in assigned:
            state.mechanics[m]["idle"] += 1

    # ---- 4. Advance clock ----------------------------------------------
    state.time = next_event_time(state)
    return state

# =========================
# REWARD
# =========================
def compute_reward(state):
    makespan  = state.time
    fatigue   = sum(m["count"] for m in state.mechanics.values())
    idle      = sum(m["idle"]  for m in state.mechanics.values())
    imbalance = (max(m["busy_until"] for m in state.mechanics.values()) -
                 min(m["busy_until"] for m in state.mechanics.values()))
    return -makespan - 0.7 * fatigue - 0.3 * idle - 0.5 * imbalance

# =========================
# ROLLOUT  (greedy simulation from a state)
# =========================
def rollout(state):
    state = deepcopy(state)
    for _ in range(100):
        if state.is_terminal():
            break
        actions = get_actions(state)
        if random.random() < 0.8:
            best = max(
                actions,
                key=lambda a: 0 if a is None else sum(state.cp.get(t, 0) for _, t in a)
            )
        else:
            best = random.choice(actions)
        state = step(state, best)
    return compute_reward(state)

# =========================
# MCTS
# =========================
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.children = []
        self.visits   = 0
        self.reward   = 0.0

    def best_child(self, c=1.4):
        return max(
            self.children,
            key=lambda x: x.reward / (x.visits + 1e-6)
                          + c * math.sqrt(math.log(self.visits + 1) / (x.visits + 1e-6))
        )

def mcts(root_state, iterations):
    root = MCTSNode(deepcopy(root_state))

    for _ in range(iterations):
        node = root

        # Selection
        while node.children:
            node = node.best_child()

        # Expansion
        if not node.state.is_terminal():
            for action in get_actions(node.state):
                child_state = step(node.state, action)
                node.children.append(MCTSNode(child_state, node, action))
            if node.children:
                node = random.choice(node.children)

        # Simulation
        reward = rollout(node.state)

        # Backpropagation
        while node:
            node.visits  += 1
            node.reward  += reward
            node = node.parent

    if not root.children:
        return None
    return root.best_child(c=0)   # exploit-only for final pick

# =========================
# DETERMINISTIC INITIAL SCHEDULE
# =========================
def deterministic_schedule(graphs, M, k, cp):
    """
    Build an initial plan treating ALL probabilistic edges as certain
    (deterministic=True in step).  This gives the 'expected-case' schedule
    before any live execution begins.
    """
    state = State(graphs, M, k, cp)
    log   = []

    max_steps = 500
    for _ in range(max_steps):
        if state.is_terminal():
            break

        free  = [m for m in range(M) if state.mechanics[m]["busy_until"] <= state.time]
        tasks = sorted(state.ready, key=lambda x: -cp.get(x, 0))

        if not free or not tasks:
            # Nothing to assign right now — just advance time and let
            # completions (processed at start of next step) unlock new tasks.
            state = step(state, None, deterministic=True)
            continue

        action = [(m, t) for m, t in zip(free, tasks)]
        log.append((state.time, action))
        state = step(state, action, deterministic=True)

    return state, log

# =========================
# MAIN RUN  (FIX: MCTS now actually executes)
# =========================
def run_mcts(graphs, M, k, iterations):
    cp = compute_critical_paths(graphs)

    # --- Initial deterministic plan (for display / warm reference) ---
    _, initial_log = deterministic_schedule(graphs, M, k, cp)

    # --- Live execution with MCTS + probabilistic task generation ---
    state = State(graphs, M, k, cp)
    log   = []

    max_steps = 500
    for _ in range(max_steps):
        if state.is_terminal():
            break

        # If no free mechanic or no ready task, advance time so completions
        # (processed at the start of the next step) can unlock new work.
        free  = [m for m in range(M) if state.mechanics[m]["busy_until"] <= state.time]
        if not free or not state.ready:
            state = step(state, None)
            continue

        # FIX: MCTS runs at every real decision point
        best_node = mcts(state, iterations)

        if best_node is None:
            action = get_actions(state)[0]
        else:
            action = best_node.action

        if action:
            log.append((state.time, action))

        state = step(state, action)

    return log, initial_log

# =========================
# UI
# =========================
st.title("Garage Scheduling — MCTS Optimizer")

uploaded_file = st.file_uploader("Upload input.txt")

if uploaded_file:
    graphs, M, k = read_input(uploaded_file)

    st.write(f"Loaded **{len(graphs)}** car graph(s), **{M}** mechanic(s), fatigue threshold **k={k}**")

    iterations = st.slider("MCTS Iterations", 50, 300, 100)

    col1, col2 = st.columns(2)

    if col1.button("Run Initial (Deterministic) Schedule"):
        _, log = deterministic_schedule(graphs, M, k, compute_critical_paths(graphs))
        rows = []
        for t, act in log:
            for m, (gid, node) in act:
                rows.append({"Time": t, "Mechanic": f"M{m}", "Task": f"G{gid}_T{node}", "Type": "planned"})

        df = pd.DataFrame(rows)
        st.subheader("Initial Plan")
        st.dataframe(df)

        fig, ax = plt.subplots(figsize=(8, 3))
        for _, row in df.iterrows():
            ax.barh(row["Mechanic"], 1, left=row["Time"], color="steelblue", edgecolor="white")
        ax.set_xlabel("Time")
        ax.set_title("Deterministic Schedule (all follow-ups assumed)")
        st.pyplot(fig)

    if col2.button("Run Live MCTS Schedule (with probabilistic tasks)"):
        with st.spinner("Running MCTS..."):
            log, initial_log = run_mcts(graphs, M, k, iterations)

        rows = []
        for t, act in log:
            for m, (gid, node) in act:
                rows.append({"Time": t, "Mechanic": f"M{m}", "Task": f"G{gid}_T{node}"})

        df = pd.DataFrame(rows)
        st.subheader("MCTS Live Schedule")
        if df.empty:
            st.warning("No tasks were scheduled. Check your input file.")
        else:
            st.dataframe(df)
            makespan = df["Time"].max() + 1 if not df.empty else 0
            st.metric("Makespan", makespan)

            fig, ax = plt.subplots(figsize=(8, 3))
            colors = plt.cm.tab10.colors
            mechanics = sorted(df["Mechanic"].unique())
            mech_idx  = {m: i for i, m in enumerate(mechanics)}
            for _, row in df.iterrows():
                ax.barh(
                    row["Mechanic"], 1, left=row["Time"],
                    color=colors[mech_idx[row["Mechanic"]] % len(colors)],
                    edgecolor="white"
                )
            ax.set_xlabel("Time")
            ax.set_title("MCTS Optimised Schedule (probabilistic execution)")
            st.pyplot(fig)