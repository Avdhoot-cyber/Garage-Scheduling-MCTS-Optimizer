# 🚗 Garage Scheduling using MCTS

## 📌 Overview
This project implements an intelligent **Garage Scheduling System** that dynamically assigns tasks to mechanics while considering:

- Task dependencies (DAG structure)
- Probabilistic task generation
- Mechanic fatigue constraints
- Real-time schedule adaptation

The system combines:
- **Deterministic scheduling (initial plan)**
- **Monte Carlo Tree Search (MCTS)** for live decision-making

---

## ⚙️ Problem Description

Each car type is represented as a **Directed Acyclic Graph (DAG)**:
- Nodes → Tasks
- Edges → Dependency + probability of spawning next task

### Constraints:
- Each task takes **1 time unit**
- After **k consecutive tasks**, a mechanic must take a **1-unit break**
- Tasks may probabilistically generate follow-up tasks

---

## 🎯 Objective

1. Generate an **initial optimal schedule** ignoring probabilities  
2. Dynamically **update the schedule** when new tasks appear  

---

## 🧠 Approach

### 1. Deterministic Scheduler
- Treat all probabilities = 1
- Compute baseline schedule using:
  - Topological ordering
  - Critical Path prioritization

---

### 2. MCTS-Based Live Scheduling

At each decision point:
- Generate possible assignments
- Use MCTS to:
  - Explore scheduling decisions
  - Simulate future outcomes
  - Select best action

#### MCTS Components:
- **Selection**: UCB-based node selection
- **Expansion**: Generate new scheduling states
- **Simulation**: Rollout using heuristic policy
- **Backpropagation**: Update rewards

---

### 3. Simulation Engine

- Event-driven (not step-by-step)
- Tracks:
  - Ready tasks
  - Tasks in progress
  - Mechanic availability
  - Fatigue & idle time

---

## 📊 Features

✅ Multi-mechanic scheduling  
✅ Fatigue-aware workload balancing  
✅ Probabilistic task generation  
✅ Critical path prioritization  
✅ Real-time adaptive scheduling  
✅ Interactive UI using Streamlit  
✅ Gantt chart visualization  

---

## 📂 Input Format
