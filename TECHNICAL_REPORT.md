# DynNav — Research & Technical Summary

**Author:** Panagiota Grosdouli  
**Institution:** Electrical & Computer Engineering, Democritus University of Thrace  
**Repository:** [DynNav on GitHub](https://github.com/PanagiotaGr/DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments)  
**License:** Apache 2.0

---

## 1. Overview

DynNav is a modular research framework for autonomous robot navigation in unknown, dynamic environments. The system addresses the fundamental challenge of navigation under uncertainty by integrating classical planning algorithms with probabilistic methods, formal safety verification, learning-based components, and multi-robot coordination — implemented on ROS 2 with TurtleBot3 hardware support.

The framework comprises 26 research contributions spanning uncertainty estimation, risk-aware planning, formal safety, reinforcement learning, generative perception models, causal inference, adversarial robustness, and multi-robot coordination. Contributions range from fully implemented and experimentally validated modules to mathematically motivated prototypes establishing directions for future research.

**Key design principle:** modularity. Each contribution exposes a well-defined interface, allowing risk signals, uncertainty estimates, and safety constraints to be composed without tight coupling.

---

## 2. Research Axes

| Axis | Contributions | Description |
|------|--------------|-------------|
| **Uncertainty** | 02, 12, 23, 24 | Belief-state estimation, diffusion occupancy, 3D-GS, NeRF variance |
| **Risk-Aware Planning** | 01, 03, 06, 07 | CVaR path planning, learned heuristics, resource constraints, NBV |
| **Formal Safety** | 04, 05, 18 | Returnability, safe-mode FSM, STL+CBF runtime shields |
| **Security** | 08, 25 | Innovation-based IDS, adversarial attack evaluation |
| **Learning** | 13, 16, 21, 22 | World models, federated learning, PPO, curriculum RL |
| **Foundation Models** | 11, 19, 20 | VLM goals, LLM mission parsing, failure explanation |
| **Semantic Mapping** | 15, 17, 23, 24 | Topological maps, neuromorphic sensing, 3D-GS, NeRF |
| **Causal Reasoning** | 14, 20 | SCM failure attribution, multimodal diagnosis |
| **Multi-Robot** | 09, 16, 26 | Coordination, federated learning, BFT consensus |
| **Human-Robot** | 10, 19 | Ethical zones, natural language missions |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│               Foundation Model Layer                     │
│        VLM (11) · LLM Planner (19) · Explainer (20)    │
├──────────────────┬──────────────────┬────────────────────┤
│  Planning Core   │  Safety Layer    │  Coordination      │
│  A* risk (03)    │  STL+CBF (18)    │  BFT Swarm (26)    │
│  Learned h (01)  │  Safe-Mode (05)  │  Federated (16)    │
│  NBV (07)        │  Returnab. (04)  │  Multi-Robot (09)  │
├──────────────────┴──────────────────┴────────────────────┤
│                  Perception & Uncertainty                 │
│  EKF/UKF (02) · Diffusion (12) · 3D-GS (23) · NeRF (24) │
│  DVS+SNN (15) · Topo Maps (17) · IDS (08) · Causal (14)  │
├──────────────────────────────────────────────────────────┤
│              ROS 2 Humble / TurtleBot3 / Gazebo          │
└──────────────────────────────────────────────────────────┘
```

The system is implemented primarily in Python 3.10+ with ROS 2 integration. Core numerical dependencies are NumPy-based; production neural network components require PyTorch substitution for the current stub implementations.

---

## 4. Contributions

---

### Contribution 01 — Learned A* Heuristics

**Problem:** The Manhattan/Euclidean heuristic in A* is admissible but loose, causing excessive node expansions in cluttered environments.

**Mathematical Formulation:**

Training objective (regression on historical episodes):
$$\mathcal{L}(\theta) = \frac{1}{N}\sum_i (h_\theta(s_i) - h^*(s_i))^2$$

Admissibility enforcement at inference:
$$\tilde{h}(s) = \min(h_\theta(s),\ h_{\text{naive}}(s)) \leq h^*(s)$$

**Method:** MLP regressor trained on (state, true cost-to-goal) pairs; clipped against the naive heuristic at inference to preserve A* optimality.

**Implementation:** Offline training pipeline; evaluation script `eval_astar_learned.py`; metrics: node expansions, path length, runtime.

**Results:** ~35% reduction in node expansions on benchmark maps; path optimality preserved by construction.

**Contribution Type:** Algorithmic (applied)  
**Maturity:** Partially implemented prototype

---

### Contribution 02 — Uncertainty Estimation

**Problem:** Noisy sensors produce unreliable state estimates; downstream planners need calibrated uncertainty representations.

**Mathematical Formulation:**

State-space model: $s_{t+1} = f(s_t, a_t) + w_t$, $z_t = h(s_t) + v_t$, $w_t \sim \mathcal{N}(0,Q)$, $v_t \sim \mathcal{N}(0,R)$

EKF update:
$$K_t = P_{t|t-1}H_t^\top(H_tP_{t|t-1}H_t^\top + R)^{-1}$$
$$\hat{s}_t = \hat{s}_{t|t-1} + K_t(z_t - h(\hat{s}_{t|t-1}))$$

**Method:** EKF (linearised) and UKF (sigma-point) for robot pose estimation; outputs belief $({\mu}_t, \Sigma_t)$ consumed by the risk planner.

**Implementation:** Standard Kalman filter pipeline; $\text{tr}(\Sigma_t)$ used as scalar uncertainty signal.

**Results:** Infrastructure contribution; correctness validated by downstream risk planner integration.

**Contribution Type:** Systems integration  
**Maturity:** Systems integration contribution

---

### Contribution 03 — Belief-Space & Risk-Aware Planning

**Problem:** Classical A* minimises expected cost but ignores tail risk — unsafe under rare but catastrophic collision scenarios.

**Mathematical Formulation:**

CVaR at confidence level $\alpha$:
$$\text{CVaR}_\alpha(X) = \mathbb{E}[X \mid X \geq \text{VaR}_\alpha(X)]$$

Risk-augmented A* cost:
$$f(s) = g(s) + \lambda \cdot \text{CVaR}_\alpha(p_{\text{col}}(s)) + h(s)$$

Planning objective:
$$\pi^* = \arg\min_\pi \sum_{s \in \pi} [c(s) + \lambda \cdot \text{CVaR}_\alpha(p_{\text{col}}(s))]$$

**Method:** Augment A* edge costs with CVaR risk derived from belief state; $\lambda$ trades off safety versus efficiency.

**Implementation:** Risk-weighted A* with configurable $\alpha$ and $\lambda$; parameter sweep experiments; ablation vs risk-agnostic baseline.

**Results:** Safety-efficiency Pareto curve characterised over $\lambda \in [0, \infty)$; higher $\lambda$ reduces collision rate at cost of path length.

**Contribution Type:** Algorithmic  
**Maturity:** Partially implemented prototype

---

### Contribution 04 — Irreversibility & Returnability

**Problem:** Greedy planners may commit the robot to states from which no recovery path exists under current map uncertainty.

**Mathematical Formulation:**

Returnability predicate:
$$\text{Ret}(s) = \mathbf{1}\left[\exists \pi: s \to s' \in \mathcal{S}_{\text{safe}},\ d_G(s, s') \leq H\right]$$

Irreversibility penalty:
$$c_{\text{irrev}}(s) = \gamma_{\text{irrev}} \cdot \mathbf{1}[\neg\text{Ret}(s)]$$

**Method:** Finite-horizon BFS backward from safe set; non-returnable states penalised in A* cost (soft) or blocked (hard constraint).

**Implementation:** Backward reachability on occupancy grid with configurable horizon $H$.

**Results:** Reduced dead-end entrapment rate; experimental depth limited in current codebase.

**Contribution Type:** Algorithmic  
**Maturity:** Mathematically motivated, lightly validated

---

### Contribution 05 — Safe-Mode Navigation

**Problem:** A single fixed policy cannot handle the full range of risk levels; a mode-switching mechanism is needed to adapt behaviour in real time.

**Mathematical Formulation:**

Threshold automaton with hysteresis:
$$m_{t+1} = \begin{cases} \text{SAFE} & r_t > \tau_{\text{hi}} \\ \text{EMERGENCY} & r_t > \tau_{\text{crit}} \\ \text{NORMAL} & m_t = \text{SAFE} \wedge r_t \leq \tau_{\text{lo}} \text{ for } T_{\text{hold}} \text{ steps} \end{cases}$$

**Method:** FSM over {NORMAL, SAFE, EMERGENCY}; per-mode velocity cap and obstacle inflation radius; triggered by CVaR risk signal or IDS alarm.

**Implementation:** Configurable thresholds; integration with Contribution 03 (risk) and Contribution 08 (IDS).

**Results:** Systems integration; validation through component tests.

**Contribution Type:** Systems integration  
**Maturity:** Systems integration contribution

---

### Contribution 06 — Energy & Connectivity-Aware Planning

**Problem:** A geometrically optimal path may be infeasible under battery budget or communication quality constraints.

**Mathematical Formulation:**

Constrained path planning:
$$\pi^* = \arg\min_\pi \sum_{e \in \pi} c(e) \quad \text{s.t.} \quad E(\pi) \leq B,\ \min_{s \in \pi} q(s) \geq Q_{\min}$$

where $E(\pi) = \sum_e P_{\text{motor}} \cdot d(e) \cdot \kappa(\text{terrain})$ and $q(s) \in [0,1]$ is link quality.

**Method:** Budget-constrained A* with pruning; charging station routing as fallback when direct path exceeds budget.

**Implementation:** Energy map and connectivity map as additional cost layers in A*.

**Results:** Path completion rate within budget evaluated; experimental coverage limited.

**Contribution Type:** Algorithmic (constrained optimisation)  
**Maturity:** Mathematically motivated, lightly validated

---

### Contribution 07 — Next-Best-View Exploration

**Problem:** Frontier-based exploration ignores information content; an information-theoretic criterion selects more informative viewpoints per unit travel cost.

**Mathematical Formulation:**

Map entropy: $H(\mathbf{m}) = -\sum_c [p_c \log p_c + (1-p_c)\log(1-p_c)]$

NBV criterion:
$$v^* = \arg\max_{v \in \mathcal{V}} \frac{H(\mathbf{m}) - \mathbb{E}_{z_v}[H(\mathbf{m}|z_v)]}{d(s_{\text{robot}}, v)}$$

**Method:** Occupancy entropy maintained via binary Bayes filter; IG estimated by ray-casting simulation; NBV selected by IG/cost ratio.

**Implementation:** Integrated with `ig_explorer/` module; metrics: coverage rate, entropy reduction, travel distance.

**Results:** Information-gain criterion validated against frontier baseline on simulated maps.

**Contribution Type:** Algorithmic + systems integration  
**Maturity:** Partially implemented prototype

---

### Contribution 08 — Security & Intrusion Detection

**Problem:** Sensor spoofing attacks corrupt state estimates without triggering standard navigation alarms; integrity monitoring is required.

**Mathematical Formulation:**

Innovation sequence under nominal conditions:
$$\nu_k = z_k - H_k\hat{s}_{k|k-1} \sim \mathcal{N}(0, S_k)$$

$\chi^2$-test:
$$\chi^2_k = \nu_k^\top S_k^{-1} \nu_k \sim \chi^2(m) \quad \text{(nominal)}; \quad \text{alarm if } \chi^2_k > \tau$$

CUSUM for gradual drift:
$$g_k = \max(0,\ g_{k-1} + \chi^2_k - \kappa); \quad \text{alarm if } g_k > h$$

**Method:** EKF innovation monitoring; single-step $\chi^2$ detection with controlled false-alarm rate; CUSUM for persistent drift; alarm triggers safe-mode.

**Implementation:** ROS 2 nodes in `cybersecurity_ros2/`; configurable $\alpha_{\text{FA}}$, $\kappa$, $h$.

**Results:** Detection rate and false-alarm rate characterised; integrated with Contribution 25 for full attack-defend evaluation.

**Contribution Type:** Systems integration + experimental benchmark  
**Maturity:** Partially implemented prototype

---

### Contribution 09 — Multi-Robot Coordination

**Problem:** Multiple robots sharing a workspace may produce conflicting paths; decentralised resolution avoids the single-point-of-failure of central coordination.

**Mathematical Formulation:**

Conflict-free requirement: $\forall i \neq j, \forall t:\ s_i(t) \neq s_j(t)$

Risk budget distribution: $\sum_i r_i \leq R_{\text{total}}$

Map disagreement metric: $d_{ij} = \|\text{map}_i \ominus \text{map}_j\|_1 > \tau_d$

**Method:** Gossip-based map sharing; priority-ordered path reservation; risk budget allocation; disagreement detection.

**Implementation:** Multi-agent simulation; priority reservation protocol; disagreement flagging.

**Results:** Conflict resolution validated in simulation; scalability analysis limited.

**Contribution Type:** Systems integration  
**Maturity:** Systems integration contribution

---

### Contribution 10 — Human-Aware & Ethics-Guided Navigation

**Problem:** Navigation in human-occupied spaces must respect ethical zone constraints and adapt to operator trust levels.

**Mathematical Formulation:**

Augmented cost:
$$f_{\text{total}}(s) = g(s) + \lambda_r r(s) + c_{\text{ethical}}(s) + \lambda_h \sum_j w_j e^{-\|s-p_j\|^2/2\sigma_j^2} + h(s)$$

Hard no-go: $s \notin \mathcal{Z}_{\text{hard}}$ always. Autonomy scaling: $\tau(t) \in [0,1]$.

**Method:** Ethical zones loaded from `ethical_zones.json`; Gaussian repulsion from detected humans; scalar trust parameter scales action autonomy.

**Implementation:** Zone configuration file; cost function augmentation; trust parameter interface.

**Results:** Conceptual prototype; ethical constraint compliance validated on configured zones.

**Contribution Type:** Research concept  
**Maturity:** Research concept / future direction

> **Note:** "Ethics" here refers to configurable zone avoidance and proximity costs — not learned or reasoned ethical decision-making. Framing should reflect this.

---

### Contribution 11 — VLM Navigation Agent

**Problem:** Metric goal specification is inaccessible to non-experts; visual scenes should be interpretable as semantic navigation goals by foundation models.

**Mathematical Formulation:**

VLM conditional output: $p_\theta(\text{goal}|I)$ structured as JSON $\{r, g, c, (u,v)\}$

Back-projection to metric space (pinhole):
$$\mathbf{x}_{\text{wp}} = \mathbf{t} + R_{\text{yaw}} \cdot \frac{d}{f_x}(u - c_x, 0, f_x)^\top$$

Confidence gating: $\text{accept} = \mathbf{1}[c \geq \tau_c]$

**Method:** HTTP client to Ollama/GPT-4V; JSON parsing with confidence gating; pixel hint back-projected via depth map to metric waypoint.

**Implementation:** Full API client; frame encoding; response parser; offline evaluation uses stub frames (no live VLM in tests).

**Results:** Architecture validated offline; accuracy metrics require live VLM deployment.

**Contribution Type:** Systems integration / research prototype  
**Maturity:** Research concept / prototype

---

### Contribution 12 — Diffusion Occupancy Maps

**Problem:** Deterministic occupancy prediction cannot represent the distribution over future obstacle configurations; CVaR risk estimation requires samples.

**Mathematical Formulation:**

DDPM forward: $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$

Cosine schedule: $\bar\alpha_t = \cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)$

Reverse denoising: $\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t,\text{cond})\right)$

CVaR-95 from $N$ samples: $\widehat{\text{CVaR}}_{0.95}(c) = \frac{1}{\lceil0.05N\rceil}\sum_{\text{top }5\%} x_0^{(i)}[c]$

**Method:** Conditioned reverse DDPM generates $N$ occupancy samples; per-cell CVaR-95 feeds risk-weighted A*.

**Implementation:** Correct DDPM pipeline with cosine schedule; score network is MLP stub (U-Net required for production); evaluation script `eval_diffusion_occupancy.py`.

**Results:** Pipeline validated; risk map statistics reported. **Limitation:** MLP stub produces uninformative samples — a trained U-Net is required for meaningful occupancy prediction.

**Contribution Type:** Algorithmic (applied generative modelling)  
**Maturity:** Partially implemented prototype

---

### Contribution 13 — Latent World Model

**Problem:** Reactive planners cannot anticipate long-horizon consequences; model-based planning via mental rollouts can preempt irreversible failures.

**Mathematical Formulation:**

RSSM (Hafner et al. 2019):
$$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}), \quad z_t \sim p_\phi(z|h_t)$$

Posterior: $z_t \sim q_\phi(z|h_t, o_t)$

Mental rollout objective:
$$G(\mathbf{a}) = \sum_{k=1}^H \gamma^{k-1} g_\phi(h_k,z_k) - \lambda_{\text{irrev}}\cdot\mathbf{1}[\neg\text{Ret}(h_H,z_H)]$$

**Method:** RSSM updates latent state from real observations; $K$ candidate action sequences scored by discounted imagined return minus irreversibility penalty; best sequence executed.

**Implementation:** RSSM with numpy MLP stubs; correct prior/posterior structure; rollout scoring with irreversibility penalty from Contribution 04.

**Results:** Framework validated; latent dynamics are random (model not trained on navigation data).

**Contribution Type:** Research prototype  
**Maturity:** Mathematically motivated, lightly validated

---

### Contribution 14 — Causal Risk Attribution

**Problem:** Statistical co-occurrence of sensor failures and collisions does not establish causation; counterfactual reasoning is required for actionable root-cause attribution.

**Mathematical Formulation:**

SCM: $V_i = f_i(\text{PA}_i, U_i)$ over causal graph sensor\_noise $\to$ loc\_error $\to$ obs\_det $\to$ risk $\to$ collision

Average Causal Effect:
$$\text{ACE}(X \to Y) = \mathbb{E}[Y|do(X=x)] - \mathbb{E}[Y|do(X=x')]$$

Root-cause ranking (Shapley approximation):
$$\phi_i = \frac{1}{N}\sum_j [Y_{\text{obs}}(U^{(j)}) - Y_{V_i=0}(U^{(j)})]$$

**Method:** Hand-crafted structural equations over 6 navigation variables; Monte Carlo ACE estimation; ablation-based root-cause ranking.

**Implementation:** Full SCM with do-calculus; counterfactual queries; root-cause ranking; 4 tests passing.

**Results:** Causal attribution produces interpretable rankings (e.g., map\_accuracy > obstacle\_detection as top contributors). **Limitation:** Structural equations are hand-crafted linear functions, not learned from data.

**Contribution Type:** Algorithmic (applied causal inference)  
**Maturity:** Partially implemented prototype

---

### Contribution 15 — Neuromorphic Sensing

**Problem:** Frame-based cameras introduce fixed-rate latency incompatible with high-speed obstacle detection; event cameras offer microsecond-resolution sensing.

**Mathematical Formulation:**

DVS event: $e=(x,y,t,p)$ iff $|\Delta L(x,y,t)| \geq C_p$

Time surface: $\mathcal{T}_p(x,y,t) = \exp(-(t - t_p^{\text{last}}(x,y))/\tau)$

LIF dynamics: $V_{t+1} = e^{-\Delta t/\tau_m}V_t + (1-e^{-\Delta t/\tau_m})I_t$; spike if $V \geq V_{\text{thresh}}$

**Method:** DVS simulator converts frame pairs to async events; 2-channel time surface fed to grid-structured LIF SNN; spike counts normalised to obstacle probability map.

**Implementation:** DVS simulator with noise model; LIF neurons with correct dynamics; SNN grid detector; 3 tests passing.

**Results:** Pipeline data flow validated. **Limitation:** SNN weights are random — no training has been performed; obstacle detection output is therefore uninformative.

**Contribution Type:** Research prototype  
**Maturity:** Research concept / prototype

---

### Contribution 16 — Federated Navigation Learning

**Problem:** Training a shared navigation policy across a robot fleet requires data sharing, which violates privacy; federated learning enables collaborative training without centralising raw data.

**Mathematical Formulation:**

FedAvg: $w^{(t+1)} = \sum_k \frac{n_k}{n} w_k^{(t)}$

Gaussian DP mechanism: $\tilde{w}_k = \text{clip}(w_k, C) + \mathcal{N}(0, \sigma^2 I)$, where $\sigma = \frac{C\sqrt{2\ln(1.25/\delta)}}{\varepsilon}$

**Method:** FedAvg with weighted aggregation; optional $(\varepsilon,\delta)$-DP noise injection per client; validation MSE tracked per round.

**Implementation:** Server + client classes; DP mechanism; synthetic data per robot (not real navigation data); 3 tests passing.

**Results:** Val MSE decreases over rounds (federated convergence validated). **Limitation:** Robots train on synthetic random data; navigation performance impact not measured.

**Contribution Type:** Algorithmic (applied federated learning)  
**Maturity:** Partially implemented prototype

---

### Contribution 17 — Topological Semantic Maps

**Problem:** Dense metric grids are computationally expensive for long-horizon planning and do not natively support semantic instruction-following.

**Mathematical Formulation:**

Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, w)$; each node $v_i$ has embedding $e_i \in \mathbb{R}^d$

Dijkstra: $\pi^* = \arg\min_{\pi: v_s \to v_g} \sum_{(i,j)\in\pi} w_{ij}$

Open-vocabulary grounding: $v^* = \arg\max_i \frac{e_i \cdot q}{\|e_i\|\|q\|}$

**Method:** Zone graph with Dijkstra planning; cosine-similarity grounding against CLIP embeddings; edge invalidation on obstacle discovery; JSON serialisation.

**Implementation:** Full graph class; Dijkstra; stub CLIP embedding (deterministic hash — not real CLIP); Levenshtein fallback; 4 tests passing.

**Results:** Planning and grounding validated with stub embeddings. **Limitation:** Real CLIP embeddings required for open-vocabulary grounding in novel environments.

**Contribution Type:** Systems integration + algorithmic  
**Maturity:** Partially implemented prototype

---

### Contribution 18 — Formal Safety Shields

**Problem:** Learning-based planners provide no hard safety guarantees; a runtime filter is needed that provably prevents unsafe commands regardless of upstream policy.

**Mathematical Formulation:**

STL robustness (Donzé & Maler 2010):
$$\rho(\square_{[a,b]}\phi, x, t) = \min_{t' \in [t+a,t+b]} \rho(\phi, x, t')$$

CBF condition (Ames et al. 2019), safe set $\mathcal{C} = \{x: h(x)\geq 0\}$:
$$\nabla h(x) \cdot u + \alpha h(x) \geq 0 \implies \mathcal{C} \text{ forward invariant}$$

Min-perturbation QP (gradient projection):
$$u^* = \arg\min_u \|u - u_{\text{des}}\|^2 \quad \text{s.t.} \quad \nabla h(x)\cdot u + \alpha h(x) \geq 0$$

**Method:** STL monitor evaluates compositional temporal specifications over trajectory; CBF filter solves QP to modify commands; SafetyShield wraps any planner.

**Implementation:** Full compositional STL ($\square, \lozenge, \wedge, \vee$); CBF with iterative gradient projection; shielded vs unshielded experiment; 4 tests passing.

**Results:** Constraint violations reduced from ~4.2 to ~0.3 per episode; mean CBF correction 0.026 m/s; path overhead < 8%.

**Contribution Type:** Algorithmic (formal methods)  
**Maturity:** Partially implemented — approaching fully experimentally supported

> **Strongest formal methods contribution in the framework.**

---

### Contribution 19 — LLM Mission Planner

**Problem:** Metric waypoint specification is inaccessible to non-expert operators; natural language mission instructions require structured parsing.

**Mathematical Formulation:**

LLM structured prediction: $p_\theta(W|x) = \prod_k p_\theta(w_k|w_{1:k-1}, x)$ → JSON waypoints

Keyword fallback: $\mathcal{M} = \text{sort-by-position}\{z \in \mathcal{Z}_{\text{known}} : z \subseteq x\}$

**Method:** HTTP client to Ollama/OpenAI API; JSON response parsing; confidence-gated acceptance; offline keyword extraction fallback.

**Implementation:** Full API client; parser; keyword fallback; Levenshtein fuzzy matching; 6 tests passing.

**Results:** Offline fallback validated; LLM accuracy requires live server evaluation.

**Contribution Type:** Systems integration  
**Maturity:** Systems integration contribution

> **Note:** This is a lightweight interface layer, not a novel planning methodology. Should be presented as such.

---

### Contribution 20 — Multimodal Failure Explainer

**Problem:** Post-failure diagnosis currently requires manual log analysis; automated multi-source evidence fusion can reduce diagnostic time.

**Mathematical Formulation:**

Failure event: $\mathcal{F} = (\tau_f, s_f, \dot{s}_f, I_f, \mathbf{x}_{1:T}, \rho_{1:T}, \mathcal{S})$

Report: $\mathcal{R} = (d_{\text{scene}}, \phi_{1:M}, V_{\text{STL}}, \mathcal{A}_f, \hat{c})$

where $\phi_{1:M}$ = SCM root causes (Contribution 14), $V_{\text{STL}}$ = STL violations (Contribution 18), $\mathcal{A}_f$ = curated corrective actions.

**Method:** Pipeline: VLM scene description → SCM causal ranking → STL violation summary → rule-based corrective actions → Markdown/JSON report.

**Implementation:** Modular pipeline; FailureEvent/FailureReport dataclasses; corrective action dictionary; 4 tests passing.

**Results:** Reports generated correctly for COLLISION, TIMEOUT, SENSOR\_FAULT types.

**Contribution Type:** Systems integration  
**Maturity:** Systems integration contribution

---

### Contribution 21 — PPO Navigation Agent

**Problem:** Classical planners require explicit maps; RL can learn navigation policies from interaction, but standard rewards ignore risk structure.

**Mathematical Formulation:**

Risk-shaped reward:
$$r(s,a) = -c_{\text{step}} + \frac{c_{\text{prog}}}{d_{\text{goal}}+\varepsilon} - \lambda_r\max(0, d_{\text{safe}}-d_{\text{obs}}) - \lambda_c\mathbf{1}[d_{\text{obs}}<r_c] + r_g\mathbf{1}[\text{goal}]$$

PPO objective:
$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t[\min(r_t(\theta)A_t,\ \text{clip}(r_t, 1\pm\varepsilon)A_t)]$$

GAE: $\hat{A}_t = \sum_{l\geq 0}(\gamma\lambda)^l\delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**Method:** Actor-critic with clipped surrogate objective, entropy regularisation, GAE advantage estimation; custom NavEnv with risk-shaped rewards.

**Implementation:** Correct PPO data flow (rollout, GAE, clipped objective); numpy MLP stubs for actor/critic; 5 tests passing.

**Results:** Training loop executes; value loss tracks. **Critical limitation:** Numpy stubs do not support backpropagation — policy is not actually updated. This is a framework prototype, not a trained agent.

**Contribution Type:** Research prototype  
**Maturity:** Partially implemented prototype (gradient updates are stubs)

---

### Contribution 22 — Curriculum RL Training

**Problem:** Training RL agents on full-difficulty environments from scratch is sample-inefficient; difficulty should be matched to current agent capability.

**Mathematical Formulation:**

5-stage difficulty: $d \in \{$easy, medium, hard, expert, extreme$\}$, each parameterised by $(N_{\text{obs}}, L_{\text{map}}, v_{\text{obs}}, \sigma_{\text{noise}}, [r_{\text{min}}, r_{\text{max}}])$

Adaptive advancement:
$$\text{stage}_{t+1} = \text{stage}_t + 1 \quad \text{if} \quad \text{SR}_W(t) = \frac{1}{W}\sum_{i=t-W+1}^t \mathbf{1}[\text{success}_i] \geq \tau_d$$

**Method:** Adaptive, fixed-schedule, and reverse curriculum strategies; CurriculumNavEnv injects difficulty parameters transparently; stage transitions logged.

**Implementation:** 5-stage ladder; rolling SR computation; adaptive/fixed/reverse strategies; 5 tests passing including stage advancement.

**Results:** Curriculum advancement from easy → medium validated; full training requires functional PPO backend.

**Contribution Type:** Algorithmic (applied curriculum learning)  
**Maturity:** Partially implemented prototype (curriculum logic correct; RL backend is stub)

---

### Contribution 23 — Gaussian Splatting Mapper

**Problem:** Dense metric maps lack continuous uncertainty representations; 3D Gaussian primitives provide compact geometry encoding with natural uncertainty via covariance.

**Mathematical Formulation:**

Gaussian primitive: $\mathcal{G}_i = (\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i, \mathbf{c}_i)$

Merge condition: $d_M(\mathbf{p}, \mathcal{G}_i) = \sqrt{(\mathbf{p}-\boldsymbol{\mu}_i)^\top\boldsymbol{\Sigma}_i^{-1}(\mathbf{p}-\boldsymbol{\mu}_i)} < \tau$

EKF update: $\boldsymbol{\mu}_i \leftarrow (1-\alpha)\boldsymbol{\mu}_i + \alpha\mathbf{p}$

Confidence decay: $\text{conf}_i^{(t)} = \rho \cdot \text{conf}_i^{(t-1)}$

2D occupancy: $O(r,c) = \min(1, \sum_i \alpha_i \exp(-d_{2D}^2/2\sigma_i^2))$

**Method:** Incremental Gaussian add/merge/decay/prune pipeline; 2D occupancy and uncertainty projection; frontier detection at known/unknown boundary.

**Implementation:** Full Gaussian SLAM data structures; occupancy, uncertainty, frontier extraction; 5 tests passing.

**Results:** Occupancy extraction and frontier detection validated. **Limitation:** No neural rendering (no RGB loss, no rasterisation) — navigation-relevant components only.

**Contribution Type:** Research prototype  
**Maturity:** Mathematically motivated, lightly validated

---

### Contribution 24 — NeRF Uncertainty Maps

**Problem:** NeRF models do not natively quantify which regions are well-observed; MC-Dropout provides epistemic uncertainty estimates for exploration guidance.

**Mathematical Formulation:**

Volume rendering: $\hat{C} = \sum_k T_k(1-e^{-\sigma_k\delta_k})\mathbf{c}_k$

MC-Dropout uncertainty: $\text{Var}[\sigma(\mathbf{x})] \approx \frac{1}{T}\sum_t \hat\sigma_t^2 - \bar\sigma^2$

Per-ray uncertainty: $u(\mathbf{r}) = \sum_k T_k(1-e^{-\bar\sigma_k\delta_k})\cdot\text{std}_{\text{MC}}[\sigma_k]$

Exploration weights: $w(r,c) \propto U(r,c)\cdot(1-O(r,c))$

**Method:** TinyNeRF with dropout; T forward passes per ray; density variance as uncertainty proxy; 2D projection and normalisation.

**Implementation:** Positional encoding; volume rendering with transmittance; MC uncertainty aggregation; 5 tests passing.

**Results:** Data flow validated; uncertainty maps output. **Critical limitation:** NeRF weights are random and untrained — uncertainty values are not meaningful.

**Contribution Type:** Research prototype  
**Maturity:** Research concept / prototype

---

### Contribution 25 — Adversarial Attack Simulator

**Problem:** Navigation systems relying on neural perception are vulnerable to adversarial perturbations; systematic evaluation requires both gradient-based and sensor-level attack simulation.

**Mathematical Formulation:**

FGSM: $x_{\text{adv}} = x + \varepsilon\cdot\text{sign}(\nabla_x\mathcal{L})$, $\|x_{\text{adv}}-x\|_\infty \leq \varepsilon$

PGD: $x^{(k+1)} = \Pi_{\mathcal{B}_\varepsilon(x)}\!\left(x^{(k)} + \alpha\cdot\text{sign}(\nabla_{x^{(k)}}\mathcal{L})\right)$

LiDAR phantom: $\mathbf{p}_i^{\text{ph}} = \mathbf{p}_{\text{robot}} + r_i[\cos\theta_i, \sin\theta_i, 0]^\top + \boldsymbol\epsilon_i$

Odometry drift: $\tilde{s}_{t+1} = s_{t+1} + \boldsymbol\delta_t$, $\boldsymbol\delta_t = \boldsymbol\delta_{t-1} + \boldsymbol\eta_t$

**Method:** FGSM/PGD via finite-difference gradient; LiDAR phantom injection, point removal, sector blinding; odometry Gaussian drift accumulation; RobustnessEvaluator runs full suite.

**Implementation:** All attack types; epsilon-bound enforcement; 6 tests passing including epsilon constraint verification.

**Results:** FGSM loss increase: 0.77; LiDAR phantom points added: 24/attack; odometry drift after 100 steps: 0.21 m. **Limitation:** Finite differences scale as O(d) — inefficient for high-dimensional inputs; backprop required for real networks.

**Contribution Type:** Algorithmic + benchmark  
**Maturity:** Partially implemented prototype

---

### Contribution 26 — Swarm Consensus Navigation

**Problem:** In a robot fleet, faulty or compromised robots may corrupt collective navigation decisions; Byzantine fault-tolerant aggregation is required.

**Mathematical Formulation:**

BFT tolerance bound: $n \geq 3f + 1$

Proposal from robot $k$: $P_k = (\pi_k, c_k, \gamma_k)$

MAD-based Byzantine detection:
$$\text{Byzantine}_k = \mathbf{1}\!\left[|c_k - \text{median}(\mathbf{c})| > \kappa\cdot\text{MAD}(\mathbf{c})\right]$$

Weighted median consensus:
$$c^* = \text{WeightedMedian}(\{c_k\}_{k\in H},\ \{\gamma_k\}_{k\in H})$$

**Method:** Each robot computes local A* plan; MAD outlier detection removes Byzantine proposals; weighted median over honest set gives agreed cost; closest honest plan selected.

**Implementation:** SwarmRobot with honest/random/constant\_bad/silent fault modes; BFTConsensus; multi-round experiment; 5 tests passing.

**Results:** Byzantine detection rate in experiments: correct identification in all simulated rounds with $f=1$, $n=6$; agreed cost consistent across rounds.

**Contribution Type:** Algorithmic (applied Byzantine fault tolerance)  
**Maturity:** Partially implemented prototype

---

## 5. Key Contributions (Top 5)

### 1. Formal Safety Shields (18) — *Strongest contribution*
Correct implementation of STL temporal logic monitoring and CBF command filtering with proven forward-invariance guarantee. The only contribution with a full experimental comparison (shielded vs unshielded) producing quantitative results. Directly applicable to any planning backend as a drop-in wrapper.

### 2. Causal Risk Attribution (14) — *Most academically novel*
Application of SCM do-calculus to navigation failure root-cause attribution is a relatively unexplored direction. The counterfactual reasoning and ACE estimation are correctly implemented. Main limitation (hand-crafted structural equations) is addressable by learning them from data — a clear path to publishable work.

### 3. CVaR Risk-Aware Planning (03) — *Most foundational*
Clean mathematical formulation, established theoretical grounding, and modularity enabling any probabilistic risk source (EKF, diffusion, NeRF) to plug into the same planner. Forms the risk layer on which multiple other contributions depend.

### 4. Diffusion Occupancy Maps (12) — *Strongest emerging-area contribution*
Correct DDPM implementation with cosine schedule and CVaR-95 risk extraction. The methodology is sound and the research direction (diffusion models for occupancy prediction) is actively studied. Requires trained U-Net score network to become a credible experimental contribution.

### 5. BFT Swarm Consensus (26) — *Cleanest multi-robot contribution*
Clear algorithmic contribution grounded in distributed systems theory. MAD-based outlier detection and weighted median are appropriate for the navigation domain. The $f < n/3$ tolerance bound is correctly respected. One of the more complete and independently testable new contributions.

---

## 6. Research Identity

DynNav characterises the author as a **systems-oriented robotics researcher** with:

- **Mathematical literacy** across multiple sub-fields: probabilistic inference, convex optimisation, formal methods, causal inference, generative modelling, RL
- **Systems thinking**: the ability to compose heterogeneous components into a coherent integrated framework
- **Research breadth**: engagement with 10+ distinct research directions within a single domain
- **Awareness of formal guarantees**: the formal safety shields contribution demonstrates understanding of the distinction between heuristic and provably safe behaviour

The primary research identity is: *uncertainty-aware and formally safe autonomous navigation*, with secondary interests in causal reasoning, generative perception models, and multi-agent robustness.

**Important qualification:** The breadth of DynNav reflects a research exploration phase rather than deep specialisation. For PhD-level work, 2–3 contributions should be taken to full experimental maturity rather than continuing to expand the module count.

---

## 7. Limitations

The following limitations apply across the framework and should be acknowledged explicitly in any academic presentation:

**Implementation depth:**
- Contributions 21 (PPO), 22 (Curriculum RL): numpy MLP stubs do not support backpropagation — policy is not actually trained
- Contributions 11 (VLM), 19 (LLM): evaluated in offline/stub mode; live API required for real accuracy metrics
- Contributions 15 (Neuromorphic), 24 (NeRF): model weights are random — outputs are not meaningful without training

**Data dependency:**
- Contributions 12, 13, 23, 24: generative models require training on real navigation data; current validation uses synthetic or random inputs
- Contribution 16 (Federated): robots train on synthetic data, not real navigation episodes; federated convergence is validated but navigation improvement is not

**Algorithmic limitations:**
- Contribution 25: finite-difference gradient approximation scales as $O(d)$ — not suitable for high-dimensional neural network inputs
- Contribution 14: structural equations are hand-crafted linear functions, limiting causal claim fidelity
- Contribution 04: returnability uses BFS approximation, not full backward reachability

**Experimental coverage:**
- Several contributions (04, 06, 09, 10) have limited quantitative experimental validation beyond unit tests
- No systematic ablation study comparing all contributions simultaneously against a common baseline

**Hardware validation:**
- Most contributions validated in Gazebo simulation; real-robot (TurtleBot3) validation is limited to select modules

---

## 8. Future Work

The following directions represent the highest-value paths from current prototype to research contribution:

**Near-term (implementation completion):**
- Replace numpy MLP stubs in Contributions 21, 22 with PyTorch modules to enable actual PPO training
- Train U-Net score network for Contribution 12 on real occupancy map datasets (e.g., ScanNet, MatterPort3D)
- Integrate real CLIP embeddings in Contribution 17 for functional open-vocabulary grounding

**Medium-term (experimental validation):**
- Systematic evaluation of Contribution 03 (CVaR planner) vs D* Lite, RRT* on dynamic benchmark environments
- Learn SCM structural equations (Contribution 14) from navigation episode data via causal structure learning
- Full attack-defend evaluation: Contribution 25 attacks against Contribution 08 IDS with quantitative TPR/FPR curves

**Research-level extensions:**
- **Contribution 18 + 12**: formally verified safety shield with diffusion-based occupancy uncertainty as the CBF risk source — combining formal guarantees with learned probabilistic maps
- **Contribution 14 + 20**: causal model learned end-to-end from failure data, enabling automated diagnosis without hand-crafted equations
- **Contribution 22 + 21**: functional curriculum-trained PPO agent with evaluation on standard navigation benchmarks (Gibson, Habitat, RoboTHOR)
- **Contribution 26 + 16**: Byzantine-robust federated learning where the aggregation uses the weighted median rather than FedAvg — directly connecting Contributions 16 and 26

---

*This document was generated from analysis of the DynNav repository structure, contribution source files, experiment scripts, test suites, and README documentation. Maturity assessments reflect the state of implementation and experimental validation as observed in the repository.*
