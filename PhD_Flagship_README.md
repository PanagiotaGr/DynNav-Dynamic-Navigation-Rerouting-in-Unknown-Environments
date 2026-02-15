# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

> A research framework for autonomous robotic navigation in unknown,
> partially observable, and dynamically evolving environments,
> integrating uncertainty modeling, risk-aware decision making,
> irreversibility reasoning, learning-augmented planning, and
> safety-aware autonomy with strong emphasis on scientific
> reproducibility and quantitative validation.

------------------------------------------------------------------------

# Research Vision

Autonomous robots operating in real-world environments must make
decisions under uncertainty, incomplete information, and dynamic
environmental changes. Traditional navigation systems often assume
reliable sensing, static environments, and full map availability. These
assumptions rarely hold outside laboratory conditions.

This repository develops a unified navigation framework where robots
explicitly reason about uncertainty, risk, irreversibility, safety,
resource constraints, system integrity, and human interaction within a
single decision-making architecture.

The goal is not only to build algorithms, but to contribute toward a
unified scientific understanding of safe autonomous navigation under
uncertainty.

------------------------------------------------------------------------

# Why This Problem Is Hard

Real-world robotic navigation must handle:

-   Incomplete or evolving maps\
-   Sensor noise and state estimation drift\
-   Dynamic obstacles\
-   Limited energy resources\
-   Communication constraints\
-   Potential cyber-physical attacks\
-   Human safety, trust, and comfort constraints

Most existing systems treat these challenges independently. This
framework investigates how they interact and how they can be addressed
jointly.

------------------------------------------------------------------------

# Core Scientific Idea

Navigation can be viewed as constrained decision making under
uncertainty.

Conceptually, the robot solves:

Minimize: Path Cost

Subject to: Risk Constraints Returnability Constraints Energy
Constraints Connectivity Constraints Security Integrity Constraints
Human Preference / Trust Constraints

This repository explores algorithmic and experimental methods for
solving this unified problem.

------------------------------------------------------------------------

# Key Scientific Concepts (With Intuition)

## Uncertainty

Represents incomplete knowledge about robot state or environment.

Simple intuition:\
The robot is never 100% sure where it is or what exists around it.

------------------------------------------------------------------------

## Risk

Expected cost of undesirable outcomes.

Simple intuition:\
Even low-probability dangerous events must be considered.

------------------------------------------------------------------------

## Belief Space Planning

Planning over probability distributions instead of single deterministic
states.

Simple intuition:\
The robot plans while considering multiple possible realities.

------------------------------------------------------------------------

## Irreversibility / Returnability

Measures whether the robot can safely return from a region.

Simple intuition:\
Avoid entering places that may become traps.

------------------------------------------------------------------------

## Safe Mode Navigation

Adaptive switch to conservative behavior when uncertainty or risk
increases.

Simple intuition:\
When things look unsafe → move slower, safer, more predictable.

------------------------------------------------------------------------

## Learned Heuristics with Guarantees

Machine learning accelerates search while preserving correctness
guarantees.

Simple intuition:\
AI helps speed up planning but never breaks optimality.

------------------------------------------------------------------------

## Security-Aware Estimation

Detection of anomalies or attacks in sensing or state estimation
pipelines.

Simple intuition:\
Detect if robot sensors or internal data are being corrupted.

------------------------------------------------------------------------

# Research Questions Explored

-   How can robots plan safely under epistemic uncertainty?
-   How can learning accelerate planning without breaking guarantees?
-   How can navigation avoid irreversible trap states?
-   How can risk tolerance be adapted online?
-   How can planning react to estimation integrity attacks?
-   How can navigation incorporate human trust and preferences?

------------------------------------------------------------------------

# Repository Knowledge Map (Recommended Reading Paths)

## Start Here (High-Level Understanding)

→ README-large info.md\
→ Abstract_and_Contributions.md

------------------------------------------------------------------------

## Scientific Claims and Validation

→ CLAIMS_EVIDENCE.md

------------------------------------------------------------------------

# Installation

## Create Environment

python -m venv venv

## Activate

Linux / macOS: source venv/bin/activate

Windows: venv`\Scripts`{=tex}`\activate`{=tex}

## Install Dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

# Quickstart

Run baseline experiment:

python contributions/01_learned_astar/experiments/eval_astar_learned.py

------------------------------------------------------------------------

# Disclaimer

For research and educational use only.\
Not validated for safety-critical deployment.

------------------------------------------------------------------------

# Author

Panagiota Grosdouli\
Electrical and Computer Engineering\
Democritus University of Thrace

------------------------------------------------------------------------

# Project Status

Active research codebase under continuous scientific development.
