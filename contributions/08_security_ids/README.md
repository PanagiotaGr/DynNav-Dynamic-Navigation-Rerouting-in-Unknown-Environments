# Contribution 08 — Security & Intrusion Detection

[![Module](https://img.shields.io/badge/Module-08-purple)](.) [![Type](https://img.shields.io/badge/Type-Cybersecurity%20%2F%20IDS-blue)](.) [![Status](https://img.shields.io/badge/Status-Core-brightgreen)](.)

## Overview

**Anomaly-based Intrusion Detection System (IDS)** for robot navigation. Detects sensor spoofing, GPS manipulation, and adversarial attacks by monitoring innovation sequences and integrity metrics in real-time.

## Research Question

> **RQ5**: How can autonomous systems remain robust under failures or adversarial conditions?

## How It Works

```
Sensor readings → innovation sequence → χ² test / CUSUM detector → alert if anomalous → safe-mode trigger
```

- **Innovation monitoring**: compares predicted vs actual sensor readings
- **χ² test**: statistical anomaly detection on residuals
- **CUSUM**: cumulative sum for gradual drift detection
- **Response**: alert → safe-mode (Contribution 05) → operator notification

## Files

```
08_security_ids/ (see also: cybersecurity_ros2/)
├── experiments/
└── results/
```

## Quick Start

```bash
python contributions/08_security_ids/experiments/eval_ids.py
```

## Detection Methods

| Method | Best For | Latency |
|--------|----------|---------|
| χ² test | Single-step anomalies | 1 step |
| CUSUM | Gradual drift/spoofing | 5-20 steps |
| Innovation bound | Sensor failures | 1 step |

## Integration

- **Triggers**: Contribution 05 (safe-mode) on detection
- **Extended by**: Contribution 25 (adversarial simulator) for attack testing
- **Extended by**: Contribution 14 (causal SCM) for root-cause attribution
- **ROS2**: `cybersecurity_ros2/` nodes for real-time monitoring
