import numpy as np

from attack_aware_ukf import AttackAwareUKF, SensorConfig, f_constant_velocity, h_position
from security_monitor_cusum import SecurityMonitorCUSUM
from sequential_ids import EWMAConfig, CUSUMConfig
from ids_mitigation_policy import IDSMitigationPolicy, MitigationConfig


def main():
    np.random.seed(0)

    T = 250
    attack_start = 80
    dt = 0.1

    # --- UKF setup ---
    n = 4
    Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])
    ukf = AttackAwareUKF(n=n, f=f_constant_velocity, Q=Q)
    ukf.set_state(np.array([0, 0, 1, 0.5], float), np.diag([0.1, 0.1, 0.2, 0.2]))

    vo_cfg = SensorConfig(name="vo", R_base=np.diag([0.05**2, 0.05**2]))
    wheel_cfg = SensorConfig(name="wheel", R_base=np.diag([0.10**2, 0.10**2]))
    ukf.add_sensor(vo_cfg, h_position)
    ukf.add_sensor(wheel_cfg, h_position)

    # --- IDS monitor: tuned params ---
    mon = SecurityMonitorCUSUM(
        sensors=["vo"],
        ewma_cfg=EWMAConfig(lam=0.02),
        cusum_cfg=CUSUMConfig(k=0.5, h=115.1),
    )

    # --- Mitigation policy ---
    mit = IDSMitigationPolicy(
        MitigationConfig(
            vo_trust_min=0.05,
            lambda_boost=0.35,
            safe_mode_steps=50,
            cooldown_recover=0.01
        )
    )

    # --- Simulate ---
    x_true = np.array([0, 0, 1, 0.5], float)
    lambda_base = 0.2

    logs = []
    for t in range(T):
        x_true = f_constant_velocity(x_true, dt)
        ukf.predict(dt)

        z_vo = x_true[:2] + np.random.randn(2) * 0.05
        z_w  = x_true[:2] + np.random.randn(2) * 0.10

        if t >= attack_start:
            bias = np.array([0.002*(t-attack_start), -0.0015*(t-attack_start)])
            z_vo = z_vo + bias

        tel_vo = ukf.update("vo", z_vo)
        _ = ukf.update("wheel", z_w)

        # IDS update from NIS (already computed in tel_vo)
        ids_state = mon.update_from_nis("vo", tel_vo["nis"])

        # mitigation outputs
        out = mit.step(alarm_vo=ids_state.alarm)

        # planner hook: effective lambda
        lambda_eff = lambda_base + out["lambda_add"]

        logs.append([
            t,
            tel_vo["nis"],
            ids_state.ewma,
            ids_state.cusum,
            1.0 if ids_state.alarm else 0.0,
            1.0 if out["safe_mode"] else 0.0,
            out["vo_trust"],
            lambda_eff
        ])

    logs = np.array(logs, float)
    header = "t,nis,ewma,cusum,alarm,safe_mode,vo_trust,lambda_eff"
    np.savetxt("ids_to_planner_hook_log.csv", logs, delimiter=",", header=header, comments="")
    print("Saved ids_to_planner_hook_log.csv")
    print("Tip: plot columns alarm/safe_mode and cusum to see trigger + latch behavior.")


if __name__ == "__main__":
    main()
