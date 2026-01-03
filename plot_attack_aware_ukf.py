import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: str):
    # Header: t,est_px,est_py,vo_nis,vo_trust,vo_infl,w_nis,w_trust,w_infl
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    t = data[:, 0]
    est_px = data[:, 1]
    est_py = data[:, 2]
    vo_nis = data[:, 3]
    vo_trust = data[:, 4]
    vo_infl = data[:, 5]
    w_nis = data[:, 6]
    w_trust = data[:, 7]
    w_infl = data[:, 8]
    return t, est_px, est_py, vo_nis, vo_trust, vo_infl, w_nis, w_trust, w_infl


def main():
    path = "attack_aware_ukf_demo_log.csv"
    t, est_px, est_py, vo_nis, vo_trust, vo_infl, w_nis, w_trust, w_infl = load_csv(path)

    # Plot 1: Trust
    plt.figure()
    plt.plot(t, vo_trust, label="VO trust")
    plt.plot(t, w_trust, label="Wheel trust")
    plt.xlabel("t (step)")
    plt.ylabel("trust")
    plt.title("Attack-aware UKF: trust vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("attack_aware_trust.png", dpi=160)

    # Plot 2: NIS (log scale helps)
    plt.figure()
    plt.plot(t, vo_nis, label="VO NIS")
    plt.plot(t, w_nis, label="Wheel NIS")
    plt.yscale("log")
    plt.xlabel("t (step)")
    plt.ylabel("NIS (log scale)")
    plt.title("Attack-aware UKF: NIS vs time")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("attack_aware_nis.png", dpi=160)

    # Plot 3: Inflation factors (optional but useful)
    plt.figure()
    plt.plot(t, vo_infl, label="VO inflation (R factor)")
    plt.plot(t, w_infl, label="Wheel inflation (R factor)")
    plt.yscale("log")
    plt.xlabel("t (step)")
    plt.ylabel("inflation factor (log scale)")
    plt.title("Attack-aware UKF: R inflation vs time")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("attack_aware_inflation.png", dpi=160)

    # Plot 4: Estimated trajectory (just for sanity)
    plt.figure()
    plt.plot(est_px, est_py)
    plt.xlabel("est_px")
    plt.ylabel("est_py")
    plt.title("Attack-aware UKF: estimated trajectory (x-y)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("attack_aware_est_traj.png", dpi=160)

    print("Saved:")
    print(" - attack_aware_trust.png")
    print(" - attack_aware_nis.png")
    print(" - attack_aware_inflation.png")
    print(" - attack_aware_est_traj.png")


if __name__ == "__main__":
    main()
