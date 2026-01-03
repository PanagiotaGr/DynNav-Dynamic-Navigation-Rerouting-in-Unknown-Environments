import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.genfromtxt("ids_to_planner_hook_log.csv", delimiter=",", skip_header=1)
    t = data[:, 0]
    nis = data[:, 1]
    ewma = data[:, 2]
    cusum = data[:, 3]
    alarm = data[:, 4]
    safe = data[:, 5]
    trust = data[:, 6]
    lam = data[:, 7]

    plt.figure()
    plt.plot(t, cusum, label="CUSUM")
    plt.plot(t, 115.1*np.ones_like(t), label="CUSUM threshold")
    plt.xlabel("t")
    plt.ylabel("CUSUM score")
    plt.title("CUSUM score and alarm threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ids_cusum_threshold.png", dpi=160)

    plt.figure()
    plt.plot(t, alarm, label="alarm (0/1)")
    plt.plot(t, safe, label="safe_mode (0/1)")
    plt.xlabel("t")
    plt.ylabel("flag")
    plt.title("Alarm vs latched safe mode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ids_alarm_safe_mode.png", dpi=160)

    plt.figure()
    plt.plot(t, trust, label="VO trust override")
    plt.plot(t, lam, label="lambda_eff")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.title("Mitigation outputs (trust + lambda)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ids_mitigation_outputs.png", dpi=160)

    print("Saved: ids_cusum_threshold.png, ids_alarm_safe_mode.png, ids_mitigation_outputs.png")


if __name__ == "__main__":
    main()
