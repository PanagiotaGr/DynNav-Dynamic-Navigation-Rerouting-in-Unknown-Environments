# add_failure_types_to_multistart.py
import pandas as pd

from failure_taxonomy import classify_failure_reason


def main():
    df = pd.read_csv("minimax_tau_multistart.csv")

    # classify fixed failures
    df["fixed_failure_type"] = df.apply(
        lambda row: classify_failure_reason(str(row["fixed_reason"])).code if int(row["fixed_success"]) == 0 else "NONE",
        axis=1,
    )

    # safe-mode failures (should be none in your latest run, but keep it general)
    df["safe_failure_type"] = df.apply(
        lambda row: classify_failure_reason(str(row["safe_reason"])).code if int(row["safe_success"]) == 0 else "NONE",
        axis=1,
    )

    # minimax failures (should be none)
    df["mm_failure_type"] = df.apply(
        lambda row: "NONE" if int(row["mm_success"]) == 1 else "OTHER",
        axis=1,
    )

    out = "minimax_tau_multistart_with_failures.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)

    # quick summary
    print("\nFixed failure types counts:")
    print(df["fixed_failure_type"].value_counts())

    print("\nSafe failure types counts:")
    print(df["safe_failure_type"].value_counts())

    print("\nMinimax failure types counts:")
    print(df["mm_failure_type"].value_counts())


if __name__ == "__main__":
    main()
