import json
import sys


def parse_command(cmd: str):
    """
    Πολύ απλός rule-based parser για VLA-style εντολές.
    Επιστρέφει ένα dict με:
      - region: one of ["full", "top-left", "top-right", "bottom", "high-uncertainty"]
      - priority: one of ["entropy", "uncertainty", "combined"]
    """

    cmd_low = cmd.lower()

    # default values
    region = "full"
    priority = "combined"

    # ====== Περιοχή χάρτη (region) ======
    if "πάνω αριστερ" in cmd_low or "top-left" in cmd_low:
        region = "top-left"
    elif "πάνω δεξ" in cmd_low or "top-right" in cmd_low:
        region = "top-right"
    elif "κάτω" in cmd_low or "bottom" in cmd_low:
        region = "bottom"
    elif "υψηλότερη αβεβαιότητα" in cmd_low or "highest uncertainty" in cmd_low:
        region = "high-uncertainty"

    # ====== Προτεραιότητα (entropy / uncertainty / combined) ======
    if "εντροπ" in cmd_low or "entropy" in cmd_low:
        priority = "entropy"
    if "αβεβαιότητα" in cmd_low or "uncertainty" in cmd_low:
        # αν αναφέρει uncertainty, το βάζουμε πάνω από entropy
        priority = "uncertainty"
    if "και τα δυο" in cmd_low or "both" in cmd_low or "combine" in cmd_low:
        priority = "combined"

    return {
        "raw_command": cmd,
        "region": region,
        "priority": priority,
    }


if __name__ == "__main__":
    # Διαβάζουμε την εντολή είτε από argv είτε από input()
    if len(sys.argv) > 1:
        user_cmd = " ".join(sys.argv[1:])
    else:
        print("Δώσε εντολή (π.χ. 'Κάλυψε την περιοχή με την υψηλότερη αβεβαιότητα στο πάνω αριστερό μέρος'):")
        user_cmd = input("> ")

    cfg = parse_command(user_cmd)
    print("[VLA] Parsed command:")
    print(cfg)

    # Αποθήκευση σε JSON για χρήση από planners
    out_path = "vla_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[VLA] Saved VLA config to: {out_path}")

