# ================================================================
# Language Conditioned Navigation Demo
# ================================================================

from language_to_navigation_policy import LanguageToNavigationPolicy


def run_sentence(text: str):
    print(f"\n[LANG] User instruction: \"{text}\"")

    mapper = LanguageToNavigationPolicy()
    result = mapper.interpret(text)

    print("[LANG-RESULT]")
    print(f"  lambda_risk        : {result.lambda_risk}")
    print(f"  speed_scale        : {result.speed_scale}")
    print(f"  avoid_right_wall   : {result.avoid_right_wall}")
    print(f"  avoid_left_wall    : {result.avoid_left_wall}")
    print(f"  increase_exploration: {result.increase_exploration}")
    print(f"  Explanation        : {result.explanation}")


def main():
    run_sentence("πήγαινε γρήγορα")
    run_sentence("πρόσεχε, είναι επικίνδυνο μπροστά")
    run_sentence("πήγαινε αργά και πρόσεχε δεξιά")
    run_sentence("εξερεύνησε πιο πολύ τον χώρο με προσοχή")
    run_sentence("πήγαινε όπως θες")  # should trigger no special mapping


if __name__ == "__main__":
    main()
