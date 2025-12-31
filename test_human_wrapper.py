from human_preference_wrapper import HumanPreferencePlannerWrapper


class DummyPlanner:
    def plan(self, state, observation, lambda_weight=1.0, **kwargs):
        print("DummyPlanner called with:")
        print("  state:", state)
        print("  observation:", observation)
        print("  lambda_weight:", lambda_weight)
        return {"path": [state, "goal"], "lambda_used": lambda_weight}


def main():
    base = DummyPlanner()

    wrapper = HumanPreferencePlannerWrapper(
        underlying_planner=base,
        human_pref_text="Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο",
        human_influence_scale=1.0,
    )

    result = wrapper.plan(
        state="s0",
        observation="obs0",
        lambda_robot=1.0,
    )
    print("Result:", result)


if __name__ == "__main__":
    main()
