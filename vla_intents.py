echo 'class VLAIntents:
    @staticmethod
    def classify(text):
        text = text.lower()
        if "uncertainty" in text or "risky" in text:
            return "HIGH_UNCERTAINTY"
        if "coverage" in text or "uncovered" in text:
            return "LOW_COVERAGE"
        if "obstacle" in text or "wall" in text:
            return "OBSTACLE_AWARE"
        if "drift" in text:
            return "DRIFT_STABILIZATION"
        return "DEFAULT"' > nav_research/vla/vla_intents.py
