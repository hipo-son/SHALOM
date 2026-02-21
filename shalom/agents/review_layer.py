class ReviewAgent:
    """
    Review Layer Agent
    Parses simulation results (e.g., VASP OUTCAR), evaluates success against goals,
    and provides feedback metrics.
    """
    def __init__(self, config=None):
        self.config = config or {}

    def review_results(self, result_data: dict) -> dict:
        """
        Analyze simulation output and determine if the search target is achieved.
        """
        # Placeholder for review logic
        return {"status": "success", "metrics": {}}
