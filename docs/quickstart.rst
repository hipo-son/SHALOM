Quickstart
==========

A simple pipeline for an autonomous material search using the SHALOM framework.

.. code-block:: python

    from shalom.core.llm_provider import LLMProvider
    from shalom.agents.design_layer import CoarseSelector, FineSelector
    
    # Initialize the provider with your preferred LLM
    llm = LLMProvider(provider_type="openai", model_name="gpt-4o")
    
    objective = "Find a stable 2D material with bandgap > 1.0eV"
    
    # Step 1: Coarse Selection
    coarse = CoarseSelector(llm)
    candidates = coarse.select(objective)
    
    # Step 2: Fine Selection & Ranking
    fine = FineSelector(llm)
    winner = fine.rank_and_select(objective, candidates)
    
    print(f"Top Material: {winner.candidate.material_name}")
    print(f"Score: {winner.score}")


