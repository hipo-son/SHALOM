Quickstart
==========

간단한 자율 소재 탐색 파이프라인 예제입니다.

.. code-block:: python

    from shalom.core import LLMProvider
    from shalom.agents.design_layer import PlannerAgent
    
    provider = LLMProvider(model="claude-3-opus")
    planner = PlannerAgent(provider)
    
    result = planner.plan("Find a stable 2D material with bandgap > 1.0eV")
    print(result)

