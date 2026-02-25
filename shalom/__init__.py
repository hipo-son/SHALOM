"""SHALOM: System of Hierarchical Agents for Logical Orchestration of Materials."""

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy import for Pipeline/PipelineConfig — requires optional LLM dependencies."""
    if name in ("Pipeline", "PipelineConfig"):
        from shalom.pipeline import Pipeline, PipelineConfig

        globals()["Pipeline"] = Pipeline
        globals()["PipelineConfig"] = PipelineConfig
        return globals()[name]
    raise AttributeError(f"module 'shalom' has no attribute {name!r}")


__all__ = ["Pipeline", "PipelineConfig", "__version__"]
