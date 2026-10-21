"""Compatibility import for callers using the original module location."""

if __package__ and "." in __package__:
    from ..analysis.eda import save_eda_outputs
else:
    from analysis.eda import save_eda_outputs

__all__ = ["save_eda_outputs"]