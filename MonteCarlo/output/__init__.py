"""Output and visualization module — fan charts, trace plots, diagnostics."""
from .visualizations import (
    plot_fan_chart,
    plot_return_distribution,
    plot_trace,
    plot_posterior_densities,
    plot_rhat_table,
    plot_ess_table,
    plot_shrinkage,
    plot_var_breakdown,
)

__all__ = [
    "plot_fan_chart",
    "plot_return_distribution",
    "plot_trace",
    "plot_posterior_densities",
    "plot_rhat_table",
    "plot_ess_table",
    "plot_shrinkage",
    "plot_var_breakdown",
]
