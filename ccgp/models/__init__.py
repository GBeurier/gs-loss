"""Genomic predictors with a common scikit-learn-style fit/predict interface."""
from .classical import GBLUP, make_classical
from .neural import NeuralRegressor, make_neural

__all__ = ["GBLUP", "make_classical", "NeuralRegressor", "make_neural"]
