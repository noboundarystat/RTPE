from .classic import ClassicTransformer
from .elasticnet import train_and_evaluate_elasticnet
from .proposed import ProposedTransformer
from .shortformer import Shortformer
from .spacetimeformer import Spacetimeformer
from .spe import SPETransformer

# Neural models: all share the skeleton in ``common.LongitudinalTransformer`` and differ
# only in how time / position enters the model.
TORCH_MODELS = {
    "Proposed": ProposedTransformer,          # sinusoidal encoding of real measurement time
    "Classic": ClassicTransformer,            # sinusoidal encoding of visit index
    "Shortformer": Shortformer,               # position-infused attention
    "SPE": SPETransformer,                    # stochastic positional encoding
    "Spacetimeformer": Spacetimeformer,       # Time2Vec + position + variable embeddings
}

ELASTICNET = "ElasticNet"
ALL_MODELS = list(TORCH_MODELS) + [ELASTICNET]

__all__ = ["TORCH_MODELS", "ELASTICNET", "ALL_MODELS", "train_and_evaluate_elasticnet"]
