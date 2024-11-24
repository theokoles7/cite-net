"""Model loader utility."""

from logging            import Logger

from models.deepwalk    import DeepWalk
from models.gcn         import GCN
from utils              import LOGGER

def load_model(
    model:          str,
    channels_in:    int =       None,
    hidden_layers:  int =       None,
    channels_out:   int =       None,
    embedding_size: int =       128,
    walk_length:    int =       40,
    window_size:    int =       5,
    num_walks:      int =       10,
    **kwargs
) -> DeepWalk:
    """# Load selected model.
    
    ## Args:
        * model             (str):              Choice of model ("citenet", "deepwalk", "gcn")

    ### DeepWalk Keyword Args:
        * embedding_size    (int, optional):    Size of each embedding vector. Defaults to 128.
        * walk_length       (int, optional):    Number of nodes in a random walk sequence. 
                                                Defaults to 40.
        * window_size       (int, optional):    In a random walk w, a node w[j] is considered 
                                                close to a node w[i] if i - window_size <= j <= 
                                                i + window_size. Defaults to 5.
        * num_walks         (int, optional):    Number of random walks to generate during fit.
        
    ### GCN Keyword Args:
        * channels_in       (int, optional):    Number of features per node in the input graph.
        * hidden_layers     (int, optional):    Number of features output by the first GCN layer.
        * channels_out      (int, optional):    Number of features output by the final GCN layer.

    ## Returns:
        * DeepWalk: DeepWalk model, initialized with DeepWalk keyword arguments provided.
    """
    # Initialize logger
    _logger:    Logger =    LOGGER.getChild("model-loader")
    
    # Log action
    _logger.info(f"Loading {model} model")
    
    # Match model selection
    match model:
        
        # CiteNet
        case "citenet":     raise NotImplementedError(f"CiteNet has not been implemented.")
        
        # DeepWalk
        case "deepwalk":    return DeepWalk(**kwargs)
        
        # GCN
        case "gcn":         return GCN(channels_in = channels_in, hidden_layers = hidden_layers, channels_out = channels_out)
        
        # Invalid selection
        case _:             raise NotImplementedError(f"Invalid model selection: {model}")