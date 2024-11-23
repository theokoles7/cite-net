"""Dataset loader utility."""

from logging            import Logger

from datasets.citeseer  import CiteSeer
from datasets.cora      import Cora
from datasets.pubmed    import PubMed
from utils              import LOGGER

def load_dataset(
    dataset:                str,
    split:                  str =       "public",
    num_train_per_class:    int =       20,
    num_val:                int =       500,
    num_test:               int =       1000,
    force_reload:           bool =      False,
    **kwargs
) -> CiteSeer | Cora | PubMed:
    """# Load selected dataset.

    ## Args:
        * dataset               (str):                  Choice of dataset ("citeseer", "cora", 
                                                        "pubmed").
        * split                 (str, optional):        The type of dataset split ("public", 
                                                        "full", "geom-gcn", "random"). If set to 
                                                        "public", the split will be the public 
                                                        fixed split from the “Revisiting 
                                                        Semi-Supervised Learning with Graph 
                                                        Embeddings” paper. If set to "full", all 
                                                        nodes except those in the validation and 
                                                        test sets will be used for training (as 
                                                        in the “FastGCN: Fast Learning with Graph 
                                                        Convolutional Networks via Importance 
                                                        Sampling” paper). If set to "geom-gcn", 
                                                        the 10 public fixed splits from the 
                                                        “Geom-GCN: Geometric Graph Convolutional 
                                                        Networks” paper are given. If set to 
                                                        "random", train, validation, and test 
                                                        sets will be randomly generated, 
                                                        according to num_train_per_class, num_val 
                                                        and num_test. Defaults to "public".
        * num_train_per_class   (int, optional):        The number of training samples per class 
                                                        in case of "random" split. Defaults to 20.
        * num_val               (int, optional):        The number of validation samples in case 
                                                        of "random" split. Defaults to 500.
        * num_test              (int, optional):        The number of test samples in case of 
                                                        "random" split. Defaults to 1000.
        * force_reload          (bool, optional):       Whether to re-process the dataset. 
                                                        Defaults to False.

    ## Returns:
        * CiteSeer | Cora | PubMed: Selected dataset.
    """
    # Get logger
    _logger:    Logger =    LOGGER.getChild("dataset-loader")
    
    # Log action
    _logger.info(f"Loading {dataset} dataset")
    
    # Match dataset selection
    match dataset:
        
        # CiteSeer
        case "citeseer":    return CiteSeer(**kwargs)
        
        # Cora
        case "cora":        return Cora(**kwargs)
        
        # PubMed
        case "pubmed":      return PubMed(**kwargs)
        
        # Invalid selection
        case _:             raise NotImplementedError(f"Invalid dataset selection: {dataset}")