"""PubMed dataset wrapper & utilities."""

from logging                    import Logger

from torch_geometric.datasets   import Planetoid

from utils                      import LOGGER

class PubMed(Planetoid):
    """PubMed extension for torch_geometric.datasets.Plantoid.
    
    The PubMed dataset consists of 19,717 scientific publications from PubMed database pertaining to 
    diabetes classified into one of three classes. The citation network consists of 44338 links. 
    Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary 
    which consists of 500 unique words.
    """
    
    # Initialize logger
    _logger:    Logger =    LOGGER.getChild("pubmed")
    
    def __init__(self,
        split:                  str =       "public",
        num_train_per_class:    int =       20,
        num_val:                int =       500,
        num_test:               int =       1000,
        force_reload:           bool =      False,
        **kwargs
    ):
        """# Initialize PubMed dataset.

        ## Args:
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
                                            
        # Initialize dataset                Defaults to False.
        """
        # Initialize dataset
        super(PubMed, self).__init__(root = "input", name = "PubMed")
        
        # Log for debugging
        self._logger.debug(f"PubMed dataset arguments: {locals()}")