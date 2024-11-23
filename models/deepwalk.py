"""DeepWalk model implementation."""

from logging                import Logger

from dgl                    import DGLGraph
from dgl.nn.pytorch         import DeepWalk

from gensim.models          import Word2Vec
from numpy                  import ndarray, where
from numpy.random           import choice
from torch                  import tensor, Tensor
from torch.nn               import Module
from torch_geometric.utils  import to_dense_adj

from utils                  import LOGGER

class DeepWalk(DeepWalk):
    """# DeepWalk Model.
    
    DeepWalk is an unsupervised graph embedding model based on random walks and the Skip-Gram model 
    from natural language processing. Its goal is to generate embeddings for each node in a graph, 
    capturing the structural information and relationships between nodes in a low-dimensional vector 
    space. These embeddings can then be used for downstream tasks like node classification.
    """
    
    # Initialize logger
    _logger:    Logger =    LOGGER.getChild("deepwalk")
    
    def __init__(self,
        g:              DGLGraph,
        emb_dim:        int =       128,
        walk_length:    int =       40,
        window_size:    int =       5,
        neg_weight:     float =     1.0,
        negative_size:  int =       1,
        fast_neg:       bool =      True,
        sparse:         bool =      True
    ):
        """# Initialize DeepWalk model.

        ## Args:
            * g             (DGLGraph):         Graph for learning node embeddings.
            * emb_dim       (int, optional):    Size of each embedding vector. Defaults to 128.
            * walk_length   (int, optional):    Number of nodes in a random walk sequence. Defaults 
                                                to 40.
            * window_size   (int, optional):    In a random walk w, a node w[j] is considered close 
                                                to a node w[i] if i - window_size <= j <= i + 
                                                window_size. Defaults to 5.
            * neg_weight    (float, optional):  Weight of the loss term for negative samples in the 
                                                total loss. Defaults to 1.0.
            * negative_size (int, optional):    Number of negative samples to use for each positive 
                                                sample. Defaults to 1.
            * fast_neg      (bool, optional):   Sample negative node pairs within a batch of random 
                                                walks. Defaults to True.
            * sparse        (bool, optional):   Gradients with respect to the learnable weights will 
                                                be sparse. Defaults to True.
        """
        # Initialize model
        super(DeepWalk, self).__init__(**locals())
    
    # def __init__(self,
    #     embedding_dim:  int =   128,
    #     window_size:    int =   5,
    #     walk_length:    int =   40,
    #     num_walks:      int =   10,
    #     **kwargs
    # ):
    #     """# Initialize DeepWalk model.

    #     ## Args:
    #         * embedding_dim (int, optional):    Size of each embedding vector. Defaults to 128.
    #         * window_size   (int, optional):    In a random walk w, a node w[j] is considered close 
    #                                             to a node w[i] if i - window_size <= j <= i + 
    #                                             window_size. Defaults to 5.
    #         * walk_length   (int, optional):    Number of nodes in a random walk sequence. Defaults 
    #                                             to 40.
    #         * num_walks     (int, optional):    Number of random walks to generate. Defaults to 10.
    #     """
    #     # Log action
    #     self._logger.info(f"Initializing DeepWalk model ({locals()})")
        
    #     # Initialize Module
    #     super(DeepWalk, self).__init__()
        
    #     # Define attributes
    #     self.embedding_dim: int =   embedding_dim
    #     self.window_size:   int =   window_size
    #     self.walk_length:   int =   walk_length
    #     self.num_walks:     int =   num_walks
        
    # def fit(self,
    #     edge_index: ndarray
    # ) -> None:
    #     """# Train model on graph and produce embeddings.

    #     ## Args:
    #         * edge_index    (ndarray):  Graph's edge index.
    #     """
    #     # Log action
    #     self._logger.info("Fitting DeepWalk model.")
        
    #     # Get adjacency matrix
    #     adjacency_matrix:   Tensor =    to_dense_adj(edge_index = edge_index).squeeze(0).numpy()
        
    #     # Generate random walks
    #     walks:              list =      self._generate_random_walks(adjacency_matrix = adjacency_matrix)
        
    #     # Train Word2Vec model
    #     self.w2v:           Word2Vec =  Word2Vec(sentences = walks, vector_size = self.embedding_dim, window = self.window_size, sg = 1, workers = 4)
        
    # def forward(self,
    #     node_indices:   list
    # ) -> Tensor:
    #     """# Feed nodes through network.

    #     ## Args:
    #         * node_indices  (list): Nodes to predict.

    #     ## Returns:
    #         * Tensor:   Node predictions.
    #     """
    #     return tensor(data = [self.w2v.wv[str(node)] for node in node_indices], dtype = float)
        
    # def _generate_random_walks(self,
    #     adjacency_matrix:   Tensor
    # ) -> list:
    #     """# Generate random walks across nodes.

    #     ## Args:
    #         * adjacency_matrix  (Tensor):   Single dense batched adjacency matrix.

    #     ## Returns:
    #         * list: List of randomly generated walks.
    #     """
    #     # Initialize walks
    #     walks:      list =  []
        
    #     # Record number of nodes
    #     num_nodes:  int =   adjacency_matrix.shape[0]
        
    #     # Log action
    #     self._logger.info(f"Generating random walks acros {num_nodes} nodes")
        
    #     # For each node...
    #     for node in range(num_nodes):
            
    #         # For each walk to be made...
    #         for w in range(1, self.num_walks + 1):
                
    #             # Initialize walk at node
    #             walk:   list =  [node]
                
    #             # Log walk for debugging
    #             self._logger.debug(f"\tWalk {w}/{self.num_walks}")
                
    #             # For each step of the walk...
    #             for s in range(1, self.walk_length):
                    
    #                 # Log step for debugging
    #                 self._logger.debug(f"\t\tStep {s}/{self.walk_length}")
                    
    #                 # Get current node
    #                 current:    int =       walk[-1]
                    
    #                 # Find neighbors
    #                 neighbors:  ndarray =   where(adjacency_matrix[current] > 0)[0]
                    
    #                 # If there are no neighbors, skip
    #                 if len(neighbors) == 0: break
                    
    #                 # Otherwise, append random neighbor to walk
    #                 walk.append(choice(neighbors))
                    
    #             # Append walk to walks
    #             walks.append(list(map(str, walk)))
                
    #     # Return generated walks
    #     return walks