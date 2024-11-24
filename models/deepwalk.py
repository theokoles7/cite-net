"""DeepWalk model implementation."""

from logging                import Logger

from gensim.models          import Word2Vec
from numpy                  import ndarray, unique, where
from numpy.random           import choice
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import accuracy_score, roc_auc_score
from torch                  import tensor, Tensor
from torch.nn               import Module
from torch_geometric.data   import Data
from torch_geometric.utils  import to_dense_adj

from utils                  import LOGGER

class DeepWalk(Module):
    """# DeepWalk Model.
    
    DeepWalk is an unsupervised graph embedding model based on random walks and the Skip-Gram model 
    from natural language processing. Its goal is to generate embeddings for each node in a graph, 
    capturing the structural information and relationships between nodes in a low-dimensional vector 
    space. These embeddings can then be used for downstream tasks like node classification.
    """
    
    # Initialize logger
    _logger:    Logger =    LOGGER.getChild("deepwalk")
    
    def __init__(self,
        embedding_size: int =       128,
        walk_length:    int =       40,
        window_size:    int =       5,
        num_walks:      int =       10,
        **kwargs
    ):
        """# Initialize DeepWalk model.

        ## Args:
            * embedding_size    (int, optional):    Size of each embedding vector. Defaults to 128.
            * walk_length       (int, optional):    Number of nodes in a random walk sequence. 
                                                    Defaults to 40.
            * window_size       (int, optional):    In a random walk w, a node w[j] is considered 
                                                    close to a node w[i] if i - window_size <= j <= 
                                                    i + window_size. Defaults to 5.
            * num_walks         (int, optional):    Number of random walks to generate during fit.
        """
        # Initialize model
        super(DeepWalk, self).__init__()
        
        # Define attributes
        self.embedding_size:    int =   embedding_size
        self.walk_length:       int =   walk_length
        self.window_size:       int =   window_size
        self.num_walks:         int =   num_walks
        
    def fit(self,
        edge_index: ndarray
    ) -> None:
        """# Train model on graph and produce embeddings.

        ## Args:
            * edge_index    (ndarray):  Graph's edge index.
        """
        # Log action
        self._logger.info("Fitting DeepWalk model")
        
        # Get adjacency matrix
        adjacency_matrix:   Tensor =    to_dense_adj(edge_index = edge_index).squeeze(0).numpy()
        
        # Generate random walks
        walks:              list =      self._generate_random_walks(adjacency_matrix = adjacency_matrix)
        
        # Train Word2Vec model
        self.w2v:           Word2Vec =  Word2Vec(sentences = walks, vector_size = self.embedding_size, window = self.window_size, sg = 1, workers = 4)
        
    def forward(self,
        node_indices:   list,
        **kwargs
    ) -> Tensor:
        """# Feed nodes through network.

        ## Args:
            * node_indices  (list): Nodes to predict.

        ## Returns:
            * Tensor:   Node predictions.
        """
        return tensor(data = [self.w2v.wv[str(node)] for node in node_indices], dtype = float)
        
    def _generate_random_walks(self,
        adjacency_matrix:   Tensor
    ) -> list:
        """# Generate random walks across nodes.

        ## Args:
            * adjacency_matrix  (Tensor):   Single dense batched adjacency matrix.

        ## Returns:
            * list: List of randomly generated walks.
        """
        # Initialize walks
        walks:      list =  []
        
        # Record number of nodes
        num_nodes:  int =   adjacency_matrix.shape[0]
        
        # Log action
        self._logger.info(f"Generating random walks acros {num_nodes} nodes")
        
        # For each node...
        for node in range(num_nodes):
            
            # For each walk to be made...
            for w in range(1, self.num_walks + 1):
                
                # Initialize walk at node
                walk:   list =  [node]
                
                # Log walk for debugging
                self._logger.debug(f"\tWalk {w}/{self.num_walks}")
                
                # For each step of the walk...
                for s in range(1, self.walk_length):
                    
                    # Log step for debugging
                    self._logger.debug(f"\t\tStep {s}/{self.walk_length}")
                    
                    # Get current node
                    current:    int =       walk[-1]
                    
                    # Find neighbors
                    neighbors:  ndarray =   where(adjacency_matrix[current] > 0)[0]
                    
                    # If there are no neighbors, skip
                    if len(neighbors) == 0: break
                    
                    # Otherwise, append random neighbor to walk
                    walk.append(choice(neighbors))
                    
                # Append walk to walks
                walks.append(list(map(str, walk)))
                
        # Return generated walks
        return walks
    
    def _train(self,
        data:   Data
    ) -> None:
        """# Train model and produce emebeddings.

        ## Args:
            * data  (Data):  Dataset's data component.
        """
        self.fit(data.edge_index)
        
    def _evaluate(self,
        data:   Data
    ) -> dict:
        """# Evaluate DeepWalk model embeddings.

        ## Args:
            * data  (Data): Data object containing node features, edge index, and masks.

        ## Returns:
            * dict:
                * accuracy: DeepWalk accuracy on nodes.
                * auc:      DeepWalk AUC score. 
        """
        # Place model in evaluation mode
        self.eval()
        
        # Form embeddings
        self.embeddings:    Tensor =    self.forward(range(data.x.size(0))).detach().cpu().numpy()
        
        # Extract labels & masks
        labels:             Tensor =    data.y.cpu().numpy()
        train_mask:         Tensor =    data.train_mask.cpu().numpy()
        test_mask:          Tensor =    data.test_mask.cpu().numpy()
        
        # Train classifier
        classifier: LogisticRegression =    LogisticRegression(max_iter = 1000)
        classifier.fit(self.embeddings[train_mask], labels[train_mask])
        
        # Make predictions
        predictions:    Tensor =    classifier.predict(self.embeddings[test_mask])
        
        # Calculate probabilities
        probabilities:  Tensor =    classifier.predict_proba(self.embeddings[test_mask])
        
        # Return accuracy & AUC score
        return {
            "accuracy": accuracy_score(labels[test_mask], predictions),
            "auc":      roc_auc_score(labels[test_mask], probabilities, multi_class = "ovr") if len(unique(labels)) > 2 else roc_auc_score(labels[test_mask], probabilities)
        }