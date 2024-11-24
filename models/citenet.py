"""CiteNet model implementation."""

from logging                import Logger

from gensim.models          import Word2Vec
from networkx               import Graph
from networkx.exception     import NetworkXError
from numpy                  import array, ndarray, unique
from numpy.random           import choice
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import accuracy_score, roc_auc_score
from torch                  import cat, float32, tensor, Tensor
from torch.nn               import Linear, Module
from torch.nn.functional    import relu
from torch_geometric.data   import Data
from torch_geometric.nn     import GATConv

from utils                  import LOGGER

class CiteNet(Module):
    """# CiteNet model.
    
    Graph Attention-Based Node Embedding (GANE) model.
    """
    
    # Initialize logger
    _logger:    Logger =    LOGGER.getChild("citenet")
    
    def __init__(self,
        channels_in:    int,
        hidden_layers:  int,
        channels_out:   int,
        **kwargs
    ):
        """# Initialize CiteNet model.

        ## Args:
            * channels_in   (int):              Number of input channels.
            * hidden_layers (int):              Number of hidden layers.
            * channels_out  (int):              Number of output channels.
        """
        # Initialize module
        super(CiteNet, self).__init__()
        
        # Define attention layers
        self.gat1:  GATConv =   GATConv(in_channels = channels_in,          out_channels = hidden_layers,   heads = 8,  concat = True,  dropout = 0.6)
        self.gat2:  GATConv =   GATConv(in_channels = hidden_layers * 8,    out_channels = hidden_layers,   heads = 1,  concat = True,  dropout = 0.6)
        
        # Define fully connected layer
        self.fc:    Linear =    Linear(in_features = hidden_layers, out_features = channels_out)
        
    def fit(self,
        data:   Data
    ) -> None:
        """# Fit model on walk embeddings with GAT.

        ## Args:
            * data  (Data): Graph data.
        """
        # Log action
        self._logger.info(f"Fitting model.")
        
        # Convert edge index to array
        edge_index: ndarray =   data.edge_index.numpy()
        
        # Generate random walks
        self.embeddings:    Tensor =    self.generate_embeddings(edge_index = edge_index, num_nodes = data.num_nodes)
        
    def forward(self,
        X:          Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """# Perform forward pass of data through model.

        ## Args:
            * X             (Tensor):   Input tensor.
            * edge_index    (Tensor):   Graph's edge index.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # Forward pass through layer 1
        X = relu(self.gat1(X, edge_index))
        
        # Forward pass through layer 2
        X = relu(self.gat2(X, edge_index))
        
        # Return classification
        return self.fc(X)
        
    def generate_embeddings(self,
        edge_index: Tensor,
        num_nodes:  int
    ) -> Tensor:
        """# Generate random walk embeddings for each node.

        ## Args:
            * edge_index    (Tensor):   Graph edge indices.
            * num_nodes     (int):      Number of nodes in graph.

        ## Returns:
            * Tensor:   Embeddings generated during walks.
        """
        # Convert edge index to graph
        graph:  Graph = Graph()
        graph.add_edges_from(edge_index.T)
        
        # Initialize walks list
        walks:  list =  []
        
        # Log action
        self._logger.info(f"Generating random walks acros {num_nodes} nodes")
        
        # For each node in graph...
        for node in range(num_nodes):
            
            # For each walk prescribed...
            for w in range(10):
                
                # Log walk for debugging
                self._logger.debug(f"\tWalk {w}/{num_nodes}")
                
                # Generate walk
                walk = self.random_walk(graph = graph, start = node, walk_length = 40)
                
                # Append to list
                walks.append([str(n) for n in walk])
                
        # Train Word2Vec model
        w2v:    Word2Vec =  Word2Vec(sentences = walks, vector_size = 128, window = 5, sg = 1, workers = 4)
        
        # Return embeddings in Tensor
        return tensor(array([w2v.wv[str(node)] for node in range(num_nodes)]), dtype = float32)
    
    def random_walk(self, 
        graph:          Graph, 
        start:          int, 
        walk_length:    int
    ) -> list:
        """
        # Perform random walk on given node.

        ## Args:
            * graph         (Graph):    The input graph.
            * start         (int):      The starting node for the random walk.
            * walk_length   (int):      The length of the random walk.

        ## Returns:
            * list: The sequence of nodes visited during the random walk.
        """
        # Initialize walk with given node
        walk:   list =  [start]
        
        # For each step prescribed...
        for s in range(walk_length - 1):
                    
            try:# Log step for debugging
                self._logger.debug(f"\t\tStep {s}/{walk_length}")
                
                # Start with given node
                current:    int =   walk[-1]
                
                # Extract neighbors
                neighbors:  list =  list(graph.neighbors(current))
                
                # Skip if there are no neighbors
                if not neighbors:   break
                
                # Step to random neighbor
                walk.append(choice(neighbors))
                
            # Node not found in graph
            except NetworkXError: break
            
        # Provide generated walk
        return walk
    
    def _train(self,
        data:   Data
    ) -> None:
        """# Train (fit) model.

        ## Args:
            * data  (Data): Graph data.
        """
        self.fit(data)
        
    def _evaluate(self,
        data:   Data
    ) -> dict:
        """# Evaluate CiteNet model.

        ## Args:
            * data  (Data): Data object containing node features, edge index, and masks.

        ## Returns:
            * dict:
                * accuracy: GCN accuracy on nodes.
                * auc:      GCN AUC score.
        """
        # Generate embeddings
        embeddings: Tensor =    self.forward(data.x, data.edge_index)
        
        # Combine embeddings
        embeddings: Tensor =    cat([tensor(embeddings), tensor(self.embeddings)], dim = 1)
        
        # Extract targets
        targets:    Tensor =    data.y.cpu().numpy()
            
        # Extract train & test masks
        train_mask:     Tensor =    data.train_mask.cpu().numpy()
        test_mask:      Tensor =    data.test_mask.cpu().numpy()
        
        # Train classifier
        classifier: LogisticRegression =    LogisticRegression(max_iter = 1000)
        classifier.fit(self.embeddings[train_mask], targets[train_mask])
        
        # Make predictions
        predictions:    Tensor =    classifier.predict(self.embeddings[test_mask])
        
        # Calculate probabilities
        probabilities:  Tensor =    classifier.predict_proba(self.embeddings[test_mask])
        
        # Return accuracy & AUC score
        return {
            "accuracy": accuracy_score(targets[test_mask], predictions),
            "auc":      roc_auc_score(targets[test_mask], probabilities, multi_class = "ovr") 
                        if len(unique(targets)) > 2 
                        else roc_auc_score(targets[test_mask], probabilities)
        }