"""Graph Convolution Network model implementation."""

from logging                import Logger

from numpy                  import ndarray
from torch                  import Tensor
from torch.nn               import Module
from torch.nn.functional    import log_softmax, nll_loss, relu
from torch.optim            import Adam
from torch_geometric.data   import Data
from torch_geometric.nn     import GCNConv

from utils                  import LOGGER

class GCN(Module):
    """# Graph Convolution Network model.

    GCN learns representations by aggregating features from neighbors. Below is a PyTorch 
    implementation.
    """
    
    # Initialize logger
    _logger:    Logger =    LOGGER.getChild("gcn")
    
    def __init__(self,
        channels_in:    int,
        hidden_layers:  int,
        channels_out:   int,
        **kwargs
    ):
        """# Initialize GCN model.

        ## Args:
            * channels_in   (int):              Number of input channels.
            * hidden_layers (int):              Number of hidden layers.
            * channels_out  (int):              Number of output channels.
        """
        # Initialize module
        super(GCN, self).__init__()
        
        # Define layers
        self.conv1: GCNConv =   GCNConv(in_channels = channels_in,      out_channels = hidden_layers)
        self.conv2: GCNConv =   GCNConv(in_channels = hidden_layers,    out_channels = channels_out)
        
    def forward(self,
        X:          Tensor,
        edge_index: ndarray
    ) -> Tensor:
        """# Perform forward pass of data through network.

        ## Args:
            * X             (Tensor):   Input tensor.
            * edge_index    (ndarray):  Graph's edge index.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # Forward pass through layer 1
        X = relu(self.conv1(X, edge_index))
        
        # Forward pass through layer 2
        return log_softmax(self.conv2(X, edge_index), dim = 1)
    
    def _train(self,
        data:           Data,
        epochs:         int =   100,
        learning_rate:  float = 0.01
    ) -> None:
        """# Train GCN model on graph.

        ## Args:
            * data          (Data):             Graph's data component.
            * epochs        (int, optional):    Number of epochs for which model will be trained. 
                                                Defaults to 100.
            * learning_rate (float, optional):  Learning rate with which optimizer will be 
                                                initialized. Defaults to 0.01.
        """
        # Log action
        self._logger.info(f"Training GCN for {epochs} epochs")
        
        # Initialize optimizer
        optimizer:  Adam =  Adam(params = self.parameters(), lr = learning_rate)
        
        # For each scheduled epoch
        for epoch in range(1, epochs + 1):
            
            # Place model in training mode
            self.train()
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Make predictions
            predictions:    Tensor =    self(data.x, data.edge_index)
            
            # Calculate loss
            loss:           Tensor =    nll_loss(predictions[data.train_mask], data.y[data.train_mask])
            
            # Back propagation
            loss.backward()
            
            # Update weights
            optimizer.step()