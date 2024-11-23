"""Run graph learning task."""

from logging                    import Logger

from torch_geometric.datasets   import Planetoid

from datasets                   import load_dataset
from utils                      import LOGGER

def run_task(
    dataset:    str,
    model:      str,
    **kwargs
) -> None:
    """_summary_

    Args:
        dataset (str): _description_
        model (str): _description_
    """
    # Initialize logger
    _logger:    Logger =    LOGGER.getChild("task")
    
    # Log action
    _logger.info(f"Initializing task ({locals()})")
    
    # Load dataset
    dataset:    Planetoid = load_dataset(dataset = dataset, **kwargs)