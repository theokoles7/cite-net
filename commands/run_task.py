"""Run graph learning task."""

from json                       import dump, dumps, load
from logging                    import Logger

from torch.nn                   import Module
from torch_geometric.data       import Data

from datasets                   import load_dataset
from models                     import load_model
from utils                      import LOGGER

def run_task(
    dataset:    str,
    model:      str,
    **kwargs
) -> None:
    """_summary_

    Args:
        data (str): _description_
        model (str): _description_
    """
    # Initialize logger
    _logger:    Logger =    LOGGER.getChild("task")
    
    # Initialization -------------------------------------------------------------------------------
    
    # Log action
    _logger.info(f"Initializing task ({locals()})")
    
    # Load data
    data:       Data =      load_dataset(dataset = dataset, **kwargs)
    
    # Load model
    model:      Module =    load_model(
        model =         model, 
        channels_in =   data.num_node_features,
        hidden_layers = 64,
        channels_out =  data.num_classes,
        **kwargs
    )
    
    # Training -------------------------------------------------------------------------------------
    
    # Train/fit model
    model._train(data[0])
    
    # Evaluation -----------------------------------------------------------------------------------
    
    # Evaluate model
    results:    dict =      model._evaluate(data[0])
    
    # Reporting ------------------------------------------------------------------------------------
    
    # Log result
    _logger.info(f"{model._get_name()} results on {dataset}: {dumps(results, indent = 2, default = str)}")
    
    # Load report for writing
    report: dict =  load(fp = open(f"output/results.json", "r"))
    
    # Update results
    report[model._get_name()][dataset] = results
    
    # Record results in file
    dump(obj = report, fp = open(f"output/results.json", "w"), indent = 2, default = str)