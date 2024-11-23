"""Console argument definitions & parsing."""

from argparse   import ArgumentParser, _ArgumentGroup, Namespace, _SubParsersAction

# Initialize parser
_parser:    ArgumentParser =    ArgumentParser(
    prog =          "cite-net",
    description =   "Graph representation learning for node classification."
)

# Initialize sub-parser
_subparser: _SubParsersAction = _parser.add_subparsers(
    dest =          "cmd",
    help =          "Graph learning commands."
)

# BEGIN ARGUMENTS ==================================================================================

# LOGGING -----------------------------------------------------------------
_logging:   _ArgumentGroup =    _parser.add_argument_group(title = "Logging")

_logging.add_argument(
    "--logging-level",
    type =          str,
    choices =       ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default =       "INFO",
    help =          "Minimum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). Defaults to 'INFO'."
)

_logging.add_argument(
    "--logging-path",
    type =          str,
    default =       "logs",
    help =          "Path at which logs will be written. Defaults to './logs/'."
)

# COMMANDS ----------------------------------------------------------------

# RUN-TASK _______________________________________
_run_task:  ArgumentParser =    _subparser.add_parser(
    name =          "run-task",
    help =          "Run a graph learning task."
)

# MODEL .................
_model:     _ArgumentGroup =    _run_task.add_argument_group("Model")

_model.add_argument(
    "model",
    type =          str,
    choices =       ["citenet", "deepwalk", "gcn"],
    default =       "citenet",
    help =          "Choice of model with which task will be run. Defaults to 'citenet'."
)

# DATASET ...............
_dataset:   _ArgumentGroup =    _run_task.add_argument_group("Dataset")

_dataset.add_argument(
    "dataset",
    type =          str,
    choices =       ["citeseer", "cora", "pubmed"],
    default =       "citeseer",
    help =          "Choice of dataset on which task will be run. Defaults to 'citeseer'."
)

_dataset.add_argument(
    "--split",
    type =          str,
    choices =       ["public", "full", "geom-gcn", "random"],
    default =       "public",
    help =          """The type of dataset split ("public", "full", "geom-gcn", "random"). If set to 
                    "public", the split will be the public fixed split from the “Revisiting 
                    Semi-Supervised Learning with Graph Embeddings” paper. If set to "full", all 
                    nodes except those in the validation and test sets will be used for training (as 
                    in the “FastGCN: Fast Learning with Graph Convolutional Networks via Importance 
                    Sampling” paper). If set to "geom-gcn", the 10 public fixed splits from the 
                    “Geom-GCN: Geometric Graph Convolutional Networks” paper are given. If set to 
                    "random", train, validation, and test sets will be randomly generated, according 
                    to num_train_per_class, num_val and num_test. Default to "public"."""
)

_dataset.add_argument(
    "--num-train-per-class",
    type =          int,
    default =       20,
    help =          """The number of training samples per class in case of "random" split. Defaults 
                    to 20."""
)

_dataset.add_argument(
    "--num-val",
    type =          int,
    default =       500,
    help =          """The number of validation samples in case of "random" split. Defaults to 500."""
)

_dataset.add_argument(
    "--num-test",
    type =          int,
    default =       1000,
    help =          """The number of test samples in case of "random" split. Defaults to 1000."""
)

_dataset.add_argument(
    "--force-reload",
    action =        "store_true",
    default =       False,
    help =          """Whether to re-process the dataset. Defaults to False."""
)

# END ARGUMENTS ====================================================================================

# Parse console arguments into name space
ARGS:       Namespace =         _parser.parse_args()