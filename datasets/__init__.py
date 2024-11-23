__all__ = ["citeseer", "cora", "pubmed", "loader"]

from datasets.citeseer  import CiteSeer
from datasets.cora      import Cora
from datasets.pubmed    import PubMed

from datasets.loader    import load_dataset