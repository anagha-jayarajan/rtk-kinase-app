
import json, os
from featurizer_lib import build_graph
_MAP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "featurizer_mapping.json")))

def smiles_to_ligand_graph(smiles):
    if not _MAP.get("verified"):
        raise NotImplementedError("Featuriser mapping is not verified")
    return build_graph(smiles, _MAP)
