import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from rdkit import Chem

H, NH, DO = 128, 4, 0.2

class _Enc(nn.Module):
    def __init__(self, in_dim, edge_dim):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, H // NH, heads=NH, edge_dim=edge_dim, dropout=DO)
        self.conv2 = GATv2Conv(H, H, heads=1, concat=False, edge_dim=edge_dim, dropout=DO)
        self.norm1, self.norm2 = nn.LayerNorm(H), nn.LayerNorm(H)
    def forward(self, x, ei, ea):
        x = F.elu(self.norm1(self.conv1(x, ei, ea))); x = F.dropout(x, DO, self.training)
        return F.elu(self.norm2(self.conv2(x, ei, ea)))

class LigandEncoder(_Enc):
    def __init__(self): super().__init__(36, 10)
class ProteinEncoder(_Enc):
    def __init__(self): super().__init__(27, 1)

class GraphPoolingRegression(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(hidden_dim*2, 256), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
    def forward(self, lx, px, lb, pb):
        return self.regressor(torch.cat([global_mean_pool(lx, lb), global_mean_pool(px, pb)], 1)).squeeze(-1)

class FastCrossAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.ligand_to_protein = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.protein_to_ligand = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ligand_norm, self.protein_norm = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
    def forward(self, lx, px, lb, pb, return_attn=False):
        L, Lm = to_dense_batch(lx, lb); P, Pm = to_dense_batch(px, pb)
        l_att, w_lp = self.ligand_to_protein(L, P, P, key_padding_mask=~Pm, need_weights=return_attn, average_attn_weights=True)
        p_att, w_pl = self.protein_to_ligand(P, L, L, key_padding_mask=~Lm, need_weights=return_attn, average_attn_weights=True)
        Lo, Po = self.ligand_norm(L + l_att)[Lm], self.protein_norm(P + p_att)[Pm]
        return (Lo, Po, w_lp, w_pl) if return_attn else (Lo, Po)

class RTKDualGraphModelFast(nn.Module):
    def __init__(self, le, pe, ca, pr):
        super().__init__()
        self.ligand_encoder, self.protein_encoder, self.cross_attention, self.pooling_regression = le, pe, ca, pr
    def forward(self, batch, return_attn=False):
        lig, prot = batch["ligand"], batch["protein"]
        le = self.ligand_encoder(lig.x, lig.edge_index, lig.edge_attr)
        pe = self.protein_encoder(prot.x, prot.edge_index, prot.edge_attr)
        out = self.cross_attention(le, pe, lig.batch, prot.batch, return_attn=return_attn)
        pred = self.pooling_regression(out[0], out[1], lig.batch, prot.batch)
        return (pred, {"w_lp": out[2], "w_pl": out[3]}) if return_attn else pred

def load_model(path, device):
    pkg = torch.load(path, map_location=device, weights_only=False)
    m = RTKDualGraphModelFast(LigandEncoder(), ProteinEncoder(), FastCrossAttention(), GraphPoolingRegression()).to(device)
    m.load_state_dict(pkg["model_state_dict"]); m.eval()
    return m, pkg

def _to_batch(g, device):
    get = (lambda k: g[k]) if isinstance(g, dict) else (lambda k: getattr(g, k))
    d = Data(x=get("x").float(), edge_index=get("edge_index").long(), edge_attr=get("edge_attr").float())
    return Batch.from_data_list([d]).to(device)

@torch.no_grad()
def interaction(model, lig_g, prot_g, smiles, device, top_k=3, min_enrich=2.0):
    batch = {"ligand": _to_batch(lig_g, device), "protein": _to_batch(prot_g, device)}
    pred, attn = model(batch, return_attn=True)
    w_lp = attn["w_lp"][0].cpu(); w_pl = attn["w_pl"][0].cpu().T
    score = torch.sqrt(w_lp * w_pl); score = score / score.mean()
    n_atoms, n_res = score.shape
    symbols = [f"A{i}" for i in range(n_atoms)]
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is not None and mol.GetNumAtoms() == n_atoms:
        symbols = [f"{a.GetSymbol()}{a.GetIdx()}" for a in mol.GetAtoms()]
    vals, idx = score.topk(min(top_k, n_res), dim=1)
    rows = [(a, symbols[a], int(r), int(r) + 1, float(v))
            for a in range(n_atoms) for v, r in zip(vals[a], idx[a]) if v >= min_enrich]
    edges = pd.DataFrame(rows, columns=["atom_idx", "atom", "residue_idx", "residue_pos", "enrichment"]) \
              .sort_values("enrichment", ascending=False).reset_index(drop=True)
    top_res = (edges.groupby("residue_pos").enrichment.agg(["sum", "count"])
                    .sort_values("sum", ascending=False).head(15).reset_index()) if len(edges) else pd.DataFrame()
    return {"pred": float(pred.item()), "score": score.numpy(), "symbols": symbols,
            "edges": edges, "top_residues": top_res}
