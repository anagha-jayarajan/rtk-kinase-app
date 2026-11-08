import os, numpy as np, torch
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors, rdPartialCharges, ChemicalFeatures
from torch_geometric.data import Data

HT, BT, BS, CT = (Chem.rdchem.HybridizationType, Chem.rdchem.BondType,
                  Chem.rdchem.BondStereo, Chem.rdchem.ChiralType)
try:
    _factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
except Exception:
    _factory = None

ELEMENTS = ["H","B","C","N","O","F","Si","P","S","Cl","Se","Br","I","Na","K","Li","Mg","Ca","Fe","Zn","Cu","As","Sn","Pt"]
OTHER_SETS = {"1": ["C","N","O","S","F","Cl","Br","I","P"], "2": ["C","N","O","S","F","Cl","Br","I"],
              "3": ["C","N","O","S","F","Cl","Br"], "4": ["C","N","O","S","F","Cl","Br","I","P","B","Si","Se"],
              "5": ["C","N","O","F","S","Cl"], "6": ["C","N","O","F","P","S","Cl","Br","I"]}

# ------------------------------------------------------------ atom candidates
ATOM_FUNCS = {}
def _a(name, f): ATOM_FUNCS[name] = f
for e in ELEMENTS: _a(f"sym_{e}", lambda a, e=e: a.GetSymbol() == e)
for k, st in OTHER_SETS.items(): _a(f"sym_other_{k}", lambda a, st=st: a.GetSymbol() not in st)
_a("atomic_num", lambda a: a.GetAtomicNum()); _a("mass", lambda a: a.GetMass())
_a("degree", lambda a: a.GetDegree())
for k in range(0, 7): _a(f"degree_{k}", lambda a, k=k: a.GetDegree() == k)
for k in range(1, 7): _a(f"degree_ge_{k}", lambda a, k=k: a.GetDegree() >= k)
_a("total_degree", lambda a: a.GetTotalDegree())
for k in range(0, 7): _a(f"total_degree_{k}", lambda a, k=k: a.GetTotalDegree() == k)
_a("num_hs", lambda a: a.GetTotalNumHs())
for k in range(0, 5): _a(f"num_hs_{k}", lambda a, k=k: a.GetTotalNumHs() == k)
for k in range(1, 5): _a(f"num_hs_ge_{k}", lambda a, k=k: a.GetTotalNumHs() >= k)
_a("explicit_hs", lambda a: a.GetNumExplicitHs())
_a("implicit_valence", lambda a: a.GetImplicitValence())
_a("total_valence", lambda a: a.GetTotalValence())
for k in range(0, 8): _a(f"total_valence_{k}", lambda a, k=k: a.GetTotalValence() == k)
_a("formal_charge", lambda a: a.GetFormalCharge())
for k in range(-2, 3): _a(f"charge_{k}", lambda a, k=k: a.GetFormalCharge() == k)
_a("charge_pos", lambda a: a.GetFormalCharge() > 0); _a("charge_neg", lambda a: a.GetFormalCharge() < 0)
_a("charge_nonzero", lambda a: a.GetFormalCharge() != 0)
for nm in ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED", "OTHER"]:
    h = getattr(HT, nm, None)
    if h is not None: _a(f"hyb_{nm}", lambda a, h=h: a.GetHybridization() == h)
_a("aromatic", lambda a: a.GetIsAromatic()); _a("in_ring", lambda a: a.IsInRing())
for k in range(3, 9): _a(f"in_ring_{k}", lambda a, k=k: a.IsInRingSize(k))
_a("num_rings", lambda a: a.GetOwningMol().GetRingInfo().NumAtomRings(a.GetIdx()))
for nm in ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]:
    c = getattr(CT, nm, None)
    if c is not None: _a(f"chiral_{nm}", lambda a, c=c: a.GetChiralTag() == c)
_a("chiral_any", lambda a: a.GetChiralTag() != CT.CHI_UNSPECIFIED)
_a("cip_R", lambda a: a.HasProp("_CIPCode") and a.GetProp("_CIPCode") == "R")
_a("cip_S", lambda a: a.HasProp("_CIPCode") and a.GetProp("_CIPCode") == "S")
_a("radical_electrons", lambda a: a.GetNumRadicalElectrons()); _a("isotope", lambda a: a.GetIsotope())

BOND_FUNCS = {}
def _b(name, f): BOND_FUNCS[name] = f
for nm in ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]:
    _b(f"bond_{nm.lower()}", lambda b, nm=nm: b.GetBondType() == getattr(BT, nm))
_b("bond_type_num", lambda b: b.GetBondTypeAsDouble()); _b("conjugated", lambda b: b.GetIsConjugated())
_b("bond_in_ring", lambda b: b.IsInRing()); _b("bond_aromatic_flag", lambda b: b.GetIsAromatic())
for k in range(3, 9): _b(f"bond_in_ring_{k}", lambda b, k=k: b.IsInRingSize(k))
for nm in ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]:
    s_ = getattr(BS, nm, None)
    if s_ is not None: _b(f"stereo_{nm}", lambda b, s_=s_: b.GetStereo() == s_)
_b("stereo_any_not_none", lambda b: b.GetStereo() != BS.STEREONONE)
_b("rotatable", lambda b: b.GetBondType() == BT.SINGLE and not b.IsInRing()
   and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1)

def _compute(funcs, items):
    out = {k: np.zeros(len(items)) for k in funcs}
    for i, it in enumerate(items):
        for k, f in funcs.items():
            try: out[k][i] = float(f(it))
            except Exception: out[k][i] = 0.0
    return out

def atom_candidates(mol):
    n = mol.GetNumAtoms(); c = _compute(ATOM_FUNCS, list(mol.GetAtoms()))
    c["gasteiger"] = np.zeros(n)
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        c["gasteiger"] = np.nan_to_num(np.array([a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()], float),
                                       posinf=0, neginf=0)
    except Exception: pass
    c["crippen_logp"], c["crippen_mr"], c["tpsa_contrib"] = np.zeros(n), np.zeros(n), np.zeros(n)
    try:
        cc = rdMolDescriptors._CalcCrippenContribs(mol)
        c["crippen_logp"] = np.array([x[0] for x in cc], float); c["crippen_mr"] = np.array([x[1] for x in cc], float)
    except Exception: pass
    try: c["tpsa_contrib"] = np.array(rdMolDescriptors._CalcTPSAContribs(mol), float)
    except Exception: pass
    c["hbond_donor"], c["hbond_acceptor"] = np.zeros(n), np.zeros(n)
    if _factory is not None:
        try:
            for f in _factory.GetFeaturesForMol(mol):
                key = {"Donor": "hbond_donor", "Acceptor": "hbond_acceptor"}.get(f.GetFamily())
                if key: c[key][list(f.GetAtomIds())] = 1.0
        except Exception: pass
    return c

def bond_candidates(mol):
    return _compute(BOND_FUNCS, list(mol.GetBonds()))

# ------------------------------------------------------------ discovery
def _fit(Y, C, tol=1e-3):
    names = list(C); M = np.stack([C[n] for n in names], 1); res = []
    for j in range(Y.shape[1]):
        y = Y[:, j]; span = float(np.ptp(y))
        if span < 1e-12:
            res.append({"col": j, "kind": "const", "value": float(y[0]), "name": "constant", "a": 0.0, "b": float(y[0]),
                        "rel_err": 0.0, "ok": True}); continue
        best = None
        for k, n in enumerate(names):
            c = M[:, k]
            if np.ptp(c) < 1e-12: continue
            a, b = np.polyfit(c, y, 1); err = float(np.abs(a * c + b - y).max()) / span
            if best is None or err < best["rel_err"] - 1e-9:
                best = {"col": j, "kind": "fit", "name": n, "a": float(a), "b": float(b), "rel_err": err}
        best["ok"] = best["rel_err"] < tol; res.append(best)
    return res

def discover(smiles_list, graphs):
    AX, AC, AS, EY, EC = [], {}, [], [], {}; used = std = 0; loops = missing = 0
    for smi, g in zip(smiles_list, graphs):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or mol.GetNumAtoms() != g.x.shape[0]: continue
        used += 1
        AX.append(g.x.numpy().astype(float)); AS += [a.GetSymbol() for a in mol.GetAtoms()]
        for k, v in atom_candidates(mol).items(): AC.setdefault(k, []).append(v)
        ei, ea = g.edge_index.numpy(), g.edge_attr.numpy().astype(float)
        bc = bond_candidates(mol); keep, bidx = [], []
        for e in range(ei.shape[1]):
            u, v = int(ei[0, e]), int(ei[1, e])
            b = mol.GetBondBetweenAtoms(u, v) if u != v else None
            if b is None:
                loops += int(u == v); missing += int(u != v)
            else: keep.append(e); bidx.append(b.GetIdx())
        exp = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        exp = np.array([[x for i, j in exp for x in (i, j)], [x for i, j in exp for x in (j, i)]]).reshape(2, -1)
        std += int(exp.shape == ei.shape and (exp == ei).all())
        if keep:
            EY.append(ea[keep])
            for k, v in bc.items(): EC.setdefault(k, []).append(v[bidx])
    if used < 50: raise RuntimeError(f"Only {used} molecules matched graph atom counts. Check the SMILES/graph alignment.")
    node = _fit(np.vstack(AX), {k: np.concatenate(v) for k, v in AC.items()})
    edge = _fit(np.vstack(EY), {k: np.concatenate(v) for k, v in EC.items()})
    X, syms = np.vstack(AX), np.array(AS)
    for r in node:
        if not r["ok"]:
            col = X[:, r["col"]]
            r["seen_on"] = {s: int((col[syms == s] > 0.5).sum()) for s in sorted(set(syms)) if (col[syms == s] > 0.5).any()}
    return {"n_mols": used, "node_dim": int(X.shape[1]), "edge_dim": int(np.vstack(EY).shape[1]),
            "edge_order_standard": std / used, "self_loops": loops, "edges_not_bonds": missing,
            "node": node, "edge": edge, "verified": False}

# ------------------------------------------------------------ build a graph from a mapping
def build_graph(smiles, mapping):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: raise ValueError("Invalid SMILES")
    n = mol.GetNumAtoms(); ac = atom_candidates(mol)
    X = np.zeros((n, mapping["node_dim"]))
    for m in mapping["node"]:
        X[:, m["col"]] = m["b"] if m["kind"] == "const" else m["a"] * ac[m["name"]] + m["b"]
    bonds = list(mol.GetBonds()); bc = bond_candidates(mol)
    E = np.zeros((2 * len(bonds), mapping["edge_dim"])); src, dst = [], []
    for i, b in enumerate(bonds):
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); src += [u, v]; dst += [v, u]
        for m in mapping["edge"]:
            val = m["b"] if m["kind"] == "const" else m["a"] * bc[m["name"]][i] + m["b"]
            E[2 * i, m["col"]] = E[2 * i + 1, m["col"]] = val
    return Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
                edge_attr=torch.tensor(E, dtype=torch.float32).reshape(-1, mapping["edge_dim"]))


# ---------------------------------------------------------------- V2_EXTRAS
from rdkit.Chem import Descriptors, Crippen
_base_atom_candidates = atom_candidates

def atom_candidates(mol):
    c = _base_atom_candidates(mol); n = mol.GetNumAtoms(); rd = rdMolDescriptors
    mol_desc = {
        "MolWt": lambda m: Descriptors.MolWt(m), "HeavyMolWt": lambda m: Descriptors.HeavyAtomMolWt(m),
        "MolLogP": lambda m: Crippen.MolLogP(m), "MolMR": lambda m: Crippen.MolMR(m),
        "TPSA": lambda m: rd.CalcTPSA(m), "LabuteASA": lambda m: rd.CalcLabuteASA(m),
        "HBD": lambda m: rd.CalcNumHBD(m), "HBA": lambda m: rd.CalcNumHBA(m),
        "RotBonds": lambda m: rd.CalcNumRotatableBonds(m), "Rings": lambda m: rd.CalcNumRings(m),
        "AromRings": lambda m: rd.CalcNumAromaticRings(m), "Heteroatoms": lambda m: rd.CalcNumHeteroatoms(m),
        "FracCSP3": lambda m: rd.CalcFractionCSP3(m), "HeavyAtoms": lambda m: m.GetNumAtoms(),
        "NumBonds": lambda m: m.GetNumBonds(), "BertzCT": lambda m: Descriptors.BertzCT(m),
    }
    for k, f in mol_desc.items():
        try: c["mol_" + k] = np.full(n, float(f(mol)))
        except Exception: c["mol_" + k] = np.zeros(n)
    idx = np.arange(n, dtype=float); c["atom_index"], c["atom_index_frac"] = idx, idx / max(n - 1, 1)
    def nb(f): return np.array([sum(f(x) for x in a.GetNeighbors()) for a in mol.GetAtoms()], float)
    c["nb_aromatic"] = nb(lambda x: x.GetIsAromatic()); c["nb_hetero"] = nb(lambda x: x.GetAtomicNum() not in (1, 6))
    c["nb_sum_atomic_num"] = nb(lambda x: x.GetAtomicNum()); c["nb_in_ring"] = nb(lambda x: x.IsInRing())
    c["gasteiger_withHs"], c["crippen_logp_withHs"], c["crippen_mr_withHs"] = np.zeros(n), np.zeros(n), np.zeros(n)
    try:
        mh = Chem.AddHs(mol); rdPartialCharges.ComputeGasteigerCharges(mh)
        g = np.array([mh.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge") for i in range(n)], float)
        c["gasteiger_withHs"] = np.nan_to_num(g, posinf=0, neginf=0)
        cc = rdMolDescriptors._CalcCrippenContribs(mh)
        c["crippen_logp_withHs"] = np.array([cc[i][0] for i in range(n)], float)
        c["crippen_mr_withHs"] = np.array([cc[i][1] for i in range(n)], float)
    except Exception: pass
    return c


# ---------------------------------------------------------------- V3_EXTRAS
from rdkit.Chem import EState
_v2_atom_candidates = atom_candidates
_PT = Chem.GetPeriodicTable()
_PAULING = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 34: 2.55, 35: 2.96, 53: 2.66}

def atom_candidates(mol):
    c = _v2_atom_candidates(mol); n = mol.GetNumAtoms()
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    def tab(fn):
        try: return np.array([fn(z) for z in zs], float)
        except Exception: return np.zeros(n)
    c["pt_rvdw"], c["pt_rcov"], c["pt_rb0"] = tab(_PT.GetRvdw), tab(_PT.GetRcovalent), tab(_PT.GetRb0)
    c["pt_outer_e"], c["pt_default_valence"] = tab(_PT.GetNOuterElecs), tab(_PT.GetDefaultValence)
    c["pauling_en"] = tab(lambda z: _PAULING.get(z, 0.0))
    c["labute_asa"], c["estate"] = np.zeros(n), np.zeros(n)
    try: c["labute_asa"] = np.array(rdMolDescriptors._CalcLabuteASAContribs(mol)[0], float)
    except Exception: pass
    try: c["estate"] = np.array(EState.EStateIndices(mol), float)
    except Exception: pass
    return c


# ---------------------------------------------------------------- V4_FAMILIES
import os as _os
from rdkit import RDConfig as _RDConfig, RDLogger as _RDLogger
from rdkit.Chem import ChemicalFeatures as _CF
_RDLogger.DisableLog('rdApp.*')
_FAC = _CF.BuildFeatureFactory(_os.path.join(_RDConfig.RDDataDir, "BaseFeatures.fdef"))
_FAMS = ['Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder',
         'Aromatic', 'Hydrophobe', 'LumpedHydrophobe']
_v3_atom_candidates = atom_candidates

def atom_candidates(mol):
    c = _v3_atom_candidates(mol); n = mol.GetNumAtoms()
    M = {f: np.zeros(n) for f in _FAMS}
    for f in _FAC.GetFeaturesForMol(mol):
        if f.GetFamily() in M:
            for i in f.GetAtomIds(): M[f.GetFamily()][i] = 1.0
    for f in _FAMS: c["fam_" + f] = M[f]
    return c
