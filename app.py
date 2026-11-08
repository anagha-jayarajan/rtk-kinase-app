import os, numpy as np, pandas as pd, torch, streamlit as st
import plotly.graph_objects as go, plotly.express as px
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, rdFingerprintGenerator
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from model import load_model, interaction

ROOT = os.environ.get("RTK_ROOT", os.path.dirname(os.path.abspath(__file__)))
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def _load_lig(row, smiles):
    f = f"{ROOT}/ligand_graphs/graph_{int(row['original_row_index']):06d}.pt"
    if os.path.exists(f):
        return torch.load(f, map_location="cpu", weights_only=False)
    from featurizer import smiles_to_ligand_graph
    return smiles_to_ligand_graph(smiles)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="RTK Dual-Graph | Kinase Potency Prediction",
                   page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- STYLE
NAVY, TEAL, INK, MUTED = "#0B1F3A", "#00A6A6", "#14213D", "#5B6B82"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color:{INK}; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding-top:0; padding-bottom:3rem; max-width:1250px; }}

.nav {{ display:flex; justify-content:space-between; align-items:center; padding:18px 4px;
        border-bottom:1px solid #E6EBF2; margin-bottom:0; }}
.nav .logo {{ font-weight:800; font-size:1.25rem; letter-spacing:-0.02em; color:{NAVY}; }}
.nav .logo span {{ color:{TEAL}; }}
.nav .links a {{ margin-left:28px; font-size:.9rem; font-weight:500; color:{MUTED}; text-decoration:none; }}

.hero {{ background:linear-gradient(120deg,{NAVY} 0%,#12385F 60%,#0E5E73 100%);
         border-radius:18px; padding:64px 56px; margin:22px 0 28px 0; color:white; position:relative; overflow:hidden; }}
.hero:after {{ content:""; position:absolute; right:-80px; top:-80px; width:380px; height:380px; border-radius:50%;
               background:radial-gradient(circle,rgba(0,166,166,.35),transparent 70%); }}
.hero .eyebrow {{ color:#5FE3E3; font-weight:600; font-size:.8rem; letter-spacing:.14em; text-transform:uppercase; }}
.hero h1 {{ font-size:2.8rem; font-weight:800; line-height:1.12; letter-spacing:-0.03em; margin:12px 0 16px 0; color:white; }}
.hero p {{ font-size:1.08rem; color:#C9D8EA; max-width:640px; line-height:1.6; margin:0; }}

.kpi {{ background:white; border:1px solid #E6EBF2; border-radius:14px; padding:20px 22px;
        box-shadow:0 2px 10px rgba(11,31,58,.04); }}
.kpi .v {{ font-size:1.9rem; font-weight:800; color:{NAVY}; letter-spacing:-0.02em; }}
.kpi .l {{ font-size:.78rem; color:{MUTED}; font-weight:600; text-transform:uppercase; letter-spacing:.08em; }}

.card {{ background:white; border:1px solid #E6EBF2; border-radius:14px; padding:22px 24px;
         box-shadow:0 2px 10px rgba(11,31,58,.04); margin-bottom:16px; }}
.section {{ font-size:1.35rem; font-weight:700; letter-spacing:-0.02em; margin:26px 0 4px 0; color:{NAVY}; }}
.sub {{ color:{MUTED}; font-size:.93rem; margin-bottom:14px; }}
.badge {{ display:inline-block; padding:4px 12px; border-radius:99px; font-size:.78rem; font-weight:600; }}
.badge.ok {{ background:#E3F7F4; color:#067A6F; }} .badge.warn {{ background:#FFF1DB; color:#9A5B00; }}

.stTabs [data-baseweb="tab-list"] {{ gap:8px; border-bottom:1px solid #E6EBF2; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; padding:10px 18px; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:{NAVY}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{TEAL}; }}
.stButton>button {{ background:{TEAL}; color:white; border:0; border-radius:10px; padding:.65rem 1.4rem; font-weight:600; }}
.stButton>button:hover {{ background:#008A8A; color:white; }}
.foot {{ text-align:center; color:{MUTED}; font-size:.82rem; margin-top:40px; padding-top:20px; border-top:1px solid #E6EBF2; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- DATA
@st.cache_resource
def get_model():
    return load_model(f"{ROOT}/dualgraph_package.pt", device)

@st.cache_data
def get_test():
    t = pd.read_csv(f"{ROOT}/test_scaffold.csv")
    p = pd.read_csv(f"{ROOT}/test_predictions_dualgraph.csv")
    return t.merge(p[["original_row_index", "y_pred"]], on="original_row_index")

@st.cache_resource
def get_train_fps():
    path = f"{ROOT}/train_scaffold.csv"
    if not os.path.exists(path): return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in pd.read_csv(path)["Canonical_isomeric_SMILES"].drop_duplicates():
        m = Chem.MolFromSmiles(str(s))
        if m is not None: fps.append(gen.GetFingerprint(m))
    return fps

model, pkg = get_model()
test_df = get_test()
kinases = pkg["kinases"]
vm, tm = pkg["val_metrics"], pkg["test_metrics"]





# DARK_FIX: readable headings on a dark page
st.markdown("""<style>
.nav { border-bottom:1px solid #2E3846 !important; }
.nav .logo { color:#FFFFFF !important; font-size:1.4rem !important; }
.nav .logo span { color:#2DD4D4 !important; }
.nav .links { color:#9FB0C6; font-size:.9rem; font-weight:500; }
.section { color:#FFFFFF !important; font-size:1.5rem !important; }
.sub { color:#B4C2D6 !important; font-size:.98rem !important; }
.stTabs [data-baseweb="tab"] p { font-size:1.02rem; }
</style>""", unsafe_allow_html=True)


# TAB_FIX: readable tabs on a dark page
st.markdown("""<style>
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:transparent !important; border-radius:8px 8px 0 0; padding:10px 18px; }
.stTabs [data-baseweb="tab"] p { color:#B4C2D6 !important; font-size:1.05rem !important; font-weight:600 !important; }
.stTabs [data-baseweb="tab"]:hover { background:rgba(45,212,212,.12) !important; }
.stTabs [data-baseweb="tab"]:hover p { color:#5FE3E3 !important; }
.stTabs [aria-selected="true"] p { color:#FFFFFF !important; font-weight:700 !important; }
.stTabs [data-baseweb="tab-highlight"] { background:#2DD4D4 !important; height:3px !important; }
.stTabs [data-baseweb="tab-border"] { background:#2E3846 !important; }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- NAV + HERO
st.markdown("""
<div class="nav"><div class="logo">RTK<span>.</span>DualGraph</div>
<div class="links">Research preview</div></div>
<div class="hero">
  <div class="eyebrow">Receptor tyrosine kinase · Structure-aware AI</div>
  <h1>Predict kinase inhibitor potency<br>and see where it binds.</h1>
  <p>A dual-graph attention network learns from the ligand's molecular graph and the kinase's residue graph,
  then exposes which atoms and residues drive each prediction.</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
for col, val, lab in [(k1, f"{tm['r2']:.2f}", "Test R²"), (k2, f"{tm['rmse']:.2f}", "Test RMSE (pIC50)"),
                      (k3, f"{tm['spearman']:.2f}", "Test Spearman ρ"), (k4, f"{len(kinases)}", "Supported kinases")]:
    col.markdown(f'<div class="kpi"><div class="v">{val}</div><div class="l">{lab}</div></div>', unsafe_allow_html=True)

tab_pred, tab_perf, tab_about = st.tabs(["Predict", "Model performance", "About & limitations"])

# ---------------------------------------------------------------- PREDICT
import requests
import py3Dmol
import matplotlib
import streamlit.components.v1 as components
from numpy.lib.stride_tricks import sliding_window_view
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem.Draw import rdMolDraw2D

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def uniprot_info(uid):
    try:
        j = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json", timeout=15).json()
        gene = (j.get("genes") or [{}])[0].get("geneName", {}).get("value", uid)
        pd_ = j.get("proteinDescription", {})
        rec = pd_.get("recommendedName") or (pd_.get("submittedName") or [{}])[0]
        doms = []
        for f in j.get("features", []):
            if f.get("type") == "Domain" and "kinase" in f.get("description", "").lower():
                doms.append((int(f["location"]["start"]["value"]), int(f["location"]["end"]["value"])))
        return {"gene": gene, "name": rec.get("fullName", {}).get("value", ""),
                "seq": j["sequence"]["value"], "domains": doms}
    except Exception:
        return {"gene": uid, "name": "", "seq": "", "domains": []}

@st.cache_data(show_spinner=False)
def gene_names(ids):
    try:
        q = " OR ".join(f"accession:{i}" for i in ids)
        r = requests.get("https://rest.uniprot.org/uniprotkb/stream",
                         params={"query": q, "fields": "accession,gene_primary", "format": "tsv"}, timeout=30)
        out = {}
        for line in r.text.strip().split("\n")[1:]:
            parts = line.split("\t")
            out[parts[0]] = parts[1] if len(parts) > 1 and parts[1] else parts[0]
        return out
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def alphafold_pdb(uid):
    try:
        j = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uid}", timeout=20).json()
        return requests.get(j[0]["pdbUrl"], timeout=60).text
    except Exception:
        return None

def ca_coords(pdb):
    out = {}
    for l in pdb.splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            out[int(l[22:26])] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return out

AA_ORDERS = ["ACDEFGHIKLMNPQRSTVWY", "ARNDCQEGHILKMFPSTWYV"]

def map_graph_to_sequence(prot_g, seq, domains):
    """Find which UniProt residues the protein-graph nodes correspond to.
    Returns {'offset', 'text', 'verified'} or None. UniProt position of node i = offset + i + 1."""
    x = (prot_g["x"] if isinstance(prot_g, dict) else prot_g.x).float().numpy()
    n = x.shape[0]
    # 1) verify using amino-acid one-hot columns inside the node features (if present)
    if len(seq) >= n:
        for s in range(0, x.shape[1] - 19):
            b = x[:, s:s + 20]
            if np.all((b == 0) | (b == 1)) and np.all(b.sum(1) == 1):
                idx = b.argmax(1)
                sq = np.frombuffer(seq.encode(), dtype=np.uint8)
                best = None
                for order in AA_ORDERS:
                    g = np.frombuffer("".join(order[i] for i in idx).encode(), dtype=np.uint8)
                    ident = (sliding_window_view(sq, n) == g).mean(1)
                    o = int(ident.argmax())
                    if best is None or ident[o] > best[1]:
                        best = (o, float(ident[o]))
                if best and best[1] >= 0.9:
                    return {"offset": best[0], "verified": True,
                            "text": f"Residue mapping verified by amino-acid match ({best[1]:.0%}): graph covers UniProt residues "
                                    f"{best[0] + 1}–{best[0] + n}."}
                break
    # 2) fall back to UniProt kinase-domain annotation if its length equals the graph size
    for (a, b_) in domains:
        if b_ - a + 1 == n:
            return {"offset": a - 1, "verified": False,
                    "text": f"Graph matches the UniProt kinase domain ({a}–{b_}, {n} residues). "
                            f"Mapping is based on length and annotation, not verified by sequence."}
    return None

def ligand_3d(mol):
    mh = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mh, randomSeed=42) != 0:
        if AllChem.EmbedMolecule(mh, randomSeed=42, useRandomCoords=True) != 0:
            return None
    try: AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
    except Exception: pass
    m = Chem.RemoveHs(mh)
    return m if m.GetNumAtoms() == mol.GetNumAtoms() else None

def xyz(p): return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}

def build_3d_view(pdb, mol3d, edges, ca, res_label, symbols, offset, n_res, n_lines, spin):
    top = edges.head(n_lines).copy()
    top["rid"] = offset + top.residue_idx.astype(int) + 1            # UniProt / AlphaFold numbering
    top = top[top.rid.isin(ca.keys())]
    if top.empty: return None
    dom_ids = [i for i in range(offset + 1, offset + n_res + 1) if i in ca]
    dom_centre = np.mean([ca[i] for i in dom_ids], axis=0)
    pocket = np.mean([ca[i] for i in top.rid.unique()], axis=0)
    d = pocket - dom_centre; d = d / (np.linalg.norm(d) + 1e-6)
    conf = mol3d.GetConformer(); pos = conf.GetPositions()
    pos = pos - pos.mean(axis=0) + (pocket + d * 6.0)                 # schematic placement beside the pocket
    for i, p in enumerate(pos): conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
    molblock = Chem.MolToMolBlock(mol3d)

    v = py3Dmol.view(width=1100, height=640)
    v.addModel(pdb, "pdb")
    v.setStyle({"model": 0}, {})                                       # hide everything...
    v.setStyle({"model": 0, "resi": f"{offset + 1}-{offset + n_res}"},  # ...except the kinase domain
               {"cartoon": {"color": "#9FB4CC", "opacity": 0.9}})
    v.addModel(molblock, "mol")
    v.setStyle({"model": 1}, {"stick": {"radius": 0.28, "colorscheme": "greenCarbon"},
                              "sphere": {"scale": 0.3, "colorscheme": "greenCarbon"}})
    for rid in top.rid.unique():
        v.addStyle({"model": 0, "resi": int(rid)}, {"stick": {"radius": 0.25, "colorscheme": "orangeCarbon"}})
        lab = res_label[int(top[top.rid == rid].residue_idx.iloc[0])]
        v.addLabel(lab, {"position": xyz(ca[int(rid)]), "fontSize": 15, "fontColor": "#14213D",
                   "backgroundColor": "white", "backgroundOpacity": 0.9, "borderColor": "#E8830C",
                   "borderThickness": 1, "inFront": True})
    cmap = matplotlib.colormaps["plasma"]
    lo, hi = float(top.enrichment.min()), float(top.enrichment.max())
    for _, e in top.iterrows():
        t = (float(e.enrichment) - lo) / (hi - lo + 1e-9)
        v.addCylinder({"start": xyz(pos[int(e.atom_idx)]), "end": xyz(ca[int(e.rid)]),
                       "radius": 0.08 + 0.08 * t, "color": matplotlib.colors.to_hex(cmap(0.1 + 0.7 * t)),
                       "dashed": True, "fromCap": 1, "toCap": 1})
    for a in sorted(set(top.atom_idx.astype(int))):
        v.addLabel(symbols[a], {"position": xyz(pos[a]), "fontSize": 13, "fontColor": "#0B6B3A",
                   "backgroundColor": "white", "backgroundOpacity": 0.85, "borderThickness": 0, "inFront": True})
    v.setBackgroundColor("white")
    v.zoomTo({"model": 1}); v.zoom(0.5)
    if spin: v.spin(True)
    return v._make_html() if hasattr(v, "_make_html") else v.write_html()

def draw_mol(mol, highlight):
    try:
        d = rdMolDraw2D.MolDraw2DCairo(560, 380)
        d.drawOptions().addAtomIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(highlight))
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return Draw.MolToImage(mol, size=(560, 380))

# ---------- page ----------
def render_predict():
    st.markdown('<div class="section">Select a ligand and a kinase</div>'
                '<div class="sub">Pick both, and the model predicts potency and draws the 3D interaction view.</div>',
                unsafe_allow_html=True)
    names = gene_names(tuple(kinases))
    kin_label = lambda u: f"{names.get(u, u)}  ·  {u}"

    src = st.radio("Ligand source", ["Test-set compound", "Custom SMILES"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        uid = st.selectbox("① Target kinase (protein)", kinases, format_func=kin_label)
    row = None
    with c2:
        if src == "Test-set compound":
            only = st.checkbox("Only compounds measured on this kinase", True)
            pool = (test_df[test_df.UniProt_ID == uid] if only else test_df).reset_index()
            if pool.empty:
                st.warning("No test compounds for this kinase. Untick the box."); return
            def lab(i):
                r_ = pool.loc[i]
                tail = "" if only else f"  ·  measured on {names.get(r_.UniProt_ID, r_.UniProt_ID)}"
                return f"Compound #{int(r_['index'])}  ·  exp. pIC50 {r_.pIC50_median:.2f}{tail}"
            sel = st.selectbox("② Ligand (small molecule)", pool.index, format_func=lab)
            row = pool.loc[sel]
            smiles = row["Canonical_isomeric_SMILES"]
            ligand_name = f"Test compound #{int(row['index'])}"
        else:
            smiles = st.text_area("② Ligand SMILES", "COc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCCCN1CCOCC1", height=100)
            ligand_name = "Custom compound"

    o1, o2, o3 = st.columns(3)
    n_lines = o1.slider("Interaction lines to draw", 3, 30, 10)
    spin = o2.checkbox("Auto-rotate 3D view", False)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES."); return
    try:
        if row is not None:
            lig_g = _load_lig(row, smiles)
        else:
            from featurizer import smiles_to_ligand_graph
            lig_g = smiles_to_ligand_graph(smiles)
    except (ImportError, NotImplementedError, FileNotFoundError) as feat_err:
        st.warning(f"Custom SMILES is disabled. Reason: {feat_err}. Test-set mode works without it.")
        return
    prot_g = torch.load(f"{ROOT}/protein_graphs/{uid}.pt", map_location="cpu", weights_only=False)

    with st.spinner("Predicting and building the 3D view..."):
        r = interaction(model, lig_g, prot_g, smiles, device)
        info = uniprot_info(uid)
        pdb = alphafold_pdb(uid)

    S = r["score"]; n_atoms, n_res = S.shape
    atom_ok = mol.GetNumAtoms() == n_atoms
    mapping = map_graph_to_sequence(prot_g, info["seq"], info["domains"]) if info["seq"] else None
    if mapping:
        off = mapping["offset"]
        res_label = [f"{info['seq'][off + i]}{off + i + 1}" for i in range(n_res)]
    else:
        off = 0
        res_label = [f"{i + 1}" for i in range(n_res)]
    edges = r["edges"].copy()
    if len(edges):
        edges["Ligand atom"] = edges["atom"]
        edges["Kinase residue"] = edges["residue_idx"].map(lambda i: res_label[i])
        edges["Enrichment (×)"] = edges["enrichment"].round(2)

    fps, sim = get_train_fps(), None
    if fps:
        fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
        sim = max(DataStructs.BulkTanimotoSimilarity(fp, fps))

    # ---- who interacts with whom
    kin_title = info["gene"] + (f" · {info['name']}" if info["name"] else "")
    st.markdown(f"""
    <div class="card" style="display:flex;gap:28px;align-items:center;flex-wrap:wrap">
      <div style="flex:1;min-width:240px">
        <div style="color:{MUTED};font-size:.75rem;font-weight:700;letter-spacing:.1em">LIGAND (SMALL MOLECULE)</div>
        <div style="font-size:1.3rem;font-weight:700;color:{NAVY}">{ligand_name}</div>
        <div style="font-family:monospace;font-size:.8rem;color:#44546A;word-break:break-all">{smiles}</div>
      </div>
      <div style="font-size:2rem;color:{TEAL}">⇄</div>
      <div style="flex:1;min-width:240px">
        <div style="color:{MUTED};font-size:.75rem;font-weight:700;letter-spacing:.1em">TARGET KINASE (PROTEIN)</div>
        <div style="font-size:1.3rem;font-weight:700;color:{NAVY}">{kin_title}</div>
        <div style="color:#44546A;font-size:.9rem">UniProt {uid} · kinase domain, {n_res} residues</div>
      </div></div>""", unsafe_allow_html=True)

    # ---- numbers
    pred = r["pred"]; ic50 = 10 ** (9 - pred)
    exp = float(row["pIC50_median"]) if (row is not None and row["UniProt_ID"] == uid) else None
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted pIC50", f"{pred:.2f}")
    m2.metric("Predicted IC50", f"{ic50:,.1f} nM")
    if exp is not None:
        m3.metric("Experimental pIC50", f"{exp:.2f}", f"error {pred - exp:+.2f}", delta_color="off")
    else:
        m3.metric("Experimental pIC50", "n/a", "no measurement for this pair", delta_color="off")
    m4.metric("Similarity to training set", f"{sim:.2f}" if sim is not None else "n/a")
    if sim is not None and sim < 0.4:
        st.markdown('<span class="badge warn">Low similarity to training chemistry. Treat as extrapolation.</span>', unsafe_allow_html=True)
    if row is not None and row["UniProt_ID"] != uid:
        st.info("This ligand was measured on a different kinase, so this pair is an extrapolation with no experimental value.")

    # ---- 3D view
    st.markdown('<div class="section">3D interaction view</div>'
                '<div class="sub">Drag to rotate · scroll to zoom · right-drag to pan.</div>', unsafe_allow_html=True)
    html3d, why = None, ""
    if pdb is None:
        why = "Could not download the AlphaFold structure (no internet, or none exists for this UniProt ID)."
    elif mapping is None:
        why = (f"Could not match your {n_res}-residue protein graph to the {len(info['seq'])}-residue UniProt sequence "
               f"(kinase domains found: {info['domains'] or 'none'}).")
    elif not atom_ok:
        why = "Ligand graph atom count differs from the RDKit molecule, so atoms cannot be placed in 3D."
    elif not len(edges):
        why = "No atom–residue pairs passed the enrichment threshold for this pair."
    else:
        ca = ca_coords(pdb)
        mol3d = ligand_3d(mol)
        if mol3d is None: why = "Could not generate a 3D conformer for this ligand."
        else: html3d = build_3d_view(pdb, mol3d, edges, ca, res_label, r["symbols"], off, n_res, n_lines, spin)
    if html3d:
        components.html(html3d, height=660)
        st.markdown(f"""<div class="sub"><b>Grey-blue cartoon</b> = {info['gene']} kinase domain (AlphaFold structure) ·
        <b style="color:#E8830C">orange sticks</b> = residues with the strongest links ·
        <b style="color:#1B9E4B">green</b> = ligand ·
        <b>dashed lines</b> = model attention links (brighter and thicker = stronger).<br>
        ⚠ The ligand position is <b>schematic</b> (placed beside the pocket, not docked). Lines show what the model
        associates, not proven physical contacts.</div>""", unsafe_allow_html=True)
        st.caption(mapping["text"])
    else:
        st.warning(why + " The 2D details below still work.")

    if len(edges):
        t = edges.iloc[0]
        st.success(f"Strongest link: ligand atom **{t['Ligand atom']}** ⇄ residue **{t['Kinase residue']}** "
                   f"of {info['gene']} ({t['Enrichment (×)']}× above average attention).")

    # ---- 2D details
    with st.expander("2D details: molecule, heatmap, residue ranking, table"):
        a, b = st.columns([1, 1])
        with a:
            hl = edges.groupby("atom_idx").enrichment.max().nlargest(5).index if (atom_ok and len(edges)) else []
            st.image(draw_mol(mol, hl), caption="Numbers = atom index. Highlighted = strongest links.")
        with b:
            if len(edges):
                show_df = edges[["Ligand atom", "Kinase residue", "Enrichment (×)"]]
                st.dataframe(show_df.head(15), use_container_width=True, height=320)
                st.download_button("Download all pairs (CSV)", show_df.to_csv(index=False), "interactions.csv", "text/csv")
        cols = np.sort(np.argsort(-S.sum(0))[:40])
        fig = go.Figure(go.Heatmap(z=S[:, cols], x=[res_label[c] for c in cols], y=r["symbols"],
                                   colorscale="Viridis", colorbar=dict(title="Enrichment ×"),
                                   hovertemplate="Ligand atom %{y}<br>Residue %{x}<br>%{z:.2f}× average<extra></extra>"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=f"{info['gene']} residue", yaxis_title="Ligand atom",
                          yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(type="category"); fig.update_yaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)
        if len(r["top_residues"]):
            tr = r["top_residues"].head(10).copy()
            tr["Residue"] = tr["residue_pos"].map(lambda p: res_label[p - 1])
            bar = px.bar(tr, x="Residue", y="sum", color_discrete_sequence=[TEAL])
            bar.update_layout(height=300, yaxis_title="Summed enrichment", xaxis=dict(type="category"),
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
            st.plotly_chart(bar, use_container_width=True)


with tab_pred:
    try:
        render_predict()
    except Exception as e:
        st.error(f"The Predict tab hit an error: {e}")
        st.exception(e)

# ---------------------------------------------------------------- PERFORMANCE
def render_perf():
    st.markdown('<div class="section">Validation and test metrics</div>', unsafe_allow_html=True)
    perf = pd.DataFrame({"Validation": vm, "Test": tm}).T[["r2", "rmse", "mae", "pearson", "spearman"]].round(3)
    perf.columns = ["R²", "RMSE", "MAE", "Pearson r", "Spearman ρ"]
    st.dataframe(perf, use_container_width=True)

    st.markdown('<div class="section">Predicted vs experimental pIC50 (test set)</div>', unsafe_allow_html=True)
    s = test_df.sample(min(5000, len(test_df)), random_state=0)
    sc = px.scatter(s, x="pIC50_median", y="y_pred", opacity=.35, color_discrete_sequence=[TEAL],
                    labels={"pIC50_median": "Experimental pIC50", "y_pred": "Predicted pIC50"})
    lo, hi = float(s.pIC50_median.min()), float(s.pIC50_median.max())
    sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#AAB8CC", dash="dash"), name="y = x"))
    sc.update_layout(height=480, plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
    st.plotly_chart(sc, use_container_width=True)

    st.markdown('<div class="section">Per-kinase accuracy (test, n ≥ 30)</div>', unsafe_allow_html=True)
    rows = [{"UniProt": t, "n": len(g), "R²": r2_score(g.pIC50_median, g.y_pred),
             "MAE": mean_absolute_error(g.pIC50_median, g.y_pred),
             "Spearman": spearmanr(g.pIC50_median, g.y_pred)[0]}
            for t, g in test_df.groupby("UniProt_ID") if len(g) >= 30]
    st.dataframe(pd.DataFrame(rows).sort_values("R²", ascending=False).round(3), use_container_width=True, height=420)

    hist = f"{ROOT}/training_history_v2.csv"
    if os.path.exists(hist):
        h = pd.read_csv(hist)
        st.markdown('<div class="section">Training history</div>', unsafe_allow_html=True)
        st.plotly_chart(px.line(h, x="epoch", y=["train_mse", "val_mse"], color_discrete_sequence=["#AAB8CC", TEAL]),
                        use_container_width=True)


with tab_perf:
    try:
        render_perf()
    except Exception as e:
        st.error(f"The Performance tab hit an error: {e}")
        st.exception(e)

# ---------------------------------------------------------------- ABOUT
with tab_about:
    st.markdown('<div class="section">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
Two GATv2 encoders embed the **ligand graph** (atoms, bonds) and the **protein graph** (residues).
Bidirectional cross-attention links atoms to residues, and a pooled regression head predicts **pIC50**.
The interaction map keeps only pairs where both attention directions agree.
""")
    st.markdown('<div class="section">Limitations</div>', unsafe_allow_html=True)
    st.warning(f"""
- Valid for the **{len(kinases)} kinases seen in training only**, not unseen kinases.
- Attention highlights learned associations. **It is not proof of a physical contact or binding site.**
- Predictions for compounds dissimilar to the training set (Tanimoto < 0.4) are extrapolations.
- Intended for research prioritisation, not as a substitute for experimental assay.
""")

st.markdown('<div class="foot">RTK DualGraph · research use only</div>', unsafe_allow_html=True)
