#!/usr/bin/env python
"""Build an allosteric-site benchmark in QASC's npz format, straight from RCSB PDB.

Operational site definition (the same convention ASBench/CASBench use: sites are
defined by the residues that directly contact the crystallographic ligand):

  anchor (orthosteric / active site) = residues whose heavy atoms come within
      CONTACT A of a *cofactor* ligand (nucleotide, NAD, FAD, SAM, CoA, PLP...)
  y (allosteric site)                = residues whose heavy atoms come within
      CONTACT A of a *drug-like* ligand (>= MIN_HEAVY heavy atoms, not a
      cofactor, not a crystallisation additive) whose contact shell is at least
      DISTAL A away from every anchor residue

An entry is kept only when both sites exist, are disjoint, and the drug-like
ligand really is distal. This is a *proxy* annotation, not expert curation: it
says "a drug-like molecule binds here, far from the catalytic site", which is
the operational definition of a candidate allosteric site used by the field.
Everything is reproducible from public RCSB files; nothing is hand-entered.

Output: data/targets/<pdbid>_<chain>.npz with cb, anchor, y, resnums (QASC format).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import urllib.request
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "targets")
CACHE = os.path.join(HERE, "data", "pdb_cache")

CONTACT = 4.5      # A, ligand-residue contact shell
DISTAL = 8.0       # A, minimum separation between the two sites
MIN_HEAVY = 12     # heavy atoms for a ligand to count as drug-like
MIN_SITE = 3       # minimum residues in each site
MIN_LEN = 100      # minimum modelled residues in the chain
MAX_LEN = 700

COFACTORS = {
    "GDP", "GTP", "GNP", "GSP", "GCP", "ATP", "ADP", "AMP", "ANP", "AGS", "ACP",
    "NAD", "NAI", "NAP", "NDP", "FAD", "FMN", "SAM", "SAH", "COA", "ACO",
    "PLP", "TPP", "U5P", "5GP", "3PG", "PEP", "UDP", "UTP", "CTP", "CDP", "TTP",
    # nucleotide analogues / deoxy forms
    "MGT", "DGT", "DTP", "DAT", "DCT", "DUT", "GCP", "GSP", "ADX", "APC", "AP5",
    # porphyrins & metal-organic cofactors: prosthetic groups, not modulators
    "HEM", "HEC", "HEA", "HEB", "HDD", "DHE", "VER", "SRM", "COH", "HEV", "MHM",
    "B12", "COB", "CBY", "BCL", "CLA", "PHO", "MQ7", "MQ8", "UQ1", "UQ2",
    # sugar-phosphate substrates
    "G1P", "G6P", "F6P", "13P", "2PG", "R5P", "RIP", "S7P", "E4P",
}

# crystallisation additives / ions / cryoprotectants — never a "site"
JUNK = {
    "HOH", "DOD", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "P6G", "2PE", "SO4",
    "PO4", "ACT", "ACY", "FMT", "CIT", "FLC", "TLA", "MPD", "IPA", "DMS", "TRS",
    "MES", "EPE", "BTB", "BIS", "CAC", "IMD", "NH4", "NO3", "SCN", "AZI", "BME",
    "DTT", "TCE", "BCT", "CO3", "MLI", "OXL", "SIN", "MAE", "PYR",
    "NA", "K", "MG", "CA", "MN", "ZN", "FE", "FE2", "NI", "CU", "CU1", "CD",
    "HG", "PB", "CO", "CS", "RB", "SR", "BA", "LI", "AL", "CL", "BR", "IOD",
    "F", "IUM", "PT", "AU", "AG", "W", "MO", "V", "SE", "XE", "KR", "AR",
    "UNX", "UNL", "PLM", "MYR", "OLA", "OLC", "LDA", "LMT", "C8E", "BOG",
    # PEG / polyol cryoprotectants
    "PE4", "PE5", "PE8", "XPE", "7PE", "12P", "15P", "P33", "PG0", "PG5", "PGF",
    "M2M", "MRD", "BU3", "PGO", "DIO", "HEZ", "TMA", "EOH", "ACN",
    # glycosylation / polysaccharide fragments — not a drug site
    "NAG", "NDG", "BMA", "MAN", "GAL", "GLC", "FUC", "XYP", "XYS", "SIA",
    "BGC", "A2G", "RAM", "NGA", "GLA", "SUC", "TRE", "MAL",
    # extra buffers
    "MPO", "PIN", "TAR", "TBR", "CHES", "BES", "ADA", "MLA", "MLT", "SPD",
    "SPM", "PUT", "TFA", "TRT", "BNG", "HTG", "OGA", "DDQ",
}

DRUGLIKE_ELEMS = {"N", "S", "F", "CL", "BR", "I"}

BACKBONE = {"N", "CA", "C", "O", "OXT"}
AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


def uniprot_of(pdb_id: str):
    """UniProt accessions of an entry (PDBe SIFTS); empty set on failure."""
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qasc-plus/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        return set(d.get(pdb_id.lower(), {}).get("UniProt", {}).keys())
    except Exception:
        return set()


def fetch(pdb_id: str) -> str | None:
    """Download (and cache) a gzipped PDB file; return its text or None."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{pdb_id}.pdb.gz")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "qasc-plus/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
        except Exception:
            return None
    try:
        with gzip.open(path, "rt", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def parse(text: str):
    """Return (residues_by_chain, ligands) from PDB text, first model only.

    residues_by_chain[chain] = list of (resnum_key, resname, {atom: xyz})
    ligands = list of (chain, resname, resnum_key, np.ndarray heavy coords)
    """
    residues = defaultdict(dict)
    order = defaultdict(list)
    ligands = defaultdict(lambda: defaultdict(list))
    for line in text.splitlines():
        rec = line[:6]
        if rec == "ENDMDL":
            break
        if rec not in ("ATOM  ", "HETATM"):
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        resname = line[17:20].strip()
        chain = line[21].strip() or "A"
        rkey = line[22:27].strip()          # resSeq + iCode
        atom = line[12:16].strip()
        elem = (line[76:78].strip() or atom[:1]).upper()
        if elem == "H" or elem == "D":
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if resname in AA3 and rec == "ATOM  " or (resname == "MSE" and rec == "HETATM"):
            key = (chain, rkey)
            if key not in residues[chain]:
                residues[chain][key] = [resname, {}]
                order[chain].append(key)
            residues[chain][key][1][atom] = xyz
        elif rec == "HETATM" and resname not in JUNK and resname not in AA3:
            ligands[(chain, resname, rkey)]["xyz"].append(xyz)
            ligands[(chain, resname, rkey)]["elem"].append(elem)
    return residues, order, ligands


def chain_arrays(residues, order, chain):
    """Per-residue Cbeta (Calpha for Gly / missing CB), heavy atoms, resnums."""
    cb, heavy, resnums = [], [], []
    for key in order[chain]:
        resname, atoms = residues[chain][key]
        pos = atoms.get("CB") or atoms.get("CA")
        if pos is None:
            continue
        cb.append(pos)
        heavy.append(np.array([v for a, v in atoms.items()], dtype=float))
        try:
            resnums.append(int("".join(c for c in key[1] if c.isdigit() or c == "-")))
        except ValueError:
            resnums.append(len(resnums) + 1)
    if not cb:
        return None
    return np.asarray(cb, float), heavy, np.asarray(resnums, int)


def contacts(heavy_list, lig_xyz, cutoff=CONTACT):
    """Indices of residues with any heavy atom within cutoff of the ligand."""
    lig = np.asarray(lig_xyz, dtype=float)
    hits = []
    for i, h in enumerate(heavy_list):
        d2 = ((h[:, None, :] - lig[None, :, :]) ** 2).sum(-1)
        if d2.min() <= cutoff * cutoff:
            hits.append(i)
    return np.asarray(hits, dtype=int)


def build_one(pdb_id: str):
    text = fetch(pdb_id)
    if not text:
        return None
    residues, order, ligands = parse(text)
    if not residues:
        return None

    # group ligands by chain
    by_chain = defaultdict(list)
    for (chain, resname, rkey), d in ligands.items():
        xyz = np.asarray(d["xyz"], float)
        if len(xyz) == 0:
            continue
        by_chain[chain].append((resname, rkey, xyz, set(d["elem"])))

    best = None
    for chain in residues:
        arr = chain_arrays(residues, order, chain)
        if arr is None:
            continue
        cb, heavy, resnums = arr
        n = len(cb)
        if not (MIN_LEN <= n <= MAX_LEN):
            continue
        ligs = by_chain.get(chain, [])
        cofs = [l for l in ligs if l[0] in COFACTORS]
        drugs = [l for l in ligs
                 if l[0] not in COFACTORS and len(l[2]) >= MIN_HEAVY
                 and (l[3] & DRUGLIKE_ELEMS)]      # PEG/glycerol are C/O only
        if not cofs or not drugs:
            continue

        # anchor = union of contacts of all cofactor copies
        anchor = np.unique(np.concatenate([contacts(heavy, l[2]) for l in cofs]))
        if len(anchor) < MIN_SITE:
            continue

        for resname, rkey, xyz, _elems in drugs:
            y_idx = contacts(heavy, xyz)
            if len(y_idx) < MIN_SITE:
                continue
            if np.intersect1d(y_idx, anchor).size:
                continue                       # overlaps the active site
            from scipy.spatial.distance import cdist
            sep = cdist(cb[y_idx], cb[anchor]).min()
            if sep < DISTAL:
                continue                       # not distal -> not allosteric
            y = np.zeros(n, dtype=int)
            y[y_idx] = 1
            cand = dict(pdb=pdb_id, chain=chain, ligand=resname, n=n,
                        cb=cb, anchor=anchor, y=y, resnums=resnums,
                        n_anchor=len(anchor), n_pos=int(y.sum()), sep=float(sep))
            if best is None or cand["sep"] > best["sep"]:
                best = cand
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    ids = json.load(open(os.path.join(HERE, "data", "candidate_ids.json")))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    kept, tried, seen_unp = [], 0, set()
    for pdb_id in ids[:limit]:
        tried += 1
        try:
            r = build_one(pdb_id)
        except Exception:
            r = None
        if r is None:
            continue
        unp = uniprot_of(pdb_id)
        if unp & seen_unp:
            continue                      # same protein family already represented
        seen_unp |= unp
        r["uniprot"] = sorted(unp)
        path = os.path.join(OUT, f"{r['pdb']}_{r['chain']}.npz")
        np.savez_compressed(path, cb=r["cb"], anchor=r["anchor"], y=r["y"],
                            resnums=r["resnums"])
        kept.append({k: r[k] for k in ("pdb", "chain", "ligand", "n", "n_anchor",
                                       "n_pos", "sep", "uniprot")})
        print(f"[{len(kept):3d}/{tried}] {r['pdb']}_{r['chain']} lig={r['ligand']:>4s} "
              f"N={r['n']:4d} anchor={r['n_anchor']:3d} pos={r['n_pos']:3d} "
              f"sep={r['sep']:.1f}A", flush=True)
        if len(kept) >= want:
            break
    json.dump(kept, open(os.path.join(HERE, "data", "manifest.json"), "w"), indent=1)
    print(f"\nkept {len(kept)} targets from {tried} candidates -> {OUT}")


if __name__ == "__main__":
    main()
