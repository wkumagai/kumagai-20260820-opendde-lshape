"""Build SFT examples for OpenDDE from the OpenFold3 PDB training subset.

OpenDDE ships as an inference-only build: its featurizer produces `ref_pos`
(idealized CCD conformers) but never ground-truth coordinates. Supervised
fine-tuning therefore needs a label tensor laid out in the model's own atom
order, which is what this module produces.

Inputs come from the per-entry `.fasta` (one record per chain, the header being
the chain id) and `.npz` (experimental coordinates annotated with chain_id /
res_id / atom_name). The bundled `.cif` is deliberately not used: OpenFold3's
preprocessed files carry no `_entity_poly` category, so OpenDDE's mmCIF path
classifies every polymer as a ligand and atomizes it.

The chain id from the FASTA header is written into the input JSON verbatim, so
the model's atom array and the label array agree on chain naming by
construction rather than by a positional guess.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

JOIN_KEYS = ("chain_id", "res_id", "atom_name")


@dataclass
class Example:
    """One fine-tuning example: model inputs plus aligned ground truth."""

    name: str
    input_json: dict
    coordinate: np.ndarray  # [N_atom, 3] in model atom order
    coordinate_mask: np.ndarray  # [N_atom] bool, False where unresolved
    n_atom: int
    n_resolved: int
    chain_perm: np.ndarray | None  # [N_atom] int, the swapped label assignment
    composition: str  # "homo" or "hetero"


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read (chain_id, sequence) pairs, preserving file order."""
    records: list[tuple[str, str]] = []
    chain_id: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if chain_id is not None:
                records.append((chain_id, "".join(chunks)))
            chain_id, chunks = line[1:], []
        elif line:
            chunks.append(line)
    if chain_id is not None:
        records.append((chain_id, "".join(chunks)))
    return records


def build_input_json(name: str, chains: list[tuple[str, str]]) -> dict:
    """One proteinChain entity per chain, keeping the deposited chain id."""
    return {
        "name": name,
        "sequences": [
            {"proteinChain": {"sequence": seq, "count": 1, "id": [chain_id]}}
            for chain_id, seq in chains
        ],
    }


def build_chain_permutation(model_atoms, chains: list[tuple[str, str]]):
    """Index array that relabels the two chains of a homodimer, or None.

    A homodimer's two chains carry the same sequence, so the model's input
    features are symmetric under exchanging them and the model cannot know
    which of the two deposited chains it is being asked to place where. Trained
    against one fixed assignment it is punished half the time for a structure
    that is correct. AF3 answers this by permuting the ground truth to whichever
    assignment the prediction actually made (supplement Section 4.2); with two
    identical chains the permutation group has exactly two elements, so the
    exact answer is a `min` over two candidates and needs no search.

    Returned as `perm` with the convention `coordinate[perm]` = the swapped
    labelling. It is an involution, which is what lets one mask serve both
    assignments in the loss.

    None means "no permutation applies": either the entry is not a two-chain
    homodimer, or the two chains' atom layouts did not come out identical, in
    which case a positional swap would silently pair up different atoms.
    """
    if len(chains) != 2:
        return None
    (id_a, seq_a), (id_b, seq_b) = chains
    if seq_a != seq_b:
        return None

    chain_ids = np.asarray(model_atoms.chain_id)
    idx_a = np.flatnonzero(chain_ids == id_a)
    idx_b = np.flatnonzero(chain_ids == id_b)
    if len(idx_a) == 0 or len(idx_a) != len(idx_b):
        return None

    # Same sequence is not by itself proof of the same atom layout: verify it,
    # because a wrong permutation is a silently corrupted label, not a crash.
    names = np.asarray(model_atoms.atom_name)
    if not np.array_equal(names[idx_a], names[idx_b]):
        return None
    res = np.asarray(model_atoms.res_id)
    if not np.array_equal(res[idx_a] - res[idx_a][0], res[idx_b] - res[idx_b][0]):
        return None

    perm = np.arange(len(chain_ids), dtype=np.int64)
    perm[idx_a] = idx_b
    perm[idx_b] = idx_a
    return perm


def build_example(fasta_path: Path, npz_path: Path, name: str) -> Example:
    """Featurize one entry and align its experimental coordinates to model order."""
    from opendde.data.inference.json_to_feature import SampleDictToFeatures
    from opendde.data.utils import map_annotations_to_atom_indices

    chains = read_fasta(fasta_path)
    input_json = build_input_json(name, chains)
    model_atoms = SampleDictToFeatures(input_json).get_atom_array()

    gt = np.load(npz_path, allow_pickle=True)
    gt_coord = gt["coord"]
    gt_keys = list(
        zip(
            gt["chain_id"].tolist(),
            gt["res_id"].tolist(),
            gt["atom_name"].tolist(),
        )
    )
    gt_index: dict[tuple, int] = {}
    for i, key in enumerate(gt_keys):
        gt_index.setdefault(key, i)  # first occurrence wins, as for altlocs

    n_atom = len(model_atoms)
    coordinate = np.zeros((n_atom, 3), dtype=np.float32)
    coordinate_mask = np.zeros(n_atom, dtype=bool)

    model_keys = zip(
        model_atoms.chain_id.tolist(),
        model_atoms.res_id.tolist(),
        model_atoms.atom_name.tolist(),
    )
    for i, key in enumerate(model_keys):
        hit = gt_index.get(key)
        if hit is None:
            continue  # atom absent from the deposited structure
        xyz = gt_coord[hit]
        if not np.isfinite(xyz).all():
            continue  # present but unresolved: stored as NaN, not omitted
        coordinate[i] = xyz
        coordinate_mask[i] = True

    perm = build_chain_permutation(model_atoms, chains)
    return Example(
        name=name,
        input_json=input_json,
        coordinate=coordinate,
        coordinate_mask=coordinate_mask,
        n_atom=n_atom,
        n_resolved=int(coordinate_mask.sum()),
        chain_perm=perm,
        composition="homo" if len({s for _, s in chains}) < len(chains) else "hetero",
    )


def select_entries(
    data_root: Path, n_entries: int, composition: str = "any"
) -> pd.DataFrame:
    """Pick two-chain, all-standard-residue protein complexes.

    Homodimers used to be excluded here because they need a
    chain-permutation-aware loss and a naive label assignment would train the
    model toward an arbitrary chain labelling. That reasoning was right and its
    consequence was not measured: FoldBench's protein-protein split is 200/279
    homomeric interfaces, so the exclusion removed the majority of the test
    distribution from training. `build_chain_permutation` supplies the missing
    feature, so the exclusion is now a `composition` choice rather than a law.

    Non-standard residues (`X` in the FASTA) stay excluded unconditionally:
    they have no CCD template to build an atom layout from.

    `composition` is "any", "homo" or "hetero". It exists so that a
    composition-matched comparison can hold the entry count fixed and vary only
    which kind of interface is shown.
    """
    if composition not in {"any", "homo", "hetero"}:
        raise ValueError(f"unknown composition {composition!r}")
    df = pd.read_csv(data_root / "index.csv")
    candidates = df[
        (df.molecule_types == "PROTEIN") & (df.chain_count == 2)
    ].sort_values("resolution")

    structure_dir = _structure_dir(data_root)
    rows = []
    for pdb_id in candidates.id:
        fasta = structure_dir / pdb_id / f"{pdb_id}.fasta"
        npz = structure_dir / pdb_id / f"{pdb_id}.npz"
        if not (fasta.exists() and npz.exists()):
            continue
        chains = read_fasta(fasta)
        seqs = [s for _, s in chains]
        if any("X" in s for s in seqs):
            continue
        is_homo = len(set(seqs)) < len(seqs)
        if composition == "homo" and not is_homo:
            continue
        if composition == "hetero" and is_homo:
            continue
        rows.append(pdb_id)
        if len(rows) >= n_entries:
            break

    if len(rows) < n_entries:
        logger.warning(
            "requested %d %s entries, found %d", n_entries, composition, len(rows)
        )
    return candidates[candidates.id.isin(rows)]


def _structure_dir(data_root: Path) -> Path:
    return (
        data_root
        / "pdb_training_set"
        / "preprocessed_pdb_data"
        / "standard"
        / "structure_files"
    )


def run(
    data_root: str,
    cache_dir: str,
    n_entries: int,
    min_resolved_fraction: float = 0.5,
    composition: str = "any",
) -> list[str]:
    """Featurize the selected entries and cache them. Returns the cached paths."""
    data_root_path = Path(data_root)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    structure_dir = _structure_dir(data_root_path)
    entries = select_entries(data_root_path, n_entries, composition)

    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    for pdb_id in entries.id:
        out_path = cache_path / f"{pdb_id}.pt"
        if out_path.exists():
            written.append(str(out_path))
            continue

        try:
            ex = build_example(
                structure_dir / pdb_id / f"{pdb_id}.fasta",
                structure_dir / pdb_id / f"{pdb_id}.npz",
                pdb_id,
            )
        except Exception as exc:  # noqa: BLE001 - one bad entry must not stop the set
            skipped.append((pdb_id, f"{type(exc).__name__}: {exc}"))
            continue

        resolved_fraction = ex.n_resolved / max(ex.n_atom, 1)
        if resolved_fraction < min_resolved_fraction:
            skipped.append((pdb_id, f"only {resolved_fraction:.1%} atoms matched"))
            continue

        if ex.composition == "homo" and ex.chain_perm is None:
            # A homodimer whose two chains did not produce identical atom
            # layouts. Training on it without a permutation is exactly the
            # failure the old exclusion was protecting against, so it is
            # dropped rather than silently trained on.
            skipped.append((pdb_id, "homodimer with no usable chain permutation"))
            continue

        torch.save(
            {
                "name": ex.name,
                "input_json": ex.input_json,
                "coordinate": torch.from_numpy(ex.coordinate),
                "coordinate_mask": torch.from_numpy(ex.coordinate_mask),
                "n_atom": ex.n_atom,
                "n_resolved": ex.n_resolved,
                "composition": ex.composition,
                "chain_perm": (
                    None if ex.chain_perm is None
                    else torch.from_numpy(ex.chain_perm)
                ),
            },
            out_path,
        )
        written.append(str(out_path))
        logger.info(
            "%s: %d atoms, %d matched (%.1f%%), %s%s",
            pdb_id,
            ex.n_atom,
            ex.n_resolved,
            100 * resolved_fraction,
            ex.composition,
            "" if ex.chain_perm is None else " (chain swap available)",
        )

    for pdb_id, reason in skipped:
        logger.warning("skipped %s: %s", pdb_id, reason)
    logger.info("cached %d examples, skipped %d", len(written), len(skipped))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--n-entries", type=int, default=8)
    parser.add_argument(
        "--composition", default="any", choices=["any", "homo", "hetero"]
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = run(
        args.data_root, args.cache_dir, args.n_entries,
        composition=args.composition,
    )
    print(json.dumps({"cached": len(paths), "cache_dir": args.cache_dir}))


if __name__ == "__main__":
    main()
