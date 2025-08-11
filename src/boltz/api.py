"""In-memory Python API for Boltz models.

This module exposes a small, well documented wrapper around the core
:mod:`boltz` models.  It mirrors the command line interface used by the
project but operates purely on Python objects and does not perform any
file system interaction.  The API is intentionally thin so that advanced
users can interact directly with the underlying :class:`torch.nn.Module`
without dealing with command line parsing or on-disk intermediate
representations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Literal

import numpy as np
import torch
from rdkit.Chem.rdchem import Mol

from boltz.data import const
from boltz.data.module.inferencev2 import collate
from boltz.data.parse.a3m import _parse_a3m
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import (
    Coords,
    Ensemble,
    Input,
    Interface,
    MSA,
    Record,
    StructureV2,
    Target,
)
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.model.models.boltz2 import Boltz2


# ---------------------------------------------------------------------------
# High level data containers
# ---------------------------------------------------------------------------


@dataclass
class ChainSpec:
    """Specification for an input chain.

    This mirrors the entries found in the YAML based interface.  Each chain
    is defined by its ``kind`` and an identifier.  For protein, DNA and RNA
    chains a primary amino/nucleotide ``sequence`` must be provided.  Ligands
    can be specified either via a SMILES string or a CCD identifier.
    """

    kind: Literal["protein", "dna", "rna", "ligand"]
    id: str | list[str]
    sequence: str | None = None
    smiles: str | None = None
    ccd: str | list[str] | None = None


@dataclass
class BindingResult:
    """Binding affinity prediction for a complex.

    Attributes
    ----------
    value:
        Predicted affinity value (for example the log dissociation constant).
    probability:
        Model estimated probability that the complex binds.
    value1, value2, probability1, probability2:
        Optional ensemble components when the model was trained with an
        affinity ensemble head.
    """

    value: float
    probability: float
    value1: float | None = None
    probability1: float | None = None
    value2: float | None = None
    probability2: float | None = None


@dataclass
class PredictionResult:
    """Container holding the result of a prediction."""

    record: Record
    structure: StructureV2
    raw: dict[str, np.ndarray | Any]

    def to_pdb(self) -> str:
        """Return the prediction as a PDB formatted string."""

        return to_pdb(self.structure, plddts=self.raw.get("plddt"))

    def to_mmcif(self) -> str:
        """Return the prediction as an mmCIF formatted string."""

        return to_mmcif(self.structure, plddts=self.raw.get("plddt"))

    @property
    def affinity(self) -> BindingResult | None:
        """Return binding affinity predictions if present.

        The returned :class:`BindingResult` exposes the aggregated affinity
        value and probability.  When the model was configured with an affinity
        ensemble, individual head predictions are also populated.
        """

        if "affinity_pred_value" not in self.raw:
            return None

        def _maybe(key: str) -> float | None:
            value = self.raw.get(key)
            if value is None:
                return None
            return float(np.array(value).squeeze())

        return BindingResult(
            value=float(np.array(self.raw["affinity_pred_value"]).squeeze()),
            probability=float(
                np.array(self.raw["affinity_probability_binary"]).squeeze()
            ),
            value1=_maybe("affinity_pred_value1"),
            probability1=_maybe("affinity_probability_binary1"),
            value2=_maybe("affinity_pred_value2"),
            probability2=_maybe("affinity_probability_binary2"),
        )


def msa_from_a3m(text: str, *, max_seqs: int | None = None) -> MSA:
    """Parse an A3M formatted alignment held in memory."""

    return _parse_a3m(StringIO(text), taxonomy=None, max_seqs=max_seqs)


def build_target(
    name: str,
    chains: Sequence[ChainSpec],
    *,
    canonicals: Mapping[str, Mol],
    mol_dir: Path | None = None,
    boltz2: bool = True,
    affinity_binder: str | None = None,
) -> Target:
    """Create a :class:`~boltz.data.types.Target` from chain specifications.

    Parameters
    ----------
    name:
        Identifier for the construct.
    chains:
        Sequence of chain specifications.
    canonicals:
        Mapping of reference molecules (typically the CCD dictionary).
    mol_dir:
        Optional directory containing additional ligands referenced by CCD
        identifiers.
    boltz2:
        Whether the :class:`StructureV2` layout should be used.
    affinity_binder:
        Optional identifier of a ligand chain for which a binding affinity
        prediction should be made.
    """

    schema: dict[str, Any] = {"version": 1, "sequences": []}
    for chain in chains:
        entry: dict[str, Any] = {}
        data: dict[str, Any] = {"id": chain.id}
        if chain.kind == "ligand":
            if chain.smiles is not None:
                data["smiles"] = chain.smiles
            elif chain.ccd is not None:
                data["ccd"] = chain.ccd
            else:
                msg = "Ligand chain requires either a SMILES string or CCD id."
                raise ValueError(msg)
        else:
            if chain.sequence is None:
                msg = f"Chain {chain.id} of kind {chain.kind} requires a sequence"
                raise ValueError(msg)
            data["sequence"] = chain.sequence
        entry[chain.kind] = data
        schema["sequences"].append(entry)

    if affinity_binder is not None:
        schema["properties"] = [{"affinity": {"binder": affinity_binder}}]

    return parse_boltz_schema(name, schema, canonicals, mol_dir, boltz2)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PairformerArgs:
    """Configuration for the Pairformer trunk.

    The fields mirror the command line options but are kept minimal in
    order to avoid a dependency on the CLI module.
    """

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """Configuration for the MSA module."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class DiffusionParams:
    """Parameters for the diffusion process used during sampling."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7.0
    step_scale: float = 1.5
    sigma_min: float = 1e-4
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class SteeringParams:
    """Parameters controlling optional FK steering."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20


# ---------------------------------------------------------------------------
# High level model interface
# ---------------------------------------------------------------------------


class BoltzModel:
    """Light‑weight wrapper around the :class:`Boltz2` model.

    The class exposes a minimal API for running structure (and optional
    affinity) predictions without relying on the command line interface
    or touching the file system.  All inputs are expected to already be
    loaded into memory as :class:`~boltz.data.types.Target` and
    :class:`~boltz.data.types.MSA` objects.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        canonicals: Mapping[str, Mol],
        device: str | torch.device | None = None,
        pairformer_args: PairformerArgs | None = None,
        msa_args: MSAModuleArgs | None = None,
        diffusion_params: DiffusionParams | None = None,
        steering_params: SteeringParams | None = None,
        recycling_steps: int = 3,
        sampling_steps: int = 30,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        affinity: bool = False,
        affinity_ensemble: bool = False,
        affinity_mw_correction: bool = True,
        use_kernels: bool = True,
    ) -> None:
        """Load a pretrained Boltz model.

        Parameters
        ----------
        checkpoint:
            Path to a ``.ckpt`` file produced during training.
        canonicals:
            Mapping from residue names to reference RDKit molecules.  These
            are used by the featurizer and must contain entries for all
            molecules that appear in the input structures.
        device:
            Device on which the model should run.  Defaults to ``"cpu"`` or
            ``"cuda"`` if available.
        pairformer_args, msa_args, diffusion_params, steering_params:
            Optional configuration objects.  When ``None`` the default
            configuration values are used.
        recycling_steps, sampling_steps, diffusion_samples, max_parallel_samples:
            Parameters controlling the sampling behaviour during
            prediction.  They correspond to the respective command line
            options.
        affinity:
            Whether the loaded checkpoint includes a binding affinity head.
            When ``True`` the returned :class:`PredictionResult` objects expose
            affinity scores via their :pyattr:`~PredictionResult.affinity`
            property.
        affinity_ensemble:
            Use the ensemble variant of the affinity head.  Only relevant when
            ``affinity`` is ``True``.
        affinity_mw_correction:
            Apply the molecular-weight correction when predicting affinity.
        use_kernels:
            Whether to use the custom CUDA kernels if available.
        """

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.canonicals = dict(canonicals)

        pairformer = asdict(pairformer_args or PairformerArgs())
        msa = asdict(msa_args or MSAModuleArgs())
        diffusion = asdict(diffusion_params or DiffusionParams())
        steering = asdict(steering_params or SteeringParams())

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            # ``predict_step`` expects these fields even if unused.
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        self.model = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            map_location=self.device,
            predict_args=predict_args,
            diffusion_process_args=diffusion,
            pairformer_args=pairformer,
            msa_args=msa,
            steering_args=steering,
            ema=False,
            affinity_prediction=affinity,
            affinity_ensemble=affinity_ensemble,
            affinity_mw_correction=affinity_mw_correction,
            use_kernels=use_kernels,
        )
        self.model.eval()

        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        target: Target,
        msas: Mapping[str, MSA],
        *,
        seed: int = 42,
    ) -> dict[str, torch.Tensor | Any]:
        """Tokenize and featurize a target for prediction.

        Parameters
        ----------
        target:
            The target structure and associated metadata.
        msas:
            Mapping from chain identifiers to MSA objects.  Chains without
            an entry will be run in single sequence mode.
        seed:
            Random seed used by the featurizer.  Using a fixed seed ensures
            deterministic feature generation.

        Returns
        -------
        dict
            Feature dictionary ready to be consumed by the model.
        """

        input_data = Input(
            structure=target.structure,
            msa=dict(msas),
            record=target.record,
            residue_constraints=target.residue_constraints,
            templates=target.templates,
            extra_mols=target.extra_mols,
        )

        tokenized = self.tokenizer.tokenize(input_data)

        molecules: dict[str, Mol] = {}
        molecules.update(self.canonicals)
        molecules.update(input_data.extra_mols)

        random = np.random.default_rng(seed)
        options = target.record.inference_options
        if options is None:
            pocket_constraints, contact_constraints = None, None
        else:
            pocket_constraints = options.pocket_constraints
            contact_constraints = options.contact_constraints

        features = self.featurizer.process(
            tokenized,
            molecules=molecules,
            random=random,
            training=False,
            max_atoms=None,
            max_tokens=None,
            max_seqs=const.max_msa_seqs,
            pad_to_max_seqs=False,
            single_sequence_prop=0.0,
            compute_frames=True,
            inference_pocket_constraints=pocket_constraints,
            inference_contact_constraints=contact_constraints,
            compute_constraint_features=True,
            compute_affinity=self.model.affinity_prediction,
        )
        features["record"] = target.record

        batch = collate([features])
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return batch

    def _to_structure(
        self,
        target: Target,
        coords: np.ndarray,
        pad_mask: np.ndarray,
        plddt: np.ndarray | None = None,
    ) -> StructureV2:
        """Construct a :class:`StructureV2` from raw model outputs."""

        structure: StructureV2 = target.structure  # type: ignore[assignment]
        structure = structure.remove_invalid_chains()

        atoms = structure.atoms.copy()
        residues = structure.residues.copy()
        valid = pad_mask.astype(bool)
        atoms["coords"] = coords[valid]
        atoms["is_present"] = True
        if plddt is not None and "plddt" in atoms.dtype.names:
            atoms["plddt"] = plddt[valid]
        residues["is_present"] = True

        coord_arr = np.array([(x,) for x in atoms["coords"]], dtype=Coords)
        ensemble = np.array([(0, len(coord_arr))], dtype=Ensemble)
        new_structure = StructureV2(
            atoms=atoms,
            bonds=structure.bonds,
            residues=residues,
            chains=structure.chains,
            interfaces=np.array([], dtype=Interface),
            mask=np.ones_like(structure.mask, dtype=bool),
            coords=coord_arr,
            ensemble=ensemble,
            pocket=structure.pocket,
        )
        return new_structure

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def predict(
        self,
        target: Target,
        msas: Mapping[str, MSA],
        *,
        seed: int = 42,
    ) -> PredictionResult:
        """Run the model on the given target.

        Parameters
        ----------
        target:
            Input target produced by :func:`boltz.data.parse.schema.parse_boltz_schema`
            or similar utilities.
        msas:
            Mapping from chain identifiers to multiple sequence alignments
            to use during prediction.
        seed:
            Random seed controlling stochastic parts of the featurisation
            pipeline.

        Returns
        -------
        PredictionResult
            A container with the predicted structure, optional affinity
            predictions and raw output arrays.
        """

        batch = self._prepare_batch(target, msas, seed=seed)
        with torch.no_grad():
            outputs = self.model.predict_step(batch, 0)

        raw: dict[str, Any] = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                raw[key] = value.detach().cpu().numpy()
            else:
                raw[key] = value

        coords = raw["coords"][0]
        mask = raw["masks"][0] if raw["masks"].ndim > 1 else raw["masks"]
        plddt = raw.get("plddt")
        if isinstance(plddt, np.ndarray):
            plddt = plddt[0]
        structure = self._to_structure(target, coords, mask, plddt)
        return PredictionResult(record=target.record, structure=structure, raw=raw)

    def predict_from_chains(
        self,
        name: str,
        chains: Sequence[ChainSpec],
        *,
        msas: Mapping[str, MSA] | None = None,
        seed: int = 42,
        mol_dir: Path | None = None,
        affinity_binder: str | None = None,
    ) -> PredictionResult:
        """High level convenience wrapper around :meth:`predict`.

        This function accepts :class:`ChainSpec` objects describing the input
        complex, builds the corresponding :class:`Target` and runs a prediction.
        When ``affinity_binder`` is provided and the model was initialised with
        affinity prediction enabled, the returned
        :class:`PredictionResult.affinity` field will contain the binding
        affinity estimate for the specified ligand chain.
        """

        target = build_target(
            name,
            chains,
            canonicals=self.canonicals,
            mol_dir=mol_dir,
            boltz2=True,
            affinity_binder=affinity_binder,
        )
        return self.predict(target, msas or {}, seed=seed)

    def predict_affinity(
        self,
        target: Target,
        msas: Mapping[str, MSA],
        *,
        seed: int = 42,
    ) -> BindingResult:
        """Predict only the binding affinity for ``target``.

        This is a thin wrapper around :meth:`predict` that returns the
        :class:`BindingResult` directly.  The model must have been initialised
        with ``affinity=True``.

        Raises
        ------
        RuntimeError
            If the underlying model does not support affinity prediction.
        """

        if not self.model.affinity_prediction:
            msg = "Model was not initialised with affinity prediction enabled."
            raise RuntimeError(msg)

        result = self.predict(target, msas, seed=seed)
        affinity = result.affinity
        if affinity is None:
            raise RuntimeError("Affinity prediction not found in model output.")
        return affinity


__all__ = [
    "BoltzModel",
    "PairformerArgs",
    "MSAModuleArgs",
    "DiffusionParams",
    "SteeringParams",
    "ChainSpec",
    "BindingResult",
    "PredictionResult",
    "msa_from_a3m",
    "build_target",
]
