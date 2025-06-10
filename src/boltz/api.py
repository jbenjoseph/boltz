"""High level Python API for Boltz."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import gemmi
import torch
import numpy as np

import yaml
from pydantic import BaseModel, Field, model_validator

from boltz.main import predict as _cli_predict
from boltz.data.write.pdb import to_pdb
from boltz.data.write.mmcif import to_mmcif
from boltz.data.types import Coords, Interface, Record, Structure, StructureV2
from pytorch_lightning.callbacks import BasePredictionWriter


###############################################################################
# Pydantic models mirroring the YAML schema used by the CLI
###############################################################################


class Modification(BaseModel):
    """Residue modification."""

    position: int
    ccd: str


class Protein(BaseModel):
    """Protein chain specification."""

    id: Union[str, List[str]]
    sequence: str
    msa: Optional[str] = None
    modifications: Optional[List[Modification]] = None
    cyclic: bool = False


class DNA(BaseModel):
    """DNA chain specification."""

    id: Union[str, List[str]]
    sequence: str
    msa: Optional[str] = None
    modifications: Optional[List[Modification]] = None
    cyclic: bool = False


class RNA(BaseModel):
    """RNA chain specification."""

    id: Union[str, List[str]]
    sequence: str
    msa: Optional[str] = None
    modifications: Optional[List[Modification]] = None
    cyclic: bool = False


class Ligand(BaseModel):
    """Ligand specification."""

    id: Union[str, List[str]]
    smiles: Optional[str] = None
    ccd: Optional[str] = None

    @model_validator(mode="after")
    def _check_smiles_or_ccd(cls, values: "Ligand") -> "Ligand":  # noqa: D401
        if not values.smiles and not values.ccd:
            raise ValueError("One of `smiles` or `ccd` must be provided")
        if values.smiles and values.ccd:
            raise ValueError("`smiles` and `ccd` are mutually exclusive")
        return values


class SequenceItem(BaseModel):
    """Wrapper for a sequence entry."""

    protein: Optional[Protein] = None
    dna: Optional[DNA] = None
    rna: Optional[RNA] = None
    ligand: Optional[Ligand] = None

    @model_validator(mode="after")
    def _only_one(cls, values: "SequenceItem") -> "SequenceItem":  # noqa: D401
        filled = [
            values.protein,
            values.dna,
            values.rna,
            values.ligand,
        ]
        if sum(v is not None for v in filled) != 1:
            raise ValueError(
                "Exactly one of protein/dna/rna/ligand must be set",
            )
        return values


class BondConstraint(BaseModel):
    atom1: tuple[str, int, str]
    atom2: tuple[str, int, str]


class PocketConstraint(BaseModel):
    binder: str
    contacts: List[tuple[str, Union[int, str]]]
    max_distance: Optional[float] = Field(default=None, alias="max_distance")


class ContactConstraint(BaseModel):
    token1: tuple[str, Union[int, str]]
    token2: tuple[str, Union[int, str]]
    max_distance: Optional[float] = Field(default=None, alias="max_distance")


class Constraint(BaseModel):
    bond: Optional[BondConstraint] = None
    pocket: Optional[PocketConstraint] = None
    contact: Optional[ContactConstraint] = None

    @model_validator(mode="after")
    def _only_one(cls, values: "Constraint") -> "Constraint":  # noqa: D401
        filled = [values.bond, values.pocket, values.contact]
        if sum(v is not None for v in filled) != 1:
            raise ValueError("Exactly one of bond/pocket/contact must be set")
        return values


class Template(BaseModel):
    cif: str
    chain_id: Optional[Union[str, List[str]]] = None
    template_id: Optional[Union[str, List[str]]] = Field(default=None, alias="template_id")


class Affinity(BaseModel):
    binder: str


class Property(BaseModel):
    affinity: Optional[Affinity] = None


class BoltzInput(BaseModel):
    """Top level input schema."""

    sequences: List[SequenceItem]
    constraints: Optional[List[Constraint]] = None
    templates: Optional[List[Template]] = None
    properties: Optional[List[Property]] = None
    version: int = 1

    def to_yaml(self) -> str:
        """Dump the object to a YAML string."""

        data = json.loads(self.json(exclude_none=True))
        return yaml.safe_dump(data, sort_keys=False)


class BoltzOptions(BaseModel):
    """Configuration options for :func:`predict`."""

    output_format: str = "mmcif"
    cache: Path = Path("~/.boltz")
    checkpoint: Optional[str] = None
    affinity_checkpoint: Optional[str] = None
    devices: int = 1
    accelerator: str = "gpu"
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    sampling_steps_affinity: int = 200
    diffusion_samples_affinity: int = 3
    max_parallel_samples: Optional[int] = None
    step_scale: Optional[float] = None
    write_full_pae: bool = False
    write_full_pde: bool = False
    num_workers: int = 2
    override: bool = False
    seed: Optional[int] = None
    use_msa_server: bool = False
    msa_server_url: str = "https://api.colabfold.com"
    msa_pairing_strategy: str = "greedy"
    use_potentials: bool = False
    model: str = "boltz2"
    method: Optional[str] = None
    affinity_mw_correction: Optional[bool] = False
    preprocessing_threads: int = 1
    max_msa_seqs: int = 8192
    subsample_msa: bool = True
    num_subsampled_msa: int = 1024
    no_trifast: bool = False

    def to_cli_args(self) -> Dict[str, Any]:
        data = self.dict()
        data["cache"] = str(Path(data["cache"]).expanduser())
        return data


###############################################################################
# In-memory prediction writers
###############################################################################

_MEM_WRITER: Optional["_MemoryWriter"] = None
_MEM_AFF_WRITER: Optional["_MemoryAffinityWriter"] = None


class _MemoryWriter(BasePredictionWriter):
    """Collect structures and confidence data in memory."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: str = "mmcif",
        boltz2: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            raise ValueError(f"Invalid output format: {output_format}")

        self.data_dir = Path(data_dir)
        self.output_format = output_format
        self.boltz2 = boltz2
        self.structures: List[str] = []
        self.coords: List[torch.Tensor] = []
        self.confidence: Dict[str, Any] = {}

        global _MEM_WRITER
        _MEM_WRITER = self

    def write_on_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        prediction: Dict[str, torch.Tensor],
        batch_indices: List[int],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if prediction["exception"]:
            return

        records: List[Record] = batch["record"]
        coords = prediction["coords"].unsqueeze(0)
        pad_masks = prediction["masks"]

        if "confidence_score" in prediction:
            argsort = torch.argsort(prediction["confidence_score"], descending=True)
            idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
        else:
            idx_to_rank = {i: i for i in range(len(records))}

        for record, coord, pad_mask in zip(records, coords, pad_masks):
            path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                structure = StructureV2.load(path)
            else:
                structure = Structure.load(path)

            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                model_coord = coord[model_idx]
                coord_unpad = model_coord[pad_mask.bool()]
                coord_np = coord_unpad.cpu().numpy()

                atoms = structure.atoms
                atoms["coords"] = coord_np
                atoms["is_present"] = True

                if self.boltz2:
                    coord_np = [(x,) for x in coord_np]
                    coord_np = np.array(coord_np, dtype=Coords)

                residues = structure.residues
                residues["is_present"] = True
                interfaces = np.array([], dtype=Interface)

                if self.boltz2:
                    new_structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_np,
                    )
                else:
                    new_structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                plddts = prediction.get("plddt")
                plddt = plddts[model_idx] if plddts is not None else None

                if self.output_format == "pdb":
                    struct_str = to_pdb(new_structure, plddts=plddt, boltz2=self.boltz2)
                else:
                    struct_str = to_mmcif(new_structure, plddts=plddt, boltz2=self.boltz2)

                self.structures.append(struct_str)
                self.coords.append(torch.tensor(coord_np, dtype=torch.float32))

                if model_idx == 0 and "plddt" in prediction:
                    conf = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        conf[key] = prediction[key][model_idx].item()
                    conf["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    conf["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][model_idx].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    self.confidence[record.id] = conf


class _MemoryAffinityWriter(BasePredictionWriter):
    """Collect affinity predictions in memory."""

    def __init__(self, data_dir: str, output_dir: str) -> None:
        super().__init__(write_interval="batch")
        self.affinity: Dict[str, Any] = {}
        global _MEM_AFF_WRITER
        _MEM_AFF_WRITER = self

    def write_on_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        prediction: Dict[str, torch.Tensor],
        batch_indices: List[int],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if prediction["exception"]:
            return
        record = batch["record"][0]
        summary = {
            "affinity_pred_value": prediction["affinity_pred_value"].item(),
            "affinity_probability_binary": prediction["affinity_probability_binary"].item(),
        }
        if "affinity_pred_value1" in prediction:
            summary["affinity_pred_value1"] = prediction["affinity_pred_value1"].item()
            summary["affinity_probability_binary1"] = prediction[
                "affinity_probability_binary1"
            ].item()
            summary["affinity_pred_value2"] = prediction["affinity_pred_value2"].item()
            summary["affinity_probability_binary2"] = prediction[
                "affinity_probability_binary2"
            ].item()
        self.affinity[record.id] = summary


class BoltzPredictor:
    """Helper class to run Boltz predictions programmatically."""

    def __init__(self, options: Optional[BoltzOptions] = None, **kwargs: Any) -> None:
        self.options = BoltzOptions() if options is None else options
        if kwargs:
            self.options = self.options.model_copy(update=kwargs)

    def predict(
        self,
        inp: BoltzInput,
        *,
        out_dir: Optional[Union[str, Path]] = None,
        keep_files: bool = False,
        **kwargs: Any,
    ) -> PredictionResult:
        opts = self.options.model_copy(update=kwargs)
        return predict(
            inp,
            out_dir=out_dir,
            options=opts,
            keep_files=keep_files,
        )


@dataclass
class PredictionResult:
    """Output from :func:`predict`.

    The ``structures`` field contains the generated structure files as strings
    (either PDB or CIF depending on the chosen output format).  Confidence and
    affinity metadata are returned as parsed dictionaries.
    """

    structures: List[str]
    confidence: Dict[str, Any]
    affinity: Optional[Dict[str, Any]] = None
    coords: Optional[List[torch.Tensor]] = None


###############################################################################
# High level prediction helper
###############################################################################


def predict(
    inp: BoltzInput,
    out_dir: Optional[Union[str, Path]] = None,
    *,
    options: Optional[BoltzOptions] = None,
    keep_files: bool = False,
    **kwargs: Any,
) -> PredictionResult:
    """Run Boltz prediction for a given :class:`BoltzInput`.

    Parameters
    ----------
    inp:
        The input description.
    out_dir:
        Optional directory where predictions will be written.  If ``None``, a
        temporary directory is used and cleaned up automatically.
    options:
        Prediction options. If ``None`` the default :class:`BoltzOptions` will
        be used.
    keep_files:
        If ``True`` and ``out_dir`` is ``None`` the temporary output directory
        is not removed.  Ignored when ``out_dir`` is provided.
    **kwargs:
        Additional keyword arguments override the values in ``options`` and are
        forwarded to :func:`boltz.main.predict`.

    Returns
    -------
    PredictionResult
        Object describing the prediction outputs.
    """

    opts = BoltzOptions() if options is None else options
    cli_args = opts.to_cli_args()
    cli_args.update(kwargs)

    if out_dir is None:
        tmp_out = tempfile.TemporaryDirectory()
        out_path = Path(tmp_out.name)
    else:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        tmp_out = None

    with tempfile.TemporaryDirectory() as tmp_in:
        tmp_path = Path(tmp_in) / "input.yaml"
        tmp_path.write_text(inp.to_yaml())

        use_memory = out_dir is None and not keep_files
        if use_memory:
            import boltz.main as _main_mod
            from boltz.data.write import writer as _writer_mod

            old_writer = _main_mod.BoltzWriter
            old_aff = _main_mod.BoltzAffinityWriter
            old_w_writer = _writer_mod.BoltzWriter
            old_w_aff = _writer_mod.BoltzAffinityWriter
            _main_mod.BoltzWriter = _MemoryWriter
            _main_mod.BoltzAffinityWriter = _MemoryAffinityWriter
            _writer_mod.BoltzWriter = _MemoryWriter
            _writer_mod.BoltzAffinityWriter = _MemoryAffinityWriter

            try:
                _cli_predict(
                    data=str(tmp_path),
                    out_dir=str(out_path),
                    **cli_args,
                )
            finally:
                _main_mod.BoltzWriter = old_writer
                _main_mod.BoltzAffinityWriter = old_aff
                _writer_mod.BoltzWriter = old_w_writer
                _writer_mod.BoltzAffinityWriter = old_w_aff

            results_dir = None
        else:
            _cli_predict(
                data=str(tmp_path),
                out_dir=str(out_path),
                **cli_args,
            )

            results_dir = out_path / f"boltz_results_{tmp_path.stem}" / "predictions" / tmp_path.stem

    if use_memory:
        mem_writer = _MEM_WRITER
        mem_aff = _MEM_AFF_WRITER
        structures = mem_writer.structures if mem_writer else []
        coords = mem_writer.coords if mem_writer else []
        conf_data = (
            list(mem_writer.confidence.values())[0] if mem_writer and mem_writer.confidence else {}
        )
        affinity_data = (
            list(mem_aff.affinity.values())[0] if mem_aff and mem_aff.affinity else None
        )
    else:
        struct_ext = cli_args.get("output_format", "mmcif")
        struct_paths = sorted(
            results_dir.glob(f"{tmp_path.stem}_model_*.{struct_ext}")
        )
        structures = [p.read_text() for p in struct_paths]
        coords: List[torch.Tensor] = []
        for p in struct_paths:
            text = p.read_text()
            if struct_ext == "pdb":
                s = gemmi.read_pdb_string(text)
            else:
                doc = gemmi.cif.read_string(text)
                s = gemmi.make_structure_from_block(doc.sole_block())
            c_list = [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for m in s
                for ch in m
                for res in ch
                for atom in res
            ]
            coords.append(torch.tensor(c_list, dtype=torch.float32))

        conf_file = results_dir / f"confidence_{tmp_path.stem}_model_0.json"
        conf_data: Dict[str, Any] = {}
        if conf_file.exists():
            conf_data = json.loads(conf_file.read_text())

        affinity_file = results_dir / f"affinity_{tmp_path.stem}.json"
        affinity_data = (
            json.loads(affinity_file.read_text()) if affinity_file.exists() else None
        )

    result = PredictionResult(
        structures=structures,
        confidence=conf_data,
        affinity=affinity_data,
        coords=coords,
    )

    if tmp_out is not None and not keep_files:
        tmp_out.cleanup()

    return result


__all__ = [
    "BoltzInput",
    "BoltzOptions",
    "PredictionResult",
    "BoltzPredictor",
    "predict",
]

