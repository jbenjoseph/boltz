# Python API

Boltz can be used directly from Python without invoking the command
line.  The :mod:`boltz.api` module exposes helpers to build input
specifications programmatically and run predictions.

```python
from boltz import BoltzInput, BoltzOptions, BoltzPredictor
```

## Building an input

Inputs mirror the YAML format used by the CLI.  For example a simple
protein--ligand complex can be described as:

```python
inp = BoltzInput(
    sequences=[
        {"protein": {"id": "A", "sequence": "ACDE"}},
        {"ligand": {"id": "L", "ccd": "ATP"}},
    ]
)
```

You can dump the YAML representation with
`inp.to_yaml()` if needed.

## Running a prediction

Use :class:`boltz.api.BoltzPredictor` to run the model programmatically.
When an output directory is not provided predictions are kept entirely in
memory so no files are written. The predictor manages temporary directories
internally and returns a
:class:`boltz.api.PredictionResult` object containing the generated
structures and metadata.

```python
predictor = BoltzPredictor(BoltzOptions(diffusion_samples=2))
result = predictor.predict(inp)
print(result.structures)      # list of CIF/PDB strings
print(result.confidence)      # parsed confidence JSON
print(result.affinity)        # affinity JSON if requested
print(result.coords[0])       # torch Tensor of atomic coordinates
```

