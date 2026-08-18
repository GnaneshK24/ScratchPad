# Image Coordinate Utility

This project provides a local, training-free utility for locating a reference
pattern in a larger grayscale image and returning its center coordinate.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run the command-line utility

```bash
python localize.py --reference path/to/reference.png --search path/to/search.png
```

Successful default output is one machine-readable line:

```text
x,y
```

The command accepts only the two images. It does not read labels, annotations,
or evaluation files.

## Create example inputs

```bash
python generate_dataset.py --architecture finfet --num-pairs 5 --output-dir demo_data --seed 42
```

This creates `reference/`, `search/`, and `ground_truth.csv` under
`demo_data/`. The public generator supports FinFET only. Generated outputs
are excluded from version control.

To view every generation option:

```bash
python generate_dataset.py --help
```

## Optional interface

```bash
python -m streamlit run app.py
```

The browser interface is optional and is useful for local inspection and
validation.

## Evaluation Evidence

[`submission_evidence/`](submission_evidence/README.md) contains the fixed-seed
synthetic FinFET evaluation evidence: a 30-pair representative subset,
ground-truth/prediction results, metadata and pair notes, tolerance and
precision-recall figures, noise-stress analysis, sub-pixel metrics, failure
analysis, and environment information. It is generated with the frozen public
generator and localizer; its report distinguishes the full synthetic pool from
the deliberately diverse selected examples.

## Notes

- Python 3.14.7 was used for final checks.
- No model download, network access, API key, or GPU is required at runtime.
- See [docs/REFERENCES.md](docs/REFERENCES.md) for technical background.
