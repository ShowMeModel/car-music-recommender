The 2019 version
================

The original notebooks of this project, kept as they were in 2019 apart from
the move into this folder. Their outputs are next to them, so they still
resolve their relative paths. **They are superseded by the `carmusic` package
at the repository root.** The main README section *What was wrong with the
2019 version* explains why. In short: the situations were averaged away
before any model saw them, and the headline F1 of 0.79 came from a label
leak.

| file | what it is |
|---|---|
| `CarMusic exploratory analysis.ipynb` | Cleaning, feature engineering, clustering, apriori. Writes the CSV and pickle files below. |
| `CarMusic deep recommender.ipynb` | Spotlight (PyTorch) factorisation models. |
| `CarMusic cf and hybrid.ipynb` | LightFM hybrid model. |
| `CarMusic balanced machine learning.ipynb` | scikit-learn classifier on synthetic negatives (the 0.790 F1). |
| `CarMusic AutoML.ipynb` | TPOT search on the same data (the 0.796 F1). |
| `tpot_car_music.py` | TPOT's exported pipeline, with its template placeholders still in place. |
| `html/` | Static HTML exports of the notebooks, outputs included. |
| `dataset.csv`, `dataset_aggr.csv`, `*.pickle`, `music_type.csv` | Outputs of the exploratory notebook. |
| `Data_InCarMusic.xlsx` | Copy of `data/Data_InCarMusic.xlsx`, where the notebooks expect it. |
| `requirements.txt` | The original, unpinned dependency list (Python 3.6 era). |
| `check_leak.py` | *Added in 2026.* Rebuilds the notebooks' negative sampling with the current stack and shows where the F1 came from. |

Running the notebooks needs a 2019-era environment: Python 3.6–3.8, pandas
below 2.0, scikit-learn below 1.2, the TPOT 0.x API, and Spotlight installed
from GitHub. `check_leak.py` needs only the main project's
dependencies:

```shell
python legacy/check_leak.py
```

Don't unpickle the `.pickle` files unless you trust them. Nothing reads them;
they are kept only as a record.
