# deepfakeguard

A small CNN that separates real faces from deepfake faces.

* **Training** - FaceForensics++ (C23): `real/` vs `fake/Deepfakes/`, one head crop per video.
* **Testing** - Celeb-DF v2 (`Test` split) plus the unseen FaceForensics++ manipulations
  Face2Face, FaceShifter and FaceSwap.

Both archives are downloaded automatically the first time the notebook runs, so it works
as-is - no manual dataset setup.

## Run it

Jupyter:

```bash
jupyter lab DeepfakeGuard-X.ipynb
```

[marimo](https://marimo.io) (this is how it was verified):

```bash
marimo convert DeepfakeGuard-X.ipynb -o DeepfakeGuard-X.py
marimo edit DeepfakeGuard-X.py
```

`deepfake_guard.py` is the same notebook already in marimo format (it can also be run
directly with `marimo edit deepfake_guard.py`).

## What happens on the first run

1. `FF++.zip` is assembled from the FaceForensics++ C23 videos on HuggingFace
   (`bitmind/FaceForensicsC23`). Only the ~800 videos that are needed are fetched, using
   HTTP range requests against the remote zip, and one head crop per video is stored as
   `real/<video>/000.png` / `fake/<method>/<video>/000.png`.
2. `CelebDF.zip` is assembled from the Celeb-DF v2 image dataset on HuggingFace
   (`thenewsupercell/celeb-df-image-dataset`) as `Celeb_V2/Test/{real,fake}/*.jpg`.
3. The CNN is trained for 80 epochs and evaluated on all four test sets.

Both archives are cached in the working directory, so later runs start straight at training.
