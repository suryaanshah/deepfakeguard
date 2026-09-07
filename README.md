# deepfakeguard

DeepfakeGuard provides a small baseline detection engine for:

- image authenticity classification (`REAL`, `SUSPICIOUS`, `DEEPFAKE`)
- document integrity classification for passport/visa/national ID-like documents (`ORIGINAL`, `SUSPICIOUS`, `TAMPERED`)

The model entry points are:

- `detect_deepfake_image(signals)`
- `detect_tampered_document(signals)`

Each function consumes computer-vision signal scores in range `[0, 1]`, computes a weighted risk score, and returns a `DetectionReport`.

## Run tests

```bash
python -m unittest discover -s tests
```
