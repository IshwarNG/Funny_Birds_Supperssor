# FunnyBirds-Suppressor

Generates synthetic bird image datasets for testing XAI (attribution) methods. Every image has a class label,
concept labels (for concept bottleneck models) and masks for every bird part. The same birds are produced many times
with different amounts and types of background noise (**suppressor variables**), so you can see how
explanation methods react when the noise gets stronger.

It uses the 3D bird renderer of [FunnyBirds](https://github.com/visinf/funnybirds) and the signal + correlated-noise
idea of [XAI-TRIS](https://arxiv.org/abs/2306.12816). This repository only generates the data; training models and
evaluating XAI methods happen in a separate project that reads the data with `fb_dataset.py`.

---

## 1. Requirements

| What | Why |
|---|---|
| Python 3.9+ | the generator |
| `numpy scipy pillow requests pyyaml` | the generator |
| `torch` | only for `fb_dataset.py` (the loader), not for generating |
| Node.js + npm | the render server |
| Chromium or Chrome | the render server draws the birds in a headless browser (no GPU needed) |
| Git | to download the FunnyBirds renderer |

## 2. Setup

**1. Download this repository**

```bash
git clone <this-repo-url> Funny_Birds
cd Funny_Birds
```

**2. Python environment**

```bash
conda create -n medxai python=3.11 -y && conda activate medxai       # or any venv
pip install numpy scipy pillow requests pyyaml
pip install torch                                                     # only if you use the loader here
```

**3. Download the FunnyBirds renderer, install it and apply the two patches**

```bash
mkdir -p Clones && git clone https://github.com/visinf/funnybirds.git Clones/funnybirds
cd Clones/funnybirds
(cd render && npm install)
patch -p1 < ../../render_patch.diff
patch -p1 < ../../render_patch_layered.diff
cd ../..
```

Check that it worked:

```bash
ls Clones/funnybirds/render/js/three.js-master/examples/data/XAI/bird01.glb    # the 3D bird models
ls Clones/funnybirds/render/node_modules | head -3                              # npm packages
grep -c "is_black_bg" Clones/funnybirds/render/page.html                        # should be 2 or more
```

If puppeteer cannot find a browser later, set `render.chromium_path` in the YAML (for example `/usr/bin/chromium`).

## 3. Run

Small test first (a few minutes):

```bash
python generate.py --config configs/layered_test.yaml
python validate.py --config configs/layered_test.yaml
```

Full dataset, in the background, with 18 parallel workers (set by `render.workers` in the YAML):

```bash
nohup env RENDER_WAIT_MS=600 python -u generate.py --config configs/full_layered.yaml > full_layered.log 2>&1 &
tail -f full_layered.log
```

* The render server starts automatically.
* If the run stops, start the same command again: it continues where it stopped.
* To stop everything: `pkill -f generate.py; pkill -f "node server.js"`.
* Change the number of workers without editing the file: add `--set render.workers=12`.
* To change what is generated (classes, images per class, noise types, alpha values, image size), copy a YAML from
  `configs/`, edit it, and give it a new `dataset.name`.

Check the result:

```bash
python validate.py --config configs/full_layered.yaml        # automatic checks, preview sheets, report
python verify_masks.py --all-variants --alpha 0.75           # mask pictures, one folder per noise type
```

## 4. What it creates

Everything goes to `datasets/<dataset.name>/`:

```
datasets/full_layered_v1/
├── scenes/       images, masks and labels of the birds (written once, shared by all noise types)
│   ├── clean/        image without noise (bird smoothed, distractor objects on top)
│   ├── raw/          the same without smoothing
│   ├── part_map/     colour-coded segmentation image
│   ├── masks/        masks of every bird part, one .npz per image
│   └── labels_train.npz, labels_test.npz, labels_meta.json, ...
├── variants/     the final datasets: one folder per noise type and alpha
│   ├── white/alpha_0.050/ ... alpha_1.000/
│   ├── corr_s2/, corr_s4/, corr_s8/, corr_s4_nobird/, pink_b1/
│   └── <noise type>/alpha_X/<train|test>/<class>/<id>.png
├── validation/   check results (after validate.py)
└── classes.json, parts.json, config_used.yaml
```

**Datasets.** Each folder `variants/<noise type>/alpha_X/` is one complete dataset. With `full_layered.yaml` that is
31 datasets (7 white-noise alphas, 6 each for `corr_s2`, `corr_s4`, `corr_s8`, and 3 each for `corr_s4_nobird`
and `pink_b1`), each with 10 classes × 250 images = 2,500 images (2,000 train, 500 test), 256 × 256 pixels.
All datasets contain the same birds, so image `000042` is the same bird everywhere. Small alpha = strong noise,
alpha 1.0 = no noise.

| Noise type | Meaning |
|---|---|
| `white` | uncorrelated noise (control, no suppressors) |
| `corr_s2`, `corr_s4`, `corr_s8` | smooth correlated noise, correlation length 2 %, 4 %, 8 % of the image: creates suppressor variables |
| `corr_s4_nobird` | same noise as `corr_s4`, but removed inside the bird (ablation) |
| `pink_b1` | 1/f noise |

**Masks** (per image, in `scenes/masks/`):

* sharp: `beak`, `eye`, `foot`, `tail`, `wing`, `body`, `bird`, `bg_objects`, `bg_canvas`, `informative`
* smoothed (soft, 0–1): `beak_smooth`, `eye_smooth`, `foot_smooth`, `tail_smooth`, `wing_smooth`, `body_smooth`, `bird_smooth`, `informative_smooth`

**Concept labels** (per image, in `scenes/labels_<split>.npz`):

| Array | Format |
|---|---|
| `class_label` | class index |
| `concept_vector` | 26 values, one-hot per part (beak 4, eye 3, foot 4, tail 9, wing 6 variants); zeros if the part is missing |
| `concept_indices` | 5 integers, variant index per part (-1 = missing) |
| `attribute_vector` | 22 values, shape and colour as separate concepts |
| `part_present` | 5 booleans, part exists on the bird |
| `part_visible`, `part_visible_pixels` | 5 booleans / counts, part is visible in the image |

## 5. Use the data in another project

Copy `fb_dataset.py` into your project:

```python
from fb_dataset import FunnyBirdsSuppressor

ds = FunnyBirdsSuppressor("datasets/full_layered_v1", split="test", variant="corr_s4", alpha=0.5)
item = ds[0]
item["image"]            # (3, 256, 256) in [0, 1]
item["class_label"]
item["concept_vector"]   # (26,)
item["masks"]            # sharp masks (K, 256, 256); ask for smoothed ones with
                         # mask_names=["bird", "bird_smooth", "wing_smooth", ...]
```

## 6. Common problems

| Problem | Fix |
|---|---|
| `node_modules not found` | run `npm install` in `Clones/funnybirds/render` |
| `bird models not found` | run `git checkout -- render/js` inside `Clones/funnybirds` |
| render server does not start / no browser | set `render.chromium_path` in the YAML; read `Clones/funnybirds/render/server.log` |
| distractor objects look black in `scenes/clean/` | the second patch (`render_patch_layered.diff`) is not applied; apply it, restart, regenerate |
| only `scenes/` exists, no `variants/` | the run is still rendering; `variants/` appears after all scenes are done |

More detail (all settings, all noise types, validation, reproducibility): `README_detailed.md`.

## Credits

* R. Hesse, S. Schaub-Meyer, S. Roth. *FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods.* ICCV 2023.
* B. Clark, R. Wilming, S. Haufe. *XAI-TRIS: Non-linear image benchmarks to quantify false positive post-hoc attribution of feature importance.* arXiv:2306.12816.

This project is independent and not affiliated with the authors of either work. Respect the licences of FunnyBirds
and of the other third-party components (three.js, puppeteer).
