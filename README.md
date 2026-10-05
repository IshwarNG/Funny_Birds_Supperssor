# FunnyBirds-Suppressor

**A configurable generator for part-annotated synthetic bird datasets with controlled suppressor variables.**

FunnyBirds-Suppressor renders 3D birds with the FunnyBirds renderer, annotates every image with a class label, several kinds of concept labels (for Concept Bottleneck Models) and pixel-accurate masks for every bird part, and then produces many *versions* of the same images in which the background carries a controlled, adjustable amount of **suppressor-variable** structure. The result is a family of datasets that share identical scenes, labels and masks and differ only in the type and strength of the suppressor noise, which makes it possible to study how attribution (XAI) methods react when suppressors become stronger.

This repository contains the **data generation** pipeline only. Training models and evaluating XAI methods are intended to live in a separate project that reads the generated data through the single-file loader `fb_dataset.py`.

---

## Table of contents

1. [Motivation and background](#1-motivation-and-background)
2. [Key features](#2-key-features)
3. [How the data are generated](#3-how-the-data-are-generated)
4. [Repository structure and file responsibilities](#4-repository-structure-and-file-responsibilities)
5. [Installation](#5-installation)
6. [Quick start](#6-quick-start)
7. [Configuration reference](#7-configuration-reference)
8. [Suppressor families](#8-suppressor-families)
9. [Output dataset layout](#9-output-dataset-layout)
10. [Annotations: classes, concepts and masks](#10-annotations-classes-concepts-and-masks)
11. [Loading the data](#11-loading-the-data)
12. [Validation](#12-validation)
13. [Exporting self-contained datasets](#13-exporting-self-contained-datasets)
14. [Performance, parallelism and resuming](#14-performance-parallelism-and-resuming)
15. [Reproducibility](#15-reproducibility)
16. [Extending the generator](#16-extending-the-generator)
17. [Design decisions and known limitations](#17-design-decisions-and-known-limitations)
18. [Troubleshooting](#18-troubleshooting)
19. [References and acknowledgements](#19-references-and-acknowledgements)

---

## 1. Motivation and background

Post-hoc attribution methods (saliency maps, Integrated Gradients, LRP, SHAP, LIME, ...) are often evaluated on anecdotal examples. Quantitative evaluation requires data for which the *truly important* input features are known by construction. The XAI-TRIS benchmark (Clark, Wilming and Haufe, 2023) builds such data and shows that attribution methods frequently assign importance to **suppressor variables**: features that have no statistical association with the target on their own, but that help a model by cancelling noise that is shared with the informative features.

In images, suppressors arise naturally when the background noise is spatially correlated with the noise on the object, for example illumination that is visible in the background and also changes the brightness of the object. A model can read the lighting from the background and use it to normalise the object, so the background becomes *useful for prediction without being class-informative*.

[FunnyBirds](https://github.com/visinf/funnybirds) (Hesse, Schaub-Meyer and Roth, ICCV 2023) provides a synthetic 3D bird dataset in which every bird is composed of parts (beak, eye, foot, tail, wing) and the class is defined by which variant of each part the bird has. It therefore comes with a natural part-level ground truth, and it supports concept-based models because each class is a fixed combination of part variants.

This project combines the two ideas:

* **FunnyBirds** supplies the scenes, the class definition, the part segmentation and the concepts.
* **XAI-TRIS** supplies the principle of mixing a signal and spatially correlated noise with a signal-to-noise parameter `alpha` so that suppressor variables emerge in a controlled way.

The generator then produces one dataset per *(suppressor family, alpha)* combination, all on the same scenes, so that differences between datasets are caused by the suppressor alone.

---

## 2. Key features

* **One command, one YAML file.** The whole dataset family is described in a YAML file. All paths and parameters are configurable; there are no hard-coded locations.
* **Render once, blend many times.** The slow 3D rendering happens once per scene. Every suppressor dataset is produced from the stored clean image in a second, fast stage that needs no renderer. New noise types or additional alpha values can be added in seconds-to-minutes without re-rendering.
* **Two independent suppressor controls.** Signal strength `alpha` and noise correlation length `sigma_frac` are separate parameters, so both can be swept.
* **Seven noise families** (white, correlated, oriented, colour-correlated, pink 1/f, natural photographs, multiplicative illumination) plus an *exclude-bird* ablation that removes the noise inside the bird.
* **Layered signal (`signal.source: layered`).** As in XAI-TRIS the signal is only the object: the bird is smoothed on its own, the 3D distractors are pasted on top afterwards, and only then the noise is added. Masks are provided both sharp and smoothed (section 3.6).
* **Dataset-level normalisation**, so that `alpha` has the same meaning for every image.
* **Concept labels in several formats**: one-hot per part variant, per-part indices, factorised attributes (for example shape and colour), part presence, and part *visibility* in the image.
* **Masks for every bird part**: beak, eye, foot, tail, wing, body, whole bird, distractor objects, empty background and an *informative-parts* mask.
* **Higher image quality**: rendering at `size x supersample` pixels and area-averaging down; part maps are rendered without anti-aliasing and downsampled with nearest-neighbour so every mask pixel has an exact part colour.
* **Shared annotations.** Masks, part maps and labels are stored once and shared by all suppressor datasets instead of being duplicated per alpha.
* **Parallel and resumable.** Rendering and blending use a configurable number of workers; interrupted runs continue where they stopped.
* **Built-in validation** with automatic consistency checks, bit-exact rebuild checks, a suppressor-strength statistic and preview images.
* **Self-contained exports** (hard links, symbolic links or copies) for handing a single dataset to an evaluation project.

---

## 3. How the data are generated

### 3.1 Pipeline overview

```
configs/*.yaml --> config.py --> generate.py
                                    |
        +---------------------------+----------------------------+
        |                                                        |
 STAGE 1  "scenes"  (slow, needs the renderer)           STAGE 2  "variants"  (fast, no renderer)
 scenes.py   classes, random scenes, labels, masks        suppressors.py   noise + alpha blending
 render.py   Node / puppeteer / three.js server           reads scenes/clean + per-scene noise seed
        |                                                        |
        v                                                        v
 datasets/<name>/scenes/                                 datasets/<name>/variants/<variant>/alpha_x/
   clean/  part_map/  masks/  labels_*.npz  index_*.json
        |                                                        |
        +--------------------> validate.py <---------------------+
        +--------------------> fb_dataset.py  (evaluation project)
```

### 3.2 Stage 1: scenes

1. **Classes.** A class is a unique combination of one variant per part (beak, eye, foot, tail, wing), drawn from `parts.json`.
2. **Scenes.** For every image the generator samples camera distance, camera pitch and roll (full rotation by default), light direction, a random number of 3D distractor objects (box, cone, sphere, cylinder, capsule; red, green, blue or yellow) and, in the *train* split, which parts to drop from the bird (see below).
3. **Rendering.** `render.py` asks the Node render server for the signal image and for the **part map** (a segmentation render in which every part has an exact RGB colour). Both are rendered at `size x supersample` pixels and downsampled.
4. **Masks and labels.** Masks are extracted from the part map by exact colour matching; concept labels are derived from what is actually present in the scene.
5. **Validation of the render.** A render is accepted only if the bird covers at least `render.min_bird_frac` of the image; otherwise it is retried (`render.attempts`) and finally dropped.

Result: the folder `scenes/`, which is independent of any suppressor.

### 3.3 Stage 2: variants

For each scene, each suppressor entry in the config and each alpha:

1. The clean image is loaded and smoothed with the XAI-TRIS signal filter *H*.
2. Noise is generated deterministically from the scene's stored `noise_seed`.
3. Signal and noise are blended with `alpha` and written as a lossless PNG.

### 3.4 The blending model

Let `s` be the smoothed clean signal in `[0, 1]` and `n` a unit-variance noise field of the chosen family.

**Signal filter H.** `s = GaussianFilter(clean, sigma_H)` per channel with `sigma_H = signal.sigma_smooth_frac * size` pixels (default `0.006 * 256 = 1.5 px`); values below `signal.threshold_support` times the channel maximum are set to zero (XAI-TRIS, Eq. 1).

**Dataset-level statistics.** From up to `signal.stats_samples` train images, evenly spaced, the generator computes the per-channel mean `mu` and the pooled standard deviation `sigma_s` of the smoothed signal and stores them in `scenes/norm_stats.json`.

**Additive families** (white, corr, oriented, color_corr, pink, natural):

```
x = clip( alpha * s + (1 - alpha) * ( c + gain * sigma_s * n ), 0, 1 )
c = clip( mu, min(3A, 0.5), max(1 - 3A, 0.5) ),   A = gain * sigma_s
```

`c` is the signal mean, moved away from 0 and 1 only when necessary so that noise on a dark signal (for example `signal.source: foreground`, bird on black) is not truncated at 0.

**Multiplicative family** (illum):

```
x = clip( s * max( 1 + (1 - alpha) * gain * n , 0.05 ), 0, 1 )
```

where `n` is a smooth grey (identical in all three channels) illumination field.

**Nominal signal-to-noise ratio** for additive families: `SNR_dB = 20 * log10( alpha / ((1 - alpha) * gain) )`. It is `-infinity` at `alpha -> 0` and undefined (noise-free) at `alpha = 1`. No SNR is reported for multiplicative families.

**Unit-variance noise.** Every family is scaled by a fixed calibration factor (RMS over six draws with a fixed seed), so `gain = 1` means the noise has the same standard deviation as the signal. Because the factor is dataset-level and not computed per image, natural image-to-image variation of the noise energy is preserved; this follows the dataset-level (Frobenius) normalisation used in XAI-TRIS.

### 3.5 Why correlated noise creates suppressors

With `corr` noise a background pixel next to the bird is correlated with the noise on the bird. A model can therefore use the background to estimate and subtract the noise on the bird, so background pixels become informative for the prediction although they contain no class information. The correlation length is `sigma_frac * size` pixels: larger values mean that more distant background pixels are correlated with the bird. `white` noise has no such correlation and serves as the control. `corr` with `exclude_bird: true` removes the noise inside the bird region, which breaks the suppressor mechanism and serves as an ablation.

### 3.6 The layered signal (`signal.source: layered`)

With `default` or `foreground` the filter H is applied to the whole rendered picture. The `layered` mode follows the XAI-TRIS definition more closely: the signal is *only the bird*, H acts *only on the bird*, and everything else is added afterwards.

```
1. render the bird alone (black background)          -> bird_img, bird_pm      (part map of the bird, nothing occluding it)
2. render bird + distractors (black background)      -> scene_img, scene_pm    (tells which pixels show a distractor)
3. smooth the bird:      S = H(bird_img)             (sigma = signal.sigma_smooth_frac * size)
4. masks of the smoothed bird and of each part       (sharp and smoothed, below)
5. paste the distractors: signal = S, except where scene_pm shows a distractor, there signal = scene_img
6. noise:   x = alpha * signal + (1 - alpha) * (G o noise)       (blending as in 3.4)
```

* **Four renders per scene** instead of two, so stage 1 takes roughly twice as long.
* **Occlusion is exact.** Distractors that are in front of the bird cover it in the signal and in all masks; distractors behind the bird do not appear where the bird is.
* **G acts on the noise only.** The noise field is smoothed (`sigma_frac`) *before* it is added; the combined image is never smoothed again, so the bird stays as sharp as H left it.
* **Distractors scale with alpha** like the bird (they are part of the signal term), but they are never blurred.
* **Background is black** (zero signal), as in XAI-TRIS. The blue FunnyBirds background of the other modes is not used.
* Stage 2 does not apply H again (`suppressors.prepare_signal` returns the stored image unchanged for `layered`).
* Requires `masks.dilate_radius: 0` and the second render patch (`render_patch_layered.diff`, section 5.3).

**Masks in layered mode** (all describe the *final* image, i.e. after occlusion by distractors):

| Mask | Meaning |
|---|---|
| `beak`, `eye`, `foot`, `tail`, `wing`, `body`, `bird`, `informative` | Sharp (unsmoothed) boolean masks of what is visible. |
| `<name>_smooth` for the same eight names | Soft masks, `uint8` 0..255 (loader: floats 0..1): the sharp mask blurred with the same sigma as H. |
| `bg_objects`, `bg_canvas` | Distractor pixels, empty background. |

The support of the smoothed masks is where the smoothed *bird* mask is at least `signal.threshold_support` (the mask analogue of the support threshold of H). All smoothed masks share this support, so the smoothed part masks add up to `bird_smooth` (up to 8-bit rounding), and every pixel of a sharp mask also has a non-zero smoothed value. The unsmoothed composite image (sharp bird + distractors) is stored in `scenes/raw/` for reference.

---

## 4. Repository structure and file responsibilities

```
Synthetic_Funny_Birds/
├── config.py            configuration, defaults, CLI overrides, file-path helper
├── render.py            client for the Node / puppeteer render server
├── scenes.py            classes, scenes, concept labels, masks
├── suppressors.py       noise families and alpha blending
├── generate.py          main script: runs both stages, resume, export
├── validate.py          consistency checks, suppressor statistic, preview sheets
├── fb_dataset.py        standalone PyTorch loader for the evaluation project
├── configs/
│   ├── quick_test.yaml  small run to check the installation
│   ├── layered_test.yaml small run of the layered signal
│   └── full.yaml        production configuration (example in section 7.4)
├── render_patch.diff            patch 1 for Clones/funnybirds/render/{server.js,page.html}
├── render_patch_layered.diff    patch 2: extra 'black' render mode in page.html (layered signal)
├── diag_consistency.py          optional: checks that part maps, masks and images agree
├── diag_parts.py                optional: how often each part is hidden / below the visibility threshold
├── datasets/            generated output (created automatically)
└── Clones/funnybirds/   upstream FunnyBirds repository (renderer, 3D models)
```

### Responsibilities

| File | Responsibility | Key functions |
|---|---|---|
| `config.py` | Holds every default (`DEFAULTS`), loads and validates the YAML, applies `--set` overrides, expands `~` and `${ENV}`, resolves relative paths against the project root, and defines all dataset paths in one class. | `load_config`, `get_config`, `resolve_path`, `alpha_tag`, `Paths` |
| `render.py` | Starts the render server if it is not running, requests the signal render and the part map, retries failures, supersamples and downsamples (area average for images, nearest-neighbour for part maps). Thread-safe. | `ensure_server`, `render_scene` |
| `scenes.py` | Everything about the *content* of a scene: unique classes, random scene parameters (camera, light, distractors, dropped parts), **all concept label formats**, **all masks**. | `make_classes`, `sample_scene`, `build_label_info`, `sample_labels`, `extract_masks`, `finalize_masks` |
| `suppressors.py` | Everything about the *background*: the noise families, signal smoothing, dataset statistics, **alpha blending**, nominal SNR. | `FAMILIES`, `make_noise`, `prepare_signal`, `signal_stats`, `compose`, `nominal_snr_db` |
| `generate.py` | Orchestration. Stage 1 (`render_one`, `build_index_and_labels`), stage 2 (`ensure_norm_stats`, `variant_one`), thread pool, atomic file writes, export. | `stage_scenes`, `stage_variants`, `export`, `main` |
| `validate.py` | Verifies a finished dataset and writes preview images and `report.json`. | `check_scenes`, `check_variants`, `ring_correlation`, `make_sheets` |
| `fb_dataset.py` | Loads any (variant, alpha) dataset or an exported folder. Depends only on NumPy, Pillow and PyTorch, so it can be copied into another project. | `FunnyBirdsSuppressor`, `list_variants`, `list_alphas` |

### Where is X implemented?

| I want to look at... | File and function |
|---|---|
| How suppressor noise is created | `suppressors.py`: `_corr`, `_pink`, ... and `make_noise` |
| How noise and image are mixed with alpha | `suppressors.py`: `compose` |
| Where the mixing is called and the PNGs are written | `generate.py`: `variant_one` |
| How concept labels (CBM) are built | `scenes.py`: `build_label_info`, `sample_labels` |
| Where visibility of parts is computed and labels are saved | `generate.py`: `build_index_and_labels` |
| How masks are extracted | `scenes.py`: `extract_masks`, `finalize_masks` |
| How random scenes (camera, distractors, dropped parts) are sampled | `scenes.py`: `sample_scene` |
| How the render server is started and queried | `render.py`: `ensure_server`, `_fetch` |
| Every default setting | `config.py`: `DEFAULTS` |

---

## 5. Installation

### 5.1 Requirements

| Component | Purpose | Notes |
|---|---|---|
| Python 3.9 or newer | generator and loader | tested in a conda environment |
| `numpy`, `scipy`, `pillow`, `requests`, `pyyaml` | generator | |
| `torch` | only `fb_dataset.py` | not needed for generation |
| Node.js and npm | render server | packages: `express`, `ejs`, `puppeteer` (installed from `package.json`) |
| Chromium / Chrome | used by puppeteer | software rendering (SwiftShader) is sufficient; no GPU is required |
| Git | restoring renderer assets | |

### 5.2 Python environment

```bash
conda activate <your-env>            # for example: conda activate medxai
pip install numpy scipy pillow requests pyyaml
pip install torch                    # only if you use fb_dataset.py in this environment
```

### 5.3 Renderer

The renderer is the unmodified FunnyBirds render server plus two small edits shipped as `render_patch.diff`.

```bash
cd Synthetic_Funny_Birds/Clones/funnybirds

# 1. restore the three.js files and the 3D bird models if they were removed
git checkout -- render/js

# 2. install the Node packages
cd render && npm install && cd ..

# 3. apply the patch once (see below)
patch -p1 < ../../render_patch.diff
```

Verify the installation:

```bash
ls render/js/three.js-master/examples/data/XAI/bird01.glb    # must exist
ls render/node_modules | head -3                              # must list packages
```

**What the patch changes**

| File | Change |
|---|---|
| `render/server.js` | Image size and supersampling factor are read from the request (`size`, `scale`). The Chromium path is no longer hard-coded; it is only set if `PUPPETEER_EXECUTABLE_PATH` is defined (done by `generate.py` from `render.chromium_path`). The wait time after page load can be set with the environment variable `RENDER_WAIT_MS` (default 200 ms). |
| `render/page.html` | Anti-aliasing is disabled when `render_mode == 'part_map'`, so that every part-map pixel has an exact part colour. |

If `patch` reports "Reversed (or previously applied) patch detected", the changes are already in place.

**Second patch (only for `signal.source: layered`).** `render_patch_layered.diff` adds a `black` render mode to `page.html` (bird and distractors on a black background). Apply it after the first patch:

```bash
cd Synthetic_Funny_Birds/Clones/funnybirds
patch -p1 < ../../render_patch_layered.diff
```

Check that both patches are active: `grep -n "antialias" render/page.html` must show `render_mode !== 'part_map'`, and `grep -n "is_black_bg" render/page.html` must show matches. Restart the render server after patching (`pkill -f "node server.js"`).

The render server listens on port **8081** (hard-coded in the upstream `server.js`). The generator starts it automatically (`render.autostart_server: true`), writes its log to `Clones/funnybirds/render/server.log` and stops it when Python exits.

---

## 6. Quick start

```bash
cd Synthetic_Funny_Birds
conda activate <your-env>

python generate.py --config configs/quick_test.yaml      # a few minutes
python validate.py --config configs/quick_test.yaml
```

Inspect:

* `datasets/quick_test/validation/masks_overlay.png`: every part mask overlaid on the clean image. Verify the colours sit on the correct parts.
* `datasets/quick_test/validation/sheet_corr_s4.png`: the same scenes at decreasing alpha.
* The validation table in the terminal should end with `all checks passed`.

Then create a production configuration (section 7.4) and run:

```bash
nohup env RENDER_WAIT_MS=600 python -u generate.py --config configs/full.yaml > full.log 2>&1 &
tail -f full.log
```

`python -u` disables output buffering so the log updates immediately; `RENDER_WAIT_MS` gives the browser more time to load the 3D models when many renders run in parallel.

### Command-line reference

```
python generate.py  --config FILE  [--set KEY=VALUE ...]  [--stage {all,scenes,variants}]
                    [--export DIR] [--export-mode {hardlink,symlink,copy}]
python validate.py  --config FILE  [--set KEY=VALUE ...]  [--samples N] [--cols N]
python config.py    --config FILE  [--set KEY=VALUE ...]      # prints the resolved configuration
```

| Option | Meaning |
|---|---|
| `--config FILE` | YAML file. Omitted keys use the defaults in `config.py`. |
| `--set KEY=VALUE` | Override any key with dotted notation, for example `--set render.size=128 dataset.seed=3`. Values are parsed as YAML, so whole lists can also be given (`--set 'suppressors=[{type: white, alphas: [0.5]}]'`), but editing the file is easier for suppressors. |
| `--stage scenes` | Only stage 1. |
| `--stage variants` | Only stage 2 (needs a finished stage 1). |
| `--export DIR` | After generation, build self-contained folders in `DIR` (section 13). |
| `--samples N` | Validation: number of images checked per split and variant (default 60). |

---

## 7. Configuration reference

All keys are optional. Relative paths are resolved against the project root (the folder containing `config.py`), so commands work from any directory.

### 7.1 `dataset`

| Key | Default | Description |
|---|---|---|
| `name` | `FunnyBirds_Suppressor` | Name of the output folder inside `out_dir`. Use a new name for every new dataset. |
| `out_dir` | `./datasets` | Parent folder of all datasets. |
| `parts_json` | `./Clones/funnybirds/render/parts.json` | Part variants (models and colours). Copied into the dataset. |
| `seed` | `0` | Random seed for classes and scenes. |
| `nr_classes` | `10` | Number of bird classes (must not exceed the number of distinct part combinations). |
| `train_per_class` | `50` | Training images per class (`0` disables the split). |
| `test_per_class` | `10` | Test images per class (`0` disables the split). |
| `min_bg_parts` | `0` | Minimum number of 3D distractor objects per image. |
| `max_bg_parts` | `35` | Upper bound (exclusive): the count is uniform in `[min, max - 1]`. |
| `drop_parts_in_train` | `true` | FunnyBirds behaviour: in half of the train images random parts are removed. Test images always show all parts. |
| `regen_classes` | `false` | Overwrite an existing `classes.json`. |
| `regen_params` | `false` | Overwrite existing scene parameters (new random scenes). |
| `camera.distance_range` | `[200, 400]` | Camera distance (integer). |
| `camera.pitch_range` | `[0, 6.2832]` | Camera pitch in radians. |
| `camera.roll_range` | `[0, 6.2832]` | Camera roll in radians. |

### 7.2 `render`

| Key | Default | Description |
|---|---|---|
| `server_url` | `http://localhost:8081` | Address of the render server. |
| `render_dir` | `./Clones/funnybirds/render` | Folder containing `server.js`. |
| `chromium_path` | `null` | Path to Chromium/Chrome if puppeteer does not find one, for example `/usr/bin/chromium`. |
| `autostart_server` | `true` | Start the server automatically when it is not running. |
| `size` | `256` | Final image side length in pixels. |
| `supersample` | `2` | Render at `size * supersample` and average down. `1` disables supersampling. |
| `workers` | `1` | Parallel render requests (also used for the blending stage). |
| `timeout` | `120` | Seconds before a render request times out. |
| `max_retries` | `5` | HTTP retries per request. |
| `attempts` | `3` | Re-renders when the bird is not visible before the scene is dropped. |
| `min_bird_frac` | `0.002` | Minimum bird area (fraction of the image) for a render to be accepted. |

### 7.3 `signal`, `masks`, `suppressors`

| Key | Default | Description |
|---|---|---|
| `signal.source` | `default` | `default`: bird plus distractors on the blue FunnyBirds background. `foreground`: bird only, on black (the part map is then rendered without distractors, so the `bg_objects` mask is empty). `layered`: bird smoothed alone, distractors pasted afterwards on black, plus smoothed masks (section 3.6). |
| `signal.sigma_smooth_frac` | `0.006` | Width of the signal filter H as a fraction of the image size (`0.006 * 256 = 1.5 px`). `0` disables smoothing. In `layered` mode it also sets the width of the smoothed masks. |
| `signal.threshold_support` | `0.05` | Values below this fraction of the channel maximum are zeroed after smoothing. |
| `signal.stats_samples` | `500` | Train images used to estimate the signal mean and standard deviation. |
| `masks.dilate_radius` | `0` | Morphological dilation of part masks. `0` is exact because part maps are rendered without anti-aliasing. |
| `masks.min_visible_frac` | `0.0002` | A part counts as *visible* above this fraction of the image area. |
| `masks.save_pngs` | `false` | Also write one PNG per mask (the `.npz` is always written). |
| `suppressors` | one `corr` entry | List of suppressor entries, see section 8. |

### 7.4 Example production configuration

This is the reference configuration used for the full dataset (10 classes, 2,500 scenes, 37 image sets per scene):

```yaml
dataset:
  name: full_v1
  seed: 0
  nr_classes: 10
  train_per_class: 200
  test_per_class: 50
  min_bg_parts: 0
  max_bg_parts: 35

render:
  size: 256
  supersample: 2
  workers: 12            # roughly half of the CPU threads; see section 14

signal:
  source: default

suppressors:
  - {type: white, alphas: &grid [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]}
  - {type: corr, sigma_frac: 0.02, alphas: *grid}
  - {type: corr, sigma_frac: 0.04, alphas: *grid}
  - {type: corr, sigma_frac: 0.08, alphas: *grid}
  - {type: corr, sigma_frac: 0.04, exclude_bird: true, name: corr_s4_nobird, alphas: [0.1, 0.2, 0.5]}
  - {type: pink, beta: 1.0, alphas: [0.1, 0.2, 0.5]}
  - {type: illum, sigma_frac: 0.15, gain: 0.5, alphas: [0.2, 0.5, 0.8]}
```

YAML anchors (`&grid`, `*grid`) let several entries share the same alpha list.

---

## 8. Suppressor families

Every entry in the `suppressors` list produces one **variant** (a folder of images) per listed alpha.

### 8.1 Common keys

| Key | Required | Description |
|---|---|---|
| `type` | yes | Family name (see below). |
| `alphas` | yes | List of values in `(0, 1]`. `1.0` is the (smoothed) signal without noise. Small values mean strong noise. |
| `name` | no | Folder name of the variant. Generated automatically if omitted (for example `corr_s4`). Names must be unique. |
| `gain` | no | Noise amplitude multiplier. Default `1.0` for additive and `0.5` for multiplicative families. |
| `exclude_bird` | no | Additive families only. `true` removes the noise inside the bird mask (ablation without suppressor relationship). |

### 8.2 Families

| `type` | Extra keys (defaults) | Description | Typical use |
|---|---|---|---|
| `white` | none | Uncorrelated Gaussian noise. | Control without suppressors. |
| `corr` | `sigma_frac` (0.04) | Gaussian-smoothed noise; correlation length `sigma_frac * size` px. The XAI-TRIS *CORR* background. | Main suppressor family. Sweep `sigma_frac`. |
| `oriented` | `sigma_x_frac` (0.08), `sigma_y_frac` (0.01) | Different smoothing along x and y, giving stripe-like structure. | Anisotropic correlations. |
| `color_corr` | `sigma_frac` (0.04), `rho` (0.8) | Correlated noise whose RGB channels share a common component (`rho` = shared fraction). | Colour-correlated backgrounds. |
| `pink` | `beta` (1.0) | Noise with a `1/f^beta` power spectrum. | Natural-image-like statistics. |
| `natural` | `dir` (required), `gray` (true) | Real photographs from a folder, centre-cropped, zero-centred. Supports `.jpg .jpeg .png .bmp .webp`. | Realistic backgrounds (for example the PASS dataset, which has no people). |
| `illum` | `sigma_frac` (0.15), `gain` (0.5) | **Multiplicative** smooth grey illumination field. With `signal.source: default` the constant background reveals the field, so background pixels become exact suppressors of the lighting on the bird. | Lighting-type suppressors. Not meaningful with a black background. |

### 8.3 Automatic names

`corr` with `sigma_frac: 0.04` becomes `corr_s4`, `oriented` becomes `oriented_x8y1`, `pink` with `beta: 1` becomes `pink_b1`, `color_corr` becomes `color_corr_s4_rho0.8`, and `exclude_bird: true` appends `_nobird`. Set `name:` explicitly to override.

### 8.4 Paired design

The noise of a sample is generated from the scene's stored `noise_seed`, so:

* All alpha values of a variant use *the same* noise field for a given image.
* `white`, `corr` and `oriented` start from the *same* raw Gaussian field for a given image, so differences between them are due to the filtering only.

### 8.5 Choosing alpha and sigma

Strongly correlated noise destroys the signal at much larger alpha than white noise does, so use a roughly logarithmic alpha grid that starts low, for example `0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0`. The XAI-TRIS paper selects, per scenario and model, the alpha at which a classifier reaches a target accuracy; training such models is outside this project, but the alpha grid should be dense enough in the interesting range. The paper smooths with sigma = 10 on 64 px images, which corresponds to `sigma_frac` of about 0.16.

---

## 9. Output dataset layout

```
datasets/<name>/
├── classes.json                      class -> part variant indices
├── parts.json                        copy of the part definitions used
├── config_used.yaml                  fully resolved configuration of the last run
│
├── scenes/                           independent of any suppressor; written once
│   ├── params_<split>.json           complete scene parameters (render params, present parts, seeds)
│   ├── index_<split>.json            [{id, class_idx, noise_seed, n_bg_objects}, ...] of valid scenes
│   ├── labels_<split>.npz            label arrays, one row per entry of the index (section 10)
│   ├── labels_meta.json              names and metadata for the label arrays
│   ├── norm_stats.json               signal mean / std used for blending
│   ├── clean/<split>/<class>/<id>.png        noise-free signal image (layered: smoothed bird + distractors)
│   ├── raw/<split>/<class>/<id>.png          layered mode only: unsmoothed bird + distractors
│   ├── part_map/<split>/<class>/<id>.png     segmentation render with exact part colours
│   └── masks/<split>/<class>/<id>.npz        boolean masks (and optional <id>_<mask>.png)
│
├── variants/                         one folder per suppressor variant
│   └── <variant>/
│       └── alpha_<a>/                for example alpha_0.100
│           ├── variant.json          type, alpha, parameters, nominal SNR, normalisation
│           └── <split>/<class>/<id>.png    the final image
│
└── validation/                       created by validate.py
    ├── report.json
    ├── sheet_<variant>.png           alpha sweep preview
    └── masks_overlay.png             mask check
```

* `<split>` is `train` or `test`; `<class>` is the integer class index; `<id>` is a six-digit scene id (`000042`).
* The same `<split>/<class>/<id>` identifies the same scene in `clean/`, `part_map/`, `masks/` and in *every* variant.
* Image files are lossless 8-bit RGB PNG at `size x size`.
* Files are written atomically (temporary file, then rename), so an interrupted run never leaves half-written files.

### Storage estimate

```
images        = scenes x (total number of alpha values over all suppressor entries)
approx. size  = images x 0.1 to 0.2 MB     (noisy 256x256 PNG; measure with  du -sh)
```

The reference configuration has 2,500 scenes and 37 alpha entries, which gives 92,500 variant images.

---

## 10. Annotations: classes, concepts and masks

### 10.1 Classes

A class is a fixed combination of one variant for each of the five parts. `classes.json` stores, for every class, the variant index per part. `class_label` in the label file equals the class index.

### 10.2 Concept labels

`labels_<split>.npz` contains one row per valid scene, in the order of `index_<split>.json`. Shapes below use the default `parts.json` (beak 4, eye 3, foot 4, tail 9, wing 6 variants).

| Array | Shape | Type | Description |
|---|---|---|---|
| `ids` | `(N,)` | int64 | Scene id. |
| `class_label` | `(N,)` | int64 | Class index. |
| `concept_vector` | `(N, 26)` | float32 | One-hot per part variant, concatenated in the order beak, eye, foot, tail, wing. All zeros for a part that is missing from the bird. |
| `concept_indices` | `(N, 5)` | int64 | Variant index per part; `-1` if the part is missing. |
| `attribute_vector` | `(N, 22)` | float32 | Factorised concepts (for example `wing:model=wing01`, `wing:color=green`). Only attributes with at least two distinct values are used. The length depends on `parts.json`. |
| `part_present` | `(N, 5)` | bool | The part is part of the bird in this scene. |
| `part_visible_pixels` | `(N, 5)` | int64 | Number of mask pixels per part. |
| `part_visible` | `(N, 5)` | bool | Present *and* above `masks.min_visible_frac` of the image area. |

`labels_meta.json` provides `concept_names`, `attribute_names`, `attribute_spec`, `parts`, `part_dims`, `mask_names`, `informative_parts` and `class_to_concept`, a `(num_classes, 26)` matrix with the canonical (complete) concept vector of every class.

**Present versus visible.** Camera pitch and roll cover the full sphere by default, so a part can be present in the scene but hidden behind the body or too small to see. Use `part_visible` when you want labels that match the image, and `part_present` when you want labels that match the scene. The validation report states the share of present-but-hidden parts.

**Missing parts.** In the train split, with probability 0.5 an image drops a random number (0 to 5, uniform) of random parts. The body is always present. Test images contain all parts. Set `dataset.drop_parts_in_train: false` for complete birds everywhere.

**Informative parts.** A part is *informative* if its variant is not identical across all classes. Only informative parts carry class information. With 10 randomly drawn classes usually all five are informative; with few classes some parts may be constant, and they are then excluded from the `informative` mask.

### 10.3 Masks

`masks/<split>/<class>/<id>.npz` holds boolean `H x W` arrays:

| Mask | Content |
|---|---|
| `beak`, `eye`, `foot`, `tail`, `wing`, `body` | Pixels of the respective part (both instances for eyes, feet and wings). |
| `bird` | Union of all parts including the body. |
| `bg_objects` | 3D distractor objects (empty when `signal.source: foreground`). |
| `bg_canvas` | Pure background (nothing rendered). |
| `informative` | Union of the masks of informative parts (the class-relevant region, excluding the body). |

In `layered` mode the file also contains the soft `<name>_smooth` masks (section 3.6).

The eight masks `beak`, `eye`, `foot`, `tail`, `wing`, `body`, `bg_objects` and `bg_canvas` cover each pixel (almost) exactly once when `dilate_radius` is 0; `bird` and `informative` are unions of them. Masks are computed from the unsmoothed geometry; the image is lightly blurred by the signal filter (about 1.5 px at 256 px), so edges can differ by one or two pixels.

---

## 11. Loading the data

Copy `fb_dataset.py` into the evaluation project. It requires NumPy, Pillow and PyTorch.

```python
from fb_dataset import FunnyBirdsSuppressor, list_variants, list_alphas

root = "datasets/full_v1"
print(list_variants(root))                 # ['corr_s2', 'corr_s4', ..., 'white']
print(list_alphas(root, "corr_s4"))        # [0.05, 0.1, ..., 1.0]

ds = FunnyBirdsSuppressor(root, split="test", variant="corr_s4", alpha=0.2)
item = ds[0]
item["image"]            # (3, H, W) float32 in [0, 1]
item["class_label"]      # scalar long
item["concept_vector"]   # (26,)
item["concept_indices"]  # (5,)
item["attribute_vector"] # (22,)
item["part_present"]     # (5,)
item["part_visible"]     # (5,)
item["masks"]            # (10, H, W) in the order of ds.mask_names
item["alpha"]            # scalar
```

Constructor arguments:

| Argument | Description |
|---|---|
| `root` | Dataset folder (or an exported folder). |
| `split` | `"train"` or `"test"`. |
| `variant`, `alpha` | Select the suppressor dataset. Omit both for an exported folder. |
| `get_masks` | Return the stacked masks (default `True`). |
| `get_part_map` | Also return the part-map image as `uint8` (default `False`). |
| `get_clean` | Also return the noise-free image (default `False`). |
| `mask_names` | Subset and order of masks, for example `["bird", "informative"]`. |
| `transform` | Callable applied to the **image only**. Do not use geometric transforms, since masks are not transformed. |

Useful attributes: `concept_names`, `attribute_names`, `part_names`, `informative_parts`, `class_to_concept`, `mask_names`, `num_classes`, `num_concepts`, `alpha`, `variant_info`.

Smoothed masks (layered datasets) are requested by name; `ds.all_mask_names` lists everything that is stored:

```python
ds = FunnyBirdsSuppressor(root, "test", variant="corr_s4", alpha=0.2,
                          mask_names=["bird", "bird_smooth", "wing", "wing_smooth", "bg_objects"],
                          get_raw=True)        # get_raw: unsmoothed composite image
```

### Sweeping suppressor strength

```python
from torch.utils.data import DataLoader

for alpha in list_alphas(root, "corr_s4"):
    ds = FunnyBirdsSuppressor(root, "test", variant="corr_s4", alpha=alpha)
    loader = DataLoader(ds, batch_size=64, num_workers=4)
    ...  # explain the same scenes at every alpha
```

Because all variants share scenes, `ds[i]` refers to the *same* scene for every variant and alpha, so explanations can be compared image by image.

### Example: fraction of attribution inside a mask

```python
import torch

def attribution_in_mask(attr, mask, eps=1e-12):
    """attr: (B,H,W) non-negative attribution, mask: (B,H,W) binary."""
    return (attr * mask).sum((1, 2)) / (attr.sum((1, 2)) + eps)

bird = ds.mask_names.index("bird")
inside = attribution_in_mask(attr.abs(), batch["masks"][:, bird])
```

### Reading the labels without PyTorch

```python
import json, numpy as np
d = "datasets/full_v1/scenes"
L = np.load(f"{d}/labels_train.npz")
meta = json.load(open(f"{d}/labels_meta.json"))
print(L.files)
print(dict(zip(meta["concept_names"], L["concept_vector"][0])))
```

---

## 12. Validation

```bash
python validate.py --config configs/full.yaml --samples 100
```

| Check | Type | Meaning |
|---|---|---|
| Label and index lengths agree | error | Detects incomplete label files. |
| Class balance | warning | Reports unequal class counts (some scenes can be dropped). |
| Missing part with non-empty mask | error | Concept label and mask must agree. |
| Present-but-hidden parts | reported | Share of present parts with no visible pixels. A few percent is expected from camera angles; a sudden jump indicates incomplete renders (see section 18). |
| Stored masks equal masks re-extracted from the part map | error | Verifies the mask pipeline. |
| Unknown-colour pixels in the part map | warning | Detects anti-aliasing or colour problems (should be about 0). |
| Missing images per variant and alpha | error | Every scene needs an image for every variant and alpha. |
| Bit-exact rebuild | error | Each checked image is rebuilt from the clean image, the stored seed and `variant.json` statistics and compared with the file on disk. A difference means the configuration changed after generation. |
| Saturated pixel fraction | reported | Share of pixels at 0 or 255; high values mean the noise is clipped. |
| Ring correlation | reported | See below. |

**Ring correlation (suppressor strength).** For each checked scene the mean noise inside the bird and the mean noise in a ring around it (gap 1 % and width 8 % of the image size) are computed; the statistic is the Pearson correlation of these two quantities across scenes. For `white` noise it is near zero. Clearly positive values mean that the background carries information about the noise on the bird, that is, suppressor variables exist. It depends only on the noise family, not on alpha, and is noisy for small `--samples`. It is `0` for `exclude_bird` variants by construction.

Outputs in `datasets/<name>/validation/`: `report.json`, `sheet_<variant>.png` (clean image plus one row per alpha) and `masks_overlay.png` (clean image and part masks side by side). The exit code is 1 if any error was found.

---

## 13. Exporting self-contained datasets

The generated layout shares scenes between variants. To hand one dataset (one variant at one alpha) to another project as a single self-contained folder:

```bash
python generate.py --config configs/full.yaml --stage variants \
                   --export ./export --export-mode hardlink
```

This creates `export/<variant>__alpha_<a>/`:

```
export/corr_s4__alpha_0.200/
├── images/<split>/<class>/<id>.png
├── scenes/            part maps, masks, labels, clean images, metadata
├── variant.json
├── classes.json
└── parts.json
```

Load it without `variant`/`alpha`:

```python
ds = FunnyBirdsSuppressor("export/corr_s4__alpha_0.200", split="test")
```

| `--export-mode` | Behaviour |
|---|---|
| `hardlink` (default) | No additional disk space on the same filesystem; falls back to copying across filesystems. Do not edit files in place. |
| `symlink` | Absolute symbolic links; the original dataset must stay where it is. |
| `copy` | Independent copy, uses additional disk space. |

---

## 14. Performance, parallelism and resuming

### 14.1 Parallelism

`render.workers` sets the size of a thread pool. In stage 1 it controls how many render requests are in flight at once; in stage 2 it controls how many images are blended concurrently. All requests go to a single Chromium instance that rasterises on the CPU, so more workers than CPU cores do not help. A reasonable setting is half of the available hardware threads (`nproc`).

With many workers, a page may need longer to load the 3D models than the default wait time, which can produce birds with missing parts. Increase the wait through the environment (the server inherits it):

```bash
RENDER_WAIT_MS=600 python generate.py --config configs/full.yaml
```

### 14.2 Measured reference

On a workstation with a 24-thread CPU, `render.workers: 12`, `size: 256`, `supersample: 2`:

| Stage | Throughput | Time for the reference configuration |
|---|---|---|
| Stage 1, scenes (`default`) | about 1.4 scenes/s | about 30 minutes for 2,500 scenes |
| Stage 1, scenes (`layered`) | not measured | expect about twice as long, because four renders per scene are needed |
| Stage 2, variants | about 10.7 scenes/s (37 images each) | about 4 minutes |

Stage 1 dominates the cost. Stage 2 is cheap, so exploring additional suppressor settings is fast.

### 14.3 Resuming and incremental changes

All files that already exist are skipped, so re-running the same command continues an interrupted run. Use this table to decide what to do after changing a setting:

| Change | Action |
|---|---|
| `render.workers`, `render.timeout`, `RENDER_WAIT_MS` | Safe at any time. |
| Add a new entry to `suppressors`, or new alpha values | `python generate.py --config ... --stage variants`. Existing images are kept. |
| Change the *parameters* of an existing entry (for example `sigma_frac`) | Give the entry a new `name` (or delete its folder under `variants/`); images with an existing file name are not regenerated. |
| Change label logic (`scenes.py`, `masks.min_visible_frac`) | `python generate.py --config ... --stage scenes`. Rendering is skipped for existing scenes and the labels are rewritten (this still starts the render server). |
| Change `render.size`, `render.supersample`, `signal.*`, `masks.dilate_radius`, `dataset.seed`, class or sample counts | Use a new `dataset.name`. |

If `signal.*` changes and `norm_stats.json` is recomputed, previously generated variant images no longer match the new statistics; `validate.py` reports this as a rebuild error.

### 14.4 Stopping

```bash
pkill -f generate.py; pkill -f "node server.js"
```

---

## 15. Reproducibility

* Classes and scene parameters are drawn from NumPy generators seeded by `dataset.seed` and saved in `classes.json` and `params_<split>.json`. Existing files are reused on later runs.
* Each scene stores its own `noise_seed`. Noise is a deterministic function of the seed, the family parameters and the image size, for a fixed NumPy/SciPy version.
* The final images of stage 2 can be rebuilt bit-exactly from the clean image, the seed and `variant.json`. `validate.py` verifies this.
* Stage 1 images depend on the renderer (browser, software rasteriser, three.js). The scene *parameters* are reproducible; pixel-identical renders on a different machine or browser version are not guaranteed.
* `config_used.yaml` stores the fully resolved configuration of the last run. Keep it together with the dataset.

---

## 16. Extending the generator

### Add a new noise family

1. In `suppressors.py` write a function `_myfamily(shape, rng, p)` that returns a float32 array of shape `(H, W, 3)` with zero mean. `p` is the YAML entry, so extra keys are available as `p.get(...)`. Scaling to unit variance is automatic.
2. Register it: `FAMILIES['myfamily'] = _myfamily`.
3. Optionally extend `default_name` for readable folder names.
4. Use it in a YAML entry and run `python generate.py --config ... --stage variants`.

A family that must be applied multiplicatively also needs to be added to `MULTIPLICATIVE` and handled in `compose`.

### Add or change a label format

Edit `scenes.build_label_info` and `scenes.sample_labels`, add the array in `generate.build_index_and_labels` if it needs mask information, expose it in `fb_dataset.__getitem__`, and re-run `--stage scenes` (no re-rendering).

### Use a different bird definition

Edit `Clones/funnybirds/render/parts.json` (or point `dataset.parts_json` to another file). Dimensions, concept names and attributes are derived from it. The part names and the part-map colours (`PART_COLORS` in `scenes.py`) must match the renderer.

---

## 17. Design decisions and known limitations

* **Alpha is a dataset-level quantity.** Noise is scaled with fixed constants rather than normalised per image, so noise energy varies naturally between images. Comparisons across alpha are therefore consistent.
* **`alpha = 1` is not identical to the clean image.** The signal filter H is applied to the signal before mixing. Set `signal.sigma_smooth_frac: 0` for exact pass-through.
* **Correlation length is relative.** `sigma_frac` is a fraction of the image size, so the same value gives the same visual structure at any resolution.
* **Suppressor strength is a property of the noise family and `sigma_frac`.** Alpha controls how much of the noise enters the image. Both should be varied.
* **Ground truth definition.** The masks mark where the *object* is. Which pixels a model actually needs is model-dependent: a model may use only a subset of the informative parts, and, with suppressors, also background pixels. The `informative` mask restricts the object ground truth to parts that differ between classes.
* **Distractor colours.** Distractor objects use red, green, blue and yellow, as in FunnyBirds; some of these also occur on bird parts.
* **Random camera.** The full-sphere camera produces birds seen from any direction, so some parts are occluded. This is intentional; use `part_visible` to account for it.
* **Layered mode is only as tested as the renderer allows.** Its pipeline was verified end to end with a synthetic renderer (occlusion, masks, rebuild check); the real `black` render mode in `page.html` should be checked once with `layered_test.yaml` and `masks_overlay.png`. Distractor edges can show a faint dark rim because they are anti-aliased against black in the image render while the mask is hard.
* **`saturated` in the validation table** counts pixels equal to 0 or 255, so it is high at `alpha = 1` for black-background signals (the black pixels themselves). Read it for the noisy alphas only.
* **Multiplicative illumination** needs a signal with a non-black background to be informative, and no nominal SNR is defined for it.
* **Single image size per dataset.** Different resolutions require different dataset names (and a new stage 1).
* **No model training or XAI evaluation in this repository.** Metrics such as importance mass accuracy or earth mover's distance are to be implemented in the evaluation project using the masks.

---

## 18. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `node_modules not found - run "npm install"` | Run `npm install` in `Clones/funnybirds/render`. |
| `bird models not found in .../js` | Restore with `git checkout -- render/js` inside `Clones/funnybirds`. |
| Puppeteer cannot find a browser | Set `render.chromium_path` (for example `/usr/bin/chromium`) in the YAML. |
| `render server did not come up within 40 s` | Read `Clones/funnybirds/render/server.log`. Check that port 8081 is free (`ss -ltnp \| grep 8081`). |
| New settings (wait time, patch) seem ignored | An older render server is still running. Stop it with `pkill -f "node server.js"` and start again. |
| The log is empty at the start | Normal: the first progress line appears after 25 scenes. Use `python -u` to avoid buffering. |
| Only `scenes/` exists, no `variants/` | Stage 1 is still running. `variants/` is created after all scenes of all splits are rendered. |
| `bird not visible after N renders - dropped` | The scene was dropped. Occasional drops are harmless; frequent ones indicate overload (lower `workers`, raise `RENDER_WAIT_MS`). |
| High present-but-hidden percentage in `validate.py` | Renders are incomplete because the page was captured too early. Raise `RENDER_WAIT_MS`, lower `render.workers`, and regenerate with a new `dataset.name`. |
| `images cannot be rebuilt from clean image + seed` | The configuration changed after generation (for example the signal settings or a suppressor parameter). Regenerate the affected variant under a new name. |
| Layered mode: distractors look black or missing in `clean/` images | The second patch is not active, so the `black` render mode is unknown to the page. Apply `render_patch_layered.diff`, restart the render server, and regenerate under a new `dataset.name`. |
| `ModuleNotFoundError: yaml` | `pip install pyyaml` in the active environment. |
| `ModuleNotFoundError: torch` | Only `fb_dataset.py` needs PyTorch. Install it in the evaluation environment. |
| Files appear in an unexpected folder | The output goes next to the `config.py` that was run (`datasets/` in the project root). Keep a single copy of the project. |
| Disk full | Estimate with section 9, reduce the number of alphas or samples, or point `dataset.out_dir` to a larger disk. |
| Shell reports `command not found` for prompt lines | Only paste the commands themselves, not the prompt or output. |

---

## 19. References and acknowledgements

This project builds on the FunnyBirds renderer (Apache License 2.0, included under `Clones/funnybirds`) and follows the data-generation principle of XAI-TRIS. It is an independent work and is not affiliated with the authors of either project. Please comply with the licences of the included third-party components (FunnyBirds, three.js, puppeteer) and of any background image collection you use.

* R. Hesse, S. Schaub-Meyer, S. Roth. **FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods.** ICCV 2023. https://github.com/visinf/funnybirds

  ```bibtex
  @inproceedings{Hesse:2023:FunnyBirds,
    title     = {Funny{B}irds: {A} Synthetic Vision Dataset for a Part-Based Analysis of Explainable {AI} Methods},
    author    = {Hesse, Robin and Schaub-Meyer, Simone and Roth, Stefan},
    booktitle = {2023 {IEEE/CVF} International Conference on Computer Vision (ICCV), Paris, France, October 2-6, 2023},
    year      = {2023}
  }
  ```

* B. Clark, R. Wilming, S. Haufe. **XAI-TRIS: Non-linear image benchmarks to quantify false positive post-hoc attribution of feature importance.** arXiv:2306.12816. https://github.com/braindatalab/xai-tris
* R. Wilming, C. Budding, K.-R. Mueller, S. Haufe. **Scrutinizing XAI using linear ground-truth data with suppressor variables.** Machine Learning, 2022.
* S. Haufe, F. Meinecke, K. Goergen, et al. **On the interpretation of weight vectors of linear models in multivariate neuroimaging.** NeuroImage 87:96-110, 2014.

**License.** Add a licence file for the generator code in this repository. The code in `Clones/funnybirds` remains under its own Apache 2.0 licence.
