# Beyond Representations: Flow-based Generative Design of Soft Grippers

Physics-Informed Conditional Flow Matching for the generative design of soft
robotic gripper fingers, with quality-diversity search over the generative
latent and grasp evaluation by a differentiable-contact FEM solver.

<p align="center">
  <img src="Figure/overview.gif"
       alt="Pipeline overview: the 128x64 design space with its boundary conditions; conditional flow matching composing fin-ray, graph-optimised and topology-optimised features into a single design; the resulting two-finger gripper evaluated by 2D FEA von Mises stress while grasping a disc; and the 3D-printed gripper performing the same grasp in hardware.">
  <br>
  <em>Design space and boundary conditions &rarr; compositional flow matching over fin-ray /
  graph-optimised / topology-optimised features &rarr; FEA grasp evaluation &rarr; hardware validation.</em>
</p>

<p align="center">
  <img src="Figure/Overview.png"
       alt="Method overview in five panels: (a) the 128x64 pixel design space and its boundary conditions; (b) the three source design representations - topology-optimised density fields, graph-based designs and parametric fin-ray designs; (c) compositional generative design, where a Voronoi design condition and Gaussian noise are carried by conditional flow matching into a design composed of topology-optimised, graph-based, fin-ray and unconditional features; (d) an evolutionary algorithm searching the conditions and image tokens of the flow model to fill an archive over strut thinness, branch density and material fraction; (e) four elite designs validated by FEA von Mises stress and on the 3D-printed gripper.">
  <br>
  <em><b>a</b> Design space and boundary conditions. <b>b</b> Source design representations:
  topology-optimised, graph-based and fin-ray. <b>c</b> Compositional generative design by
  conditional flow matching. <b>d</b> Evolutionary search over the model's conditions and image
  tokens. <b>e</b> Validation of elite designs in FEA and hardware.</em>
</p>

The repository contains: training, generation, evolution and
simulation, with no external project dependencies.

```
training/     stage 1 condition encoder -> stage 2 PI-CFM DiT
evolution/    Voronoi+token genome -> CFM -> design -> implicit FEM grasp -> 3-D CVT MAP-Elites
```

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

A CUDA GPU is required: the FEM solver runs in NVIDIA Warp and the DiT trains on GPU.
Tested on Python 3.12 / CUDA 12.4 / RTX A5000 and H100.

## 2. Data and checkpoints

Neither is tracked in git — see [`data/README.md`](data/README.md) and
[`checkpoints/README.md`](checkpoints/README.md).

```
data/mechanism_dataset/   (topopt)
data/finray_dataset/      (finray)
data/graph_dataset/       (graph)
```

## 3. Training

Two stages, **in order** — the DiT is trained against a frozen, pre-trained
condition encoder, so stage 2 without stage 1 trains against random features.

```bash
cd training
./run_training.sh              # both stages
./run_training.sh encoder      # stage 1 only  -> checkpoints/design_encoder/encoder_best.pt
./run_training.sh dit          # stage 2 only  -> checkpoints/picfm/<run_name>/
```

| stage | script | what it trains |
|---|---|---|
| 1 | `pretrain_encoder.py` | `SpatialStyleEncoder`: BC-CNN reconstruction + design-space classification + fusion |
| 2 | `train_picfm_dit.py`  | Lumina-DiT flow matching with the mechanism-FEM physics loss |

The launcher is written for a **single local desktop GPU** (no SLURM). Batch
sizes default lower than the published run; override via the environment:

```bash
DATA_ROOT=../data ENC_EPOCHS=150 DIT_BATCH=64 ./run_training.sh
```

Stage-2 hyper-parameters live in [`training/configs/train_dit.yaml`](training/configs/train_dit.yaml);
any scalar can be overridden on the command line (`--lr`, `--total_steps`, …).

## 4. Evolution

Quality-diversity search over the generative model's own inputs. A genome is

```
genome = [ theta (5K) | z_token (512) ]                      D = 572 at K = 12

theta     2K Voronoi site positions + 3K per-site class logits (topopt / graph /
          finray). Rasterised to a hard, pure-class RGB image -> the CONDITION.
z_token   one noise value per DiT token (32 x 16), upsampled to the pixel grid ->
          the flow's starting x0. Not a condition: the search explores the
          generator's conditioning and its noise jointly.
```

decoded by the trained PI-CFM into a 128 x 64 binary design, then evaluated by
closing one soft finger on a rigid disc and lifting it.

```bash
cd evolution
./run_evolution.sh smoke        # 2 tiny iterations — checks the whole pipeline
./run_evolution.sh              # the full run  -> ../runs/me_<timestamp>/
```

**Archive.** 3-D CVT MAP-Elites, 2000 cells, LM-MA-ES emitters (64 x 32 = 2048
designs per iteration). Descriptor axes:

| axis | meaning |
|---|---|
| `strut_complexity` | fraction of material in thin struts (<= 2 px half-thickness) |
| `branch_density`   | skeleton branch-points per unit skeleton length |
| `material_fraction`| filled fraction of the design space |

**Objective.** `1 + novelty + 0.5 * (0.2 * tanh(pull_off / f_ref) + 0.8 * wrap)`,
zero for an invalid design. Novelty dominates on purpose: the run is meant to
illuminate the space of morphologies, not to hill-climb one grasp.

**Physics.** Implicit backward-Euler PNCG with IPC log-barrier contact,
self-collision and friction:

| | |
|---|---|
| material | E = 1.9 MPa, mu = 0.343 |
| step | dt = 2 ms |
| close | 1.10 s, 10 mm cosine-eased stroke |
| pull | 3.03 s over 30 mm |
| object | rigid disc, r = 14 mm |
| opening | 52 mm two-finger equivalent (one finger sees 26 mm to the disc centre) |

Everything is pinned in [`evolution/evaluate.py`](evolution/evaluate.py).

```
evolution/
  genome -> design      generate/   PI-CFM DiT, Voronoi rasteriser, per-token CFG
  design -> metrics     sim/        implicit PNCG-IPC solver, socket + mesh builder
  descriptors.py        the three archive axes
  evaluate.py           the pinned operating point + composite score
  map_elites.py         CVT archive, LM-MA-ES emitters, the ask/tell loop
```

---

## Licence

CSIRO Open Source Software Licence Agreement (a variation of the BSD/MIT
licence) — see [`LICENSE`](LICENSE).

## Citation

*(to be added on publication)*
