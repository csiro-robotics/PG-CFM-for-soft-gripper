# Checkpoints

Not tracked in git (see `.gitignore`). Two are needed:

| file | produced by | used by |
|---|---|---|
| `condition_encoder.pt` | `training/run_training.sh encoder` | stage 2 of training |
| `picfm_dit.pt` | `training/run_training.sh dit` | `evolution/` — decodes a genome into a design |

Stage 2 writes into `picfm/<run_name>/`, as `checkpoint_<iteration>.pt` every
`save_every` steps plus `model_final.pt` at the end. Point the search at whichever
one you want:

```bash
# from the repo root — pick the final model, or any intermediate checkpoint
ln -s picfm/run_dit_v2/model_final.pt checkpoints/picfm_dit.pt
```

`evolution/` looks for `checkpoints/picfm_dit.pt` by default. To use a different file
without the symlink, give a path relative to where you run the script:

```bash
cd evolution
CHECKPOINT=../checkpoints/picfm/run_dit_v2/model_final.pt ./run_evolution.sh
```

To reproduce the paper's designs without retraining, download the released checkpoints
from the CSIRO Data Access Portal (<https://doi.org/10.25919/j8yy-c617>) and place the
DiT at `checkpoints/picfm_dit.pt`.

The released `condition_encoder.pt` is the encoder the released DiT was trained
against — its 20 tensors are bit-identical to the `cond_encoder.*` weights inside
`picfm_dit.pt`. Note that stage 1 selects its checkpoint on design-space
classification accuracy alone, which saturates before `--fusion_start` enables the
fusion path, so the fusion weights in the released encoder are at initialisation.
