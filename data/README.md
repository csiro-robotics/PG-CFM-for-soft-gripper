# Datasets

Not tracked in git — download from the CSIRO Data Access Portal:
<https://doi.org/10.25919/j8yy-c617>. Three condition datasets are expected:

```
data/topopt_dataset/   # topopt
data/finray_dataset/   # finray
data/graph_dataset/    # graph
```

Each holds 3000 `data_XXXXX.npz` files with a `geometry` field (128 x 64, binary)
plus the boundary-condition channels the encoder consumes: `BCx`, `BCy`,
`inputx`, `inputy`, `outputx`, `outputy`, and the scalars `objective_value`,
`volume_fraction`, `penal`, `rmin`, `disp_mag`. Paths are set in
`training/configs/train_dit.yaml` and can be overridden with `DATA_ROOT`.

Two conventions worth stating, because they decide the sign of the physics loss:

* `inputx`/`inputy` are prescribed **displacements** (equal in magnitude to
  `disp_mag`), not forces — they enter the FEM solve as non-zero Dirichlet BCs.
* `outputx`/`outputy` are `-1` at the output pixel, meaning *maximise* motion in
  that direction. The output DOF is clamped and the mechanism loss is
  `output_sign * R_out`, which a good mechanism drives negative.

All three datasets share one boundary-value problem: the BCs sit on row 127,
columns 0-4, the prescribed input on row 127, columns 59-63, and the output at
pixel (0, 0). Only the geometry and the scalars vary between samples.
`objective_value` is defined differently by each generator and is **not**
comparable across the three datasets.
