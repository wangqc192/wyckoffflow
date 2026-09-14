# WyckoffFlow Lightning

This repository trains composition-conditioned Wyckoff flow and space-group
models with Hydra and PyTorch Lightning.

## Training

Install the locked environment:

~~~bash
uv sync --frozen
~~~

Run a short pipeline check on data/mini:

~~~bash
uv run python -m models.run \
  data=mini \
  train.trainer.fast_dev_run=true \
  logging=none
~~~

The default data path is `data/mp20`. The first run parses each CSV and writes a
same-name `.pt` cache next to it; later runs reuse that cache while it is newer
than the CSV.

Train the discrete flow model:

~~~bash
uv run python -m models.run experiment=discrete_flow
~~~

The full MP20 configuration uses prototype-frequency resampling with
`prototype_sampling_alpha=0.25` to reduce the dominance of frequent structural
prototypes. Set `data.datamodule.prototype_sampling_alpha=0` to run the original
uniform-sampling baseline; `0.5` is the stronger inverse-sqrt comparison.

Train the space-group model:

~~~bash
uv run python -m models.run experiment=space_group
~~~

Hydra overrides can adjust any setting:

~~~bash
uv run python -m models.run \
  experiment=discrete_flow \
  data.datamodule.batch_size.train=128 \
  model.model_config.composition_encoder_dim=128 \
  train.trainer.max_epochs=10
~~~

Runs are stored below outputs/<date>/<time>_<experiment>/. Each run contains
the resolved hparams.yaml, Hydra metadata, CSV logs, and checkpoints/ with
best.ckpt, last.ckpt, and periodic epoch checkpoints.

Resume the full Lightning state with:

~~~bash
uv run python -m models.run \
  experiment=discrete_flow \
  resume_from=/absolute/path/to/last.ckpt
~~~

Resumed training continues writing checkpoints to the directory containing the
selected checkpoint.

Load either model checkpoint for inference:

~~~python
from models.common.checkpoint import load_model

model, _, config = load_model("outputs/<date>/<run>", device="cuda")
~~~

Plot the epoch-level training and validation loss for a run:

~~~bash
uv run python scripts/plot_loss.py outputs/<date>/<run>
~~~

The script also accepts a direct `metrics.csv` path. Use `--output` to select
the output PNG path; otherwise it writes `loss.png` next to `metrics.csv`.

Extract human-readable Wyckoff samples from a sampling output:

~~~bash
uv run python scripts/extract_wyckoff_samples.py \
  --input_pt /path/to/samples.pt \
  --output_csv /path/to/samples.csv
~~~

The CSV contains the generated formula, conditioned formula, space group,
and occupied Wyckoff sites. The `wyckoff_occupancy` value uses the form
`space_group_element_countxWyckoff_site`, for example
`220_Li2x1a_Li3x2b_Ca1x16c`. Equivalent occupancy strings with different
component ordering are canonicalized, and duplicate occupancies for the same
`target_index` are kept once. The `count` column records how many generated
samples were merged into each row.

An input example for formula-conditioned sampling is provided at
`example/example.csv`:

The examples use real material space groups and complete conventional-cell
formula counts (they are not reduced formulas).

The 50-target example generated from
`/home/wangqc/NextCrystal/examples/shotgunII/*.vasp` is available as
`example/shotgunII.csv`.

~~~bash
uv run python scripts/sample_wy.py \
  --model_path /path/to/checkpoint_or_run \
  --formula_file example/shotgunII.csv \
  --num_evals 500 \
  --batch_size 1 \
  --save_path outputs/example
~~~

`--num_evals` applies the same sampling count to every formula; it is not stored
in the formula CSV.

`--flow_steps` optionally overrides the checkpoint's inference step count for one
sampling run. If omitted, the checkpoint value is used and the effective value is
stored in the output `.pt` file. The Bash pipeline accepts this as its optional
sixth argument; use a different `RESULT_NAME` for each step-count comparison.

Sampling enforces the conditioned composition by default. Use
`--no-count_conserving` to disable this repair step.
If a formula and space group cannot realize the requested complete composition,
the sampling script reports that pair and skips it while continuing with the
remaining records.

To prepare the full MP20 test split as sampling input, use the conventional CIF
stored in `cif.conv`; this preserves complete conventional-cell formula counts:

~~~bash
uv run python scripts/prepare_test_input.py \
  --input_csv data/mp20/test.csv \
  --output_csv example/input_test.csv
~~~

`target_index` preserves the source row number from `test.csv` through sampling
and extraction, so records with the same formula and space group are not mixed.
The evaluator expands the equivalent Wyckoff settings stored in
`wyckoff_spglib`; component order does not affect a match. Evaluate G-W-A Top-K
matches with:

~~~bash
uv run python scripts/eval_gwa.py \
  --target_path data/mp20/test.csv \
  --gen_path results/test/top-1.csv \
  --top_k 1 \
  --output_path results/test/top-1_gwa.csv \
  --summary_path results/test/top-1_gwa.json
~~~

Run sampling, extraction, and evaluation together by passing the model and one
sampling count to the Bash pipeline:

~~~bash
scripts/sample_and_eval_gwa.sh \
  outputs/2026-09-10/10-27-26_discrete_flow/checkpoints/best.ckpt \
  20 \
  test
~~~

The third argument is the output folder name under the checkpoint's training
run directory. For the command above, whose checkpoint uses 100 flow steps, this
writes the raw samples to
`outputs/2026-09-10/10-27-26_discrete_flow/test/top-20_flow_steps-100.pt`, the
generated templates to the corresponding `.csv`, and the per-target evaluation
to `top-20_flow_steps-100_gwa.csv`. The G-W-A Top-K match-rate summary is written
to `top-20_flow_steps-100_gwa.json` in the same folder, including the matched and
total material counts. Passing the training run directory instead of a checkpoint uses
the same output location. The optional positional arguments are `INPUT_CSV`,
`TARGET_CSV`, and `FLOW_STEPS`, in that order. The same `example/input_test.csv`
can be reused
for any Top-K because the Bash argument is passed to
`sample_wy.py --num_evals` and `eval_gwa.py --top_k`.

The three data/mini splits contain the same 64 examples and are intended only
for pipeline checks; use `data=mini` for that check. The default `data=mp20`
configuration points to the full training data.
