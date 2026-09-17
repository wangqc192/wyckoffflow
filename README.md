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
fourth positional argument; use a different `RESULT_NAME` for each step-count
comparison.

Sampling enforces the conditioned composition by default. With
`count_conserving` enabled, the GPU phase runs all input batches first and writes one
intermediate `.logits.pt` payload containing each batch's final masked `zero_logits`
and `inf_logits`. Only after that file has been completely written does CPU DP begin.
The decoder processes the saved batches with one shared process pool, uses prefix
slices instead of repeatedly scanning the full merged graph array, and shows a
per-graph progress bar. Logit tensors are shared with workers rather than copied once
per graph. The default is 52 CPU worker processes; override it with
`--cpu-workers N`. Use `--no-count_conserving` to disable this final
composition-conserving sampling step. If a formula and space group cannot realize the
requested complete composition, the sampling script reports that pair and skips it
while continuing with the remaining records.

There are two sampling modes. The default `n-shot` mode keeps the original
with-replacement behavior: every target runs the GPU flow `--num_evals` times and
each final graph is repaired independently with stochastic CPU DP, so duplicate
templates are allowed. Use `top-n` when every target must produce distinct
templates:

~~~bash
scripts/sample_and_eval_gwa.sh \
  /path/to/checkpoint_or_run \
  20 \
  test \
  --sampling-mode top-n
~~~

In `top-n` mode each target runs the GPU flow once and saves its final masked
`zero_logits` and `inf_logits`; after all GPU batches have been saved, CPU DP
returns up to the requested number of distinct exact-composition templates. The
decoder expands its fixed-site beam when necessary, and never copies an existing
template to fill a missing Top-K slot. If the feasible template space is smaller
than K, the output reports and contains the actual number of unique templates.
The raw logits and output names include the `-top-n` suffix so they do not collide
with `n-shot` results. `--topn-beam-size` controls the initial CPU DP beam.

The equivalent direct entry-point options are `--sampling-mode n-shot` and
`--sampling-mode top-n`; `--sample-mode` is an accepted alias. Both modes use the
same exact composition constraint and default to 52 CPU worker processes, which
can be changed with `--cpu-workers N`.

Top-N ranking is performed by the CPU dynamic program over the final masked logits.
It ranks complete exact-composition assignments rather than independently ranking
Wyckoff variables, so every returned candidate is a valid complete template.

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
the same output location. The optional positional arguments are `FLOW_STEPS`,
`INPUT_CSV`, and `TARGET_CSV`, in that order. The same `example/input_test.csv`
can be reused for any Top-K because the Bash argument is passed to
`sample_wy.py --num_evals` and `eval_gwa.py --top_k`.

To skip GPU flow and reuse the matching
`top-<K>_flow_steps-<N>.logits.pt` already present in the result folder, append
`--reuse-logits`:

~~~bash
scripts/sample_and_eval_gwa.sh \
  outputs/2026-09-10/10-27-26_discrete_flow/checkpoints/best.ckpt \
  1 \
  mp20_test_set \
  1 \
  --reuse-logits
~~~

The pipeline finds the logits file from `NUM_EVALS` and `FLOW_STEPS`, skips model
loading and GPU inference, reruns only CPU DP repair, then extracts the corrected
CSV and evaluates it. If `FLOW_STEPS` is omitted and exactly one matching Top-K
logits file exists in the result folder, that file is selected automatically. If
multiple matching files exist, pass `FLOW_STEPS` to select one. The alias
`--skip-sample-logits` has the same behavior. CPU repair uses 52 workers by default;
for example, append `--cpu-workers 32` to use 32 processes. The flag applies both to
fresh sampling and `--reuse-logits` runs.

## Formula-to-structure generation

Generate concrete CIF and POSCAR candidates directly from one complete chemical
formula:

~~~bash
uv run python scripts/generate_structure.py \
  --formula Ga4Te4 \
  --flow-checkpoint outputs/<date>/<run>/checkpoints/best.ckpt \
  --output-dir outputs/formula_generation/Ga4Te4
~~~

The command runs one integrated chain:

1. the released NextCrystal predictor masks space groups whose Wyckoff
   multiplicities cannot realize the requested complete composition, then ranks
   the remaining Space-group Top-K;
2. this repository samples composition-conserving Wyckoff templates separately
   under each ranked space group and removes duplicate complete
   `G-W-A-W-A-...` sequences. The default is `n-shot`; pass
   `--sampling-mode top-n` to run one GPU flow per space-group condition and
   obtain distinct CPU-DP candidates without replacement;
3. the symmetry-aware model from `/home/wangqc/DiffCSP` expands each template
   into lattice parameters and fractional coordinates;
4. the standardized samples are exported to both CIF and POSCAR.

The defaults use these external model assets without copying checkpoints or the
external repositories into this repository:

- `/home/wangqc/NextCrystal/artifacts/mp_20/spacegroup.ckpt`;
- the checkpoint supplied through `--flow-checkpoint`;
- `/home/wangqc/DiffCSP/ckpt/CSP-mp20-sym`, sampled with 100 integration
  steps, coordinate annealing slope 5, and batch size 50.

Use `--nextcrystal-root`, `--nextcrystal-checkpoint`, `--diffcsp-repo`,
`--diffcsp-checkpoint`, and `--diffcsp-python` to override those locations.
`--space-group-top-k`, `--templates-per-space-group`, and
`--template-pool-size` control the hierarchy. The input is treated as a complete
conventional-cell composition and is never reduced; for example, `Ga4Te4` stays
`Ga4Te4` rather than becoming `GaTe`. Some formula/space-group pairs are not
Wyckoff-realizable, and duplicate templates are removed, so the final candidate
count can be smaller than `space-group Top-K × templates per space group`; neither
sampling mode pads the result by repeating a template.

Use `--prepare-only` to stop after the first two stages. The output directory
contains:

- `space_groups.csv`: ranked NextCrystal predictions;
- `templates.csv` and `templates.pt`: selected Wyckoff templates;
- `run.json`: formula, random seed, checkpoints, backend, and effective sampling
  parameters needed to reproduce the run;
- `diffcsp_templates.csv`: DiffCSP's conventional
  `formula,num_evals,pressure,wyckoff` input representation;
- `diffcsp_queries.json`: the same templates as structured symmetry queries;
- `diffcsp_sample.pt`: standardized DiffCSP samples;
- `structures.csv`, `cif/*.cif`, and `poscar/*.vasp`: indexed final structures.

DiffCSP++ remains available as an alternative backend:

~~~bash
uv run python scripts/generate_structure.py \
  --formula Ga4Te4 \
  --flow-checkpoint outputs/<date>/<run>/checkpoints/best.ckpt \
  --output-dir outputs/formula_generation/Ga4Te4_diffcsppp \
  --structure-backend diffcsppp
~~~

This backend always invokes `scripts/run_nextcrystal_diffcsppp.py sample` with
`--batch-size 128 --num-shards 4`. All four shards are produced and merged;
small formula-only jobs therefore include valid empty shard files. Override its
external assets with `--diffcsppp-repo`, `--diffcsppp-checkpoint`, and
`--diffcsppp-python`.

Evaluate DiffCSP structures generated from the unique MP20 test Top-20
templates in two stages. First convert the extracted templates to DiffCSP's
`wyckoff_info.csv` format while retaining an index manifest:

~~~bash
uv run python scripts/prepare_diffcsp_templates.py \
  --input outputs/<run>/mp20_test_set/top-20.csv \
  --output outputs/<run>/mp20_test_set/diffcsp_top20/wyckoff_info.csv \
  --manifest outputs/<run>/mp20_test_set/diffcsp_top20/manifest.csv
~~~

For parallel sampling, `--start-index` and `--limit` can write contiguous input
shards. Pass the resulting sample `.pt` files to `--samples` in index order.

Generate one structure per unique template with the symmetry-aware DiffCSP
sampler and the MP20 checkpoint settings (`N=100`, coordinate annealing slope
5, batch size 50):

~~~bash
scripts/run_diffcsp_template_shard.sh \
  /path/to/diffcsp/python \
  0 \
  /path/to/DiffCSP/symmetry-worktree \
  /path/to/DiffCSP/ckpt/CSP-mp20-sym \
  outputs/<run>/mp20_test_set/diffcsp_top20/wyckoff_info.csv \
  outputs/<run>/mp20_test_set/diffcsp_top20/samples \
  outputs/<run>/mp20_test_set/diffcsp_top20/sample.log
~~~

Evaluate all generated structures against `cif.conv`:

~~~bash
/path/to/diffcsp/python scripts/eval_diffcsp_templates.py \
  --samples outputs/<run>/mp20_test_set/diffcsp_top20/samples.pt \
  --manifest outputs/<run>/mp20_test_set/diffcsp_top20/manifest.csv \
  --targets data/mp20/test.csv \
  --diffcsp-scripts /path/to/DiffCSP/scripts \
  --output outputs/<run>/mp20_test_set/diffcsp_top20/matches.csv \
  --summary outputs/<run>/mp20_test_set/diffcsp_top20/summary.json \
  --top-k 20 \
  --workers 32
~~~

The reported DiffCSP StructureMatcher Top-20 match rate is material-level: a
test material is a hit when at least one of its generated unique Top-20
templates passes DiffCSP's composition and structure validity checks and
matches. The summary also includes the raw StructureMatcher rate without those
validity filters. The matcher uses `ltol=0.3`, `stol=0.5`, `angle_tol=10`,
primitive-cell comparison, and volume scaling.

The three data/mini splits contain the same 64 examples and are intended only
for pipeline checks; use `data=mini` for that check. The default `data=mp20`
configuration points to the full training data.
