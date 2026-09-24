import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("flow_steps", [None, 7])
@pytest.mark.parametrize("reuse_logits", [False, True])
@pytest.mark.parametrize(
    "sampling_mode,composition_flag",
    [(None, None), ("greedy", None), ("n-shot", "--no-enforce-composition")],
)
def test_sampling_steps_do_not_require_checkpoint_config(
    tmp_path, flow_steps, reuse_logits, sampling_mode, composition_flag
):
    expected_steps = 100 if flow_steps is None else flow_steps
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    result_dir = run_dir / "mp20_test_set"
    suffix = f"-{sampling_mode}" if sampling_mode == "greedy" else ""
    sample_path = result_dir / f"top-1_flow_steps-{expected_steps}{suffix}"
    logits_path = Path(f"{sample_path}.logits.pt")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    result_dir.mkdir()
    logits_path.touch()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "uv-calls.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$UV_CALL_LOG"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["UV_CALL_LOG"] = str(call_log)
    command = [
        str(repo_root / "scripts/sample_and_eval_gwa.sh"),
        str(checkpoint),
        "1",
        "mp20_test_set",
    ]
    if flow_steps is not None:
        command.append(str(flow_steps))
    if reuse_logits:
        command.append("--reuse-logits")
    if sampling_mode:
        command.extend(["--sampling-mode", sampling_mode])
    if composition_flag:
        command.append(composition_flag)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert calls[0].startswith("run python scripts/sample_wy.py")
    assert f"--save_path {sample_path}" in calls[0]
    if sampling_mode:
        assert f"--sampling-mode {sampling_mode}" in calls[0]
    if composition_flag:
        assert composition_flag in calls[0]
    if reuse_logits:
        assert "Skipping GPU flow" in completed.stdout
        assert f"--logits_path {logits_path}" in calls[0]
        assert "--reuse-logits" in calls[0]
        assert "--num-samples 1" in calls[0]
        assert "--formula_file" not in calls[0]
    else:
        assert f"--flow_steps {expected_steps}" in calls[0]
        assert "--formula_file example/input_test.csv" in calls[0]
        assert "--reuse-logits" not in calls[0]
    assert "--cpu-workers 52" in calls[0]
    assert calls[1].startswith("run python scripts/extract_wyckoff_samples.py")
    assert calls[2].startswith("run python scripts/eval_gwa.py")
