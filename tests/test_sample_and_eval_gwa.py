import os
import subprocess
from pathlib import Path


def test_reuse_logits_skips_gpu_flow(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    result_dir = run_dir / "mp20_test_set"
    logits_path = result_dir / "top-1_flow_steps-7.logits.pt"
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
    completed = subprocess.run(
        [
            str(repo_root / "scripts/sample_and_eval_gwa.sh"),
            str(checkpoint),
            "1",
            "mp20_test_set",
            "7",
            "--reuse-logits",
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert "Skipping GPU flow" in completed.stdout
    assert len(calls) == 3
    assert calls[0].startswith("run python scripts/sample_wy.py")
    assert f"--logits_path {logits_path}" in calls[0]
    assert "--reuse-logits" in calls[0]
    assert "--cpu-workers 52" in calls[0]
    assert "--formula_file" not in calls[0]
    assert calls[1].startswith("run python scripts/extract_wyckoff_samples.py")
    assert calls[2].startswith("run python scripts/eval_gwa.py")
