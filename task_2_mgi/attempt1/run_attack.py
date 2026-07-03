#!/usr/bin/env python3
"""Main pipeline: calibrate proxy DCB, run C&W attacks, build submission.npz."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from config import (  # noqa: E402
    CLASS_SIZE,
    DEFAULT_SUBMISSION_JPEG_QUALITY,
    DIRECTION_BLOCKS,
    IMAGE_SIZE,
    V1_ATTACK_DIRECTIONS,
    AttackConfig,
    PathsConfig,
    setup_oned_tokenizer_path,
)
from cw_attack import attack_batch  # noqa: E402
from submission_io import build_submission, load_reference_images, save_submission_npz  # noqa: E402
from proxy_dcb import (  # noqa: E402
    calibrate_thresholds,
    load_calibration,
    load_vqgan,
    save_calibration,
    smoke_check,
    uint8_to_tensor,
)

try:
    from job_progress import bind_job, complete, fail, report
except ImportError:
    def bind_job(*_a, **_k):  # type: ignore
        return None

    def report(*_a, **_k):  # type: ignore
        pass

    def complete(*_a, **_k):  # type: ignore
        pass

    def fail(*_a, **_k):  # type: ignore
        pass


def run_smoke(paths: PathsConfig, cfg: AttackConfig) -> int:
    from smoke_test_proxy import main as smoke_main

    return smoke_main()


def run_calibrate(
    vqgan,
    images: np.ndarray,
    paths: PathsConfig,
    cfg: AttackConfig,
    batch_size: int = 16,
) -> dict:
    device = cfg.device
    t_m = uint8_to_tensor(images[0:CLASS_SIZE], device)
    t_n = uint8_to_tensor(images[CLASS_SIZE : 2 * CLASS_SIZE], device)
    t_g = uint8_to_tensor(images[2 * CLASS_SIZE : 3 * CLASS_SIZE], device)
    cal = calibrate_thresholds(vqgan, t_m, t_n, t_g, eps=cfg.eps, batch_size=batch_size)
    save_calibration(paths.calibration_path, cal)
    print(f"Saved calibration -> {paths.calibration_path}")
    print(json.dumps(cal, indent=2))
    if not smoke_check(cal):
        print("WARNING: smoke check failed — proxy may not separate G from M/N", file=sys.stderr)
    return cal


def run_attack_phase(
    vqgan,
    originals: np.ndarray,
    cal: dict,
    paths: PathsConfig,
    cfg: AttackConfig,
    directions: tuple[str, ...],
    resume: bool = True,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    tau_g = cal["tau_G"]
    alpha = cal["alpha"]
    attacked: dict[str, np.ndarray] = {}

    total_images = sum(
        min(count, limit) if limit is not None else count
        for name, _, _, count, mode in DIRECTION_BLOCKS
        if name in directions and mode != "original"
    )
    bind_job("task_2", attempt=1, owner=os.environ.get("USER"), total_steps=total_images, unit="images")
    done = 0

    def on_progress(local_i: int, local_n: int, direction: str) -> None:
        nonlocal done
        if local_i % 10 == 0 or local_i == local_n:
            report(done + local_i, total_images, phase="attack", message=f"{direction} {local_i}/{local_n}")

    for name, slot_start, source_start, count, mode in DIRECTION_BLOCKS:
        if name not in directions or mode == "original":
            continue
        n_attack = min(count, limit) if limit is not None else count
        print(f"\n=== Attacking {name} ({n_attack}/{count} images, mode={mode}) ===")
        src = originals[source_start : source_start + n_attack]
        adv, logs = attack_batch(
            vqgan,
            src,
            direction=mode,
            tau_g=tau_g,
            alpha=alpha,
            cfg=cfg,
            output_dir=paths.output_dir,
            start_index=source_start,
            resume=resume,
            on_progress=on_progress,
            checkpoint_key=name,
        )
        if n_attack < count:
            block = originals[source_start : source_start + count].copy()
            block[:n_attack] = adv
            attacked[name] = block
        else:
            attacked[name] = adv
        done += n_attack
        success_rate = sum(1 for e in logs if e.get("success")) / max(len(logs), 1)
        mean_mse = float(np.mean([e.get("mse", 0.0) for e in logs]))
        jpeg_q = cfg.submission_jpeg_quality
        print(
            f"  success_rate={success_rate:.1%} (deployable uint8"
            f"{f'+jpeg q={jpeg_q}' if jpeg_q is not None else ''})"
            f"  mean_mse={mean_mse:.6f}"
        )

        summary_path = paths.output_dir / f"attack_summary_{name}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "direction": name,
                    "success_rate": success_rate,
                    "mean_mse": mean_mse,
                    "logs": [{k: v for k, v in e.items() if k != "image"} for e in logs],
                },
                indent=2,
            )
        )

    return attacked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MGI adversarial attack pipeline")
    p.add_argument(
        "--phase",
        choices=("smoke", "calibrate", "attack", "assemble", "all"),
        default="all",
    )
    p.add_argument("--directions", type=str, default=",".join(V1_ATTACK_DIRECTIONS))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Attack only the first N images per direction (fast iteration)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_SUBMISSION_JPEG_QUALITY,
        help="JPEG q for deployable success check and assemble (must match submit)",
    )
    p.add_argument(
        "--no-jpeg-check",
        action="store_true",
        help="Validate success on uint8 only (skip JPEG roundtrip in attack)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = PathsConfig()
    setup_oned_tokenizer_path(paths.oned_tokenizer_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    directions_arg = tuple(d.strip() for d in args.directions.split(",") if d.strip())
    stage2_needed = [d for d in directions_arg
                     if d in ("M_N", "N_M", "G_M", "G_N")]
    if args.phase in ("attack", "all") and stage2_needed:
        print(
            "WARNING: run_attack.py is Stage-1 only. Directions "
            f"{stage2_needed} need Stage-2 control (M/N) and are NOT solved here:\n"
            "  - M_N / N_M are left as UNMODIFIED originals (score 0).\n"
            "  - G_M / G_N share the same objective and cannot both be correct.\n"
            "Use attack_combined.py for these directions, then assemble with\n"
            "build_submission_v2.py. run_attack.py is faithful only for M_G, N_G.",
            file=sys.stderr,
        )

    cfg = AttackConfig()
    if args.device:
        cfg.device = args.device
    elif not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.c is not None:
        cfg.c = args.c
    if args.kappa is not None:
        cfg.kappa = args.kappa
    if args.no_jpeg_check:
        cfg.submission_jpeg_quality = None
    else:
        cfg.submission_jpeg_quality = args.jpeg_quality

    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())

    if args.phase == "smoke":
        return run_smoke(paths, cfg)

    if not paths.tokenizer_ckpt.is_file():
        print(f"ERROR: tokenizer not found: {paths.tokenizer_ckpt}", file=sys.stderr)
        return 1
    if not paths.data_dir.is_dir():
        print(f"ERROR: data dir not found: {paths.data_dir}", file=sys.stderr)
        return 1

    print(f"Loading images from {paths.data_dir}")
    originals = load_reference_images(paths.data_dir)
    vqgan = load_vqgan(paths.tokenizer_ckpt, device=cfg.device)

    cal: dict | None = None
    attacked: dict[str, np.ndarray] = {}

    try:
        if args.phase in ("calibrate", "all"):
            cal = run_calibrate(vqgan, originals, paths, cfg)
            if not smoke_check(cal):
                print("Smoke check failed after calibration", file=sys.stderr)

        if args.phase in ("attack", "all"):
            if cal is None:
                if paths.calibration_path.is_file():
                    cal = load_calibration(paths.calibration_path)
                else:
                    print("ERROR: run calibration first", file=sys.stderr)
                    return 1
            attacked = run_attack_phase(
                vqgan,
                originals,
                cal,
                paths,
                cfg,
                directions,
                resume=not args.no_resume,
                limit=args.limit,
            )

        if args.phase in ("assemble", "all"):
            if not paths.calibration_path.is_file() and cal is None:
                print("ERROR: calibration missing for assemble", file=sys.stderr)
                return 1
            # Load attacked blocks from checkpoints if not in memory
            for name, _, source_start, count, mode in DIRECTION_BLOCKS:
                if mode == "original" or name in attacked:
                    continue
                ckpt_dir = paths.output_dir / "checkpoints" / name
                if not ckpt_dir.is_dir():
                    continue
                block = np.empty((count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                for i in range(count):
                    ckpt = ckpt_dir / f"{source_start + i:04d}.json"
                    if ckpt.is_file():
                        data = json.loads(ckpt.read_text())
                        block[i] = np.array(data["image"], dtype=np.uint8)
                    else:
                        block[i] = originals[source_start + i]
                attacked[name] = block

            submission = build_submission(originals, attacked)
            save_submission_npz(
                submission,
                paths.submission_path,
                jpeg_quality=args.jpeg_quality,
            )

        complete("MGI attack pipeline finished")
        return 0
    except Exception as exc:
        fail(str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
