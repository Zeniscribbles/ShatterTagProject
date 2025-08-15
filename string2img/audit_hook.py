
"""
Audit helper for ShatterTag periodic evaluations.

Usage from your training script:
    from audit_hook import maybe_run_audit

    # after finishing an epoch (or at desired cadence):
    maybe_run_audit(
        epoch=i_epoch + 1,
        every=args.audit_eval_every,
        data_dir=args.audit_data_dir or args.data_dir,
        encoder_ckpt=encoder_ckpt_path,
        decoder_ckpt=decoder_ckpt_path,
        out_dir=os.path.join(args.output_dir, "audits", f"epoch_{i_epoch+1:03d}"),
        image_resolution=args.image_resolution,
        bit_length=args.bit_length,
        batch_size=args.audit_batch_size,
        cuda=args.cuda,
        seed=args.audit_seed,
        limit_images=args.audit_limit_images,
        save_images=args.audit_save_images,
        save_npz=args.audit_save_npz,
        tar_images=args.audit_tar_images,
        threshold=args.audit_threshold,
        eval_script_path=args.audit_script_path,
    )

Author: Amanda + Chansen
"""

import os
import sys
import subprocess


def _which_python():
    # Try to pick the same interpreter that's running training
    return sys.executable or "python"


def maybe_run_audit(
    epoch: int,
    every: int,
    data_dir: str,
    encoder_ckpt: str,
    decoder_ckpt: str,
    out_dir: str,
    image_resolution: int,
    bit_length: int,
    batch_size: int = 128,
    cuda: str = "cuda",
    seed: int = 123,
    limit_images: int | None = None,
    save_images: str = "failures",  # none|failures|all
    save_npz: bool = False,
    tar_images: bool = False,
    threshold: float = 0.99,
    eval_script_path: str = "eval_exhaustive.py",
):
    if every is None or every <= 0:
        return  # auditing disabled
    if epoch % every != 0:
        return

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        _which_python(),
        eval_script_path,
        "--data_dir", data_dir,
        "--encoder_path", encoder_ckpt,
        "--decoder_path", decoder_ckpt,
        "--output_dir", out_dir,
        "--image_resolution", str(image_resolution),
        "--bit_length", str(bit_length),
        "--batch_size", str(batch_size),
        "--cuda", str(cuda),
        "--seed", str(seed),
        "--save_images", save_images,
        "--threshold", str(threshold),
    ]

    if save_npz:
        cmd.append("--save_npz")
    if tar_images:
        cmd.append("--tar_images")
    if limit_images is not None:
        cmd.extend(["--limit_images", str(limit_images)])

    print(f"[AUDIT] Running: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("[AUDIT] Completed")
    except subprocess.CalledProcessError as e:
        print("[AUDIT] FAILED")
        print("Return code:", e.returncode)
        if e.stdout:
            print("STDOUT:\n", e.stdout)
        if e.stderr:
            print("STDERR:\n", e.stderr)
