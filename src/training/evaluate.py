"""Stage 4: closed-loop rollout evaluation on LIBERO.

Loads a trained checkpoint and runs the policy inside LIBERO's MuJoCo simulator,
measuring task success rate per suite. Uses receding-horizon control: plan K=10
actions per forward pass, execute the first H (default 5), then replan.

Usage:
    MUJOCO_GL=egl python -m src.training.evaluate \\
        --checkpoint checkpoints/final.pt \\
        --suite libero_spatial \\
        --episodes 10 \\
        --execute-horizon 5 \\
        --output runs/spatial-run1/eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.libero_dataset import preprocess_image  # noqa: E402
from src.model.vla import LIBEROVLA, VLAConfig  # noqa: E402


# LIBERO envs return observations where the scene hasn't settled yet. Run a few
# no-op steps after reset so physics stabilises before the policy takes over.
NUM_SETTLE_STEPS = 10


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[LIBEROVLA, dict, dict | None, int]:
    """Load trained VLA from a training checkpoint blob."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = VLAConfig(**blob["vla_config"])
    model = LIBEROVLA(cfg)
    missing, unexpected = model.load_state_dict(blob["model_trainable"], strict=False)
    if unexpected:
        print(f"[ckpt] unexpected keys: {unexpected[:5]}")
    model = model.to(device)
    model.eval()
    return (
        model,
        blob["action_stats"],
        blob.get("proprio_stats"),
        int(blob.get("step", 0)),
    )


def denormalize_actions(a_norm: np.ndarray, stats: dict) -> np.ndarray:
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    return (a_norm + 1.0) / 2.0 * (hi - lo) + lo


def normalize_proprio(p: np.ndarray, stats: dict) -> np.ndarray:
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    rng = np.maximum(hi - lo, 1e-6)
    out = 2.0 * (p - lo) / rng - 1.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def extract_proprio(obs: dict) -> np.ndarray:
    """ee_pos (3) + gripper_qpos (2) → (5,)."""
    ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    return np.concatenate([ee_pos, gripper], axis=0)


@torch.no_grad()
def plan_chunk(
    model: LIBEROVLA,
    obs: dict,
    instruction: str,
    device: torch.device,
    max_text_len: int,
    use_amp: bool,
    proprio_stats: dict | None,
) -> np.ndarray:
    """Run one forward pass, return (K, 7) normalised actions."""
    pixel_values = preprocess_image(obs["agentview_image"]).unsqueeze(0).to(device)
    tok = model.tokenizer(
        [instruction],
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)

    wrist_pv = None
    if model.config.use_wrist:
        wrist_img = obs["robot0_eye_in_hand_image"]
        wrist_pv = preprocess_image(wrist_img).unsqueeze(0).to(device)

    proprio_t = None
    if model.config.use_proprio:
        assert proprio_stats is not None, (
            "checkpoint has use_proprio=True but no proprio_stats in blob"
        )
        p = extract_proprio(obs)
        p_norm = normalize_proprio(p, proprio_stats)
        proprio_t = torch.from_numpy(p_norm).unsqueeze(0).to(device)

    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            wrist_pixel_values=wrist_pv,
            proprio=proprio_t,
        )
    actions_norm = out["actions"].float().squeeze(0).cpu().numpy()  # (K, 7)
    return actions_norm


def rollout_episode(
    env,
    model: LIBEROVLA,
    instruction: str,
    init_state: np.ndarray,
    action_stats: dict,
    proprio_stats: dict | None,
    device: torch.device,
    max_text_len: int,
    use_amp: bool,
    max_steps: int,
    execute_horizon: int,
    save_frames: bool,
) -> dict:
    """Run one rollout. Returns {success, steps, frames?}."""
    env.reset()
    env.set_init_state(init_state)
    dummy_action = np.zeros(7, dtype=np.float32)
    dummy_action[-1] = -1.0  # gripper open
    for _ in range(NUM_SETTLE_STEPS):
        obs, _, _, _ = env.step(dummy_action)

    frames: list[np.ndarray] = []
    action_buffer: list[np.ndarray] = []
    success = False
    steps_taken = 0
    t_start = time.time()

    for step in range(max_steps):
        if save_frames:
            frames.append(obs["agentview_image"][::-1].copy())

        if not action_buffer:
            actions_norm = plan_chunk(
                model,
                obs,
                instruction,
                device,
                max_text_len,
                use_amp,
                proprio_stats,
            )  # (K, 7)
            actions_real = denormalize_actions(actions_norm, action_stats)
            take = min(execute_horizon, actions_real.shape[0])
            action_buffer = [actions_real[i] for i in range(take)]

        action = action_buffer.pop(0)
        obs, _, done, _ = env.step(action)
        steps_taken = step + 1

        if env.check_success():
            success = True
            break
        if done:
            break

    return {
        "success": success,
        "steps": steps_taken,
        "wallclock": time.time() - t_start,
        "frames": frames if save_frames else None,
    }


def save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per task (capped by available init states)")
    parser.add_argument("--execute-horizon", type=int, default=5,
                        help="How many actions to execute before replanning (H <= K)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max env steps per episode")
    parser.add_argument("--num-flow-steps", type=int, default=10,
                        help="Flow matching Euler integration steps")
    parser.add_argument("--max-text-len", type=int, default=32)
    parser.add_argument("--task-index", type=int, default=None,
                        help="Restrict to one task index (for debugging)")
    parser.add_argument("--save-failed-videos", action="store_true")
    parser.add_argument("--output", required=True,
                        help="Output directory for results.json + videos")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[eval] device={device} amp={use_amp}")

    ckpt_path = Path(args.checkpoint)
    print(f"[eval] loading {ckpt_path}")
    model, action_stats, proprio_stats, train_step = load_model(ckpt_path, device)
    model.action_head.num_flow_steps = args.num_flow_steps
    print(f"[eval] checkpoint step={train_step} "
          f"use_wrist={model.config.use_wrist} use_proprio={model.config.use_proprio}")
    if model.config.use_proprio and proprio_stats is None:
        raise RuntimeError(
            "Checkpoint has use_proprio=True but no proprio_stats saved"
        )

    # Import LIBERO here — MUJOCO_GL must be set first.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bm = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = bm.n_tasks
    print(f"[eval] suite={args.suite} n_tasks={n_tasks}")

    task_indices = [args.task_index] if args.task_index is not None else list(range(n_tasks))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "failures"

    results = {
        "suite": args.suite,
        "checkpoint": str(ckpt_path),
        "train_step": train_step,
        "execute_horizon": args.execute_horizon,
        "num_flow_steps": args.num_flow_steps,
        "max_steps": args.max_steps,
        "episodes_per_task": args.episodes,
        "tasks": [],
    }

    overall_success = 0
    overall_total = 0
    t_eval_start = time.time()

    for task_idx in task_indices:
        task = bm.get_task(task_idx)
        instruction = task.language
        task_bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        print(f"\n[task {task_idx:>2}] {instruction}")
        init_states = bm.get_task_init_states(task_idx)
        n_ep = min(args.episodes, len(init_states))

        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl, camera_heights=128, camera_widths=128
        )

        task_successes = 0
        task_results = []
        for ep in range(n_ep):
            out = rollout_episode(
                env=env,
                model=model,
                instruction=instruction,
                init_state=init_states[ep],
                action_stats=action_stats,
                proprio_stats=proprio_stats,
                device=device,
                max_text_len=args.max_text_len,
                use_amp=use_amp,
                max_steps=args.max_steps,
                execute_horizon=args.execute_horizon,
                save_frames=args.save_failed_videos,
            )
            task_successes += int(out["success"])
            task_results.append({
                "episode": ep,
                "success": out["success"],
                "steps": out["steps"],
                "wallclock": round(out["wallclock"], 2),
            })
            mark = "ok" if out["success"] else "--"
            print(f"  ep {ep:>2}  {mark}  steps={out['steps']:>3}  "
                  f"({out['wallclock']:.1f}s)")

            if args.save_failed_videos and not out["success"] and out["frames"]:
                video_path = videos_dir / f"task{task_idx:02d}_ep{ep:02d}.mp4"
                save_video(out["frames"], video_path)

        env.close()
        rate = task_successes / n_ep if n_ep else 0.0
        print(f"[task {task_idx:>2}] success {task_successes}/{n_ep} ({rate:.0%})")

        results["tasks"].append({
            "task_index": task_idx,
            "name": task.name,
            "language": instruction,
            "success_rate": rate,
            "successes": task_successes,
            "episodes": n_ep,
            "details": task_results,
        })
        overall_success += task_successes
        overall_total += n_ep

    results["overall_success_rate"] = (
        overall_success / overall_total if overall_total else 0.0
    )
    results["overall_successes"] = overall_success
    results["overall_episodes"] = overall_total
    results["total_wallclock"] = round(time.time() - t_eval_start, 1)

    print("\n" + "=" * 60)
    print(f"OVERALL: {overall_success}/{overall_total} "
          f"({results['overall_success_rate']:.1%}) "
          f"in {results['total_wallclock']:.0f}s")
    print("=" * 60)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
