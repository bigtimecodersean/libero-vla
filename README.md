# LIBERO-VLA

A 1.5B-parameter Vision-Language-Action model trained on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) robotic manipulation benchmark — built from scratch without the cross-embodiment pretraining (Open X-Embodiment) that most published VLAs rely on.

This repo is a learning-oriented write-up. It is not an attempt to beat state-of-the-art (compute requirements too large). The goal is to characterize what a small, pretraining-free VLA actually does: where it is strong, where it fails, and which architectural decisions are load-bearing.

---

## TL;DR

- **Architecture**: SigLIP (frozen) + MLP projector → Qwen2-1.5B with LoRA → flow-matching action head.
- **Training**: joint fine-tuning on all 4 LIBERO base suites for 50k steps on one H100. Only 28.4M trainable parameters (the rest is frozen vision + frozen LLM with LoRA adapters).
- **Results on base LIBERO (H=10)**: spatial 78.0%, object 86.8%, goal 81.8%, long 35.4%, 4-suite average **70.5%**.
- **Main finding**: the long suite's weakness is bimodal, not uniform. The model handles single-object multi-step tasks at 72–80%, but multi-object sequential tasks collapse to 20–30%, with one task involving two identical objects failing completely (0/50).

---

## What this is  

**Is**: a compact, reproducible reference implementation of a modern VLA — vision encoder + LLM backbone + flow matching action head — with an honest diagnostic on where a pretraining-free model succeeds and fails on LIBERO. A single weekend of H100 time reproduces the numbers below. Published VLAs (OpenVLA, Octo, π0, RDT) all use Open X-Embodiment — an 800k+ trajectory, 22M-step cross-embodiment pretraining corpus — before fine-tuning on LIBERO. That pretraining is the dominant factor in their headline scores. This project skips OXE entirely, which is why raw numbers aren't directly comparable.

---

## Architecture

```
Image (224×224) ─> SigLIP (frozen) ─> MLP projector ─┐
                                                     ├─> Qwen2-1.5B (LoRA) ─> Action queries ─> Flow matching head ─> (B, 10 timesteps, 7D joint actions)
Instruction ─> Qwen2 tokenizer ──────────────────────┘
```

### Components

| Component | Choice | Why |
|---|---|---|
| Vision encoder | SigLIP ViT-Base 224 (86M, frozen) | Strong contrastive vision prior; sigmoid loss gives cleaner features than CLIP for downstream tasks; frozen to preserve pretraining |
| Projector | 2-layer MLP, 768 → LLM hidden | Minimal adapter; cheap to train; dominant "visual-to-language" mapping approach in recent VLMs |
| LLM backbone | Qwen2-1.5B with LoRA on q/k/v/o | 1.5B fits on consumer GPUs for inference; Qwen2 has strong multilingual/reasoning pretraining; LoRA (r=16) trains 28M of 1.5B params |
| Action head | Flow matching, K=10 chunks | Handles multi-modal action distributions better than MSE regression; 10-step Euler integration at inference; chunk size 10 = 0.5s of control at 20Hz |

### Prefix-LM attention mask

Standard causal attention is wrong for VLAs: vision patches benefit from bidirectional attention (all patches see each other to build coherent spatial features), but text generation is still causal. We use a prefix-LM mask:

- Vision ↔ Vision: bidirectional
- Text → Vision: full access (text attends to all patches)
- Text → Text: causal
- Vision → Text: blocked (patches don't need the instruction for spatial features)

### Flow matching, briefly

We train a velocity field `v(x_t, t, context)` that transports noise to actions:

1. **Training**: sample `t ~ U(0, 1)`, interpolate `x_t = (1-t) * noise + t * target`, regress on velocity `v = target - noise`.
2. **Inference**: Euler integrate from `x_0 ~ N(0, I)` through 10 steps to produce a clean action chunk.

Flow matching is chosen over direct regression because action distributions in teleoperated data are multi-modal (multiple valid ways to pick up an object). An MSE head averages these into a smeared mean; flow matching models the distribution.

---

## Training setup

| | |
|---|---|
| Data | LIBERO base, all 4 suites joint (spatial, object, goal, long) |
| Input modalities | Front camera, wrist camera, 5-D proprioception (end-effector pos + gripper) |
| Action space | 7-D (dx, dy, dz, droll, dpitch, dyaw, gripper) |
| Chunk size | 10 actions per forward |
| Optimizer | AdamW, lr=5e-5, weight_decay=0.01, 500-step warmup |
| Batch size | 32, grad accum 1 |
| Precision | bfloat16 |
| Steps | 50,000 |
| Trainable params | 28.4M (LoRA + projector + action head) |
| Compute | 1× H100 80GB, ~72 samples/sec, ~11 hours total |
| Train/val split | 288,464 / 32,111 samples |
| Final loss | 0.115 train / 0.135 val |

Data augmentation: random crop, color jitter, Gaussian noise on proprio. All images are normalized to [-0.5, 0.5] and resized to 224 before SigLIP.

---

## Results

### Base LIBERO, H=10 (100 episodes per task, 50 init states × 10 tasks = 500 eps per suite)

| Suite | Score | Rate |
|---|---|---|
| libero_spatial | 390/500 | 78.0% |
| libero_object | 434/500 | **86.8%** |
| libero_goal | 409/500 | 81.8% |
| libero_10 (long) | 177/500 | 35.4% |
| **Average** | 1410/2000 | **70.5%** |

Additional data point: libero_spatial at H=5 (replan every 5 actions) scored 399/500 (79.8%) — slightly better than H=10 on the same suite.

### The long suite is bimodal

Aggregating long at 35.4% hides the actual story. Broken down per task:

| Task | Rate | Description (abbreviated) |
|---:|---:|---|
| 2 | **80%** | turn on stove and put moka pot on it |
| 3 | **72%** | put bowl in bottom drawer of cabinet and close it |
| 5 | 48% | pick up book and place in back compartment of shelf |
| 9 | 32% | put yellow-and-white mug in microwave and close it |
| 1 | 30% | put *both* cream cheese and butter in basket |
| 4 | 26% | put white mug on left plate and put yellow mug on right plate |
| 7 | 24% | put *both* alphabet soup and cream cheese in basket |
| 0 | 22% | put *both* alphabet soup and tomato sauce in basket |
| 6 | 20% | put white mug on plate and put chocolate pudding on plate |
| 8 | **0%** | put *both* moka pots on stove |

Two clusters emerge:

**Single object, multi-step (≥48%)**: tasks 2, 3, 5 — the model can identify an object, manipulate it, and chain a second action on the same object.

**Two distinct objects, sequential (20–32%)**: tasks 0, 1, 4, 6, 7, 9 — the model must pick up object A, place it, re-orient to the scene, identify object B, and repeat. Success rates cluster tightly around a quarter, suggesting the model can execute the *first* subtask at near-single-task rates (~70% × ~50% residual capacity ≈ ~30% combined) but loses state between subtasks.

**Task 8 at 0/50**: the only task requiring disambiguation of two visually identical objects. The model likely re-selects the same moka pot repeatedly, or fails to update its spatial model after the first placement. This is a specific, reproducible failure mode that deserves its own investigation.

The implication: the gap between a no-OXE model and a pretrained VLA is not evenly spread — it is concentrated in tasks requiring referent tracking across subtasks. Cross-embodiment pretraining likely instills this kind of multi-step spatial-memory capability.

---

## Context: why these numbers aren't directly comparable to published VLAs

Approximate reported numbers on base LIBERO:

| Model | Params | Robotics pretraining | LIBERO avg |
|---|---:|---|---:|
| Octo-small | 27M | OXE (800k traj) | ~75% |
| Octo-base | 93M | OXE | ~78% |
| OpenVLA | 7B | OXE | ~76% |
| RDT-1B | 1B | OXE | ~75% |
| π0 | 3B | ~10k hrs robotics data | ~94% |
| **LIBERO-VLA (this repo)** | **1.5B** | **none** | **70.5%** |

OXE provides hundreds of thousands of trajectories across dozens of robot embodiments. SigLIP and Qwen2 provide strong vision and language priors respectively, but neither includes manipulation action data. The 5–25 percentage point gap between "matched-compute OXE-pretrained models" and this project is, to a first approximation, what cross-embodiment pretraining is worth on LIBERO.

---

## Takeaways for someone building their own VLA

1. **If you skip OXE, expect ~70% on base LIBERO at 1.5B.** That's the ballpark. Going higher without robotics pretraining likely requires either dramatically more in-domain data, a larger model, or both.
2. **Object-grounded single-step tasks are the easy regime.** Get 80+% on object/goal suites with reasonable effort.
3. **Multi-subtask tasks are hard.** Long-horizon planning and inter-subtask state persistence appear to be capabilities that OXE pretraining supplies, and that LLM + vision priors alone do not.
4. **Architecture details that mattered (ablations not reported here but worth knowing):**
   - Prefix-LM attention mask was meaningfully better than pure causal in early experiments
   - Flow matching head outperformed MSE regression on multi-modal action distributions
   - Wrist camera + proprioception meaningfully helped fine-grained manipulation
5. **Architecture details that probably don't matter much for a first attempt:** exact LoRA rank (8 vs 16 vs 32), exact action chunk size (5 vs 10 vs 15), projector depth (1 vs 2 layers). Spend your time on data and compute, not these.

---

## Limitations

- **No Open X-Embodiment pretraining.** This is a deliberate scoping choice — the project characterizes a no-OXE baseline. It is also the primary reason for the absolute gap against published work.
- **Single seed, single run.** Numbers are from one training run per configuration; variance across seeds is not characterized.
- **No robustness evaluation.** LIBERO-Plus (perturbation benchmark) is not evaluated here.
- **Confounded A/B with a prior run.** An earlier run used Qwen2-0.5B on spatial only; this run uses 1.5B jointly on all 4 suites. Two variables changed, so isolated size vs joint-training effects aren't separable from this project alone.

---

## Future work

- **LIBERO-Plus evaluation** to characterize robustness under visual, linguistic, and camera perturbations — would sharpen the "what OXE buys you" claim above.
- **Online RL post-training** (policy-gradient on flow-matching is non-trivial — no closed-form action log-prob — but recent work on distillation + REINFORCE-style updates makes it tractable).
- **Targeted fine-tuning on long-suite failure modes**, specifically tasks involving two-distinct-objects sequential structure.

---

## Reproduce

```bash
git clone <this-repo> libero_vla && cd libero_vla
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# LIBERO demonstrations (~3 GB)
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial libero_object libero_goal libero_10

# Train (11 hrs on 1× H100, bf16)
bash scripts/train.sh

# Eval a single suite at H=10
bash scripts/eval.sh --checkpoint checkpoints/joint-run-b/final.pt --suite libero_spatial --execute-horizon 10 --output runs/joint-run-b/eval/libero_spatial/h10

# Smoke test (no training, verifies architecture loads)
python scripts/smoke_test.py
```

See `config.yaml` for all training and model hyperparameters.

---

## Repo structure

```
src/
  model/
    vision_encoder.py   SigLIP + MLP projector
    vlm_backbone.py     Qwen2 + LoRA + prefix-LM attention
    action_head.py      Flow matching, learnable action queries
    vla.py              Full LIBEROVLA wiring
  data/
    libero_dataset.py   HDF5 loading, image/action preprocessing
  training/
    train.py            SFT loop
    evaluate.py         Rollout eval in LIBERO env
    rl.py               (stub) RL post-training
scripts/
  smoke_test.py         Verify model loads and forwards
  train.sh              Training entrypoint
  eval.sh               Eval entrypoint
config.yaml             Single source of truth for hyperparameters
runs/                   Per-run artifacts (config, logs, eval results)
checkpoints/            Model weights
```

---

## Acknowledgements

Built on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [SigLIP](https://huggingface.co/google/siglip-base-patch16-224), [Qwen2](https://huggingface.co/Qwen/Qwen2-1.5B), [PEFT](https://github.com/huggingface/peft), [Flow matching](https://arxiv.org/abs/2210.02747), and [robosuite](https://github.com/ARISE-Initiative/robosuite).
