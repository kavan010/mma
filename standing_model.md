# Standing Model — Project Reference

## What This Is

2D ragdoll trained with PPO to stand, then curriculum-extended to balance under impulses and walk. C++ OpenGL sim (`mma.cpp`) ↔ Python PPO (`train.py`) over UDP. 20 parallel envs.

## Curriculum Plan

1. **Standing** — upright at correct height, both feet down
2. **Balance** — load standing checkpoint, apply random torso impulses, reward recovery
3. **Locomotion** — load balance checkpoint, add forward velocity target, reward gait

Architecture and state/action dims are fixed across all phases so weights transfer.

---

## State (39 per env)

11 bones × 3 = 33: `[sin(angle), cos(angle), angVel]` per bone, in order:
`head, body, armL, armR, forearmL, forearmR, legL, legR, calfL, calfR, hip`

Then 6 extra:

| idx | value | notes |
|-----|-------|-------|
| 33 | hip.pos.y / 600 | standing height target |
| 34 | hip.pos.x / 800 | horizontal position |
| 35 | hip.vel.y / 600 | vertical velocity |
| 36 | hip.vel.x / 800 | horizontal drift |
| 37 | calfL contact | 1.0 if floor |
| 38 | calfR contact | 1.0 if floor |

`angVel` values (indices 2,5,8...32) are divided by 20 inside `forward()` before the network sees them.

## Actions (10)

One target angle per joint: `body→head, body→hip, body→armR, body→armL, armR→forearmR, armL→forearmL, hip→legR, hip→legL, legR→calfR, legL→calfL`. Clamped to `[-π, π]` before sending. C++ PD controller drives each joint toward target.

---

## Hyperparameters

| param | value | note |
|-------|-------|------|
| lr | 3e-4 → 1e-5 | cosine decay over N iters |
| gamma | 0.99 | |
| gae_lambda | 0.95 | |
| clip_ratio | 0.2 | |
| ent_coef | 0.01 | fixed |
| value_coef | 0.5 | |
| grad_norm | 0.5 | unified across all params |
| K_epochs | 10 | |
| minibatches | 4 | batch = N_samples // 4 |
| log_std init | -0.5 | std ≈ 0.61, clamp [-3, 0] |
| T | 1024 | steps per rollout per env |
| N | 10000 | total iterations |

## Reward

```python
upright = exp(-2.0 * (body_ang - π/2)²)       # 0→1, peaks when body vertical
height  = exp(-100.0 * (hip_y - STAND_HEIGHT)²) # 0→1, peaks at standing height
reward  = 0.2 + upright + height                 # max 2.2/step
```

Done if `hip_y < STAND_HEIGHT/2` or `|body_ang - π/2| > 0.9`.

**`STAND_HEIGHT = 0.23` is a geometry estimate — verify with the measure loop before a long run.**

---

## Before Training Checklist

- [ ] Comment out the test loop (line ~247) — it's `while True:` and blocks training
- [ ] Verify `STAND_HEIGHT` with the measure loop at the bottom (uncomment, run briefly with loaded checkpoint, read printed hip_y at steady state)
- [ ] `NUM_ENVS = 20` in both `train.py` and `mma.cpp`

---

## Changelog

### Reward simplification
- Replaced `3.0 * upright * height` (multiplicative — zero gradient unless both conditions met simultaneously) with `0.2 + upright + height` (additive — independent gradient signal for each term)
- Removed: `feet_contact`, `jitter`, `drift`, `sym`, `space` — redundant with or premature for plain standing
- Survival bonus `0.2` added — trains the model to not fall before it trains to stand

### NaN / sim stability fixes
- C++: clamped `vel` to ±3000 px/s (was unbounded; caused divergence when hip velocity was added to state)
- C++: wrapped joint angle error to `[-π, π]` in `applyTorque` (accumulated angle drift was sending huge error to PD even with clamped actions)
- Python: `bad = ~isfinite(next_state).all(dim=1)` folded into `done` mask — NaN rows force-reset immediately and never enter the buffer
- `value_clip_eps` raised from 0.5 to 10.0 (then removed entirely — with returns ~O(10), a clip of 0.5 prevented the critic from learning)

### PPO hyperparameter cleanup
- `lr`: 2e-4 → 3e-4
- `K_epochs`: 8 → 10
- `ENT_START/ENT_END` scheduled entropy removed → fixed `ENT_COEF = 0.01` (std already decays naturally as policy converges; scheduling is redundant)
- `log_std` init: -1.0 → -0.5 (std ≈ 0.61 at start)
- `log_std` clamp upper: -0.5 → 0.0 (was identical to init, zeroing the entropy gradient from step 1)
- Minibatches: fixed `batch_size=2048` → `N_samples // 4` (scales with env count, standard 4-minibatch PPO)
- Grad clip: two separate actor/critic calls → one unified `clip_grad_norm_(model.parameters(), 0.5)`
- Value loss: clipped max formulation removed → plain MSE `(v_pred - returns)²`
- Non-finite gradient skip removed (NaN source is fixed upstream; guard was hiding bugs)
