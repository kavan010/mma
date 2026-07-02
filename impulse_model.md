# Impulse Balance Model — Phase 2

## Goal
Load standing checkpoint → apply random impulses to skeleton → train to recover upright. Eventually: balance against thrown physical blocks.

## Curriculum Plan
1. **Standing** ✅ iter 260, reward ~2.05
2. **Balance** ← current, iter ~934, reward ~2.07
3. **Locomotion** — future

---

## What Changed from Standing

### Physics (mma.cpp)
- `maybeApplyImpulse()` added to `Skeleton` — fires every 90 steps, picks a random bone weighted toward torso (50% body, 20% head, 10% each arm, 5% each leg), applies a lateral velocity delta
- Impulse magnitude controlled by Python via `-68.0, mag` UDP signal each iteration
- Foot friction added: calf bottom endpoints only, `vel.x *= 0.05f` when grounded — prevents feet sliding freely under hits
- Old fake friction (`vel.x *= 0.8`) removed from `checkBorderCollision`

### Python (train.py)
- Loads from `WHOLE_BODY_POLICY.pt` (standing checkpoint) first run, then saves/loads `IMPULSE_POLICY.pt`
- `get_impulse_mag(phase_iter)` sends curriculum scale each iteration:
  - phase < 50: 1000, < 150: 2000, < 300: 3000, else: 4000
- `isDone` angle threshold relaxed: `> 0.9` → `> 1.5` (give model room to recover before reset)
- Reward unchanged from standing: `0.2 + upright + height`

### What was NOT changed
- State dim (39), action dim (10), network architecture — all identical, weights transfer directly
- GAE, PPO update, K_epochs, minibatches — unchanged
- Model does not see the applied force — learns blind from proprioception only

---

## Observations so far
- Reward held near 2.05 through small impulses, climbed to 2.07 at ~934 iters
- std pinned at 1.003 (clamp ceiling) — policy still maximally exploratory, fine for now
- Arms/legs absorb up to 4000 px/s, torso saturates around 1000 — expected, torso is heaviest bone
- Model visibly lifts legs and tilts body to resist pushes — early balance behavior present

---

## Physical Blocks (Phase 2b, later)
Requires adding a dynamic rigid body with contact resolution against skeleton bones. Do this once balance policy is solid at 4000 px/s.

---

## Problems Log

### Impulse only on torso → no generalization to head/limb hits
Applied to random bones (weighted) instead. Covers the full contact distribution the thrown-block scenario will create.

### Feet sliding under hits (no grip)
Old `checkBorderCollision` friction was a multiplier (`vel.x *= 0.8`) — only fired during bounce events, never converged to zero. Replaced with per-frame `vel.x *= 0.05f` on grounded calf endpoints only.

### Curriculum resetting on checkpoint reload
`get_impulse_mag(iteration - start_iteration)` resets phase_iter to 0 each load. Fine since all stages are short — model re-ramps in <50 iters.
