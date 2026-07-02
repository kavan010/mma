Here's what the research shows, grounded in the actual DeepMimic/SFV source data (I pulled the real character definition files, not just the paper text) plus biomechanics literature.

Physics engine
Bullet (via PyBullet) is what DeepMimic/SFV actually used — Peng's group has since moved newer work to MuJoCo/Isaac, and that's the general industry drift too. For your case:

MuJoCo: better contact stability at small timesteps, faster, more accurate — but you're writing this in raw C++ already, and MuJoCo's C API is usable standalone (doesn't require Python).
PyBullet: worse accuracy at small dt but is what SFV/DeepMimic proved out this exact task on, huge amount of reference code (xbpeng/DeepMimic repo) to compare against.
Given you already hand-roll your own solver in mma2D.cpp, you have two real options: (1) keep hand-rolling in 3D using quaternions/spherical joints, or (2) adopt MuJoCo as the backend and just write the RL/reward loop in Python. I'd lean toward MuJoCo for 3D — ball-and-socket joints (hips, shoulders) are miserable to get right with a custom impulse solver (your 2D code only needs revolute joints; 3D needs quaternion-based angular constraints, which is a big jump in complexity and easy to get subtly wrong/unstable).
Ground-truth character spec (DeepMimic humanoid3d.txt + humanoid3d_ctrl.txt, from Peng's actual repo)
15 bodies, total mass 45 kg, height 1.62 m (SFV/DeepMimic uses the same character):

Body	Mass (kg)	Shape	Torque limit (N·m)	Kp	Kd
root (pelvis)	6.0	sphere r0.18	—	—	—
chest	14.0	sphere r0.22	200	1000	100
neck	2.0	sphere r0.205	50	100	10
hip (×2)	4.5	capsule r0.11 h0.3	200	500	50
knee (×2)	3.0	capsule r0.10 h0.31	150	500	50
ankle (×2)	1.0	box 0.177×0.055×0.09	90	400	40
shoulder (×2)	1.5	capsule r0.09 h0.18	100	400	40
elbow (×2)	1.0	capsule r0.08 h0.135	60	300	30
wrist (×2)	0.5	sphere r0.08	—	0	0
That's 15 bones, not 11 — 3D needs a root/pelvis body split from the hip, a separate neck, and wrists, none of which your 2D hip+body combo needed. Notice: Kd ≈ Kp/10 everywhere, not the critical-damping formula you're using in applyTorque (d = 2*sqrt(k*(I_A+I_B))). DeepMimic just uses a fixed empirical ratio, tuned once, same across all motions — simpler than analytically deriving critical damping per joint, and it's what shipped. Worth trying both; empirical ratio is less likely to blow up when inertia estimates are off.

Are target angles + stiffness (PD) valid? Yes — confirmed, this is exactly the architecture
Your 2D approach — torque = kp*(targetAngle - angle) - kd*angVel, clamped to a torque limit — is the DeepMimic/SFV controller (they call it stable PD / SPD, an implicit variant that's less prone to blowing up at high Kp, but this is a numerical-integration nicety, not a different concept). Action space = target angles, one per joint (or an axis-angle/quaternion target for spherical joints in 3D), fed into local per-joint PD controllers running at the physics rate while the policy runs at a lower rate (e.g. 30 Hz policy vs 1200 Hz simulation/PD in DeepMimic, 60 Hz physics is fine as a starting point). Keep target angles. It's the right call and it's proven to produce backflips, kicks, spins, martial-arts moves specifically (SFV's demo reel includes exactly these).

Torque/mass scaling
Numbers above are absolute — they work because they're paired with the 45 kg/1.62 m body. If you scale your character's total mass/height, scale torque roughly by mass_ratio * length_ratio (torque = force × lever arm, force ∝ mass) — don't keep DeepMimic's raw numbers if your character is heavier. Sanity check against real biomechanics: peak knee extension is ~2.9–3.5 N·m/kg body mass, i.e. for a 45 kg person that's ~130–155 N·m — DeepMimic's 150 N·m knee limit lines up almost exactly with real human peak isometric torque. That's a good validation method for any joint you're unsure about: torque_limit ≈ (N·m per kg from literature) × character_mass_kg.

For explosive movements (jump, backflip) specifically: PD torque limits alone often aren't the bottleneck — it's more often (a) contact/friction handling at the feet being unstable, (b) Kp too low to reach targets fast enough within a control step, or (c) integration timestep too large causing the stiff PD system to go unstable before torque saturates. If your character can't jump high enough, raise Kp/torque limit for hips+knees first (that's where jump power comes from biomechanically — ~80% of vertical jump impulse is hip+knee extension, ankle is a smaller contributor), not uniformly across all joints.

Bone shapes
Capsules for limbs (hip/knee/shoulder/elbow), spheres for chest/root/neck/wrist, a box only for the ankle/foot (flat contact surface matters for balance/friction). This matches both DeepMimic and MuJoCo's stock humanoid — capsules give cheap, robust swept-collision for limb-vs-ground and limb-vs-limb contact, which matters a lot once you're doing kicks and falls, unlike your 2D version's rectangle+circle-cap bones (fine in 2D, awkward to generalize to 3D contact manifolds).

2D vs 3D — what actually changes
Not just "more bones" — the joint math changes fundamentally: your 2D Joint::solve/applyTorque treats angle as a scalar and rA/rB as 2D rotations. In 3D, hips and shoulders are spherical (ball) joints (3 DoF, target = quaternion, controlled via axis-angle PD, not a single angle), while knees/elbows/ankles stay effectively 1–2 DoF hinges. This is the single biggest engineering jump from your existing code — it's not an extension of the 2D math, it's a different constraint formulation (quaternion error terms instead of scalar angle wrapping). This is the strongest reason to consider MuJoCo instead of hand-rolling the 3D solver yourself.

Sources:

DeepMimic paper (arXiv 1804.02717) / xbpeng/DeepMimic repo (character files: humanoid3d.txt, humanoid3d_ctrl.txt)
SFV: Reinforcement Learning of Physical Skills from Videos (arXiv 1810.03599)
SFV project page
PyBullet DeepMimic announcement
OpenAI Gym MuJoCo humanoid.xml
A Review of Nine Physics Engines for RL Research (arXiv 2407.08590)
Maximum voluntary joint torque model, lower limb (PMC6820133)
Winter's anthropometric segment mass tables (Chaffin et al. 1999 breakdown)