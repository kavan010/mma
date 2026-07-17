import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
import math
import socket
import struct
from time import sleep
import os

# =============================================================================
# DeepMimic / SFV-style physics-based imitation learning (3D humanoid).
#
# This is the BASE. It defines the full state/action/reward CONTRACT so you do
# not have to change the observation later and retrain from scratch. Every input
# the paper's reward terms need (per-link pose, velocity, end-effectors, CoM,
# phase) is already in the state vector, even though a first "stand still" run
# only exercises the pose term.
#
# What the C++ side (src/mma.cpp) must provide to match this file is marked with
# [C++ TODO]. Until those are wired up, the reference defaults to a captured
# standing pose so the whole loop runs end-to-end as a smoke test.
# =============================================================================


# --------------------------------------------------------------------------- #
#  Body definition — must match the bone order/anchors in mma.cpp
# --------------------------------------------------------------------------- #
# Link order (root first). 14 links.
LINK_NAMES = [
    "pelvis", "abs", "chest", "head",          # 0..3  (root = pelvis = 0)
    "thighR", "thighL", "calfR", "calfL",      # 4..7
    "footR",  "footL",                         # 8..9
    "armR",   "armL",  "forearmR", "forearmL", # 10..13
]
NUM_LINKS = len(LINK_NAMES)

# Per-link mass (kg), same values as the Bone() constructors in mma.cpp.
LINK_MASS = torch.tensor([
    5.0, 7.0, 8.0, 3.0,      # pelvis, abs, chest, head
    4.5, 4.5, 2.5, 2.5,      # thighs, calves
    1.0, 1.0,                # feet
    1.8, 1.8, 1.2, 1.2,      # arms, forearms
])
TOTAL_MASS = LINK_MASS.sum()

# Actuated joints as (child_link_idx, parent_link_idx), matching the Joint list
# in mma.cpp. 13 joints -> the policy outputs a 3D axis-angle target per joint.
JOINTS = [
    (1, 0),   # waist      abs   <- pelvis
    (2, 1),   # spine      chest <- abs
    (3, 2),   # neck       head  <- chest
    (4, 0),   # hipR       thighR<- pelvis
    (5, 0),   # hipL       thighL<- pelvis
    (6, 4),   # kneeR      calfR <- thighR
    (7, 5),   # kneeL      calfL <- thighL
    (8, 6),   # ankleR     footR <- calfR
    (9, 7),   # ankleL     footL <- calfL
    (10, 2),  # shoulderR  armR  <- chest
    (11, 2),  # shoulderL  armL  <- chest
    (12, 10), # elbowR     forearmR <- armR
    (13, 11), # elbowL     forearmL <- armL
]
NUM_JOINTS = len(JOINTS)

# End-effectors used by r_endeff: hands (forearm tips) and feet.
END_EFFECTORS = [8, 9, 12, 13]  # footR, footL, forearmR, forearmL

# 1 sim unit = 5 cm. Convert positions/linear-vel to meters so the paper's
# reward coefficients (tuned in metres) stay calibrated.
UNIT_M = 0.05


# --------------------------------------------------------------------------- #
#  State / action layout  (THE CONTRACT — implement the C++ export to match)
# --------------------------------------------------------------------------- #
#  Per env the C++ sends, in order:
#     [0]        phase  phi in [0,1]
#     [1]        root height (pelvis world y, in sim units)
#     then for each of the 14 links, all in the ROOT HEADING FRAME
#     (root at origin, x-axis = pelvis facing dir projected on ground):
#         rel_pos   (3)   link_center - root_center
#         quat      (4)   orientation, w,x,y,z
#         lin_vel   (3)
#         ang_vel   (3)
#  => 2 + 14*13 = 184 floats per env.
PER_LINK = 3 + 4 + 3 + 3   # 13
STATE_DIM = 2 + NUM_LINKS * PER_LINK   # 184
ACTION_DIM = 3 * NUM_JOINTS            # 39  (axis-angle target per joint)


# --------------------------------------------------------------------------- #
#  Hyperparameters
# --------------------------------------------------------------------------- #
NUM_ENVS   = 20
N          = 100000     # PPO iterations
T          = 1024       # steps per rollout (per env)
K_epochs   = 4
GAMMA      = 0.95       # DeepMimic uses a short horizon (~1s at 30Hz)
LAM        = 0.95
ENT_COEF   = 0.0        # paper uses fixed covariance; keep exploration in log_std
CLIP       = 0.2
LR         = 3e-4
CONTROL_HZ = 30.0       # policy queried at 30 Hz (paper); C++ substeps internally
ACTION_SCALE = 0.3      # policy output is a RESIDUAL (radians) on the ref target

# Motion / phase
MOTION_FPS   = 30.0
MOTION_CYCLIC = True    # True for idle/walk; False for one-shot strikes

# Early termination: end episode if head/chest drop below these world heights
# (sim units). Mirrors the paper's "torso or head touches the ground".
ET_HEAD_MIN  = 24.0
ET_CHEST_MIN = 18.0

# Imitation reward weights (paper §5.3)
W_POSE, W_VEL, W_END, W_COM = 0.65, 0.10, 0.15, 0.10


# --------------------------------------------------------------------------- #
#  Quaternion helpers (w,x,y,z), batched over [...,4]
# --------------------------------------------------------------------------- #
def q_conj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0])

def q_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)

def q_angle(q):
    # rotation magnitude of a (near-)unit quaternion, robust near identity
    w = q[..., 0].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w)

def q_to_rotvec(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w = q[..., 0:1].clamp(-1.0, 1.0)
    xyz = q[..., 1:4]
    s = xyz.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = 2.0 * torch.atan2(s, w)
    return (angle / s) * xyz

def joint_local_quats(link_quats):
    # link_quats: [N,14,4] in heading frame -> [N,13,4] child-relative-to-parent
    out = []
    for (c, p) in JOINTS:
        out.append(q_mul(q_conj(link_quats[:, p]), link_quats[:, c]))
    return torch.stack(out, dim=1)


# --------------------------------------------------------------------------- #
#  Reference motion
# --------------------------------------------------------------------------- #
class Reference:
    """Keyframed/mocap reference. Holds, per frame, everything the reward and
    the residual-action target need, all in the root heading frame:
        joint_q  [F,13,4]  joint local rotations
        link_p   [F,14,3]  link positions relative to root (sim units)
        link_av  [F,14,3]  link angular velocities
        com      [F,3]     center of mass (sim units)
    """
    def __init__(self, joint_q, link_p, link_av, com):
        self.joint_q = joint_q
        self.link_p  = link_p
        self.link_av = link_av
        self.com     = com
        self.F = joint_q.shape[0]
        self.joint_target = q_to_rotvec(joint_q)  # [F,13,3] PD targets

    def frame(self, phase):
        # phase: [N] in [0,1] -> nearest frame index (upgrade to slerp later)
        idx = torch.round(phase * (self.F - 1)).long().clamp(0, self.F - 1)
        return idx

    @staticmethod
    def idle_from_state(s0):
        """Build a 1-frame 'stand still' reference from a live state sample so
        the pipeline runs before any motion file exists. Velocities = 0."""
        _, _, rel_pos, quat, _, _ = parse_state(s0)
        link_p = rel_pos.mean(0, keepdim=True)              # [1,14,3]
        link_q = quat[:1]                                   # [1,14,4]
        joint_q = joint_local_quats(link_q)                 # [1,13,4]
        link_av = torch.zeros(1, NUM_LINKS, 3)
        com = (link_p * LINK_MASS.view(1, -1, 1)).sum(1) / TOTAL_MASS
        return Reference(joint_q, link_p, link_av, com)

    @staticmethod
    def load(path):
        """[UPGRADE PATH] load a keyframed motion exported as .npz with arrays
        joint_q[F,13,4], link_p[F,14,3], link_av[F,14,3], com[F,3]."""
        d = np.load(path)
        t = lambda k: torch.tensor(d[k], dtype=torch.float32)
        return Reference(t("joint_q"), t("link_p"), t("link_av"), t("com"))


# --------------------------------------------------------------------------- #
#  State parsing + normalization
# --------------------------------------------------------------------------- #
def parse_state(s):
    # s: [N, STATE_DIM] -> phase, root_h, rel_pos[N,14,3], quat[N,14,4],
    #                       lin_vel[N,14,3], ang_vel[N,14,3]
    phase  = s[:, 0]
    root_h = s[:, 1]
    body   = s[:, 2:].view(-1, NUM_LINKS, PER_LINK)
    rel_pos = body[..., 0:3]
    quat    = body[..., 3:7]
    lin_vel = body[..., 7:10]
    ang_vel = body[..., 10:13]
    return phase, root_h, rel_pos, quat, lin_vel, ang_vel

def _build_state_scale():
    scale = torch.ones(STATE_DIM)
    scale[1] = 0.05                          # root height (~20 units -> ~1)
    for i in range(NUM_LINKS):
        b = 2 + PER_LINK * i
        scale[b:b+3]     = 0.1               # rel_pos
        scale[b+3:b+7]   = 1.0               # quat
        scale[b+7:b+10]  = 0.02              # lin_vel
        scale[b+10:b+13] = 0.1               # ang_vel
    return scale
STATE_SCALE = _build_state_scale()


# --------------------------------------------------------------------------- #
#  Actor-Critic  (DeepMimic sizing: 1024 -> 512, ReLU, separate nets)
# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        def trunk(out):
            return nn.Sequential(
                nn.Linear(STATE_DIM, 1024), nn.ReLU(),
                nn.Linear(1024, 512),       nn.ReLU(),
                nn.Linear(512, out),
            )
        self.actor_net  = trunk(ACTION_DIM)
        self.critic_net = trunk(1)
        # fixed-ish exploration; paper uses a constant diagonal covariance
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), math.log(0.15)))

    def forward(self, s):
        x = (s * STATE_SCALE).clamp(-5.0, 5.0)
        mean = self.actor_net(x)
        std = torch.exp(self.log_std.clamp(-3.0, 0.0))
        return Normal(mean, std), self.critic_net(x)

model = ActorCritic()
opt = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N, eta_min=1e-5)


# --------------------------------------------------------------------------- #
#  Imitation reward (paper §5.3), evaluated from state vs reference@phase
# --------------------------------------------------------------------------- #
def imitation_reward(s, ref):
    phase, root_h, rel_pos, quat, lin_vel, ang_vel = parse_state(s)
    idx = ref.frame(phase)

    # ---- pose: sum of squared joint-rotation differences (unit independent)
    jq_sim = joint_local_quats(quat)                 # [N,13,4]
    jq_ref = ref.joint_q[idx]                        # [N,13,4]
    dq = q_mul(q_conj(jq_sim), jq_ref)
    pose_err = (q_angle(dq) ** 2).sum(-1)            # [N]
    r_pose = torch.exp(-2.0 * pose_err)

    # ---- velocity: joint local angular-velocity difference
    av_sim = torch.stack([ang_vel[:, c] - ang_vel[:, p] for (c, p) in JOINTS], 1)
    av_ref = torch.stack([ref.link_av[idx, c] - ref.link_av[idx, p]
                          for (c, p) in JOINTS], 1)
    vel_err = ((av_sim - av_ref) ** 2).sum(-1).sum(-1)
    r_vel = torch.exp(-0.1 * vel_err)

    # ---- end-effector: position error (converted to metres)
    ee_sim = rel_pos[:, END_EFFECTORS] * UNIT_M
    ee_ref = ref.link_p[idx][:, END_EFFECTORS] * UNIT_M
    end_err = ((ee_sim - ee_ref) ** 2).sum(-1).sum(-1)
    r_end = torch.exp(-40.0 * end_err)

    # ---- center of mass (metres)
    com_sim = (rel_pos * LINK_MASS.view(1, -1, 1)).sum(1) / TOTAL_MASS * UNIT_M
    com_ref = ref.com[idx] * UNIT_M
    com_err = ((com_sim - com_ref) ** 2).sum(-1)
    r_com = torch.exp(-10.0 * com_err)

    return W_POSE * r_pose + W_VEL * r_vel + W_END * r_end + W_COM * r_com

def is_done(s):
    phase, root_h, rel_pos, quat, lin_vel, ang_vel = parse_state(s)
    head_y  = root_h + rel_pos[:, 3, 1]
    chest_y = root_h + rel_pos[:, 2, 1]
    fallen = (head_y < ET_HEAD_MIN) | (chest_y < ET_CHEST_MIN)
    bad = ~torch.isfinite(s).all(dim=1)
    return fallen | bad


# --------------------------------------------------------------------------- #
#  Rollout buffer + GAE  (structure follows the working 2D trainer)
# --------------------------------------------------------------------------- #
class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []
    def store(self, s, a, r, d, lp, v):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.dones.append(d.float()); self.log_probs.append(lp)
        self.values.append(v.squeeze(-1))
    def compute_gae(self, last_state):
        with torch.no_grad():
            _, last_v = model(last_state)
        values = self.values + [last_v.squeeze(-1)]
        gae = torch.zeros(NUM_ENVS)
        adv = [None] * len(self.rewards)
        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + GAMMA * values[t+1] * mask - values[t]
            gae = delta + GAMMA * LAM * mask * gae
            adv[t] = gae.clone()
        self.advantages = adv
        self.returns = [a + v for a, v in zip(adv, self.values)]
        return torch.stack(self.rewards).mean().item()
    def clear(self):
        self.__init__()
buffer = RolloutBuffer()


# --------------------------------------------------------------------------- #
#  UDP bridge to the C++ sim  (same protocol style as 2D)
# --------------------------------------------------------------------------- #
class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv.bind((ip, recv_port))
        self.send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, send_port)
    def _recv(self, n):
        data, _ = self.recv.recvfrom(n * 4)
        return torch.tensor(struct.unpack(f"{n}f", data), dtype=torch.float32)
    def _send(self, vals):
        self.send.sendto(struct.pack(f"{len(vals)}f", *vals), self.addr)
    def get_state(self):
        self._send([-100.0] * ACTION_DIM)                    # [C++ TODO] "just read" sentinel
        return self._recv(STATE_DIM * NUM_ENVS).view(NUM_ENVS, STATE_DIM)
    def step(self, targets):                                  # targets: [NUM_ENVS, ACTION_DIM]
        self._send(targets.flatten().tolist())
        s = self._recv(STATE_DIM * NUM_ENVS).view(NUM_ENVS, STATE_DIM)
        return s
    def reset(self, env_idx, phase):
        # [C++ TODO] set env to reference pose at `phase` (RSI). Until the C++
        # accepts a pose, it re-inits to the default standing pose; we still pass
        # phase so Python's phase clock is seeded correctly.
        self._send([-69.0, float(env_idx), float(phase)])
udp = UDP("127.0.0.1", 5006, 5005)


# --------------------------------------------------------------------------- #
#  PPO update
# --------------------------------------------------------------------------- #
def update():
    states  = torch.stack(buffer.states).view(-1, STATE_DIM)
    actions = torch.stack(buffer.actions).view(-1, ACTION_DIM)
    old_lp  = torch.stack(buffer.log_probs).view(-1)
    returns = torch.stack(buffer.returns).view(-1)
    adv     = torch.stack(buffer.advantages).view(-1)
    adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).clamp(-5, 5)

    n = states.shape[0]
    bs = n // 4
    for _ in range(K_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, bs):
            idx = perm[start:start+bs]
            dist, values = model(states[idx])
            new_lp = dist.log_prob(actions[idx]).sum(-1)
            ratio = torch.exp((new_lp - old_lp[idx]).clamp(-20, 20))
            surr = torch.min(ratio * adv[idx],
                             torch.clamp(ratio, 1-CLIP, 1+CLIP) * adv[idx])
            actor_loss  = -surr.mean()
            critic_loss = (values.squeeze(-1) - returns[idx]).pow(2).mean()
            ent = dist.entropy().sum(-1).mean()
            loss = actor_loss + 0.5 * critic_loss - ENT_COEF * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
    return actor_loss.item(), critic_loss.item()


# --------------------------------------------------------------------------- #
#  Train
# --------------------------------------------------------------------------- #
print("Booting sim link...")
state = udp.get_state()

# Reference: load a motion file if present, else stand-still from live pose.
if os.path.exists("motion.npz"):
    ref = Reference.load("motion.npz")
    print(f"Loaded reference motion: {ref.F} frames")
else:
    ref = Reference.idle_from_state(state)
    print("No motion.npz -> using captured standing pose as 1-frame reference")

phase = torch.rand(NUM_ENVS)          # per-env phase clock (RSI)
dphase = (1.0 / CONTROL_HZ) * (MOTION_FPS / max(ref.F, 1)) if ref.F > 1 else 0.0

start_iter = 0
if os.path.exists("POLICY_3D.pt"):
    ck = torch.load("POLICY_3D.pt")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"]); start_iter = ck["iteration"] + 1
    print(f"Resumed at iteration {start_iter}")

for iteration in range(start_iter, N):
    buffer.clear()
    for t in range(T):
        # inject the reference phase into the observation the policy sees
        state[:, 0] = phase
        with torch.no_grad():
            dist, value = model(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        # residual control: PD target = reference target(phase) + policy offset
        ref_target = ref.joint_target[ref.frame(phase)]          # [N,13,3]
        offset = (action * ACTION_SCALE).view(NUM_ENVS, NUM_JOINTS, 3)
        targets = (ref_target + offset).view(NUM_ENVS, ACTION_DIM)

        next_state = udp.step(targets)
        next_state[:, 0] = phase
        reward = imitation_reward(next_state, ref)
        done = is_done(next_state)

        buffer.store(state, action, reward, done, log_prob, value)

        # advance phase clock; wrap (cyclic) or terminate at clip end
        phase = phase + dphase
        if MOTION_CYCLIC:
            phase = phase % 1.0
        else:
            done = done | (phase >= 1.0)

        state = next_state
        if done.any():
            for i in range(NUM_ENVS):
                if done[i]:
                    p0 = float(torch.rand(1))          # RSI: random restart phase
                    udp.reset(i, p0)
                    phase[i] = p0
            sleep(0.005)
            fresh = udp.get_state()
            state[done.bool()] = fresh[done.bool()]

    avg_r = buffer.compute_gae(state)
    a_loss, c_loss = update()
    scheduler.step()
    print(f"[{iteration}] reward {avg_r:.3f} | actor {a_loss:.3f} | "
          f"critic {c_loss:.3f} | std {torch.exp(model.log_std).mean():.3f}")

    if iteration % 10 == 0 and iteration > 0:
        torch.save({"iteration": iteration, "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict()}, "POLICY_3D.pt")
