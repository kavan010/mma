import torch
torch.set_num_threads(1)  # tiny model, more threads just adds contention with the C++ sim process
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
import math
import socket
import struct
from time import sleep
import os


# params
NUM_ENVS   = 25
NUM_LINKS  = 14
NUM_JOINTS = 13

NUM_SKILLS = 6
TARGET_DIM = 3
RAW_STATE_DIM = 2 + NUM_LINKS * 13          # what the C++ sim actually sends per env
state_dim = RAW_STATE_DIM + NUM_SKILLS + TARGET_DIM  # what the policy/critic sees
action_dim = 3 * NUM_JOINTS

ENT_COEF   = 0.0
UNIT_M       = 0.05
ET_HEAD  = 26.0
ET_CHEST = 18.0

N = 100000
T = 1024
K_epochs = 4

JOINTS = [(7,6),(8,7),(13,8),(9,8),(10,8),(11,9),(12,10),(4,6),(5,6),(2,4),(3,5),(0,2),(1,3)]
CHILD  = torch.tensor([c for c, p in JOINTS])
PARENT = torch.tensor([p for c, p in JOINTS])
END_EFFECTORS = [0, 1, 11, 12]
LINK_MASS = torch.tensor([1., 1, 2.5, 2.5, 4.5, 4.5, 5, 7, 8, 1.8, 1.8, 1.2, 1.2, 3])

ROOT = 6
ROOT_ANCHORS = [
    (7, 0, torch.tensor([0.,  1.8,  0.0]), torch.tensor([0.0, -1.8, 0.0])),  # abs
    (4, 7, torch.tensor([0., -1.6,  1.5]), torch.tensor([-4.6, 0.0, 0.0])),  # thighR
    (5, 8, torch.tensor([0., -1.6, -1.5]), torch.tensor([-4.6, 0.0, 0.0])),  # thighL
]


def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw*bw - ax*bx - ay*by - az*bz,
                        aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw], dim=-1)
def qconj(q):
    return q * torch.tensor([1., -1, -1, -1])
def qrot(q, v):
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]

def derive_root_quat(joint_q, link_p):
    # Kabsch/orthogonal-Procrustes alignment.
    F = joint_q.shape[0]
    root_q = torch.zeros(F, 4)
    for i in range(F):
        A = torch.zeros(3, 3)
        for child, jrow, anchorA, anchorB in ROOT_ANCHORS:
            d_local = anchorA - qrot(joint_q[i, jrow], anchorB)
            d_world = link_p[i, child] - link_p[i, ROOT]
            A += torch.outer(d_world, d_local)
        U, _, Vh = torch.linalg.svd(A)
        d = torch.sign(torch.det(U @ Vh))
        R = U @ torch.diag(torch.tensor([1.0, 1.0, d])) @ Vh
        w = torch.clamp(torch.sqrt(torch.clamp(1 + R[0,0] + R[1,1] + R[2,2], min=0)) / 2, min=1e-8)
        q = torch.stack([w, (R[2,1]-R[1,2])/(4*w), (R[0,2]-R[2,0])/(4*w), (R[1,0]-R[0,1])/(4*w)])
        root_q[i] = q / q.norm()
    return root_q


# ------------------------- model core ----------------------
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # actor
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim)
        )
        
        # fixed logstd following what deepmimic did
        self.register_buffer("log_std", torch.full((action_dim,), math.log(0.05)))

        # critic
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, s):
        s_in = s.clone()
        body = s_in[:, 2:RAW_STATE_DIM].view(-1, NUM_LINKS, 13)
        body[..., 1] -= s_in[:, 1:2]  # link y: absolute -> root-relative, before root itself is scaled
        s_in[:, 1] *= 0.05
        body[..., 0:3] *= 0.2
        body[..., 7:13] *= 0.05
        s_in = torch.clamp(s_in, -5.0, 5.0)

        mean = self.actor_net(s_in)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        value = self.critic_net(s_in)

        return dist, value
model = ActorCritic()
opt = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N, eta_min=1e-5)

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done.float())
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze(-1))

    def compute_gae(self, gamma=0.95, lam=0.95):
        avg_reward = torch.stack(self.rewards).mean().item()
        with torch.no_grad():
            _, last_value = model(state_batch)
        values = self.values + [last_value.squeeze(-1)]

        gae = torch.zeros(NUM_ENVS)
        adv = []
        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            adv.insert(0, gae.clone())

        self.advantages = adv
        self.returns = [a + v for a, v in zip(adv, self.values)]
        print(f"[{iteration}] reward: {avg_reward:.3f} | advantages mean: {torch.stack(adv).mean().item():.3f}")
        return avg_reward

    # reward n' done functions
    def getReward(self, state):
        i = ref.idx(state[:, 0])
        body = state[:, 2:RAW_STATE_DIM].view(-1, NUM_LINKS, 13)
        pos, quat, ang = body[..., 0:3], body[..., 3:7], body[..., 10:13]

        q_sim = qmul(qconj(quat[:, PARENT]), quat[:, CHILD])
        dq = qmul(qconj(q_sim), ref.joint_q[i])
        pose_err = (2.0 * torch.acos(dq[..., 0].abs().clamp(max=1.0))).pow(2).sum(-1)

        dq_root = qmul(qconj(quat[:, ROOT]), ref.root_q[i])
        pose_err = pose_err + (2.0 * torch.acos(dq_root[..., 0].abs().clamp(max=1.0))).pow(2)

        r_pose = torch.exp(-2.0 * pose_err)

        dv = (ang[:, CHILD] - ang[:, PARENT]) - ref.joint_av[i]
        r_vel = torch.exp(-0.1 * dv.pow(2).sum(-1).sum(-1))

        de = (pos[:, END_EFFECTORS] - ref.link_p[i][:, END_EFFECTORS]) * UNIT_M
        r_end = torch.exp(-40.0 * de.pow(2).sum(-1).sum(-1))

        com = (pos * LINK_MASS.view(1, -1, 1)).sum(1) / LINK_MASS.sum()
        dc = (com - ref.com[i]) * UNIT_M
        r_com = torch.exp(-10.0 * dc.pow(2).sum(-1))

        # drift penalty
        root_vel_xz = body[:, ROOT, 7:10:2]  # pelvis vel.x, vel.z (raw, unrelativized)
        r_drift = torch.exp(-1.0 * root_vel_xz.pow(2).sum(-1))

        return 0.65 * r_pose + 0.05 * r_vel + 0.15 * r_end + 0.0 * r_com + 0.15 * r_drift
    def isDone(self, state):
        body = state[:, 2:RAW_STATE_DIM].view(-1, NUM_LINKS, 13)
        head_y  = body[:, 13, 1]  # link y is world-absolute, not root-relative
        chest_y = body[:, 8, 1]
        return (head_y < ET_HEAD) | (chest_y < ET_CHEST)

    def clear(self):
        self.__init__()
buffer = RolloutBuffer()

class Reference:
    def __init__(self, state):
        if os.path.exists("motion.npz"):
            d = np.load("motion.npz")
            self.joint_q  = torch.tensor(d["joint_q"],  dtype=torch.float32)
            self.link_p   = torch.tensor(d["link_p"],   dtype=torch.float32)
            self.joint_av = torch.tensor(d["joint_av"], dtype=torch.float32)
            self.com      = torch.tensor(d["com"],      dtype=torch.float32)
            print(f"Loaded reference motion: {self.joint_q.shape[0]} frames")
        else:
            body = state[0, 2:RAW_STATE_DIM].view(NUM_LINKS, 13)
            quat = body[:, 3:7]
            self.joint_q  = qmul(qconj(quat[PARENT]), quat[CHILD]).unsqueeze(0)
            self.link_p   = body[:, 0:3].unsqueeze(0)
            self.joint_av = torch.zeros(1, NUM_JOINTS, 3)
            self.com      = (self.link_p * LINK_MASS.view(1, -1, 1)).sum(1) / LINK_MASS.sum()
            print("No motion.npz -> using captured standing pose as 1-frame reference")

        self.root_q = derive_root_quat(self.joint_q, self.link_p)
        self.F = self.joint_q.shape[0]
        self.dphase = 1.0 / self.F if self.F > 1 else 0.0

    def idx(self, phase):
        return torch.round(phase * (self.F - 1)).long().clamp(0, self.F - 1)
    def target(self, phase, action):
        # policy outputs the ABSOLUTE PD target directly, so the motion is stored
        # in the network weights. The reference is used only in the reward (getReward),
        # never in the action path -> at test time no reference is needed.
        return action.view(NUM_ENVS, NUM_JOINTS, 3)

def pad_skill(raw):
    n = raw.shape[0]
    skills = torch.zeros(n, NUM_SKILLS)
    skills[:, 0] = 1.0
    targets = torch.zeros(n, TARGET_DIM)
    return torch.cat([raw, skills, targets], dim=1)

class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self, num_floats=RAW_STATE_DIM * NUM_ENVS):
        data, _ = self.recv_sock.recvfrom(num_floats * 4)
        return struct.unpack(f"{num_floats}f", data)
    def send_actions(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)

    def get_state(self):
        self.send_actions([-100.0] * action_dim)
        raw = torch.tensor(self.receive_state(), dtype=torch.float32).view(NUM_ENVS, RAW_STATE_DIM)
        return pad_skill(raw)

    def step(self, target):
        self.send_actions(target.flatten().tolist())
        raw = torch.tensor(self.receive_state(), dtype=torch.float32).view(NUM_ENVS, RAW_STATE_DIM)
        raw[:, 0] = phase
        s = pad_skill(raw)
        return s, buffer.getReward(s), buffer.isDone(s)
    def send_reset(self, env_idx, ph):
        self.send_actions([-69.0, float(env_idx), float(ph)])
udp = UDP("127.0.0.1", 5006, 5005)

def updateModel(K_epochs):
    states        = torch.stack(buffer.states).view(-1, state_dim)
    actions       = torch.stack(buffer.actions).view(-1, action_dim)
    old_log_probs = torch.stack(buffer.log_probs).view(-1)
    returns       = torch.stack(buffer.returns).view(-1).clamp(-300, 300)
    advantages    = torch.stack(buffer.advantages).view(-1)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.clamp(-5, 5)

    N_samples = states.shape[0]
    batch_size = N_samples // 4

    for epoch in range(K_epochs):
        perm = torch.randperm(N_samples)
        for start in range(0, N_samples, batch_size):
            idx = perm[start:start + batch_size]

            dist, values = model(states[idx])
            new_log_probs = dist.log_prob(actions[idx]).sum(-1)
            entropy = dist.entropy().sum(-1)

            ratio = torch.exp((new_log_probs - old_log_probs[idx]).clamp(-20, 20))
            clipped = torch.clamp(ratio, 0.8, 1.2)
            surr = torch.min(ratio * advantages[idx], clipped * advantages[idx])
            surr = torch.where(advantages[idx] < 0, torch.max(surr, 3.0 * advantages[idx]), surr)

            actor_loss   = -surr.mean()
            critic_loss  = (values.squeeze(-1) - returns[idx]).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + ENT_COEF * entropy_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

    print(
        f"loss {loss.item():.2f} | "
        f"actor {actor_loss.item():.3f} | "
        f"critic {critic_loss.item():.3f} | "
        f"std {torch.exp(model.log_std).mean().item():.3f}"
    )

# ---------------------- load reference n' model ----------------------
print("Starting training loop...")
state_batch = udp.get_state()
phase = torch.rand(NUM_ENVS)
ref = Reference(state_batch)

start_iteration = 0
if os.path.exists("POLICY_3D.pt"):
    ckpt = torch.load("POLICY_3D.pt")
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_iteration = ckpt['iteration'] + 1
    print(f"Resuming from iteration {start_iteration}")


# ---------------------- train ----------------------
for iteration in range(start_iteration, N):
    current_std = max(0.05, 0.15 - (0.10 * (iteration / 1100.0)))
    model.log_std.data.fill_(math.log(current_std))
    buffer.clear()

    # data collection
    for t in range(T):
        state_batch[:, 0] = 0
        with torch.no_grad():
            dist, value = model(state_batch)
        action = torch.clamp(dist.sample(), -math.pi, math.pi)
        log_prob = dist.log_prob(action).sum(-1)

        # step and store data
        target = ref.target(phase, action)
        next_state, reward, done = udp.step(target.view(NUM_ENVS, action_dim))
        bad = ~torch.isfinite(next_state).all(dim=1)
        done = done | bad
        buffer.store(state_batch, action, reward, done, log_prob, value)

        # reset envs
        state_batch = next_state
        phase = (phase + ref.dphase) % 1.0
        if done.any():
            for i in range(NUM_ENVS):
                if done[i]:
                    phase[i] = float(torch.rand(1))
                    udp.send_reset(i, phase[i])
            sleep(0.005)
            fresh = udp.get_state()
            for i in range(NUM_ENVS):
                if done[i]:
                    state_batch[i] = fresh[i]

    # advantage estimation
    buffer.compute_gae()

    # update model
    updateModel(K_epochs)

    # update lr and save
    scheduler.step()
    if (iteration % 10 == 0 and iteration > 0):
        torch.save({
            'iteration': iteration,
            'model':     model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, "POLICY_3D.pt")
