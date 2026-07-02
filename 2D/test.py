import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import socket
import struct
from time import sleep
import os

# params
NUM_ENVS = 1
action_dim = 10
state_dim = 39
MAX_ANGLE = math.pi
CHECKPOINT = "IMPULSE_POLICY.pt"


# ------------------------- model core ----------------------
class ActorCritic(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()

        # actor
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # critic
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        s_in = s.clone()
        s_in[:, 2:33:3] = s_in[:, 2:33:3] / 20.0
        mean = self.actor_net(s_in)
        std = torch.exp(self.log_std.clamp(-3, 0.0))
        dist = Normal(mean, std)

        value = self.critic_net(s_in)

        return dist, value
model = ActorCritic()
model.eval()


class Env:
    # reward n' done functions (kept identical to training script,
    # only used here so udp.step() still works without changes)
    def getReward(self, state):
        b_ang = torch.atan2(state[:, 3], state[:, 4])
        upright = torch.exp(-4.0 * (b_ang - math.pi / 2) ** 2)

        hip_y = state[:, 33]
        height = torch.exp(-40.0 * (hip_y - 0.23) ** 2)

        hip_x = state[:, 34]
        tgt_x = (torch.round(hip_x * 6) / 6)
        drift = torch.exp(-12.0 * (hip_x - tgt_x) ** 2)

        l_ang = torch.atan2(state[:, 18], state[:, 19])
        r_ang = torch.atan2(state[:, 21], state[:, 22])
        sym = torch.exp(-4.0 * (l_ang + r_ang - math.pi) ** 2)
        space = torch.exp(-5.0 * (torch.abs(l_ang - r_ang) - 0.35) ** 2)

        foot_l = state[:, 37]
        foot_r = state[:, 38]
        feet_contact = foot_l * foot_r

        jitter = 0.006 * torch.sum(state[:, 2:33:3] ** 2, dim=-1)

        return (
            3.0 * upright * height * drift
            + 0.2 * sym
            + 0.2 * space
            + 0.2 * feet_contact
            - jitter
            + 0.3
        )

    def isDone(self, state):
        hip_y = state[:, 33]
        hip_low = hip_y < 0.115

        b_ang = torch.atan2(state[:, 3], state[:, 4])
        fallen = torch.abs(b_ang - math.pi / 2) > 1.5

        return hip_low | fallen
buffer = Env()


class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self, num_floats=state_dim * NUM_ENVS):
        data, _ = self.recv_sock.recvfrom(num_floats * 4)
        return struct.unpack(f"{num_floats}f", data)

    def send_actions(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)

    def get_state(self):
        self.send_actions([-100.0, -100.0])
        flat = torch.tensor(self.receive_state(), dtype=torch.float32)
        s = flat.view(NUM_ENVS, state_dim)
        return s

    def step(self, target_angle):
        self.send_actions(target_angle.tolist())
        self.send_actions([-100.0, -100.0])
        flat = torch.tensor(self.receive_state(), dtype=torch.float32)
        s = flat.view(NUM_ENVS, state_dim)
        return s, buffer.getReward(s), buffer.isDone(s)

    def send_reset(self, env_idx):
        self.send_actions([-69.0, float(env_idx)])
udp = UDP("127.0.0.1", 5006, 5005)

# ---------------------- load model ----------------------
print("Loading policy...")
if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint from iteration {ckpt.get('iteration', '?')}")
    else:
        model.load_state_dict(ckpt)  # handles legacy flat checkpoint
        print("Loaded legacy model")
else:
    print(f"WARNING: {CHECKPOINT} not found, running with randomly initialized weights")

# ---------------------- run ----------------------
print("Starting test loop...")
state_batch = udp.get_state()

while True:
    with torch.no_grad():
        dist, value = model(state_batch)
        action = dist.mean  # deterministic action for testing/inference

    action_np = torch.clamp(action, -MAX_ANGLE, MAX_ANGLE).numpy().flatten()
    next_state, reward, done = udp.step(action_np)

    print(f"reward: {reward.mean().item():.3f}")

    state_batch = next_state
    if done.any():
        for i in range(NUM_ENVS):
            if done[i]:
                udp.send_reset(i)
        sleep(0.005)
        fresh = udp.get_state()
        for i in range(NUM_ENVS):
            if done[i]:
                state_batch[i] = fresh[i]