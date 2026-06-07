import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import socket
import struct

NUM_ENVS = 5
action_dim = 10
state_dim = 34
MAX_ANGLE = math.pi


class ActorCritic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(hidden, 1)

    def forward(self, s):
        h = self.shared(s)
        mean = self.mean(h)
        std = torch.exp(self.log_std.clamp(-3, 0.5))
        dist = Normal(mean, std)
        value = self.value(h)
        return dist, value

model = ActorCritic()
model.load_state_dict(torch.load("WHOLE_BODY_POLICY.pt"))
model.eval()
print("Model loaded.")

def isDone(state):
    # ---- Body (torso) tilted too far from vertical ----
    body_angle  = torch.atan2(state[:, 3], state[:, 4])
    body_fallen = torch.abs(body_angle - math.pi / 2) > 0.8   # ~45° tolerance

    # ---- Hip too low = legs gave out / collapsed ----
    hip_low = state[:, 33] < 0.12

    # ---- Head flopped too far from vertical ----
    head_angle   = torch.atan2(state[:, 0], state[:, 1])
    head_flopped = torch.abs(head_angle - math.pi / 2) > 1.2  # ~70° tolerance

    # ---- Legs crossed ----
    legs_crossed = state[:, 22] < (state[:, 19] - 0.3)
    
    rel_sinR  = state[:, 27] * state[:, 22] - state[:, 28] * state[:, 21]
    rel_cosR  = state[:, 28] * state[:, 22] + state[:, 27] * state[:, 21]
    rel_calfR = torch.atan2(rel_sinR, rel_cosR)

    rel_sinL  = state[:, 24] * state[:, 19] - state[:, 25] * state[:, 18]
    rel_cosL  = state[:, 25] * state[:, 19] + state[:, 24] * state[:, 18]
    rel_calfL = torch.atan2(rel_sinL, rel_cosL)

    knee_blown = (torch.abs(rel_calfR) > 1.4) | (torch.abs(rel_calfL) > 1.4)  # ~80°

    return body_fallen | hip_low | legs_crossed | knee_blown


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
        return flat.view(NUM_ENVS, state_dim)

    def step(self, target_angles):
        self.send_actions(target_angles.tolist())
        self.send_actions([-100.0, -100.0])
        flat = torch.tensor(self.receive_state(), dtype=torch.float32)
        return flat.view(NUM_ENVS, state_dim)

    def send_reset(self, env_idx):
        self.send_actions([-69.0, float(env_idx)])

udp = UDP("127.0.0.1", 5006, 5005)
state = udp.get_state()

print("Running model (Ctrl+C to stop)...")
while True:
    with torch.no_grad():
        dist, _ = model(state)
        action = dist.mean
        action = torch.clamp(action, -MAX_ANGLE, MAX_ANGLE)

    state = udp.step(action.flatten())

    # Reset environments that have fallen
    # done = isDone(state)
    # if done.any():
    #     for i in range(NUM_ENVS):
    #         if done[i]:
    #             udp.send_reset(i)
    #     fresh = udp.get_state()
    #     state[done] = fresh[done]