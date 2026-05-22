import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import socket
import struct

NUM_ENVS = 3
action_dim = 4
state_dim = 16
MAX_ANGLE = math.pi


class ActorCritic(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.5)
        self.value = nn.Linear(hidden, 1)

    def forward(self, s):
        h = self.shared(s)
        mean = self.mean(h)
        std = torch.exp(self.log_std.clamp(-3, 0.5))
        dist = Normal(mean, std)
        value = self.value(h)
        return dist, value
model = ActorCritic()
model.load_state_dict(torch.load("BIPEDAL_TWOJOINT_STANDING_POLICY.pt"))
model.eval()
print("Model loaded.")


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
        action = dist.mean  # use mean action (no randomness)
        action = torch.clamp(action, -MAX_ANGLE, MAX_ANGLE)

    state = udp.step(action.flatten())