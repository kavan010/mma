import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import socket
import struct
import time

# --- Constants matching train.py ---
NUM_ENVS = 1
state_dim = 10
action_dim = 2
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
        return Normal(mean, std), self.value(h)

# --- Load Model ---
model = ActorCritic()
try:
    model.load_state_dict(torch.load("TORSO_BALANCING_POLICY.pt"))
    print("Model loaded successfully.")
except:
    print("Could not find TORSO_BALANCING_POLICY.pt, using random weights.")
model.eval()

class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self):
        # We expect state_dim * NUM_ENVS floats
        data, _ = self.recv_sock.recvfrom(state_dim * NUM_ENVS * 4)
        return struct.unpack(f"{state_dim * NUM_ENVS}f", data)

    def send_actions(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)

    def get_state(self):
        self.send_actions([-100.0] * (NUM_ENVS * action_dim))
        flat = torch.tensor(self.receive_state(), dtype=torch.float32)
        return flat.view(NUM_ENVS, state_dim)

    def step(self, actions):
        self.send_actions(actions.tolist())
        self.send_actions([-100.0] * (NUM_ENVS * action_dim))
        flat = torch.tensor(self.receive_state(), dtype=torch.float32)
        return flat.view(NUM_ENVS, state_dim)

    def send_reset(self, env_idx):
        self.send_actions([-69.0, float(env_idx)])

def is_done(state):
    # Matches the termination criteria in train.py
    hip_angle = torch.atan2(state[:, 0], state[:, 1])
    return torch.abs(hip_angle) > 0.8

udp = UDP("127.0.0.1", 5006, 5005)
state = udp.get_state()

print("Starting Inference...")
while True:
    with torch.no_grad():
        # Use mean for testing (deterministic) instead of sampling
        dist, _ = model(state)
        action = dist.mean
        
    # Clamp to ensure we don't send crazy values to the C++ joints
    action_clamped = torch.clamp(action, -MAX_ANGLE, MAX_ANGLE)
    state = udp.step(action_clamped.flatten())
    
    # Check if we need to reset
    if is_done(state).any():
        print("Fell over! Resetting...")
        udp.send_reset(0)
        time.sleep(0.1)
        state = udp.get_state()

    # Reduced sleep for smoother visualization compared to train.py
    time.sleep(1.0/60.0)