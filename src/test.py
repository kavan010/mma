import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import socket
import struct
import time

NUM_ENVS = 2
state_dim = 3
MAX_ANGLE = math.pi

class ActorCritic(nn.Module):
    def __init__(self, state_dim=3, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.value = nn.Linear(hidden, 1)
    def forward(self, s):
        h = self.shared(s)
        mean = torch.tanh(self.mean(h)) * MAX_ANGLE
        std = torch.exp(self.log_std.clamp(-3, 0.5))
        return Normal(mean, std), self.value(h)

model = ActorCritic()
model.load_state_dict(torch.load("SINGLE_PENDULUM_POLICY.pt"))
model.eval()

class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self):
        data, _ = self.recv_sock.recvfrom(state_dim * NUM_ENVS * 4)
        return struct.unpack(f"{state_dim * NUM_ENVS}f", data)
    def send_actions(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)
    def get_state(self):
        self.send_actions([-100.0, -100.0])
        flat = torch.tensor(self.receive_state(), dtype=torch.float32)
        return flat.view(NUM_ENVS, state_dim)

    def step(self, actions):
        self.send_actions(actions.tolist())
        self.send_actions([-100.0, -100.0])
        flat = torch.tensor(self.receive_state(), dtype=torch.float32)
        return flat.view(NUM_ENVS, state_dim)

udp = UDP("127.0.0.1", 5006, 5005)
state = udp.get_state()

while True:
    with torch.no_grad():
        dist, _ = model(state)
        action = dist.mean

    state = udp.step(action.flatten())
    print(f"angles: {state[:, 0].tolist()}  actions: {action.flatten().tolist()}")

    time.sleep(0.3)

