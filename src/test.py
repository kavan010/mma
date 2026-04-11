from time import time, sleep

import torch
import torch.nn as nn
from torch.distributions import Normal
import socket
import struct
import math


# --- UDP (same as yours) ---
class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self, num_floats=6):
        data, _ = self.recv_sock.recvfrom(num_floats * 4)
        return struct.unpack(f"{num_floats}f", data)
    def send_actions(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)
    def get_state(self):
        self.send_actions([-100.0])
        s = torch.tensor(self.receive_state(), dtype=torch.float32)
        return s
    def step(self, target_angle):
        self.send_actions([target_angle])
        self.send_actions([-100.0])
        s = torch.tensor(self.receive_state(), dtype=torch.float32)
        return s


# --- POLICY (same architecture) ---
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, s):
        h = self.net(s)
        mean = self.mean(h)
        std = self.log_std.exp()
        return Normal(mean, std)


# --- LOAD MODEL ---
policy = Policy()
policy.load_state_dict(torch.load("ppo_double_pendulum.pt"))
policy.eval()

def isFailed(state):
    sin1, cos1, sin2, cos2, angVel1, angVel2 = state
    theta1 = torch.atan2(sin1, cos1)
    theta2 = torch.atan2(sin2, cos2)
    err1 = torch.atan2(torch.sin(theta1 - math.pi), torch.cos(theta1 - math.pi))
    err2 = torch.atan2(torch.sin(theta2 - math.pi), torch.cos(theta2 - math.pi))
    return abs(err2) > 1.5 or abs(err1) > 1.5

# --- RUN ---
udp = UDP("127.0.0.1", 5006, 5005)

state = udp.get_state()

while True:
    with torch.no_grad():
        dist = policy(state)
        raw_action = dist.mean
        action = 0.5 * torch.tanh(raw_action)

    if (isFailed(state)):
        print("Episode ended. Resetting environment.")
        udp.send_actions([-69.0])  # Signal to reset
        state = udp.get_state()
        continue

    target_angle = math.pi + action.item()

    state = udp.step(target_angle)

    print(f"Action: {action.item():.4f}")
    #sleep(1/15)