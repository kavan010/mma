import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import math
import socket
import struct


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
udp = UDP("127.0.0.1", 5006, 5005)

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
        std = torch.exp(self.log_std.clamp(-3, 0.5))
        return Normal(mean, std)
policy = Policy()
opt = optim.Adam(policy.parameters(), lr=1e-4)