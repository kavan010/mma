import torch
import torch.nn as nn
from torch.distributions import Normal
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
        std = self.log_std.exp()
        return Normal(mean, std)

policy = Policy()
policy.load_state_dict(torch.load("double_pendulum_policy.pt"))
policy.eval()
print("Model loaded.")

def get_state():
    udp.send_actions([-100.0])
    angle1, angVel1, angle2, angVel2, joint_y, tip_y = udp.receive_state()
    return torch.tensor([angle1, angVel1, angle2, angVel2, joint_y, tip_y], dtype=torch.float32)

def step(target_angle):
    udp.send_actions([target_angle])
    udp.send_actions([-100.0])
    angle1, angVel1, angle2, angVel2, joint_y, tip_y = udp.receive_state()
    return torch.tensor([angle1, angVel1, angle2, angVel2, joint_y, tip_y], dtype=torch.float32)

def reward(state):
    joint_y = state[4].item()
    tip_y   = state[5].item()
    return tip_y - joint_y

def is_failed(state):
    joint_y = state[4].item()
    tip_y   = state[5].item()
    return tip_y < joint_y - 0.1

NUM_EPISODES = 2000
STEPS = 400

for episode in range(NUM_EPISODES):
    state = get_state()
    rewards = []

    for t in range(STEPS):
        with torch.no_grad():
            dist = policy(state)
            action = dist.mean                  # deterministic, no exploration
            target_angle = float(torch.remainder(action, 2 * math.pi))

        next_state = step(target_angle)
        r = reward(next_state)
        rewards.append(r)
        state = next_state

        joint_y = next_state[4].item()
        tip_y   = next_state[5].item()
        print(f"  t={t:03d} | rod1={next_state[0].item():.3f} | rod2={next_state[2].item():.3f} | joint_y={joint_y:.3f} | tip_y={tip_y:.3f} | r={r:.4f}")

        if is_failed(next_state):
            udp.send_actions([-69.0])
            print("  !! FAILED")
            break

    avg = sum(rewards) / len(rewards)
    print(f"\nEP {episode} | Steps: {len(rewards)} | Avg Reward: {avg:.4f}\n")