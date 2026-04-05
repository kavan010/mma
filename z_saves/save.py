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

def get_state():
    udp.send_actions([-100.0])
    angle1, angVel1, angle2, angVel2, x, y = udp.receive_state()
    s = torch.tensor([angle1, angVel1, angle2, angVel2, x, y], dtype=torch.float32)
    return s
def step(target_angle):
    udp.send_actions([target_angle])
    udp.send_actions([-100.0])
    angle1, angVel1, angle2, angVel2, x, y = udp.receive_state()
    s = torch.tensor([angle1, angVel1, angle2, angVel2, x, y], dtype=torch.float32)
    return s
# def reward(state):
#     joint_y = state[4].item()
#     tip_y   = state[5].item()
#     return -(1-tip_y)*10.0
def is_failed(state):
    return state[5].item() < state[4].item()
def reward(state):
    tip_angle = state[2].item() * 6.28  # unnormalize
    upright_error = tip_angle - (math.pi / 2)
    reward = math.cos(upright_error)  # -1 when fallen, +1 when upright
    reward += 0.1  # survival bonus per timestep
    reward -= 0.001 * action**2  # small action penalty to discourage jitter

policy = Policy()
opt = optim.Adam(policy.parameters(), lr=1e-4)


for episode in range(10000):
    log_probs = []
    rewards = []
    state = get_state()
    for t in range(400):
        dist = policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        target_angle = float(torch.remainder(action, 2*math.pi))
        next_state = step(target_angle)
        log_probs.append(log_prob)

        if is_failed(next_state):
            fall_penalty = -15.0 * (1 + (400 - t) / 200)   # range -5 to -10
            rewards.append(fall_penalty)
            udp.send_actions([-69.0])
            break

        rewards.append(reward(next_state))
        state = next_state

    if len(log_probs) < 2:
        print(f"EP: {episode}, skipped")
        continue

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = sum(-lp * G for lp, G in zip(log_probs, returns)) / len(log_probs)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    opt.step()

    print(f"EP: {episode}, Steps: {len(log_probs)}, Avg Reward: {sum(rewards)/len(rewards):.4f}")

torch.save(policy.state_dict(), "double_pendulum_policy.pt")



