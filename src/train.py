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
        std = self.log_std.exp()
        return Normal(mean, std)
policy = Policy()

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.net(s)
value_net = ValueNet()

def reward(state):
    sin1, cos1, sin2, cos2, angVel1, angVel2 = state
    theta1 = torch.atan2(sin1, cos1)
    theta2 = torch.atan2(sin2, cos2)
    err1 = torch.atan2(torch.sin(theta1 - math.pi), torch.cos(theta1 - math.pi))
    err2 = torch.atan2(torch.sin(theta2 - math.pi), torch.cos(theta2 - math.pi))
    r = torch.cos(err1) + torch.cos(err2)
    r -= 0.01 * (angVel1**2 + angVel2**2)
    return r
def isFailed(state):
    sin1, cos1, sin2, cos2, angVel1, angVel2 = state
    theta1 = torch.atan2(sin1, cos1)
    theta2 = torch.atan2(sin2, cos2)
    err1 = torch.atan2(torch.sin(theta1 - math.pi), torch.cos(theta1 - math.pi))
    err2 = torch.atan2(torch.sin(theta2 - math.pi), torch.cos(theta2 - math.pi))
    return abs(err2) > 1.5 or abs(err1) > 1.5

opt = optim.Adam(policy.parameters(), lr=1e-4)
value_opt = optim.Adam(value_net.parameters(), lr=1e-4)


T = 1
# for iteration in range(N):
while True:
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    state = udp.get_state()
    for t in range(T):
        dist = policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = 0.5 * torch.tanh(action)
        value = value_net(state)

        next_state = udp.step(action.item())
        r = reward(state)
        print(f"Action: {action.item():.3f} | Reward: {r.item():.3f}")
        done = isFailed(state)

        states.append(state)
        actions.append(action)
        rewards.append(r)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)

        state = next_state

        if done:
            udp.send_actions([-69.0])

    # advantages = compute_advantages(rewards, values)
    # for _ in range(K_epochs):
    #     update_policy(states, actions, advantages)
        

torch.save(policy.state_dict(), "ppo_double_pendulum.pt")