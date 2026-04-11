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
value_opt = optim.Adam(value_net.parameters(), lr=3e-4)


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
def compute_advantages(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    advantages = []
    advantage = 0

    values = values + [last_value]
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        advantage = delta + gamma * lam * mask * advantage
        advantages.insert(0, advantage)
    advantages = torch.stack(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #print(f"Advantages (normalized): {advantages}")
    return advantages
def update_policy(states, actions, old_log_probs, advantages, returns):
    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs).detach()
    advantages = advantages.detach()
    returns = returns.detach()

    dist = policy(states)
    new_log_probs = dist.log_prob(actions).sum(dim=-1)
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

    values_pred = value_net(states).squeeze()
    value_loss = (returns - values_pred).pow(2).mean()
    entropy = dist.entropy().mean()

    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    opt.zero_grad()
    value_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
    opt.step()
    value_opt.step()
    
    return entropy


N = 1000
T = 1000
K_epochs = 10
for iteration in range(N):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    # printing stuffs
    episode_rewards = []
    episode_lengths = []
    episode_len = 0
    episode_reward = 0.0
    num_resets = 0

    state = udp.get_state()
    for t in range(T):
        dist = policy(state)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum()
        action = raw_action.clamp(-0.5, 0.5)
        value = value_net(state)

        next_state = udp.step(action.item())
        r = reward(next_state)
        done = isFailed(next_state)

        # print(f"log_prob: {log_prob.item()}, Action: {action.item()}, Reward: {r.item()}, Done: {done}")
        states.append(state)      # ← move appends BEFORE the done block
        actions.append(raw_action)
        rewards.append(r)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)

        episode_reward += r.item()
        episode_len += 1

        if done:
            udp.send_actions([-69.0])
            rewards[-1] = rewards[-1] - 50.0
            episode_reward -= 50.0          # keep display in sync with actual reward
            next_state = udp.get_state()
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_len)
            episode_reward = 0.0
            episode_len = 0
            num_resets += 1

        state = next_state

    # printing stufs
    if episode_len > 0:
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_len)

    last_value = value_net(state)
    values = [v.detach() for v in values]
    last_value = last_value.detach()
    last_value_det = last_value.detach()
    advantages = compute_advantages(rewards, values, dones, last_value_det)
    returns = (advantages + torch.stack(values).squeeze()).detach()

    for _ in range(K_epochs):
        entropy = update_policy(states, actions, log_probs, advantages, returns)


    # printing stufs
    mean_ep_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    mean_ep_len    = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
    mean_ep_reward_per_step = mean_ep_reward / mean_ep_len if mean_ep_len > 0 else 0

    print(
        f"Iter {iteration:4d} | "
        f"EpRew: {mean_ep_reward:7.2f} | "
        f"EpLen: {mean_ep_len:6.1f} | "
        f"Rew/Step: {mean_ep_reward_per_step:5.3f} | "
        f"Resets: {num_resets:3d} | "
        f"Entropy: {entropy:.3f} | "
        f"logstd: {policy.log_std.item():.3f} | "
        f"LastVal: {last_value_det.item():.3f}"
    )
    print(f"log_std grad: {policy.log_std.grad}, log_std val: {policy.log_std.item():.4f}\n")
        

torch.save(policy.state_dict(), "ppo_double_pendulum.pt")