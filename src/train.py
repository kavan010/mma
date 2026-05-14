import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import math
import socket
import struct
from time import sleep

NUM_ENVS = 3
action_dim = 2
state_dim = 10

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

        # policy head
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.5)

        # value head
        self.value = nn.Linear(hidden, 1)

    def forward(self, s):
        h = self.shared(s)

        # ---- policy ----
        mean = self.mean(h)
        std = torch.exp(self.log_std.clamp(-3, 0.5))
        dist = Normal(mean, std)

        # ---- value ----
        value = self.value(h)

        return dist, value
model = ActorCritic()
opt = optim.Adam(model.parameters(), lr=1e-4)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done.float())
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze(-1))
    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        last_v = last_value.squeeze(-1)
        values = self.values + [last_v]

        gae = torch.zeros(NUM_ENVS)
        adv = []

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            # check with future state if prediction was opt or pess
            delta = self.rewards[t] + gamma * values[t+1] * mask - values[t]
            # check over time
            gae = delta + gamma * lam * mask * gae
            adv.insert(0, gae)

        self.advantages = adv
        self.returns = [a + v for a, v in zip(adv, self.values)]

    def clear(self):
        self.__init__()
buffer = RolloutBuffer()


def getReward(state):
    hip_angle  = torch.atan2(state[:, 0], state[:, 1])
    hip_y_norm = state[:, 9] / 0.11

    # --- Normalize angle to [0,1]
    angle_norm = (torch.cos(hip_angle) + 1) / 2

    # --- Normalize height to [0,1] using known range
    height_norm = (hip_y_norm - 0.27) / (1.10 - 0.27)
    height_norm = torch.clamp(height_norm, 0.0, 1.0)

    # --- True 60 / 40 mix
    main_reward = 0.6 * angle_norm + 0.4 * height_norm
    main_reward = main_reward - 0.7

    # --- Penalties
    stability_penalty = 0.02 * state[:, 2] ** 2
    leg_penalty = 0.005 * (state[:, 5] ** 2 + state[:, 8] ** 2)

    return main_reward - stability_penalty - leg_penalty

def isDone(state):
    hip_angle = torch.atan2(state[:, 0], state[:, 1])
    done = torch.abs(hip_angle) > 0.8
    return done

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
        s = flat.view(NUM_ENVS, state_dim)  
        return s
    def step(self, target_angle):
        self.send_actions(target_angle.tolist())
        self.send_actions([-100.0, -100.0])
        flat = torch.tensor(self.receive_state(), dtype=torch.float32)
        s = flat.view(NUM_ENVS, state_dim)  
        return s, getReward(s), isDone(s)
    def send_reset(self, env_idx):
        self.send_actions([-69.0, float(env_idx)])
udp = UDP("127.0.0.1", 5006, 5005)



N = 500
T = 1024
K_epochs = 10
state_batch = udp.get_state()

    

print("Starting training loop...")
for iteration in range(N):
    buffer.clear()


    # --- data collection ---
    for t in range(T):
        with torch.no_grad():
            dist, value = model(state_batch)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        # step and store data
        action_np = torch.clamp(action, -MAX_ANGLE, MAX_ANGLE).detach().numpy().flatten()
        next_state, reward, done = udp.step(action_np)
        buffer.store(state_batch, action, reward, done, log_prob, value)

        # reset envs
        for i in range(NUM_ENVS):
            if done[i]:
                udp.send_reset(i)
        state_batch = next_state
        if done.any():
            fresh = udp.get_state()
            for i in range(NUM_ENVS):
                if done[i]:
                    state_batch[i] = fresh[i]


    # --- advantage estimation ---
    avg_reward = torch.stack(buffer.rewards).mean().item()
    with torch.no_grad():
        _, last_value = model(state_batch)
    buffer.compute_gae(last_value)
    print(f"[{iteration}] reward: {avg_reward:.3f} | advantages mean: {torch.stack(buffer.advantages).mean().item():.3f}")


    # --- model update ---
    clip_eps = 0.2
    for epoch in range(K_epochs):
        states        = torch.stack(buffer.states)
        actions       = torch.stack(buffer.actions)
        old_log_probs = torch.stack(buffer.log_probs)
        advantages    = torch.stack(buffer.advantages)
        returns       = torch.stack(buffer.returns)

        # ---- FLATTEN HERE ----
        T_roll, Nenv, S = states.shape

        states        = states.view(T_roll * Nenv, S)
        actions       = actions.view(T_roll * Nenv, action_dim)
        old_log_probs = old_log_probs.view(T_roll * Nenv)
        advantages    = advantages.view(T_roll * Nenv)
        returns       = returns.view(T_roll * Nenv)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist, values  = model(states)
        new_log_probs = dist.log_prob(actions).sum(-1)
        entropy       = dist.entropy().sum(-1)

        ratio          = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        critic_loss    = nn.MSELoss()(values.squeeze(-1), returns)
        entropy_loss   = -entropy.mean()

        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        print("loss:", loss.item())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        if epoch == K_epochs - 1:
            print(f"  loss: {loss.item():.2f} | actor: {actor_loss.item():.3f} | critic: {critic_loss.item():.3f} | std: {torch.exp(model.log_std).tolist()}")



torch.save(model.state_dict(), "TORSO_BALANCING_POLICY.pt")




# test loop 
# while True:
#     udp.send_actions([-100.0, -100.0])
#     flat = torch.tensor(udp.receive_state(), dtype=torch.float32)
#     state_batch = flat.view(NUM_ENVS, state_dim)   # update state_batch!
#     print("reward: ", getReward(state_batch))
#     done = isDone(state_batch)
#     for i in range(NUM_ENVS):
#         if done[i]:
#             print(f"env {i} done, resetting")
#             udp.send_reset(i)