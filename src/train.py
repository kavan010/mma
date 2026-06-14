import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import math
import socket
import struct
from time import sleep
import os

# params
NUM_ENVS = 20
action_dim = 10
state_dim = 34
N = 10000
T = 1024
K_epochs = 8
ENT_START, ENT_END = 0.01, 0.001
MAX_ANGLE = math.pi


# ------------------------- model core ----------------------
class ActorCritic(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()

        # actor
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden), 
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # critic 
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden), 
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        # Forward pass through the separate networks
        mean = self.actor_net(s)
        std = torch.exp(self.log_std.clamp(-3, 0.5))
        dist = Normal(mean, std)

        value = self.critic_net(s)

        return dist, value
model = ActorCritic()
opt = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N, eta_min=1e-5)

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
    def compute_gae(self, gamma=0.99, lam=0.95):
        avg_reward = torch.stack(self.rewards).mean().item()
        with torch.no_grad():
            _, last_value = model(state_batch)

        last_v = last_value.squeeze(-1)
        values = self.values + [last_v]

        gae = torch.zeros(NUM_ENVS)
        adv = []

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]  # 0.0 if the episode ended here
            
            # True temporal separation masking
            delta = self.rewards[t] + gamma * values[t+1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            
            # FIX: .clone() preserves the temporal state of each environment at index t
            adv.insert(0, gae.clone()) 

        self.advantages = adv
        self.returns = [a + v for a, v in zip(adv, self.values)]
        print(f"[{iteration}] reward: {avg_reward:.3f} | advantages mean: {torch.stack(buffer.advantages).mean().item():.3f}")

    # reward n' done functions
    def getReward(self, state):
        body_angle = torch.atan2(state[:, 3], state[:, 4])
        upright = torch.exp(-4.0 * (body_angle - math.pi / 2) ** 2)
        hip_y = state[:, 33]
        height = torch.clamp((hip_y - 0.08) / (0.25 - 0.08), 0.0, 1.0)

        # penalize legs being asymmetric
        legL_angle = torch.atan2(state[:, 18], state[:, 19])
        legR_angle = torch.atan2(state[:, 21], state[:, 22])
        leg_symmetry = torch.exp(-3.0 * (legL_angle - legR_angle) ** 2)

        return 2.0 * upright * height * leg_symmetry + 0.3
    def isDone(self, state):
        hip_low = state[:, 33] < 0.08
        body_angle = torch.atan2(state[:, 3], state[:, 4])
        body_fallen = torch.abs(body_angle - math.pi / 2) > 0.9  # tighter than 1.2
        return hip_low | body_fallen

    def clear(self):
        self.__init__()
buffer = RolloutBuffer()

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
        return s, buffer.getReward(s), buffer.isDone(s)
    def send_reset(self, env_idx):
        self.send_actions([-69.0, float(env_idx)])
udp = UDP("127.0.0.1", 5006, 5005)

def updateModel():
    clip_eps  = 0.2
    batch_size = 2048
    ent_coeff  = ENT_START * (ENT_END / ENT_START) ** (iteration / N)

    states        = torch.stack(buffer.states).view(-1, state_dim)
    actions       = torch.stack(buffer.actions).view(-1, action_dim)
    old_log_probs = torch.stack(buffer.log_probs).view(-1)
    returns       = torch.stack(buffer.returns).view(-1)
    advantages    = torch.stack(buffer.advantages).view(-1)
    
    # Extract old values to use for robust value clipping
    old_values    = torch.stack(buffer.values).view(-1)
    
    # Normalize advantages safely
    advantages    = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    N_samples     = states.shape[0]

    for epoch in range(K_epochs):
        perm = torch.randperm(N_samples)
        for start in range(0, N_samples, batch_size):
            idx = perm[start:start + batch_size]

            dist, values  = model(states[idx])
            new_log_probs = dist.log_prob(actions[idx]).sum(-1)
            entropy       = dist.entropy().sum(-1)
            ratio         = torch.exp(new_log_probs - old_log_probs[idx])
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

            # Actor update remains clean
            actor_loss   = -torch.min(ratio * advantages[idx], clipped_ratio * advantages[idx]).mean()
            
            # ROBUST CRITIC UPDATE: Standard PPO Value Function Clipping
            # This restricts the critic network from changing its output too wildly in a single step
            v_pred = values.squeeze(-1)
            v_clipped = old_values[idx] + torch.clamp(v_pred - old_values[idx], -clip_eps, clip_eps)
            v_loss1 = (v_pred - returns[idx]).pow(2)
            v_loss2 = (v_clipped - returns[idx]).pow(2)
            critic_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

            entropy_loss = -entropy.mean()
            loss         = actor_loss + 0.5 * critic_loss + ent_coeff * entropy_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            
    print(f"  loss: {loss.item():.2f} | actor: {actor_loss.item():.3f} | critic: {critic_loss.item():.3f} | std: {torch.exp(model.log_std).tolist()}")


# ---------------------- load model ----------------------
print("Starting training loop...")
state_batch = udp.get_state()
start_iteration = 0
if os.path.exists("WHOLE_BODY_POLICY.pt"):
    ckpt = torch.load("WHOLE_BODY_POLICY.pt")
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_iteration = ckpt['iteration'] + 1
        print(f"Resuming from iteration {start_iteration}")
    else:
        model.load_state_dict(ckpt)  # handles your old flat checkpoint
        print("Loaded legacy model, starting from iteration 0")


# ---------------------- train ----------------------
for iteration in range(start_iteration, N):
    buffer.clear()

    # data collection 
    for t in range(T):
        with torch.no_grad():
            dist, value = model(state_batch)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        # step and store data
        action_np = torch.clamp(action, -MAX_ANGLE, MAX_ANGLE).detach().numpy().flatten()
        next_state, reward, done = udp.step(action_np)
        # reward = reward - 0.05 * (action ** 2).sum(-1)
        buffer.store(state_batch, action, reward, done, log_prob, value)

        # reset envs
        state_batch = next_state
        if done.any():
            for i in range(NUM_ENVS):
                if done[i]:
                    udp.send_reset(i)
            sleep(0.005)
            fresh = udp.get_state()
            for i in range(NUM_ENVS):
                if done[i]:
                    state_batch[i] = fresh[i]

    # advantage estimation 
    buffer.compute_gae()
    
    #  model update 
    updateModel()

    # update lr and save 
    scheduler.step()
    if (iteration % 10 == 0 and iteration > 0):
        torch.save({
            'iteration': iteration,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, "WHOLE_BODY_POLICY.pt")


# final save
torch.save(model.state_dict(), "WHOLE_BODY_POLICY.pt")


















# # test loop 
# while True:
#     udp.send_actions([-100.0, -100.0])
#     flat = torch.tensor(udp.receive_state(), dtype=torch.float32)
#     state_batch = flat.view(NUM_ENVS, state_dim)
#     print("reward: ", buffer.getReward(state_batch))
#     done = buffer.isDone(state_batch)
#     for i in range(NUM_ENVS):
#         if done[i]:
#             print(f"env {i} done, resetting")
#             udp.send_reset(i)
