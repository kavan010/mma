import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import socket
import struct
import numpy as np
import os

print("Starting trainer...")

class UDPCommunicator:
    def __init__(self, ip, recv_port, send_port):
        self.ip = ip
        self.send_port = send_port
        self.recv_port = recv_port

        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((self.ip, self.recv_port))
        self.recv_sock.setblocking(True)

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (self.ip, self.send_port)

    def receive_state(self, num_floats=86):
        """Receives and unpacks a list of floats from the C++ simulation."""
        data, _ = self.recv_sock.recvfrom(num_floats * 4)  # 4 bytes per float
        return struct.unpack(f"{num_floats}f", data)
    def send_actions(self, actions):
        """Packs and sends a list of floats (actions) to the C++ simulation."""
        msg = struct.pack(f"{len(actions)}f", *actions)
        self.send_sock.sendto(msg, self.send_addr)
udp = UDPCommunicator(ip="127.0.0.1", recv_port=5006, send_port=5005)


class Trainer:
    STATE_DIM = 86
    ACTION_DIM = 20

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Network (shared trunk + actor/critic heads) ──
        self.net    = nn.Sequential(
            nn.Linear(self.STATE_DIM, 256), 
            nn.Tanh(),   
            nn.Linear(256, 256),
            nn.Tanh()).to(self.device)
        self.actor  = nn.Linear(256, self.ACTION_DIM).to(self.device)
        self.critic = nn.Linear(256, 1).to(self.device)
        self.log_std = nn.Parameter(torch.full((self.ACTION_DIM,), -1.0, device=self.device))

        self.opt = optim.Adam(
            list(self.net.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()) +
            [self.log_std], lr=3e-4)

        # ── PPO hypers ──
        self.gamma, self.lam, self.clip = 0.99, 0.95, 0.2
        self.epochs, self.batch_size    = 10, 1024

        # ── Rollout buffers ──
        self.states, self.actions, self.log_probs, self.rewards, self.values = [], [], [], [], []

    def step(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h    = self.net(s)
            mean = self.actor(h)
            dist = Normal(mean, self.log_std.exp())
            act  = dist.sample()
            lp   = dist.log_prob(act).sum()
            val  = self.critic(h).squeeze()

        raw = act.cpu().numpy()

        # scale inline — delta angle ±0.1 rad, stiffness ∈ (0, 1)
        scaled = raw.copy()
        for i in range(0, self.ACTION_DIM, 2):
            scaled[i]   = np.tanh(raw[i]) * 0.1
            scaled[i+1] = (np.tanh(raw[i+1]) + 1) * 0.5

        reward = self.compute_reward(state)

        self.states.append(state)
        self.actions.append(raw)
        self.log_probs.append(lp)
        self.rewards.append(reward)
        self.values.append(val.item())

        return scaled, reward

    def compute_reward(self, state):
        joint_angles    = state[66::2]
        joint_stiffness = state[67::2]

        angle_err = np.mean(np.abs(joint_angles))
        stiff_err = np.mean(np.abs(joint_stiffness - 0.2))

        return -(angle_err + stiff_err)

    # PPO update
    def train(self):
        if len(self.states) < self.batch_size:
            return

        states  = torch.tensor(np.array(self.states),  dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        old_lps  = torch.stack(self.log_probs).to(self.device).detach()

        # GAE advantages
        vals = np.array(self.values)
        rewards = np.array(self.rewards)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nv    = vals[t+1] if t+1 < len(vals) else 0.0
            delta = rewards[t] + self.gamma * nv - vals[t]
            gae   = delta + self.gamma * self.lam * gae
            adv[t] = gae
        ret = adv + vals

        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(self.epochs):
            h    = self.net(states)
            dist = Normal(self.actor(h), self.log_std.exp())
            lps  = dist.log_prob(actions).sum(dim=1)
            ent  = dist.entropy().sum(dim=1).mean()

            ratio = torch.exp(lps - old_lps)
            actor_loss  = -torch.min(ratio * adv_t,
                                     torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv_t).mean()
            critic_loss = (ret_t - self.critic(h).squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * ent   # entropy bonus keeps exploration
            self.optimizer_step(loss)

        print(f"updated | mean_ret={ret.mean():.3f}  std={self.log_std.exp().mean().item():.3f}")
        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear()

    def optimizer_step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(                        # prevents exploding gradients
            list(self.net.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()) +
            [self.log_std], max_norm=0.5)
        self.opt.step()

    def save(self, path="model.pt"):
        torch.save({"net": self.net.state_dict(), "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(), "log_std": self.log_std.data}, path)
    def load(self, path="model.pt"):
        d = torch.load(path)
        self.net.load_state_dict(d["net"]); self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"]); self.log_std.data = d["log_std"]
trainer = Trainer()


if os.path.exists("model.pt"):
    trainer.load()
    print("Resuming from saved model.")
else:
    print("No saved model found, starting fresh.")



# ------------- Main loop ---------------
step = 0
while True:
    state = np.array(udp.receive_state(), dtype=np.float32)

    # step() returns already-scaled action ready to send
    scaled_action, reward = trainer.step(state)
    udp.send_actions(scaled_action)

    trainer.train()
    step += 1

    if step % 10000 == 0:
        trainer.save()
    if step % 5000 == 0:
        print(f"step: {step}  buffer: {len(trainer.states)}")
    if step % 200 == 0:
        print(f"reward: {reward:.4f}  action mean: {np.mean(scaled_action):.4f}")