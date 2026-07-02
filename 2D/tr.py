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
state_dim = 39
N = 10000
T = 1024
K_epochs = 4
ENT_COEF = 0.003
MAX_ANGLE = math.pi
STAND_ANGLE = math.pi / 2
STAND_HEIGHT = 0.23
IMPULSE_WINDOW        = 30
IMPULSE_COOLDOWN      = 15
IMPULSE_STEP          = 5.0
IMPULSE_START         = 300.0
IMPULSE_MAX           = 3000.0
IMPULSE_THRESH        = 1.65
IMPULSE_DEMOTE_THRESH = 1.55


_nan_seen = set()
def nan_check(name, x, extra=""):
    if name in _nan_seen or not torch.is_tensor(x):
        return False
    bad = ~torch.isfinite(x)
    if bad.any():
        _nan_seen.add(name)
        n = bad.sum().item()
        sample = x.flatten()[bad.flatten()][:5].tolist()
        it = globals().get("iteration", "?")
        print(f"[PY-NAN] '{name}' iter={it} {extra} -> {n}/{x.numel()} bad, sample={sample}")
        return True
    return False


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
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

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
        s_in = s.clone()
        s_in = torch.clamp(s_in, -5.0, 5.0)
        s_in[:, 2:33:3] = s_in[:, 2:33:3] / 20.0

        mean = self.actor_net(s_in)
        std = torch.exp(self.log_std.clamp(-3, 0.0))
        dist = Normal(mean, std)

        value = self.critic_net(s_in)

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
        rewards_stacked = torch.stack(self.rewards)
        nan_check("rewards (stacked, pre-GAE)", rewards_stacked)
        avg_reward = rewards_stacked.mean().item()
        rewards = self.rewards
        dones = self.dones
        values = self.values

        with torch.no_grad():
            _, last_value = model(state_batch)
        nan_check("last_value (bootstrap)", last_value)

        last_v = last_value.squeeze(-1)
        values = values + [last_v]

        gae = torch.zeros(NUM_ENVS)
        adv = []

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]

            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            if nan_check("gae delta", delta, f"t={t}"):
                print(f"[PY-NAN]   -> reward[t]={rewards[t]} values[t+1]={values[t+1]} values[t]={values[t]} mask={mask}")
            gae = delta + gamma * lam * mask * gae

            adv.insert(0, gae.clone())

        self.advantages = adv
        nan_check("advantages (raw, post-GAE)", torch.stack(adv))
        self.returns = [a + v for a, v in zip(adv, self.values)]
        print(f"[{iteration}] reward: {avg_reward:.3f} | advantages mean: {torch.stack(buffer.advantages).mean().item():.3f}")
        return avg_reward

    # reward n' done functions
    def getReward(self, state):
        b_ang = torch.atan2(state[:, 3], state[:, 4])
        upright = torch.exp(-2.0 * (b_ang - STAND_ANGLE) ** 2)

        hip_y = state[:, 33]
        height = torch.exp(-100.0 * (hip_y - STAND_HEIGHT) ** 2)

        return 0.2 + upright + height
    def isDone(self, state):
        hip_y = state[:, 33]
        hip_low = hip_y < STAND_HEIGHT / 2

        b_ang = torch.atan2(state[:, 3], state[:, 4])
        fallen = torch.abs(b_ang - STAND_ANGLE) > 1.4

        return hip_low | fallen

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

def updateModel(impulse_mag):
    states        = torch.stack(buffer.states).view(-1, state_dim)
    actions       = torch.stack(buffer.actions).view(-1, action_dim)
    old_log_probs = torch.stack(buffer.log_probs).view(-1)
    returns       = torch.stack(buffer.returns).view(-1).clamp(-300, 300)
    advantages    = torch.stack(buffer.advantages).view(-1)

    nan_check("states (into updateModel)", states)
    nan_check("actions (into updateModel)", actions)
    nan_check("old_log_probs (into updateModel)", old_log_probs)
    nan_check("returns (into updateModel, pre-clamp value)", returns)
    nan_check("advantages (into updateModel, pre-clamp)", advantages)

    advantages = advantages.clamp(-20, 20)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.clamp(-5, 5)  # one outlier sample can't dominate the batch mean
    nan_check("advantages (normalized)", advantages)

    N_samples = states.shape[0]
    batch_size = N_samples // 4

    for epoch in range(K_epochs):
        perm = torch.randperm(N_samples)

        for b, start in enumerate(range(0, N_samples, batch_size)):
            idx = perm[start:start + batch_size]
            tag = f"epoch={epoch} batch={b}"

            dist, values = model(states[idx])
            nan_check("actor mean (in update step)", dist.mean, tag)
            nan_check("critic values (in update step)", values, tag)
            new_log_probs = dist.log_prob(actions[idx]).sum(-1)
            entropy = dist.entropy().sum(-1)

            log_ratio = (new_log_probs - old_log_probs[idx]).clamp(-20, 20)
            nan_check("log_ratio (pre-exp)", log_ratio, tag + f" max={log_ratio.max().item():.1f} min={log_ratio.min().item():.1f}")
            ratio = torch.exp(log_ratio)
            nan_check("ratio (post-exp)", ratio, tag)
            clipped = torch.clamp(ratio, 0.8, 1.2)

            surr = torch.min(ratio * advantages[idx], clipped * advantages[idx])
            # dual-clip: when advantage is negative, a huge ratio makes the surrogate unboundedly
            # negative. cap it at 3*A so one bad sample can't dominate the batch.
            surr = torch.where(advantages[idx] < 0, torch.max(surr, 3.0 * advantages[idx]), surr)
            actor_loss = -surr.mean()
            critic_loss  = (values.squeeze(-1) - returns[idx]).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + ENT_COEF * entropy_loss
            nan_check("loss", loss.unsqueeze(0), tag)

            opt.zero_grad()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            nan_check("grad_norm (clip input)", grad_norm.unsqueeze(0), tag + " -- non-finite norm means the clip scale factor broadcasts NaN/Inf to every param")
            opt.step()

            with torch.no_grad():
                for name, p in model.named_parameters():
                    nan_check(f"param:{name}", p, tag + " (right after opt.step())")

    print(
        f"loss {loss.item():.2f} | "
        f"actor {actor_loss.item():.3f} | "
        f"critic {critic_loss.item():.3f} | "
        f"std {torch.exp(model.log_std).mean().item():.3f} | "
        f"maxImp {impulse_mag:.1f}"
    )

# ---------------------- load model ----------------------
impulse_mag      = IMPULSE_START  # defaults; overridden below if checkpoint has them
impulse_cooldown = 0
recent_rewards   = []

print("Starting training loop...")
state_batch = udp.get_state()
start_iteration = 0
if os.path.exists("IMPULSE_POLICY.pt"):
    ckpt = torch.load("IMPULSE_POLICY.pt")
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_iteration = ckpt['iteration'] + 1
        if 'impulse_mag' in ckpt:
            impulse_mag      = ckpt['impulse_mag']
            impulse_cooldown = ckpt.get('impulse_cooldown', 0)
            print(f"Resuming from iteration {start_iteration}, impulse_mag={impulse_mag:.0f}")
        else:
            print(f"Resuming from iteration {start_iteration}")
    else:
        model.load_state_dict(ckpt)  # handles old flat checkpoint
        print("Loaded legacy model, starting from iteration 0")

# impulse formula was redesigned (now calibrated in old-equivalent body velocity units) —
# old checkpoint's impulse_mag is in different units, so always restart curriculum here
impulse_mag      = IMPULSE_START
impulse_cooldown = 0
recent_rewards   = []

# ---------------------- train ----------------------
for iteration in range(start_iteration, N):
    buffer.clear()

    udp.send_actions([-68.0, impulse_mag])

    # data collection
    for t in range(T):
        nan_check("state_batch (model input, collect)", state_batch, f"t={t}")
        with torch.no_grad():
            dist, value = model(state_batch)
        nan_check("actor mean (collect)", dist.mean, f"t={t}")
        nan_check("critic value (collect)", value, f"t={t}")
        action = torch.clamp(dist.sample(), -MAX_ANGLE, MAX_ANGLE)
        log_prob = dist.log_prob(action).sum(-1)

        # step and store data
        action_np = action.detach().numpy().flatten()
        next_state, reward, done = udp.step(action_np)
        nan_check("next_state (raw from C++/UDP)", next_state, f"t={t}")
        nan_check("reward (computed from next_state)", reward, f"t={t}")
        bad = ~torch.isfinite(next_state).all(dim=1)
        done = done | bad
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
    avg_reward = buffer.compute_gae()

    # curriculum: advance impulse only after sustained performance over a window,
    # with a cooldown so the policy has time to adapt before being tested again.
    recent_rewards.append(avg_reward)
    if len(recent_rewards) > IMPULSE_WINDOW:
        recent_rewards.pop(0)

    if impulse_cooldown > 0:
        impulse_cooldown -= 1
    elif len(recent_rewards) == IMPULSE_WINDOW:
        avg = sum(recent_rewards) / IMPULSE_WINDOW
        if avg >= IMPULSE_THRESH:
            impulse_mag = min(impulse_mag + IMPULSE_STEP, IMPULSE_MAX)
            impulse_cooldown = IMPULSE_COOLDOWN
            recent_rewards.clear()
            print(f"  --> impulse advanced to {impulse_mag:.0f} (locked for {IMPULSE_COOLDOWN} iters)")
        elif avg < IMPULSE_DEMOTE_THRESH and impulse_mag > IMPULSE_START:
            impulse_mag = max(impulse_mag - IMPULSE_STEP, IMPULSE_START)
            impulse_cooldown = IMPULSE_COOLDOWN
            recent_rewards.clear()
            print(f"  --> impulse demoted to {impulse_mag:.0f} (locked for {IMPULSE_COOLDOWN} iters)")

    #  model update
    updateModel(impulse_mag)

    # update lr and save 
    scheduler.step()
    if (iteration % 10 == 0 and iteration > 0):
        torch.save({
            'iteration':        iteration,
            'model':            model.state_dict(),
            'optimizer':        opt.state_dict(),
            'scheduler':        scheduler.state_dict(),
            'impulse_mag':      impulse_mag,
            'impulse_cooldown': impulse_cooldown,
        }, "IMPULSE_POLICY.pt")


# final save
torch.save({
    'iteration':        N - 1,
    'model':            model.state_dict(),
    'optimizer':        opt.state_dict(),
    'scheduler':        scheduler.state_dict(),
    'impulse_mag':      impulse_mag,
    'impulse_cooldown': impulse_cooldown,
}, "IMPULSE_POLICY.pt")



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


# # measure standing pose
# state_batch = udp.get_state()
# while True:
#     with torch.no_grad():
#         dist, _ = model(state_batch)
#     action = torch.clamp(dist.mean, -MAX_ANGLE, MAX_ANGLE)
#     udp.send_actions(action.detach().numpy().flatten().tolist())
#     udp.send_actions([-100.0, -100.0])
#     flat = torch.tensor(udp.receive_state(), dtype=torch.float32)
#     state_batch = flat.view(NUM_ENVS, state_dim)
#     b_ang = torch.atan2(state_batch[:, 3], state_batch[:, 4])
#     hip_y = state_batch[:, 33]
#     print(f"b_ang {b_ang.mean().item():.3f} | hip_y {hip_y.mean().item():.3f} | reward {buffer.getReward(state_batch).mean().item():.3f}")


