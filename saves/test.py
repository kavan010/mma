import torch
torch.set_num_threads(1)
import torch.nn as nn
import math, socket, struct
from time import sleep

# ---- config ----
NUM_ENVS = 1     # how many envs to run
DPHASE   = 0.0    # phase increment per step. 0 for a still frame; set 1/num_frames for a motion clip.

NUM_LINKS, NUM_JOINTS = 14, 13
NUM_SKILLS, TARGET_DIM = 6, 3
RAW_STATE_DIM = 2 + NUM_LINKS * 13
state_dim = RAW_STATE_DIM + NUM_SKILLS + TARGET_DIM
action_dim = 3 * NUM_JOINTS
ET_HEAD, ET_CHEST = 26.0, 18.0


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(0.15)))
        self.critic_net = nn.Sequential(   # kept only so load_state_dict matches
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1))

    def act(self, s):
        s_in = s.clone()
        body = s_in[:, 2:RAW_STATE_DIM].view(-1, NUM_LINKS, 13)
        body[..., 1] -= s_in[:, 1:2]  # link y: absolute -> root-relative, before root itself is scaled
        s_in[:, 1] *= 0.05
        body[..., 0:3] *= 0.2
        body[..., 7:13] *= 0.05
        s_in = torch.clamp(s_in, -5.0, 5.0)
        mean = self.actor_net(s_in)   # deterministic mean action = absolute PD target
        return torch.clamp(mean, -math.pi, math.pi)

model = ActorCritic()
model.load_state_dict(torch.load("POLICY_3D.pt")['model'])
model.eval()


def isDone(state):
    body = state[:, 2:RAW_STATE_DIM].view(-1, NUM_LINKS, 13)
    return (body[:, 13, 1] < ET_HEAD) | (body[:, 8, 1] < ET_CHEST)


def pad_skill(raw):
    skills = torch.zeros(raw.shape[0], NUM_SKILLS); skills[:, 0] = 1.0
    return torch.cat([raw, skills, torch.zeros(raw.shape[0], TARGET_DIM)], dim=1)


class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)
    def send(self, a): self.send_sock.sendto(struct.pack(f"{len(a)}f", *a), self.send_addr)
    def recv(self):
        n = RAW_STATE_DIM * NUM_ENVS
        data, _ = self.recv_sock.recvfrom(n * 4)
        return torch.tensor(struct.unpack(f"{n}f", data), dtype=torch.float32).view(NUM_ENVS, RAW_STATE_DIM)
    def get_state(self):
        self.send([-100.0] * action_dim)
        return pad_skill(self.recv())
    def step(self, target):
        self.send(target.flatten().tolist())
        return pad_skill(self.recv())
    def send_reset(self, i, ph): self.send([-69.0, float(i), float(ph)])

udp = UDP("127.0.0.1", 5006, 5005)


# ---- run ----
print("Running policy (no reference, no training)...")
state = udp.get_state()
phase = torch.zeros(NUM_ENVS)

while True:
    state[:, 0] = phase
    with torch.no_grad():
        target = model.act(state)          # absolute PD target straight from the net
    state = udp.step(target)
    phase = (phase + DPHASE) % 1.0

    done = isDone(state) | ~torch.isfinite(state).all(dim=1)
    if done.any():
        for i in range(NUM_ENVS):
            if done[i]:
                phase[i] = 0.0
                udp.send_reset(i, phase[i])
        sleep(0.005)
        fresh = udp.get_state()
        for i in range(NUM_ENVS):
            if done[i]:
                state[i] = fresh[i]
