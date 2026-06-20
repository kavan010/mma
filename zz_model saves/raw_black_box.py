# raw blackbox log_prob training

import torch
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(23, 128), 
    nn.Tanh(), 
    nn.Linear(128, 128), 
    nn.Tanh(), 
    nn.Linear(128, 20)
)
log_std = nn.Parameter(torch.zeros(20))
opt = torch.optim.Adam(list(net.parameters()) + [log_std], lr=1e-3)

def black_box(action):
    target = torch.ones(20) * 0.5
    return -float((action - target).abs().mean())

for step in range(10000):
    state = torch.rand(23)

    mean = net(state)
    dist = torch.distributions.Normal(mean, log_std.exp())
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()

    reward = black_box(action)

    loss = -log_prob * reward

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 500 == 0:
        print(f"step {step:5d} | reward {reward:.4f} | mean output {float(mean.mean()):.4f}")