import torch
from torch.distributions import Normal

torch.manual_seed(0)

# -------------------------------------------------
# Target values
# -------------------------------------------------
target = torch.tensor([
    5, 4.95, 3, 4, 0, 0, 1, 2, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
], dtype=torch.float32)

# First 10 in [0, 6.28], last 10 in [0, 1]
ranges = torch.tensor([6.28]*10 + [1.0]*10)
target_norm = target / ranges


# -------------------------------------------------
# Black box (no gradients)
# -------------------------------------------------
def black_box(action_norm: torch.Tensor) -> torch.Tensor:
    # action_norm: [BATCH, 20]
    error = torch.abs(action_norm - target_norm).mean(dim=1)
    return -error  # reward in [-1, 0]


# -------------------------------------------------
# Policy parameters (one per dimension)
# -------------------------------------------------
DIM = 20
mean_param = torch.zeros(DIM, requires_grad=True)
log_std    = torch.zeros(DIM, requires_grad=True)

optimizer = torch.optim.Adam([mean_param, log_std], lr=0.03)

baseline = 0.0
BATCH = 128   # 🔥 essential for high dimensions

# -------------------------------------------------
# Training loop
# -------------------------------------------------
for step in range(15000):
    std = torch.exp(log_std)
    dist = Normal(mean_param, std)

    # ---- sample MANY actions ----
    actions = dist.sample((BATCH,))                 # [BATCH, 20]
    log_probs = dist.log_prob(actions).sum(dim=1)  # [BATCH]

    rewards = black_box(actions)                   # [BATCH]
    reward_mean = rewards.mean().item()

    # ---- baseline (variance reduction) ----
    baseline = 0.99 * baseline + 0.01 * reward_mean
    advantages = rewards - baseline

    # ---- averaged REINFORCE loss ----
    loss = -(log_probs * advantages.detach()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ---- keep things stable ----
    with torch.no_grad():
        mean_param.clamp_(0.0, 1.0)

        # gradually reduce exploration
        log_std -= 0.001
        log_std.clamp_(-6, 1)

    if step % 1000 == 0:
        print(f"step {step:5d} | reward {reward_mean:.6f}")

torch.save({
    "mean": mean_param.detach(),
    "log_std": log_std.detach(),
}, "policy.pt")
# -------------------------------------------------
# Convert back to original scale
# -------------------------------------------------
final_action = mean_param.detach() * ranges

print("\nFinal learned values:\n", final_action)
print("\nTarget values:\n", target)