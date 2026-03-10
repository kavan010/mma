import socket
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# --- Configuration ---
IP, SEND_PORT, RECV_PORT = "127.0.0.1", 5005, 5006
NUM_JOINTS = 10
NUM_BONES = 11
STATE_DIM = NUM_BONES * 6
ACTION_DIM = NUM_JOINTS * 2
SAVE_PATH = "mma_model.pth"
BATCH_SIZE = 256

# Stiffness range the model can express.
# WHY NOT 3e6: At 3e6 stiffness with typical bone inertia (~25000), one frame
# of torque application gives ~2 rad/frame overshoot → perpetual oscillation.
# The C++ damping (10000) is tuned for ~2.5e6 default and can't auto-scale.
# Empirically, ~800k is near the stability ceiling for dt=1/60 with this inertia.
# The model can still stand — it just uses slightly more angle travel to do it.
MAX_STIFFNESS = 800_000.0

# Stiffness lerp rate: how fast stiffness changes per frame toward the target.
# Prevents instant jumps from 0 → 800k which would spike angular velocity.
# At 0.15, it takes ~10 frames to reach full stiffness — smooth ramp-up.
STIFFNESS_LERP = 0.15

# --- Neural Network ---
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, ACTION_DIM)
        )
        self.critic = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.ones(ACTION_DIM) * -0.8)

    def forward(self, x):
        return self.actor(x), self.critic(x)

# --- Setup ---
device = torch.device("cpu")
model = PolicyNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

try:
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    print(f"Loaded existing model weights from {SAVE_PATH}.")
except:
    print("Starting training from scratch.")

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind((IP, RECV_PORT))
recv_sock.setblocking(False)

# Track current stiffness per joint for lerping
current_stiffness = np.ones(NUM_JOINTS) * 500_000.0

# --- Helper Functions ---
def normalize_state(raw_state):
    """
    Normalize all 6 features per bone to roughly [-1, 1] for the network.

    CRITICAL: divisors must cover the ACTUAL range seen during physics, not just
    typical values. Angular velocity during explosions or joint-solver corrections
    can hit 1000+ rad/s. Dividing by 10 gives 100+ into the network → critic
    sees out-of-distribution input → value estimate explodes → advantage explodes
    → policy update is garbage.

    Divisors chosen to cover ~99th percentile of real observed values:
      - Position: window is 800x600, so ±400 / ±300 is absolute max
      - Linear velocity: 500 covers fast falls and joint-correction artifacts  
      - Angle: π covers full rotation
      - Angular velocity: 500 covers constraint-solver spikes (was 10 — far too small)
    """
    state = np.array(raw_state, dtype=np.float32)
    for i in range(0, len(state), 6):
        state[i]   /= 400.0   # X Position
        state[i+1] /= 300.0   # Y Position
        state[i+2] /= 500.0   # X Velocity (was 100 — too small for explosion frames)
        state[i+3] /= 500.0   # Y Velocity (was 100 — too small for explosion frames)
        state[i+4] /= 3.14    # Angle
        state[i+5] /= 500.0   # Angular Velocity (was 10 — critically too small)
    return state

def calculate_reward(state_raw):
    body_angle = state_raw[4]
    head_y     = state_raw[7]
    head_vy    = state_raw[9]

    height_reward  = (head_y + 250.0) / 350.0
    upright_reward = -0.08 * (body_angle - 1.57) ** 2
    rising_bonus   = 0.001 * max(0.0, min(head_vy, 200.0))

    reward = height_reward + upright_reward + rising_bonus
    return float(np.clip(reward, -5.0, 5.0))

# --- Training Loop ---
states, actions, log_probs, rewards, values = [], [], [], [], []

print("\n--- Training Active ---")
print("1. Press 'R' in mma.exe to reset the ragdoll.")
print("2. Press Ctrl+C in this terminal to safely save and exit.")
batch_counter = 0

try:
    while True:
        # 1. Drain buffer — only react to the latest frame
        latest_data = None
        while True:
            try:
                data, _ = recv_sock.recvfrom(NUM_BONES * 6 * 4)
                latest_data = data
            except BlockingIOError:
                break

        if latest_data is None:
            continue

        state_raw = struct.unpack('f' * (NUM_BONES * 6), latest_data)
        norm_state = normalize_state(state_raw)
        state_t = torch.FloatTensor(norm_state).to(device)

        # 2. Select Action
        mu, value = model(state_t)
        std = torch.exp(model.log_std)
        dist = Normal(mu, std)
        action = dist.sample()

        states.append(state_t)
        actions.append(action)
        log_probs.append(dist.log_prob(action).sum())
        rewards.append(calculate_reward(state_raw))
        values.append(value.squeeze())

        # 3. Send to Physics
        action_np = action.detach()
        sent_actions = []
        for i in range(NUM_JOINTS):
            angle = float(torch.tanh(action_np[i*2]).item() * 3.14)

            # Target stiffness from sigmoid → (0, MAX_STIFFNESS)
            target_stiff = float(torch.sigmoid(action_np[i*2+1]).item() * MAX_STIFFNESS)

            # Lerp current stiffness toward target — prevents instant jumps
            # that cause angular velocity spikes and perpetual oscillation.
            # current = current + lerp_rate * (target - current)
            current_stiffness[i] += STIFFNESS_LERP * (target_stiff - current_stiffness[i])

            sent_actions.extend([angle, current_stiffness[i]])

        send_sock.sendto(struct.pack('f'*len(sent_actions), *sent_actions), (IP, SEND_PORT))

        # 4. Batch Update
        if len(states) >= BATCH_SIZE:
            b_log_probs = torch.stack(log_probs)
            b_rewards   = torch.FloatTensor(rewards)
            b_values    = torch.stack(values)

            b_advantage = b_rewards - b_values.detach()
            b_advantage = (b_advantage - b_advantage.mean()) / (b_advantage.std() + 1e-8)

            policy_loss = -(b_log_probs * b_advantage).mean()
            value_loss  = nn.functional.mse_loss(b_values, b_rewards)
            total_loss  = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            states, actions, log_probs, rewards, values = [], [], [], [], []

            torch.save(model.state_dict(), SAVE_PATH)
            if batch_counter % 10 == 0:
                print(
                    f"Batch {batch_counter:4d} | "
                    f"Avg Reward: {b_rewards.mean():.4f} | "
                    f"Avg Value: {b_values.mean().item():.4f} | "
                    f"Head Y: {state_raw[7]:.1f}"
                )
            batch_counter += 1

except KeyboardInterrupt:
    print("\n[Ctrl+C] Detected. Stopping training...")

finally:
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Final model successfully saved to {SAVE_PATH}. Safe to close.")
    send_sock.close()
    recv_sock.close()