import torch
import torch.nn as nn
import socket
import struct

anglesGood = torch.tensor([5, 4.95, 3, 4, 0, 0, 1, 2, 0, 0], dtype=torch.float32)
stiffGood  = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1],    dtype=torch.float32)
target     = torch.cat([anglesGood, stiffGood])

class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive(self, num_floats=23):
        data, _ = self.recv_sock.recvfrom(4096)
        return struct.unpack(f"{num_floats}f", data[:num_floats * 4])
    def send(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)
    def receive_latest(self, num_floats=23):
        self.recv_sock.setblocking(False)
        latest = None
        while True:
            try:
                data, _ = self.recv_sock.recvfrom(4096)
                latest = data
            except BlockingIOError:
                break
        self.recv_sock.setblocking(True)
        if latest is None:
            data, _ = self.recv_sock.recvfrom(4096)  # nothing buffered, wait for next
            latest = data
        return struct.unpack(f"{num_floats}f", latest[:num_floats * 4])
udp = UDP("127.0.0.1", 5006, 5005)


# give birth
net = nn.Sequential(
    nn.Linear(23, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 20)
)
log_std = nn.Parameter(torch.full((20,), -1.0))
opt = torch.optim.Adam(list(net.parameters()) + [log_std], lr=1e-3)

# load the boomer model
try:
    net.load_state_dict(torch.load("model.pt"))
    print("loaded model")
except:
    print("starting fresh")

print("----------------------------------------------\n\n")
# live loop
for step in range(1):

    # receieve goods
    raw = udp.receive_latest(23)
    state = torch.tensor(raw, dtype=torch.float32)

    # package and ship
    mean = net(state)
    dist = torch.distributions.Normal(mean, log_std.exp())
    action = dist.sample()
    log_prob = dist.log_prob(action).mean()

    udp.send(action.tolist())        

    # take payment and calc profit
    new_raw   = udp.receive_latest(23)
    new_state = torch.tensor(new_raw, dtype=torch.float32)
    reward    = -(new_state[:20] - target).abs().mean()

    # layoffs and cuts for more profit
    loss = -reward * log_prob
    opt.zero_grad()
    loss.backward()
    opt.step()

    print("\nraw data from mma.cpp:")
    print(state)

    print("\noutput from net:")
    print(mean)

    print("\naction:")
    print(action)

    print("\nraw new data:")
    print(new_state)

    print("\n    reward:" + str(reward.item()) + "   |   loss: " + str(loss.item()))


    if step % 100 == 0:
        # print("\nraw data from mma.cpp:")
        # print(state)

        # print("\noutput from net:")
        # print(mean)

        # print("\naction:")
        # print(action)

        vals = new_state.detach()
        print(f"\nstep {step:6d} | reward {reward.item():.4f}")
        print(f"  {'joint':<8} {'current':>8} {'target':>8} {'diff':>8}")
        for i in range(10):
            curr_a, tgt_a = vals[i].item(), target[i].item()
            curr_s, tgt_s = vals[i+10].item(), target[i+10].item()
            print(f"  joint {i:<3} | angle {curr_a:6.2f} -> {tgt_a:6.2f} ({curr_a-tgt_a:+.2f}) | stiff {curr_s:5.2f} -> {tgt_s:5.2f} ({curr_s-tgt_s:+.2f})")

    if step % 10000 == 0 and step > 0:
        torch.save(net.state_dict(), "model.pt")