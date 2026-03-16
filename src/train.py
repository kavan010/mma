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

    def receive(self, num_floats=20):
        data, _ = self.recv_sock.recvfrom(num_floats * 4)
        return struct.unpack(f"{num_floats}f", data)
    def send(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)
udp = UDP("127.0.0.1", 5006, 5005)


net = nn.Sequential(
    nn.Linear(20, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 20)
)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

try:
    net.load_state_dict(torch.load("model.pt"))
    print("loaded model")
except:
    print("starting fresh")

for step in range(50000):
    state     = torch.cat([torch.rand(10) * 6.28, torch.rand(10)]).unsqueeze(0)
    action    = net(state)
    new_state = state + action

    reward = -(new_state.squeeze() - target).abs().mean()
    loss   = -reward

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        vals = new_state.squeeze().detach()
        print(f"\nstep {step:6d} | reward {reward.item():.4f}")
        for i in range(10):
            print(f"  joint {i} | angle {vals[i]:.4f} (target {target[i]:.2f}) | stiff {vals[i+10]:.4f} (target {target[i+10]:.2f})")

    if step % 10000 == 0 and step > 0:
        torch.save(net.state_dict(), "model.pt")

torch.save(net.state_dict(), "model.pt")