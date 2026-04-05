import torch
import torch.nn as nn
import socket
import struct

class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive(self, num_floats=23):
        data, _ = self.recv_sock.recvfrom(num_floats * 4)
        return struct.unpack(f"{num_floats}f", data)

    def send(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)

udp = UDP("127.0.0.1", 5006, 5005)

net = nn.Sequential(
    nn.Linear(23, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 20)
)
net.load_state_dict(torch.load("model.pt"))
net.eval()

# state = torch.cat([torch.rand(10) * 6.28, torch.rand(10), torch.rand(3)]).unsqueeze(0)
state = torch.tensor(udp.receive(), dtype=torch.float32).unsqueeze(0)
print(state)
vals = (net(state)).squeeze().detach()
print("\n\n")
print(vals)
udp.send(vals.tolist())

# step = 0
# while True:
#     state     = torch.tensor(udp.receive(), dtype=torch.float32).unsqueeze(0)
#     vals      = (state + net(state)).squeeze().detach()
#     udp.send(vals.tolist())

#     if step % 10 == 0:
#         print(f"\nstep {step:6d}")
#         for i in range(10):
#             print(f"  joint {i} | angle {vals[i]:.4f} (target {target[i]:.2f}) | stiff {vals[i+10]:.4f} (target {target[i+10]:.2f})")
#     step += 1