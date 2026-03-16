import torch
import torch.nn as nn

class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self, num_floats=86):
        data, _ = self.recv_sock.recvfrom(num_floats * 4)
        return struct.unpack(f"{num_floats}f", data)
    def send_actions(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)

if torch.cuda.is_available():
    current = torch.cuda.current_device()
    print("Using GPU:", torch.cuda.get_device_name(current))
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")


net = nn.Sequential(
    nn.Linear(2, 8), 
    nn.Tanh(), 
    nn.Linear(8, 2))
net.load_state_dict(torch.load("model.pt"))
net.eval()

state = torch.tensor([[6.0, 1.0]])
action = net(state)
print(action)
print("means output is: " + str(state[0, 0].item() + action[0, 0].item()), "and stiffness output is: " + str(state[0, 1].item() + action[0, 1].item()))