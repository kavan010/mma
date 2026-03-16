class UDPCommunicator:
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