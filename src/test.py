import socket, struct, random

IP, SEND_PORT, RECV_PORT = "127.0.0.1", 5005, 5006
J, B, V = 10, 11, 6
SIZE = B * V * 4

print(f"Sending random physics data to {IP}:{SEND_PORT}")

send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv.bind((IP, RECV_PORT))
recv.setblocking(False)
recv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)

print(f"Listening for observation data on {IP}:{RECV_PORT}")

try:
    while True:
        vals = [v for _ in range(J) for v in (random.uniform(-3.14,3.14), random.uniform(0,1e6))]
        send.sendto(struct.pack('f'*len(vals), *vals), (IP, SEND_PORT))

        latest = None
        while True:
            try: latest,_ = recv.recvfrom(SIZE)
            except BlockingIOError: break

        if latest and len(latest)==SIZE:
            state = struct.unpack('f'*(B*V), latest)
            body_x, body_y = state[:2]
            # print(f"Body position: ({body_x:.2f}, {body_y:.2f})")

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    print("Closing sockets.")
    send.close()
    recv.close()