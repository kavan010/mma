import socket
import struct

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv_sock.bind(("127.0.0.1", 5006))

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# get current state from C++
data, _ = recv_sock.recvfrom(20 * 4)
vals = list(struct.unpack("20f", data))

angles    = vals[:10]
stiffness = vals[10:]

joint     = int(input("joint index (0-9): "))
angle     = float(input("target angle (radians): "))
stiff     = float(input("stiffness (0-1): "))

angles[joint]    = angle
stiffness[joint] = stiff

buf = []
for i in range(10):
    buf.append(angles[i])
    buf.append(stiffness[i])

send_sock.sendto(struct.pack("20f", *buf), ("127.0.0.1", 5005))
print(f"sent joint {joint} → angle {angle}, stiffness {stiff}")

# Joint: targetAngle=5     current angle: 5.0          stiffness=1
# Joint: targetAngle=4.95  current angle: 4.8          stiffness=1
# Joint: targetAngle=3     current angle: 2.9          stiffness=1
# Joint: targetAngle=4     current angle: 4.0          stiffness=1
# Joint: targetAngle=0     current angle: 2.9e-05      stiffness=1
# Joint: targetAngle=0     current angle: 0.00306511   stiffness=1
# Joint: targetAngle=1     current angle: 0.96783      stiffness=1
# Joint: targetAngle=2     current angle: 2.00725      stiffness=1
# Joint: targetAngle=0     current angle: 0.0030508    stiffness=1
# Joint: targetAngle=0     current angle: -0.000891685 stiffness=1