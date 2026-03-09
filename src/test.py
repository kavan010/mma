import socket
import struct
import random
import time

# Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
NUM_JOINTS = 10 # Your mma.cpp has 10 joints

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Sending random physics data to {UDP_IP}:{UDP_PORT}...")

try:
    while True:
        # Create a list of (targetAngle, stiffness) for each joint
        # targetAngle: -3.14 to 3.14 | stiffness: 0 to 1,000,000
        data_to_send = []
        for _ in range(NUM_JOINTS):
            data_to_send.append(random.uniform(-3.14, 3.14)) # targetAngle
            data_to_send.append(random.uniform(0, 1000000.0)) # stiffness

        # Pack into binary format: 'f' is a 4-byte float
        # 'f' * (NUM_JOINTS * 2) creates a string like 'ffff...'
        packet = struct.pack('f' * (NUM_JOINTS * 2), *data_to_send)
        
        sock.sendto(packet, (UDP_IP, UDP_PORT))
        
        time.sleep(1/60.0) # Match 60 FPS
except KeyboardInterrupt:
    print("Stopped by user.")