import numpy as np

LINK_MASS = np.array([1., 1, 2.5, 2.5, 4.5, 4.5, 5, 7, 8, 1.8, 1.8, 1.2, 1.2, 3])
FEET = [0, 1]

d = np.load("motion.npz")
link_p = d["link_p"]  # (frames, 14, 3)

com = (link_p * LINK_MASS[None, :, None]).sum(1) / LINK_MASS.sum()  # (frames, 3)
feet_xz = link_p[:, FEET][:, :, [0, 2]].mean(1)                      # (frames, 2)
com_xz = com[:, [0, 2]]

offset = com_xz - feet_xz
dist = np.linalg.norm(offset, axis=-1)

print(f"CoM-to-feet offset (x,z): {offset[0]}")
print(f"distance: {dist[0]:.3f}  (0 = perfectly centered)")
