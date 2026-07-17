import numpy as np

data = np.loadtxt("baked_motion.csv", delimiter=",")
F = data.shape[0]

# Columns are already in mma.cpp's bones[]/joints[] order, and train.py's link/joint
# lists are defined to match that order exactly — no reindexing needed here.
np.savez("motion.npz",
         joint_q=data[:, 0:52].reshape((F, 13, 4)),    # (w,x,y,z) per joint
         link_p=data[:, 52:94].reshape((F, 14, 3)),
         joint_av=data[:, 94:133].reshape((F, 13, 3)), # per-joint angvel, parent-relative
         com=data[:, 133:136].reshape((F, 3)))