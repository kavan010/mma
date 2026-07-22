import numpy as np

raw = np.loadtxt("baked_motion.csv", delimiter=",")
F = raw.shape[0]

# animate.cpp appends trailing columns past 136 purely for its own editor bookkeeping
# (keyframe flag, pelvis world orientation, pelvis unwrapped angle) plus, from column
# 144, root (pelvis) linear + angular velocity — real training data, just baked later
# than the original 0-135 block, so it isn't inside that block's column numbering.
data = raw[:, :136]
root_vel    = raw[:, 144:147] if raw.shape[1] >= 150 else np.zeros((F, 3))
root_ang_vel = raw[:, 147:150] if raw.shape[1] >= 150 else np.zeros((F, 3))

# Columns are already in mma.cpp's bones[]/joints[] order, and train.py's link/joint
# lists are defined to match that order exactly — no reindexing needed here.
np.savez("motion.npz",
         joint_q=data[:, 0:52].reshape((F, 13, 4)),    # (w,x,y,z) per joint
         link_p=data[:, 52:94].reshape((F, 14, 3)),
         joint_av=data[:, 94:133].reshape((F, 13, 3)), # per-joint angvel, parent-relative
         com=data[:, 133:136].reshape((F, 3)),
         root_vel=root_vel,         # pelvis linear velocity, world-frame
         root_ang_vel=root_ang_vel) # pelvis angular velocity, world-frame