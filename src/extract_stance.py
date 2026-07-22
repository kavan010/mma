import numpy as np, torch

JOINTS = [(7,6),(8,7),(13,8),(9,8),(10,8),(11,9),(12,10),(4,6),(5,6),(2,4),(3,5),(0,2),(1,3)]
ROOT = 6
ROOT_ANCHORS = [
    (7, 0, torch.tensor([0.,  1.8,  0.0]), torch.tensor([0.0, -1.8, 0.0])),
    (4, 7, torch.tensor([0., -1.6,  1.5]), torch.tensor([-4.6, 0.0, 0.0])),
    (5, 8, torch.tensor([0., -1.6, -1.5]), torch.tensor([-4.6, 0.0, 0.0])),
]

def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw*bw - ax*bx - ay*by - az*bz,
                        aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw], dim=-1)
def qconj(q):
    return q * torch.tensor([1., -1, -1, -1])
def qrot(q, v):
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]

def derive_root_quat(joint_q, link_p):
    A = torch.zeros(3, 3)
    for child, jrow, anchorA, anchorB in ROOT_ANCHORS:
        d_local = anchorA - qrot(joint_q[jrow], anchorB)
        d_world = link_p[child] - link_p[ROOT]
        A += torch.outer(d_world, d_local)
    U, _, Vh = torch.linalg.svd(A)
    d = torch.sign(torch.det(U @ Vh))
    R = U @ torch.diag(torch.tensor([1.0, 1.0, d])) @ Vh
    w = torch.clamp(torch.sqrt(torch.clamp(1 + R[0,0] + R[1,1] + R[2,2], min=0)) / 2, min=1e-8)
    q = torch.stack([w, (R[2,1]-R[1,2])/(4*w), (R[0,2]-R[2,0])/(4*w), (R[1,0]-R[0,1])/(4*w)])
    return q / q.norm()

d = np.load("motion.npz")
joint_q = torch.tensor(d["joint_q"][0], dtype=torch.float32)  # (13,4) relative target quats
link_p  = torch.tensor(d["link_p"][0],  dtype=torch.float32)  # (14,3) world positions

root_q = derive_root_quat(joint_q, link_p)
world_q = [None] * 14
world_q[ROOT] = root_q
for jidx, (child, parent) in enumerate(JOINTS):
    world_q[child] = qmul(world_q[parent], joint_q[jidx])

def axis_angle(q):
    w, v = q[0], q[1:]
    s = v.norm()
    if s < 1e-6: return torch.zeros(3)
    return (2.0 * torch.atan2(s, w)) * (v / s)

BONES = ["footR","footL","calfR","calfL","thighR","thighL","pelvis","abs","chest","armR","armL","forearmR","forearmL","head"]

print("const float STANCE_POS[14][3] = {")
for i in range(14):
    p = link_p[i]
    print(f"    {{{p[0]:.6f}f, {p[1]:.6f}f, {p[2]:.6f}f}},   // {BONES[i]}")
print("};")

print("const float STANCE_QUAT[14][4] = { // w x y z, same bone order")
for i in range(14):
    q = world_q[i]
    print(f"    {{{q[0]:.6f}f, {q[1]:.6f}f, {q[2]:.6f}f, {q[3]:.6f}f}},")
print("};")

print("const float STANCE_TARGET[13][3] = { // PD targets (axis-angle), joint order = Skeleton::joints")
for jidx in range(13):
    a = axis_angle(joint_q[jidx])
    print(f"    {{{a[0]:.6f}f, {a[1]:.6f}f, {a[2]:.6f}f}},")
print("};")
