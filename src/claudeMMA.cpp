// 3D physics-based humanoid for RL motion imitation (DeepMimic / SFV style).
// Same architecture as the 2D version (Engine / Bone / Joint / Skeleton / Data),
// upgraded to real 3D rigid-body dynamics:
//   - quaternion orientation + 3x3 inertia tensors
//   - 3D point-to-point (ball) constraint solver with Baumgarte stabilization
//   - spherical quaternion PD controller (target orientation per joint)
//   - 15 bones, masses / torque limits / Kp / Kd taken from DeepMimic humanoid3d
//
// build: g++ src/mma.cpp -o mma -fopenmp -lglfw -lGLEW -lGL -ldl -lpthread
#define GLM_ENABLE_EXPERIMENTAL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fcntl.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <ctime>
#include <unistd.h>
using namespace glm; using namespace std;

// ------------------------ Engine & Constants ----------------------
// SI units: meters, kilograms, seconds. Character ~1.6m / 45kg (DeepMimic humanoid).
const int   MAX_ENV    = 8;
const int   NUM_JOINTS = 14;                 // 15 bones - 1 (root has no parent joint)
const int   NUM_BONES  = 15;
const int   ACTION_DIM = NUM_JOINTS * 4;     // target quaternion per joint (wxyz)
const int   STATE_DIM  = NUM_BONES * 13 + 2; // pos3 + quat4 + vel3 + angVel3, + 2 foot contacts
int         NUM_ENV    = 4;
const vec3  G(0.0f, -9.81f, 0.0f);

// Global torque scale. DeepMimic's limits already permit flips/kicks on a 45kg body,
// so 1.0 is physically grounded; bump if a heavier/taller character needs more.
float TORQUE_SCALE = 1.0f;

struct Engine {
    GLFWwindow* window;
    int WIDTH = 1200, HEIGHT = 800;

    Engine() {
        if (!glfwInit()) { cerr << "Failed to initialize GLFW\n"; exit(EXIT_FAILURE); }
        window = glfwCreateWindow(WIDTH, HEIGHT, "3D MMA RL", nullptr, nullptr);
        if (!window) { cerr << "Failed to create GLFW window\n"; glfwTerminate(); exit(EXIT_FAILURE); }
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        glewInit();
        int fbW, fbH; glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        glEnable(GL_NORMALIZE);
        GLfloat lpos[] = { 4.0f, 8.0f, 6.0f, 1.0f };
        GLfloat lamb[] = { 0.35f, 0.35f, 0.4f, 1.0f };
        glLightfv(GL_LIGHT0, GL_POSITION, lpos);
        glLightfv(GL_LIGHT0, GL_AMBIENT,  lamb);
    }

    // set up perspective camera each frame (no GLU dependency)
    void run() {
        glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (float)WIDTH / (float)HEIGHT;
        mat4 proj = perspective(radians(45.0f), aspect, 0.05f, 100.0f);
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(value_ptr(proj));

        // pull the camera back enough to see the whole row of environments
        float cx = (NUM_ENV - 1) * 0.5f * 1.6f;
        vec3 eye(cx, 1.2f, 4.5f + NUM_ENV * 0.4f);
        vec3 center(cx, 0.9f, 0.0f);
        mat4 view = lookAt(eye, center, vec3(0, 1, 0));
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(value_ptr(view));
    }

    void drawGround() {
        glDisable(GL_LIGHTING);
        glColor3f(0.18f, 0.20f, 0.24f);
        glBegin(GL_QUADS);
            glVertex3f(-50, 0, -50); glVertex3f(50, 0, -50);
            glVertex3f( 50, 0,  50); glVertex3f(-50, 0,  50);
        glEnd();
        glColor3f(0.28f, 0.30f, 0.34f);
        glBegin(GL_LINES);
        for (int i = -20; i <= 20; i++) {
            glVertex3f((float)i, 0.001f, -20); glVertex3f((float)i, 0.001f, 20);
            glVertex3f(-20, 0.001f, (float)i); glVertex3f(20, 0.001f, (float)i);
        }
        glEnd();
        glEnable(GL_LIGHTING);
    }
};
Engine engine;

// cross-product (skew-symmetric) matrix M such that M*a = v x a
static mat3 skew(const vec3& v) {
    return mat3(vec3(0, v.z, -v.y), vec3(-v.z, 0, v.x), vec3(v.y, -v.x, 0));
}

// --------------------------- Drawing helpers ----------------------
static void glVtx(const vec3& v) { glVertex3f(v.x, v.y, v.z); }
static void glNrm(const vec3& v) { glNormal3f(v.x, v.y, v.z); }

static void drawSphere(float r, int stacks = 12, int slices = 16) {
    for (int i = 0; i < stacks; i++) {
        float t0 = pi<float>() * (-0.5f + (float)i / stacks);
        float t1 = pi<float>() * (-0.5f + (float)(i + 1) / stacks);
        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; j++) {
            float p = 2.0f * pi<float>() * (float)j / slices;
            for (float t : { t0, t1 }) {
                vec3 n(cos(t) * cos(p), sin(t), cos(t) * sin(p));
                glNrm(n); glVtx(n * r);
            }
        }
        glEnd();
    }
}

// capsule along local Y: cylinder of half-length h + two hemisphere caps of radius r
static void drawCapsule(float r, float h, int slices = 16) {
    glBegin(GL_QUAD_STRIP);
    for (int j = 0; j <= slices; j++) {
        float p = 2.0f * pi<float>() * (float)j / slices;
        vec3 n(cos(p), 0, sin(p));
        glNrm(n); glVtx(vec3(n.x * r, -h, n.z * r));
        glNrm(n); glVtx(vec3(n.x * r,  h, n.z * r));
    }
    glEnd();
    for (int cap = 0; cap < 2; cap++) {
        float dir = cap == 0 ? 1.0f : -1.0f, yoff = cap == 0 ? h : -h;
        for (int i = 0; i < 6; i++) {
            float t0 = 0.5f * pi<float>() * (float)i / 6.0f;
            float t1 = 0.5f * pi<float>() * (float)(i + 1) / 6.0f;
            glBegin(GL_QUAD_STRIP);
            for (int j = 0; j <= slices; j++) {
                float p = 2.0f * pi<float>() * (float)j / slices;
                for (float t : { t0, t1 }) {
                    vec3 n(cos(t) * cos(p), sin(t) * dir, cos(t) * sin(p));
                    glNrm(n); glVtx(vec3(n.x * r, yoff + n.y * r, n.z * r));
                }
            }
            glEnd();
        }
    }
}

static void drawBox(vec3 he) {
    vec3 n[6] = { {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1} };
    glBegin(GL_QUADS);
    for (int f = 0; f < 6; f++) {
        glNrm(n[f]);
        vec3 a = n[f], u = vec3(a.y, a.z, a.x), v = cross(a, u);
        vec3 c = a * he;
        glVtx(c + (u + v) * he); glVtx(c + (u - v) * he);
        glVtx(c - (u + v) * he); glVtx(c - (u - v) * he);
    }
    glEnd();
}

// ------------------------ Bodies & Physics ------------------------
enum Shape { SPHERE, CAPSULE, BOX };

struct Bone {
    vec3 pos, vel = vec3(0), angVel = vec3(0);
    quat orient = quat(1, 0, 0, 0);
    float mass, invMass;
    mat3 IbodyInv;               // inverse inertia in body frame (diagonal)
    Shape shape;
    vec3 dims;                   // SPHERE:(r,-,-)  CAPSULE:(r,halfLen,-)  BOX: half extents
    vec3 color = vec3(0.85f, 0.85f, 0.88f);

    Bone(vec3 p, Shape s, vec3 d, float m) : pos(p), mass(m), shape(s), dims(d) {
        invMass = 1.0f / m;
        vec3 I(1.0f);
        if (s == SPHERE) {
            float r = d.x, v = 0.4f * m * r * r;         // solid sphere 2/5 m r^2
            I = vec3(v, v, v);
        } else if (s == CAPSULE) {
            float r = d.x, L = 2.0f * d.y;               // approximate as solid cylinder, long axis = Y
            I.y = 0.5f * m * r * r;
            I.x = I.z = (1.0f / 12.0f) * m * (3.0f * r * r + L * L);
        } else { // BOX, dims = half extents
            float x = 2 * d.x, y = 2 * d.y, z = 2 * d.z;
            I.x = (1.0f / 12.0f) * m * (y * y + z * z);
            I.y = (1.0f / 12.0f) * m * (x * x + z * z);
            I.z = (1.0f / 12.0f) * m * (x * x + y * y);
        }
        IbodyInv = mat3(0.0f);
        IbodyInv[0][0] = 1.0f / I.x; IbodyInv[1][1] = 1.0f / I.y; IbodyInv[2][2] = 1.0f / I.z;
    }

    mat3 worldInvInertia() const {
        mat3 R = mat3_cast(orient);
        return R * IbodyInv * transpose(R);
    }
    vec3 worldPoint(const vec3& local) const { return pos + orient * local; }

    // integrate orientation from world-frame angular velocity
    void integrateOrientation(float dt) {
        quat w(0, angVel.x, angVel.y, angVel.z);
        orient = normalize(orient + 0.5f * dt * (w * orient));
    }

    // contact points as (worldPoint, radius) for ground collision
    void contacts(vector<pair<vec3, float>>& out) const {
        if (shape == SPHERE) {
            out.push_back({ pos, dims.x });
        } else if (shape == CAPSULE) {
            out.push_back({ worldPoint(vec3(0,  dims.y, 0)), dims.x });
            out.push_back({ worldPoint(vec3(0, -dims.y, 0)), dims.x });
        } else {
            for (int sx = -1; sx <= 1; sx += 2)
            for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2)
                out.push_back({ worldPoint(vec3(sx * dims.x, sy * dims.y, sz * dims.z)), 0.0f });
        }
    }

    void draw() const {
        mat4 M = translate(mat4(1.0f), pos) * mat4_cast(orient);
        glPushMatrix();
        glMultMatrixf(value_ptr(M));
        glColor3f(color.r, color.g, color.b);
        if (shape == SPHERE)       drawSphere(dims.x);
        else if (shape == CAPSULE) drawCapsule(dims.x, dims.y);
        else                       drawBox(dims);
        glPopMatrix();
    }
};

struct Joint {
    Bone *A, *B;                 // parent A, child B
    vec3 anchorA_local, anchorB_local;
    quat targetRel;              // desired orientation of B relative to A (RL action target)
    float kp, kd, maxTorque;

    Joint(Bone* a, Bone* b, vec3 anchorA, vec3 anchorB, float kp, float kd, float maxT)
        : A(a), B(b), anchorA_local(anchorA), anchorB_local(anchorB),
          kp(kp), kd(kd), maxTorque(maxT) {
        targetRel = normalize(inverse(a->orient) * b->orient);
    }

    // 3D point-to-point constraint: keep the two anchor points coincident
    void solve(float dt) {
        vec3 rA = A->orient * anchorA_local, rB = B->orient * anchorB_local;
        vec3 pA = A->pos + rA, pB = B->pos + rB;
        vec3 C = pB - pA;                                   // position error

        vec3 vRel = (B->vel + cross(B->angVel, rB)) - (A->vel + cross(A->angVel, rA));
        float beta = 0.2f;
        vec3 bias = (beta / dt) * C;                        // Baumgarte stabilization

        mat3 IA = A->worldInvInertia(), IB = B->worldInvInertia();
        mat3 sA = skew(rA), sB = skew(rB);
        mat3 K = mat3(A->invMass + B->invMass) - sA * IA * sA - sB * IB * sB;

        vec3 P = inverse(K) * (-(vRel + bias));

        A->vel    -= A->invMass * P;
        B->vel    += B->invMass * P;
        A->angVel -= IA * cross(rA, P);
        B->angVel += IB * cross(rB, P);
    }

    // spherical PD: drive B's orientation (relative to A) toward targetRel
    void applyPD(float dt) {
        quat qRel = normalize(inverse(A->orient) * B->orient);
        quat qErr = normalize(targetRel * inverse(qRel));
        if (qErr.w < 0) qErr = -qErr;                       // shortest arc

        vec3 v(qErr.x, qErr.y, qErr.z);
        float s = length(v);
        float angle = 2.0f * atan2(s, qErr.w);
        vec3 axisA = (s > 1e-6f) ? v / s : vec3(0);
        vec3 errWorld = A->orient * (axisA * angle);        // rotation error in world frame

        vec3 relW = B->angVel - A->angVel;
        vec3 torque = kp * errWorld - kd * relW;

        float lim = maxTorque * TORQUE_SCALE;
        float tl = length(torque);
        if (tl > lim) torque *= lim / tl;

        vec3 imp = torque * dt;
        A->angVel -= A->worldInvInertia() * imp;
        B->angVel += B->worldInvInertia() * imp;
    }
};

struct Skeleton {
    vector<Bone*> bones;
    vector<Joint> joints;
    vec3 startPos;
    int idx;
    Bone *root, *chest, *neck;
    Bone *hipR, *kneeR, *ankleR, *shoR, *elbR, *wriR;
    Bone *hipL, *kneeL, *ankleL, *shoL, *elbL, *wriL;

    Skeleton(vec3 p, int idx) : startPos(p), idx(idx) { init(p); }

    // DeepMimic humanoid3d: mass / torque limits, and humanoid3d_ctrl: Kp / Kd
    void init(vec3 base, float jitter = 0.0f) {
        auto rnd = [&]() { return ((rand() % 1000) / 1000.0f - 0.5f) * 2.0f * jitter; };
        for (auto b : bones) delete b;
        bones.clear(); joints.clear();

        auto B = [&](vec3 p, Shape s, vec3 d, float m) {
            Bone* b = new Bone(base + p, s, d, m);
            bones.push_back(b); return b;
        };

        // ---- Bones (positions in a rough standing pose, Y up) ----
        root   = B(vec3(0.00, 1.00, 0.00), SPHERE,  vec3(0.18, 0, 0),          6.0f);
        chest  = B(vec3(0.00, 1.40, 0.00), SPHERE,  vec3(0.22, 0, 0),         14.0f);
        neck   = B(vec3(0.00, 1.85, 0.00), SPHERE,  vec3(0.205, 0, 0),         2.0f);

        hipR   = B(vec3( 0.09, 0.73, 0.00), CAPSULE, vec3(0.11, 0.15, 0),       4.5f);
        kneeR  = B(vec3( 0.09, 0.42, 0.00), CAPSULE, vec3(0.10, 0.155, 0),      3.0f);
        ankleR = B(vec3( 0.09, 0.06, 0.03), BOX,     vec3(0.045, 0.0275, 0.09), 1.0f);
        shoR   = B(vec3( 0.30, 1.30, 0.00), CAPSULE, vec3(0.09, 0.09, 0),       1.5f);
        elbR   = B(vec3( 0.30, 1.10, 0.00), CAPSULE, vec3(0.08, 0.0675, 0),     1.0f);
        wriR   = B(vec3( 0.30, 0.95, 0.00), SPHERE,  vec3(0.08, 0, 0),          0.5f);

        hipL   = B(vec3(-0.09, 0.73, 0.00), CAPSULE, vec3(0.11, 0.15, 0),       4.5f);
        kneeL  = B(vec3(-0.09, 0.42, 0.00), CAPSULE, vec3(0.10, 0.155, 0),      3.0f);
        ankleL = B(vec3(-0.09, 0.06, 0.03), BOX,     vec3(0.045, 0.0275, 0.09), 1.0f);
        shoL   = B(vec3(-0.30, 1.30, 0.00), CAPSULE, vec3(0.09, 0.09, 0),       1.5f);
        elbL   = B(vec3(-0.30, 1.10, 0.00), CAPSULE, vec3(0.08, 0.0675, 0),     1.0f);
        wriL   = B(vec3(-0.30, 0.95, 0.00), SPHERE,  vec3(0.08, 0, 0),          0.5f);

        root->color = chest->color = vec3(0.55f, 0.75f, 0.95f);

        // ---- Joints (parent, child, anchorParent, anchorChild, Kp, Kd, torqueLim) ----
        // Kp/Kd/torqueLim from DeepMimic humanoid3d_ctrl + humanoid3d.
        auto J = [&](Bone* a, Bone* b, vec3 aa, vec3 ab, float kp, float kd, float tq) {
            joints.push_back(Joint(a, b, aa, ab, kp, kd, tq));
        };
        J(root,  chest, vec3(0, 0.18, 0),        vec3(0, -0.22, 0),  1000, 100, 200); // chest
        J(chest, neck,  vec3(0, 0.22, 0),        vec3(0, -0.205, 0),  100,  10,  50); // neck
        J(chest, shoR,  vec3( 0.19, 0.11, 0),    vec3(0,  0.09, 0),   400,  40, 100); // R shoulder
        J(shoR,  elbR,  vec3(0, -0.09, 0),       vec3(0,  0.0675, 0), 300,  30,  60); // R elbow
        J(elbR,  wriR,  vec3(0, -0.0675, 0),     vec3(0,  0.08, 0),    50,   5,  30); // R wrist
        J(chest, shoL,  vec3(-0.19, 0.11, 0),    vec3(0,  0.09, 0),   400,  40, 100); // L shoulder
        J(shoL,  elbL,  vec3(0, -0.09, 0),       vec3(0,  0.0675, 0), 300,  30,  60); // L elbow
        J(elbL,  wriL,  vec3(0, -0.0675, 0),     vec3(0,  0.08, 0),    50,   5,  30); // L wrist
        J(root,  hipR,  vec3( 0.09, -0.12, 0),   vec3(0,  0.15, 0),   500,  50, 200); // R hip
        J(hipR,  kneeR, vec3(0, -0.15, 0),       vec3(0,  0.155, 0),  500,  50, 150); // R knee
        J(kneeR, ankleR,vec3(0, -0.155, 0),      vec3(0,  0.0275, 0.03), 400, 40, 90); // R ankle
        J(root,  hipL,  vec3(-0.09, -0.12, 0),   vec3(0,  0.15, 0),   500,  50, 200); // L hip
        J(hipL,  kneeL, vec3(0, -0.15, 0),       vec3(0,  0.155, 0),  500,  50, 150); // L knee
        J(kneeL, ankleL,vec3(0, -0.155, 0),      vec3(0,  0.0275, 0.03), 400, 40, 90); // L ankle

        // Snap children onto parents (joints are ordered parent-first so it cascades)
        for (auto& j : joints) {
            vec3 err = j.A->worldPoint(j.anchorA_local) - j.B->worldPoint(j.anchorB_local);
            j.B->pos += err;
        }
        // Drop the assembled skeleton so its lowest contact point rests on the ground
        float minY = 1e9f;
        for (Bone* b : bones) {
            vector<pair<vec3, float>> cs; b->contacts(cs);
            for (auto& c : cs) minY = std::min(minY, c.first.y - c.second);
        }
        for (Bone* b : bones) b->pos.y -= minY;

        // Capture the assembled pose as the default PD target (holds a stand)
        for (auto& j : joints) j.targetRel = normalize(inverse(j.A->orient) * j.B->orient);
    }

    void step(float dtFull) {
        const int SUB = 10;          // substeps -> effective ~600Hz PD, matches DeepMimic stability
        const int ITER = 8;          // constraint iterations per substep
        float dt = dtFull / SUB;

        for (int s = 0; s < SUB; s++) {
            for (Bone* b : bones) {
                b->vel += G * dt;
                b->vel    *= 1.0f / (1.0f + 0.05f * dt);   // light drag
                b->angVel *= 1.0f / (1.0f + 0.05f * dt);
            }
            for (int it = 0; it < ITER; it++) {
                for (Joint& j : joints) { j.applyPD(dt); j.solve(dt); }
            }
            for (Bone* b : bones) {
                b->angVel = clamp(b->angVel, vec3(-40.0f), vec3(40.0f));
                b->vel    = clamp(b->vel,    vec3(-40.0f), vec3(40.0f));
                b->pos += b->vel * dt;
                b->integrateOrientation(dt);
                groundCollision(b);
            }
        }
    }

    void groundCollision(Bone* b) {
        const float rest = 0.0f, mu = 0.9f;
        vector<pair<vec3, float>> cs; b->contacts(cs);
        for (auto& c : cs) {
            vec3 p = c.first; float r = c.second;
            float pen = r - p.y;
            if (pen <= 0.0f) continue;
            vec3 n(0, 1, 0), rr = p - b->pos;
            mat3 Iinv = b->worldInvInertia();
            vec3 vp = b->vel + cross(b->angVel, rr);
            float vn = dot(vp, n);
            if (vn < 0.0f) {
                vec3 rn = cross(rr, n);
                float denom = b->invMass + dot(n, cross(Iinv * rn, rr));
                float jn = -(1.0f + rest) * vn / denom;
                vec3 P = jn * n;
                b->vel    += b->invMass * P;
                b->angVel += Iinv * cross(rr, P);
                // Coulomb friction
                vp = b->vel + cross(b->angVel, rr);
                vec3 vt = vp - dot(vp, n) * n;
                float vtl = length(vt);
                if (vtl > 1e-5f) {
                    vec3 t = vt / vtl, rt = cross(rr, t);
                    float dT = b->invMass + dot(t, cross(Iinv * rt, rr));
                    float jt = clamp(-vtl / dT, -mu * jn, mu * jn);
                    vec3 Pt = jt * t;
                    b->vel    += b->invMass * Pt;
                    b->angVel += Iinv * cross(rr, Pt);
                }
            }
            b->pos += n * pen * 0.8f;   // positional correction
        }
    }

    void draw() { for (Bone* b : bones) b->draw(); }
    void reset() { init(startPos, 0.03f); }
};

vector<Skeleton*> envs;

// ------------------------ File Connections ------------------------
struct Data {
    int sock, sendSock;
    sockaddr_in server, python;
    float recvBuffer[MAX_ENV * ACTION_DIM + 8];
    float stateBuffer[MAX_ENV * STATE_DIM];

    Data() {
        sock     = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        sendSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        fcntl(sock, F_SETFL, O_NONBLOCK);              // don't block the render loop

        server.sin_family = AF_INET; server.sin_port = htons(5005);
        server.sin_addr.s_addr = INADDR_ANY;
        python.sin_family = AF_INET; python.sin_port = htons(5006);
        inet_pton(AF_INET, "127.0.0.1", &python.sin_addr);
        bind(sock, (sockaddr*)&server, sizeof(server));
    }

    // returns true if python requested a state snapshot
    bool receiveData() {
        int n = recv(sock, (char*)recvBuffer, sizeof(recvBuffer), 0);
        if (n < (int)(2 * sizeof(float))) return false;

        if (recvBuffer[0] == -100.0f) return true;                 // request state
        if (recvBuffer[0] == -69.0f) { int e = (int)recvBuffer[1];
            if (e >= 0 && e < (int)envs.size()) envs[e]->reset(); return false; }
        if (recvBuffer[0] == -67.0f) { TORQUE_SCALE = recvBuffer[1]; return false; }

        if (n == (int)(NUM_ENV * ACTION_DIM * sizeof(float))) {    // per-joint target quats
            int i = 0;
            for (Skeleton* env : envs)
                for (Joint& j : env->joints) {
                    quat q(recvBuffer[i], recvBuffer[i+1], recvBuffer[i+2], recvBuffer[i+3]);
                    i += 4;
                    if (dot(q, q) > 1e-6f) j.targetRel = normalize(q);
                }
        }
        return false;
    }

    void sendData() {
        int i = 0;
        for (Skeleton* env : envs) {
            for (Bone* b : env->bones) {
                stateBuffer[i++] = b->pos.x; stateBuffer[i++] = b->pos.y; stateBuffer[i++] = b->pos.z;
                stateBuffer[i++] = b->orient.w; stateBuffer[i++] = b->orient.x;
                stateBuffer[i++] = b->orient.y; stateBuffer[i++] = b->orient.z;
                stateBuffer[i++] = b->vel.x; stateBuffer[i++] = b->vel.y; stateBuffer[i++] = b->vel.z;
                stateBuffer[i++] = b->angVel.x; stateBuffer[i++] = b->angVel.y; stateBuffer[i++] = b->angVel.z;
            }
            auto grounded = [](Bone* f) {
                vector<pair<vec3, float>> cs; f->contacts(cs);
                for (auto& c : cs) if (c.first.y <= c.second + 0.02f) return 1.0f;
                return 0.0f;
            };
            stateBuffer[i++] = grounded(env->ankleL);
            stateBuffer[i++] = grounded(env->ankleR);
        }
        sendto(sendSock, (char*)stateBuffer, i * sizeof(float), 0, (sockaddr*)&python, sizeof(python));
    }
};
Data* dataManager;

// ------------------------ MAIN ------------------------
int main() {
    srand(time(0) * 1234567891ULL ^ (uint64_t)clock());

    for (int i = 0; i < NUM_ENV; i++)
        envs.push_back(new Skeleton(vec3(i * 1.6f, 0, 0), i));

    dataManager = new Data();

    float dt = 1.0f / 60.0f;
    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();
        engine.drawGround();

        bool gotAction = dataManager->receiveData();

        for (Skeleton* env : envs) { env->step(dt); env->draw(); }

        if (gotAction) dataManager->sendData();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
