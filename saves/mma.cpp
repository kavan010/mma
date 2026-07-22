#define GLM_ENABLE_EXPERIMENTAL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;

// ================= vars ================= //
const int NUM_ENV = 1;
// --- Animation Globals ---
struct Keyframe {
    float time;
    std::vector<glm::vec3> angles;
    glm::vec3 pelvisOffset = glm::vec3(0);
    // Pelvis rotation as three INDEPENDENT unwrapped angles (radians) about the fixed
    // world X/Y/Z axes, applied in X->Y->Z order — NOT an axis-angle vector. A wrapped
    // axis-angle (recovered from a quaternion via acos) is clamped to [-pi, pi], so a
    // continuous spin like a backflip's 270 degrees -> 365 degrees would fold back and
    // interpolate the short way (270 -> 0) instead of continuing forward. Keeping each
    // axis as a plain unbounded float lets it keep counting past 360 degrees, so
    // lerp-ing between two keyframes' angles is always monotonic in the direction you
    // actually rotated. Only used for editing/interpolation — the baked CSV still
    // writes ordinary per-frame absolute quaternions (dense enough that no single
    // frame-to-frame delta ever needs to be >180 degrees), which is what train.py's
    // per-frame quaternion comparison already expects and needs no change for.
    glm::vec3 pelvisRot = glm::vec3(0);
};

// modes for making animation and testing (M toggles; playback lives in the timeline UI via V/C)
enum AppMode { PHYSICS, ANIMATE };
AppMode currentMode = ANIMATE;

// mouse routing: timeline + axis gizmo get first pick, camera gets the rest (defined after the UI structs)
void onMouseButton(GLFWwindow* win, int button, int action, int mods);
void onCursorMove(GLFWwindow* win, double x, double y);

// ================= camera ================= //
struct Camera {
    float radius = 50.0f;
    float azimuth = 0.0f;
    float elevation = M_PI / 2.0f;
    float orbitSpeed = 0.01f;
    double zoomSpeed = 10.0;
    float panSpeed = 0.05f;
    bool dragging = false, panning = false;
    double lastX = 0.0, lastY = 0.0;
    vec3 target = vec3(0.0f);

    vec3 position() const {
        float e = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        return target + vec3(radius * sin(e) * cos(azimuth), radius * cos(e), radius * sin(e) * sin(azimuth));
    }
    void processMouseMove(double x, double y) {
        if (dragging) {
            azimuth += float(x - lastX) * orbitSpeed;
            elevation = glm::clamp(elevation - float(y - lastY) * orbitSpeed, 0.01f, float(M_PI) - 0.01f);
        }
        if (panning) {
            // move the target along the view plane (camera right + up)
            vec3 fwd = normalize(target - position());
            vec3 right = normalize(cross(fwd, vec3(0, 1, 0)));
            vec3 up = cross(right, fwd);
            float scale = panSpeed * radius / 50.0f; // pan slower when zoomed in
            target += (-right * float(x - lastX) + up * float(y - lastY)) * scale;
        }
        lastX = x; lastY = y;
    }
    void processMouseButton(int button, int action, GLFWwindow* win) {
        bool* state = button == GLFW_MOUSE_BUTTON_LEFT ? &dragging
                    : button == GLFW_MOUSE_BUTTON_MIDDLE ? &panning : nullptr;
        if (!state) return;
        if (action == GLFW_PRESS) { *state = true; glfwGetCursorPos(win, &lastX, &lastY); }
        else if (action == GLFW_RELEASE) *state = false;
    }
    void processScroll(double, double yoffset) {
        radius = glm::max(1.0, radius - yoffset * zoomSpeed);
    }
};
Camera camera;

// ================= engine ================= //
struct Engine {
    GLFWwindow* window;
    int width = 800, height = 600;
    GLuint shaderProgram;
    GLint modelLoc, viewLoc, projLoc, colorLoc;

    const char* vertexShaderSource = R"glsl(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 model, view, projection;
        void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); }
    )glsl";

    const char* fragmentShaderSource = R"glsl(
        #version 330 core
        out vec4 FragColor;
        uniform vec4 objectColor;
        void main() { FragColor = objectColor; }
    )glsl";

    Engine() {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11); // GLEW needs a GLX context; native Wayland breaks glewInit()
        if (!glfwInit()) exit(-1);
        glfwWindowHintString(GLFW_X11_CLASS_NAME, "mma3d"); // lets Hyprland target this window with a float rule
        glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "mma3d");
        window = glfwCreateWindow(width, height, "3D Viewer", NULL, NULL);
        glfwMakeContextCurrent(window);
        glewInit();
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL); // let edge lines win against the face they sit on
        glEnable(GL_POLYGON_OFFSET_FILL); // push filled surfaces back so outline lines stay fully visible
        glPolygonOffset(1.0f, 1.0f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        shaderProgram = compileShader();
        modelLoc = glGetUniformLocation(shaderProgram, "model");
        viewLoc  = glGetUniformLocation(shaderProgram, "view");
        projLoc  = glGetUniformLocation(shaderProgram, "projection");
        colorLoc = glGetUniformLocation(shaderProgram, "objectColor");

        glfwSetWindowUserPointer(window, &camera);
        glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
            onMouseButton(win, button, action, mods);
        });
        glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
            onCursorMove(win, x, y);
        });
        glfwSetScrollCallback(window, [](GLFWwindow* win, double x, double y) {
            ((Camera*)glfwGetWindowUserPointer(win))->processScroll(x, y);
        });
    }

    GLuint compileShader() {
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vertexShaderSource, NULL);
        glCompileShader(vs);
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fragmentShaderSource, NULL);
        glCompileShader(fs);
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        return prog;
    }
    void createVAO(GLuint& VAO, GLuint& VBO, const float* data, size_t count) {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, count * sizeof(float), data, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }
    void beginFrame() {
        glfwGetWindowSize(window, &width, &height); // keep UI layout + cursor math valid after a resize
        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        mat4 proj = perspective(radians(45.0f), (float)width / height, 0.1f, 1000.0f);
        mat4 view = lookAt(camera.position(), camera.target, vec3(0, 1, 0));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, value_ptr(proj));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, value_ptr(view));
    }
    mat3 crossMatrix(vec3 v) {
        return mat3(vec3(0, v.z, -v.y), vec3(-v.z, 0, v.x), vec3(v.y, -v.x, 0));
    }
};
Engine engine;

// ================= ground plane ================= //
struct Grid {
    GLuint VAO, VBO;
    int vertexCount;
    GLuint lineVAO, lineVBO;
    int lineVertexCount;

    Grid(float size = 500.0f, int divisions = 50) {
        float half = size / 2.0f;
        vector<float> vertices = {
            -half, 0, -half,   half, 0, -half,   half, 0, half,
            -half, 0, -half,   half, 0, half,    -half, 0, half,
        };
        vertexCount = vertices.size() / 3;
        engine.createVAO(VAO, VBO, vertices.data(), vertices.size());

        vector<float> lines;
        float step = size / divisions;
        for (int i = 0; i <= divisions; ++i) {
            float x = -half + i * step, z = -half + i * step;
            lines.insert(lines.end(), {x, 0, -half, x, 0, half});
            lines.insert(lines.end(), {-half, 0, z, half, 0, z});
        }
        lineVertexCount = lines.size() / 3;
        engine.createVAO(lineVAO, lineVBO, lines.data(), lines.size());
    }

    void draw() {
        mat4 model = mat4(1.0f);
        glUniformMatrix4fv(engine.modelLoc, 1, GL_FALSE, value_ptr(model));
        glDepthMask(GL_FALSE);

        glUniform4f(engine.colorLoc, 0.05f, 0.05f, 0.05f, 0.7f);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);

        glUniform4f(engine.colorLoc, 1.0f, 1.0f, 1.0f, 0.06f);
        glBindVertexArray(lineVAO);
        glDrawArrays(GL_LINES, 0, lineVertexCount);

        glDepthMask(GL_TRUE);
    }
};
Grid grid;

// ================= bone ================= //
struct Bone {
    // physics
    vec3 pos, vel = vec3(0), angVel = vec3(0);
    quat orient = quat(1, 0, 0, 0);
    vec3 dims; // box: half extentions | capsule: x = half length, y = radius, z is useless
    float mass, invMass;
    mat3 invInertiaBody = mat3(0.0f);
    int shape; // 1 = box, 2 = capsule

    // 2D rendering
    vec3 color;
    GLuint VAO = 0, VBO = 0, lineVAO = 0, lineVBO = 0;
    int vertexCount = 0, lineVertexCount = 0;

    Bone(vec3 p, vec3 d, float m, int shape, vec3 c = vec3(0.85f))
        : pos(p), dims(d), mass(m), shape(shape), color(c) {
        invMass = 1.0f / m;
        vec3 I;
        if (shape == 1) {
            vec3 f = 2.0f * dims;
            I = mass / 12.0f * vec3(f.y * f.y + f.z * f.z, f.x * f.x + f.z * f.z, f.x * f.x + f.y * f.y);
        } else {
            float L = 2.0f * dims.x, r = dims.y;
            float side = mass / 12.0f * (3 * r * r + L * L);
            I = vec3(0.5f * mass * r * r, side, side);
        }
        invInertiaBody[0][0] = 1.0f / I.x;
        invInertiaBody[1][1] = 1.0f / I.y;
        invInertiaBody[2][2] = 1.0f / I.z;
        buildMesh();
    }

    void buildMesh() {
        vector<float> tris, lines;
        auto pushTri  = [&](vec3 p)         { tris.insert(tris.end(), {p.x, p.y, p.z}); };
        auto pushLine = [&](vec3 a, vec3 b) { lines.insert(lines.end(), {a.x, a.y, a.z, b.x, b.y, b.z}); };
        auto pushLoop = [&](const vector<vec3>& pts) {
            for (size_t i = 0; i < pts.size(); i++) pushLine(pts[i], pts[(i + 1) % pts.size()]);
        };
        //box
        if (shape == 1) {
            float hx = dims.x, hy = dims.y, hz = dims.z;
            vec3 corner[8] = {
                { hx,  hy,  hz}, { hx,  hy, -hz}, { hx, -hy,  hz}, { hx, -hy, -hz},
                {-hx,  hy,  hz}, {-hx,  hy, -hz}, {-hx, -hy,  hz}, {-hx, -hy, -hz},
            };
            int face[6][4] = {
                {0, 1, 3, 2}, {4, 5, 7, 6}, {0, 1, 5, 4},
                {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 3, 7, 5},
            };
            for (auto& f : face) {
                pushTri(corner[f[0]]); pushTri(corner[f[1]]); pushTri(corner[f[2]]);
                pushTri(corner[f[0]]); pushTri(corner[f[2]]); pushTri(corner[f[3]]);
            }
            int edge[12][2] = {
                {0,1},{1,3},{3,2},{2,0}, {4,5},{5,7},{7,6},{6,4}, {0,4},{1,5},{3,7},{2,6},
            };
            for (auto& e : edge) pushLine(corner[e[0]], corner[e[1]]);

        //capsule
        } else {
            // capsule along local x: cylinder wall, end domes, ring + stadium outlines
            float L = dims.x, r = dims.y;
            int seg = 24, halfSeg = 12;
            auto ringPoint = [&](float x, float a) { return vec3(x, r * cos(a), r * sin(a)); };
            auto domePoint = [&](float capX, float dir, float ringA, float sectorA) {
                return vec3(capX + dir * r * sin(ringA), r * cos(ringA) * cos(sectorA), r * cos(ringA) * sin(sectorA));
            };
            for (int j = 0; j < seg; j++) {
                float a1 = 2 * M_PI * j / seg, a2 = 2 * M_PI * (j + 1) / seg;
                vec3 t1 = ringPoint(L, a1), t2 = ringPoint(L, a2), b1 = ringPoint(-L, a1), b2 = ringPoint(-L, a2);
                pushTri(t1); pushTri(b1); pushTri(t2);
                pushTri(t2); pushTri(b1); pushTri(b2);
            }
            for (int end = 0; end < 2; end++) {
                float capX = end == 0 ? L : -L, dir = end == 0 ? 1.0f : -1.0f;
                for (int i = 0; i < halfSeg; i++) {
                    float r1 = M_PI / 2 * i / halfSeg, r2 = M_PI / 2 * (i + 1) / halfSeg;
                    for (int j = 0; j < seg; j++) {
                        float a1 = 2 * M_PI * j / seg, a2 = 2 * M_PI * (j + 1) / seg;
                        vec3 p1 = domePoint(capX, dir, r1, a1), p2 = domePoint(capX, dir, r1, a2);
                        vec3 p3 = domePoint(capX, dir, r2, a1), p4 = domePoint(capX, dir, r2, a2);
                        pushTri(p1); pushTri(p2); pushTri(p3);
                        pushTri(p2); pushTri(p4); pushTri(p3);
                    }
                }
            }
            for (float x : {L, -L}) {
                vector<vec3> ring;
                for (int i = 0; i < seg; i++) {
                    float t = 2 * M_PI * i / seg;
                    ring.push_back(vec3(x, r * cos(t), r * sin(t)));
                }
                pushLoop(ring);
            }
            for (int plane = 0; plane < 2; plane++) {
                vector<vec3> s;
                for (int i = 0; i <= halfSeg; i++) {
                    float a = -M_PI/2 + M_PI * i / halfSeg;
                    s.push_back(vec3( L + r*cos(a), plane ? 0 : r*sin(a), plane ? r*sin(a) : 0));
                }
                for (int i = 0; i <= halfSeg; i++) {
                    float a =  M_PI/2 + M_PI * i / halfSeg;
                    s.push_back(vec3(-L + r*cos(a), plane ? 0 : r*sin(a), plane ? r*sin(a) : 0));
                }
                pushLoop(s);
            }
        }

        vertexCount = tris.size() / 3;
        lineVertexCount = lines.size() / 3;
        engine.createVAO(VAO, VBO, tris.data(), tris.size());
        engine.createVAO(lineVAO, lineVBO, lines.data(), lines.size());
    }
    void draw() {
        mat4 model = translate(mat4(1.0f), pos) * mat4_cast(orient);
        glUniformMatrix4fv(engine.modelLoc, 1, GL_FALSE, value_ptr(model));
        glUniform4f(engine.colorLoc, color.r, color.g, color.b, 1.0f);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);

        vec3 lc = color * 0.25f;
        glLineWidth(2.0f);
        glUniform4f(engine.colorLoc, lc.r, lc.g, lc.b, 1.0f);
        glBindVertexArray(lineVAO);
        glDrawArrays(GL_LINES, 0, lineVertexCount);
    }
};

// ================= joint ================= //
struct Joint {
    Bone *A, *B;
    vec3 anchorA, anchorB;
    float maxTorque = 200.0f;

    // --- controller properties ---
    vec3 targetAngle = vec3(0); // axis-angle rotation of B relative to A
    float stiffness = 1.0f;

    Joint(Bone* a, Bone* b, vec3 anchorA, vec3 anchorB, vec3 targetAngle, float stiffness = 1.0f, float maxTorque = 200.0f)
        : A(a), B(b), anchorA(anchorA), anchorB(anchorB), maxTorque(maxTorque), targetAngle(targetAngle), stiffness(stiffness) {}

    void solve(float dt) {
        mat3 RA = mat3_cast(A->orient), RB = mat3_cast(B->orient);
        vec3 rA = RA * anchorA, rB = RB * anchorB;
        mat3 invIA = RA * A->invInertiaBody * transpose(RA);
        mat3 invIB = RB * B->invInertiaBody * transpose(RB);

        vec3 C = (B->pos + rB) - (A->pos + rA);
        vec3 relVel = (B->vel + cross(B->angVel, rB)) - (A->vel + cross(A->angVel, rA));

        float beta = 0.2f;
        vec3 Cdot = relVel + (beta / dt) * C;

        mat3 sA = engine.crossMatrix(rA), sB = engine.crossMatrix(rB);
        mat3 K = mat3(A->invMass + B->invMass) - sA * invIA * sA - sB * invIB * sB;
        vec3 P = -inverse(K) * Cdot;

        A->vel -= P * A->invMass;
        B->vel += P * B->invMass;
        A->angVel -= invIA * cross(rA, P);
        B->angVel += invIB * cross(rB, P);
    }

    void applyTorque(float dt) {
        float t = length(targetAngle);
        quat qTarget = t < 1e-6f ? quat(1, 0, 0, 0) : angleAxis(t, targetAngle / t);
        quat qRel = inverse(A->orient) * B->orient;
        quat dq = qTarget * inverse(qRel);
        if (dq.w < 0) dq = -dq;

        // orientation error as world-space axis-angle
        vec3 v = vec3(dq.x, dq.y, dq.z);
        float s = length(v);
        vec3 error = s < 1e-6f ? vec3(0) : A->orient * ((2.0f * atan2(s, dq.w)) * (v / s));
        vec3 angVel = B->angVel - A->angVel;

        float k = stiffness * maxTorque; // kp
        float d = k / 10.0f;

        vec3 torque = k * error - d * angVel;
        if (length(torque) > maxTorque) torque = maxTorque * normalize(torque);

        vec3 impulse = torque * dt;
        mat3 RA = mat3_cast(A->orient), RB = mat3_cast(B->orient);
        A->angVel -= (RA * A->invInertiaBody * transpose(RA)) * impulse;
        B->angVel += (RB * B->invInertiaBody * transpose(RB)) * impulse;
    }
};

// ================= skeleton ================= //
struct Skeleton {
    vec3 pos; int idx;
    vector<Bone*> bones;
    vector<Joint> joints;
    Bone *footR, *footL, *calfR, *calfL, *thighR, *thighL, *pelvis, *abs, *chest, *head, *armR, *armL, *forearmR, *forearmL;
    Bone *ball, *clavR, *clavL; // visual only, not simulated: joint spheres + clavicles

    Skeleton(vec3 pos, int id) : pos(vec3(pos)), idx(id) { init(); }

    void init() {
        for (Bone* b : bones) delete b;
        joints.clear();

        quat down = angleAxis(-float(M_PI) / 2.0f, vec3(0, 0, 1));

        // world-space joint heights for the standing pose (joint gaps included)
        float ankleY = 1.2f, kneeY = 10.2f, hipY = 20.2f, hipZ = 2.0f;
        float shoulderY = 31.2f, shoulderZ = 3.8f;

        //                                    pos                      hfl   rad   -      kg
        pelvis   = new Bone(vec3(pos.x, 21.8f, pos.z),                    vec3(0.0f, 2.2f, 0.0f),  5.0f, 2);
        abs      = new Bone(vec3(pos.x, 25.4f, pos.z),                    vec3(0.0f, 1.75f, 0.0f),  7.0f, 2);
        chest    = new Bone(vec3(pos.x, 29.6f, pos.z),                    vec3(0.0f, 2.65f, 0.0f),  8.0f, 2);
        head     = new Bone(vec3(pos.x, 34.6f, pos.z),                    vec3(0.0f, 2.25f, 0.0f),  3.0f, 2);

        thighR   = new Bone(vec3(0, (hipY + kneeY) / 2, hipZ),    vec3(2.8f, 1.35f, 0.0f),  4.5f, 2);
        thighL   = new Bone(vec3(0, (hipY + kneeY) / 2, -hipZ),   vec3(2.8f, 1.35f, 0.0f),  4.5f, 2);
        calfR    = new Bone(vec3(0, (kneeY + ankleY) / 2, hipZ),  vec3(2.7f, 1.3f, 0.0f),  2.5f, 2);
        calfL    = new Bone(vec3(0, (kneeY + ankleY) / 2, -hipZ), vec3(2.7f, 1.3f, 0.0f),  2.5f, 2);
        footR    = new Bone(vec3(1.5f, 0.6f, hipZ),               vec3(2.7f, 0.6f, 1.0f),  1.0f, 1); // box half-extents
        footL    = new Bone(vec3(1.5f, 0.6f, -hipZ),              vec3(2.7f, 0.6f, 1.0f),  1.0f, 1);

        armR     = new Bone(vec3(0, shoulderY - 3.0f, shoulderZ), vec3(1.9f, 1.0f, 0.0f),  1.8f, 2);
        armL     = new Bone(vec3(0, shoulderY - 3.0f, -shoulderZ),vec3(1.9f, 1.0f, 0.0f),  1.8f, 2);
        forearmR = new Bone(vec3(0, shoulderY - 8.5f, shoulderZ), vec3(1.5f, 0.9f, 0.0f),  1.2f, 2);
        forearmL = new Bone(vec3(0, shoulderY - 8.5f, -shoulderZ),vec3(1.5f, 0.9f, 0.0f),  1.2f, 2);

        for (Bone* b : {thighR, thighL, calfR, calfL, armR, armL, forearmR, forearmL}) b->orient = down;

        bones = { footR, footL, calfR, calfL, thighR, thighL, pelvis, abs, chest, armR, armL, forearmR, forearmL, head };

        // visual-only bodies (not simulated): joint sphere + shoulder clavicles
        ball  = new Bone(vec3(0), vec3(0.0f, 0.85f, 0.0f), 1.0f, 2, vec3(0.20f, 0.20f, 0.28f));
        clavR = new Bone(vec3(0), vec3(1.2f, 1.0f, 0.0f), 1.0f, 2);
        clavL = new Bone(vec3(0), vec3(1.2f, 1.0f, 0.0f), 1.0f, 2);

        float gap = 0.3f;
        auto top = [=](Bone* b) { return vec3(-b->dims.x - b->dims.y - gap, 0, 0); };
        auto bot = [=](Bone* b) { return vec3( b->dims.x + b->dims.y + gap, 0, 0); };
        float hp = float(M_PI) / 2.0f;


        //                     A       B         anchor A (local)                anchor B (local)      target (axis-angle)  stiff  maxTorque
        joints.push_back(Joint(pelvis, abs,      vec3(0,  1.8f, 0),              vec3(0, -1.8f, 0),    vec3(0, 0, 0),   2.0f, 6000.0f)); // waist
        joints.push_back(Joint(abs,    chest,    vec3(0,  1.8f, 0),              vec3(0, -2.4f, 0),    vec3(0, 0, 0),   2.0f, 6000.0f)); // spine
        joints.push_back(Joint(chest,  head,     vec3(0,  2.9f, 0),              vec3(0, -2.1f, 0),    vec3(0, 0, 0),   2.0f, 1000.0f)); // neck
        joints.push_back(Joint(chest,  armR,     vec3(0,  1.6f,  shoulderZ),     top(armR),            vec3(0, 0, -hp), 2.0f, 4000.0f)); // shoulder R
        joints.push_back(Joint(chest,  armL,     vec3(0,  1.6f, -shoulderZ),     top(armL),            vec3(0, 0, -hp), 2.0f, 4000.0f)); // shoulder L
        joints.push_back(Joint(armR,   forearmR, bot(armR),                      top(forearmR),        vec3(0, 0, 0),   2.0f, 3000.0f)); // elbow R
        joints.push_back(Joint(armL,   forearmL, bot(armL),                      top(forearmL),        vec3(0, 0, 0),   2.0f, 3000.0f)); // elbow L
        joints.push_back(Joint(pelvis, thighR,   vec3(0, -1.6f,  hipZ-0.5f),          top(thighR),          vec3(0, 0, -hp), 4.0f, 6000.0f)); // hip R
        joints.push_back(Joint(pelvis, thighL,   vec3(0, -1.6f, -hipZ+0.5f),          top(thighL),          vec3(0, 0, -hp), 4.0f, 6000.0f)); // hip L
        joints.push_back(Joint(thighR, calfR,    bot(thighR),                    top(calfR),           vec3(0, 0, 0),   4.0f, 6000.0f)); // knee R
        joints.push_back(Joint(thighL, calfL,    bot(thighL),                    top(calfL),           vec3(0, 0, 0),   4.0f, 6000.0f)); // knee L
        joints.push_back(Joint(calfR,  footR,    bot(calfR),                     vec3(-0.7f, 0.25f, 0), vec3(0, 0, hp), 5.0f, 4000.0f)); // ankle R
        joints.push_back(Joint(calfL,  footL,    bot(calfL),                     vec3(-0.7f, 0.25f, 0), vec3(0, 0, hp), 5.0f, 4000.0f)); // ankle L

        // snap child positions onto their anchors to remove any residual init error
        for (Joint& j : joints) {
            vec3 err = (j.A->pos + j.A->orient * j.anchorA) - (j.B->pos + j.B->orient * j.anchorB);
            j.B->pos += err;
        }
    }

    void draw() {
        for (Bone* b : bones) b->draw();

        // dark sphere at every joint anchor
        for (Joint& j : joints) {
            ball->pos = j.A->pos + j.A->orient * j.anchorA;
            ball->draw();
        }
        // clavicles: pinned to the chest, capsule axis turned to run along z toward each shoulder
        float hp = float(M_PI) / 2.0f;
        clavR->pos = chest->pos + chest->orient * vec3(0, 1.6f, 2.1f);
        clavR->orient = chest->orient * angleAxis(-hp, vec3(0, 1, 0));
        clavL->pos = chest->pos + chest->orient * vec3(0, 1.6f, -2.1f);
        clavL->orient = chest->orient * angleAxis(hp, vec3(0, 1, 0));
        clavR->draw();
        clavL->draw();
    }

    void step(float dt) {
        // ---- Euler Integrate Gravity ----
        for (Bone* b : bones) {
            b->vel += vec3(0, -9.81f, 0) * dt;
            b->vel    *= 1.0f / (1.0f + 0.5f * dt);
            b->angVel *= 1.0f / (1.0f + 0.5f * dt);
        }

        // ---- Solve Joints & Apply Torque ----
        int iters = 15;
        for (int i = 0; i < iters; i++) {
            if (i % 2 == 0) for (int n = 0; n < (int)joints.size(); n++)  { joints[n].solve(dt); joints[n].applyTorque(dt / iters); }
            else            for (int n = joints.size() - 1; n >= 0; n--)  { joints[n].solve(dt); joints[n].applyTorque(dt / iters); }
        }

        // ---- Euler Integrate & collisions ----
        for (Bone* b : bones) {
            b->vel    = glm::clamp(b->vel, vec3(-100.0f), vec3(100.0f));
            b->angVel = glm::clamp(b->angVel, vec3(-50.0f), vec3(50.0f));
            b->pos += b->vel * dt;
            quat spin(0.0f, b->angVel.x, b->angVel.y, b->angVel.z);
            b->orient = normalize(b->orient + 0.5f * dt * (spin * b->orient));

            // dont fall through floor
            floorCollision(b, dt);
        }
    }
    void updateKinematics() {
        // for making .npz reference
        for (Joint& j : joints) {
            float t = glm::length(j.targetAngle);
            glm::quat qTarget = t < 1e-6f ? glm::quat(1, 0, 0, 0) : glm::angleAxis(t, j.targetAngle / t);
            j.B->orient = j.A->orient * qTarget;

            //Snap child position so the anchors perfectly attach
            glm::vec3 anchorWorldA = j.A->pos + j.A->orient * j.anchorA;
            glm::vec3 anchorWorldB = j.B->orient * j.anchorB;
            j.B->pos = anchorWorldA - anchorWorldB;
        }
    }

    // FK only propagates outward from the pelvis, so posing (e.g. a crouch) leaves
    // the feet floating/clipping instead of the hips moving to meet the floor.
    // Call on demand (F key) rather than every frame, so it doesn't fight an
    // intentional pelvis-height (Q/E) move.
    void snapToGround() {
        float lowest = glm::min(footR->pos.y - footR->dims.y, footL->pos.y - footL->dims.y);
        vec3 shift(0, -lowest, 0);
        for (Bone* b : bones) b->pos += shift;
    }
    // One-directional version of snapToGround: only pushes the body UP if the
    // lowest foot has gone below the floor, never pulls it down. Used for
    // interpolated frames (scrubbing/playback and the baked CSV) where a foot
    // dipping underground between two authored keyframes is never intentional —
    // unlike F, this never moves a pose that's already clear of the floor.
    void clampAboveGround() {
        float lowest = glm::min(footR->pos.y - footR->dims.y, footL->pos.y - footL->dims.y);
        if (lowest >= 0.0f) return;
        vec3 shift(0, -lowest, 0);
        for (Bone* b : bones) b->pos += shift;
    }
    void floorCollision(Bone* b, float dt) {
        float restitution = 0.0f, slop = 0.01f, percent = 0.4f, friction = 1.2f;
        float margin = 0.05f;

        // contact points: box corners (r = 0) or capsule endpoints (r = radius)
        vector<vec3> pts;
        float r;
        if (b->shape == 1) {
            r = 0.0f;
            for (int i = 0; i < 8; i++)
                pts.push_back(vec3(i & 1 ? b->dims.x : -b->dims.x,
                                   i & 2 ? b->dims.y : -b->dims.y,
                                   i & 4 ? b->dims.z : -b->dims.z));
        } else {
            r = b->dims.y;
            pts = { vec3(b->dims.x, 0, 0), vec3(-b->dims.x, 0, 0) };
        }

        mat3 R = mat3_cast(b->orient);
        mat3 invI = R * b->invInertiaBody * transpose(R);
        vec3 n(0, 1, 0);

        // iterate the contact set so simultaneous contacts (flat foot = 4 corners) converge
        // together — a single sequential pass systematically favors the first-processed
        // corner and torques the body over.
        // accumulated impulses per contact: friction budget = friction * total normal
        // impulse this contact actually carries (the foot transmits the whole leg/body
        // load through the joints, far more than the foot bone's own weight)
        float maxPen = 0.0f;
        vector<float> jnAcc(pts.size(), 0.0f), jtAcc(pts.size(), 0.0f);
        for (int iter = 0; iter < 8; iter++) {
            for (size_t i = 0; i < pts.size(); i++) {
                vec3 contact = b->pos + R * pts[i];
                float penetration = r - contact.y;
                if (penetration <= -margin) continue;
                if (iter == 0) maxPen = glm::max(maxPen, penetration);

                vec3 rw = contact - b->pos;
                vec3 vel = b->vel + cross(b->angVel, rw);
                float vn = dot(vel, n);

                float denom = b->invMass + dot(cross(invI * cross(rw, n), rw), n);
                if (denom == 0) continue;

                // normal impulse, accumulated & clamped non-negative so iterations converge
                float jn = -(1.0f + restitution) * vn / denom;
                float jnNew = glm::max(jnAcc[i] + jn, 0.0f);
                jn = jnNew - jnAcc[i];
                jnAcc[i] = jnNew;
                b->vel += n * (jn * b->invMass);
                b->angVel += invI * cross(rw, n * jn);

                // Coulomb friction against the accumulated normal impulse, floored by the
                // gravity support impulse so a resting contact still has static friction.
                float budget = friction * glm::max(jnAcc[i], b->mass * 9.81f * dt) - jtAcc[i];
                vel = b->vel + cross(b->angVel, rw); // recompute after normal impulse
                vec3 vt = vel - dot(vel, n) * n;
                float vtLen = length(vt);
                if (vtLen > 1e-6f && budget > 0.0f) {
                    vec3 t = vt / vtLen;
                    float denomT = b->invMass + dot(cross(invI * cross(rw, t), rw), t);
                    if (denomT > 0.0f) {
                        float jt = glm::min(vtLen / denomT, budget);
                        jtAcc[i] += jt;
                        b->vel -= t * (jt * b->invMass);
                        b->angVel -= invI * cross(rw, t * jt);
                    }
                }
            }
        }
        // one positional correction for the deepest point, not per-contact — pushing per
        // corner over-lifts a flat foot by up to 4x
        b->pos += n * (percent * glm::max(maxPen - slop, 0.0f));
    }

};
vector<Skeleton*> envs {
    // new Skeleton(vec3(0, 0, -20), 1),
    new Skeleton(vec3(0, 0, 0), 0),
    // new Skeleton(vec3(0, 0, 20), 1),
};

// ================= axis gizmo ================= //
// X/Y/Z picks a rotation axis on the selected joint (ESC releases it). Two input
// modes, toggled with UP: mouse mode = click-drag to rotate; arrow mode =
// LEFT/RIGHT keys rotate. Rotation happens in the parent bone's frame, pivoting
// at the joint anchor.
struct AxisGizmo {
    int axis = -1;          // -1 = off, 0/1/2 = x/y/z
    int jointIdx = 0;       // shared "selected joint" (DOWN cycles it in KeyControl)
    bool dragging = false;  // left mouse held: the drag rotates the joint, not the camera
    bool useArrows = false; // UP toggles: false = click-drag mouse, true = LEFT/RIGHT keys
    float sens = 0.008f;    // radians per pixel of mouse movement
    GLuint VAO = 0, VBO = 0;

    void toggle(int a) { axis = (axis == a) ? -1 : a; if (axis < 0) dragging = false; }
    void release() { axis = -1; dragging = false; }
    vec3 dir() { return axis == 0 ? vec3(1, 0, 0) : axis == 1 ? vec3(0, 1, 0) : vec3(0, 0, 1); }

    void rotate(Joint& j, float dx) {
        vec3& ta = j.targetAngle;
        float len = length(ta);
        quat q = len < 1e-6f ? quat(1, 0, 0, 0) : angleAxis(len, ta / len);
        q = angleAxis(dx * sens, dir()) * q; // pre-multiply = spin about the parent-frame axis
        if (q.w < 0) q = -q;
        vec3 v(q.x, q.y, q.z);
        float s = length(v);
        ta = s < 1e-6f ? vec3(0) : (2.0f * atan2(s, q.w)) * (v / s);
    }

    // Builds the pelvis quaternion from three INDEPENDENT unwrapped angles (see the
    // comment on Keyframe::pelvisRot) instead of treating the vec3 as an axis-angle —
    // applied X then Y then Z, each about the fixed world axis.
    static quat pelvisQuat(vec3 eulerXYZ) {
        return angleAxis(eulerXYZ.z, vec3(0, 0, 1))
             * angleAxis(eulerXYZ.y, vec3(0, 1, 0))
             * angleAxis(eulerXYZ.x, vec3(1, 0, 0));
    }

    // Pelvis isn't a Joint (it's the FK root, so it has no parent/anchor to pivot
    // on) — rotate it in place about the picked world axis. Unlike joint rotation,
    // this accumulates as a plain unbounded angle per axis (never wrapped through a
    // quaternion round-trip), so a continuous multi-revolution spin (e.g. a
    // backflip) keeps counting past 360 degrees instead of folding back to the
    // short way when interpolated between keyframes.
    void rotatePelvis(Skeleton* e, glm::vec3& accum, float dx) {
        if (axis >= 0) accum[axis] += dx * sens;
        e->pelvis->orient = pelvisQuat(accum);
    }

    void draw(Skeleton* e) {
        if (axis < 0) return;
        if (jointIdx == (int)e->joints.size()) { // pseudo-joint: pelvis
            vec3 anchor = e->pelvis->pos;
            vec3 d = e->pelvis->orient * dir();
            vec3 a = anchor - d * 8.0f, b = anchor + d * 8.0f;
            float verts[6] = { a.x, a.y, a.z, b.x, b.y, b.z };
            if (!VAO) {
                glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO);
                glBindVertexArray(VAO); glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(0);
            }
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
            mat4 model(1.0f);
            glUniformMatrix4fv(engine.modelLoc, 1, GL_FALSE, value_ptr(model));
            vec3 c = axis == 0 ? vec3(1.0f, 0.25f, 0.25f) : axis == 1 ? vec3(0.3f, 1.0f, 0.3f) : vec3(0.35f, 0.55f, 1.0f);
            glUniform4f(engine.colorLoc, c.r, c.g, c.b, 1.0f);
            glDisable(GL_DEPTH_TEST);
            glLineWidth(2.5f);
            glDrawArrays(GL_LINES, 0, 2);
            glEnable(GL_DEPTH_TEST);
            return;
        }
        Joint& j = e->joints[jointIdx];
        vec3 anchor = j.A->pos + j.A->orient * j.anchorA; // the joint is the origin
        vec3 d = j.A->orient * dir();
        vec3 a = anchor - d * 8.0f, b = anchor + d * 8.0f;
        float verts[6] = { a.x, a.y, a.z, b.x, b.y, b.z };
        if (!VAO) {
            glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO);
            glBindVertexArray(VAO); glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
        }
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
        mat4 model(1.0f);
        glUniformMatrix4fv(engine.modelLoc, 1, GL_FALSE, value_ptr(model));
        vec3 c = axis == 0 ? vec3(1.0f, 0.25f, 0.25f) : axis == 1 ? vec3(0.3f, 1.0f, 0.3f) : vec3(0.35f, 0.55f, 1.0f);
        glUniform4f(engine.colorLoc, c.r, c.g, c.b, 1.0f);
        glDisable(GL_DEPTH_TEST); // always visible through the body
        glLineWidth(2.5f);
        glDrawArrays(GL_LINES, 0, 2);
        glEnable(GL_DEPTH_TEST);
    }
};
AxisGizmo gizmo;

// ================= timeline UI ================= //
// 2D bar at the bottom: drag the playhead to scrub, drag keyframe dots to move
// them, SPACE = add/replace keyframe at the playhead, V/C = play/pause,
// ENTER = bake first->last keyframe to baked_motion.csv (also reloaded as
// editable keyframes on restart).
struct TimelineUI {
    std::vector<Keyframe> keys; // always kept sorted by time
    float span = 5.0f;          // hard capacity: loaded animations longer than this are squashed to fit
    float cursor = 0.0f;        // playhead time
    bool playing = false;
    bool scrubbing = false;     // left-drag on the track
    int dragKey = -1;           // index of the dot being dragged, -1 = none
    int selKey = -1;            // last-clicked dot (stays selected after release; DELETE removes it)
    float barH = 46.0f, pad = 18.0f;
    GLuint VAO = 0, VBO = 0;

    // --- view window: what the track actually displays, separate from span (the
    // hard 5s capacity). fitView() resets it to exactly [first, last] keyframe —
    // matching what saveCSV bakes — and is called whenever keyframes structurally
    // change (load/add/paste/delete). -/= (zoom()) shrink/grow it around the
    // playhead so zooming in keeps whatever moment the cursor is on under roughly
    // the same screen position; zooming persists until keyframes change again.
    float viewStart = 0.0f, viewLen = 5.0f; // starts equal to span (empty timeline, nothing to fit yet)
    float minViewLen = 0.1f; // don't let zoom-in shrink the view to something unusable

    void fitView() {
        if (keys.size() < 2) { viewStart = 0.0f; viewLen = span; return; }
        viewStart = keys.front().time;
        viewLen = glm::max(keys.back().time - keys.front().time, 0.05f);
    }
    void zoom(float factor) {
        float t = cursor; // anchor: keep this moment under the same screen position
        float frac = viewLen > 1e-6f ? (t - viewStart) / viewLen : 0.5f;
        viewLen = glm::clamp(viewLen * factor, minViewLen, span);
        viewStart = glm::clamp(t - frac * viewLen, 0.0f, glm::max(span - viewLen, 0.0f));
    }

    // --- layout (window px, y measured from the top like the cursor) ---
    float barTop()  { return engine.height - barH - 10.0f; }
    float trackY()  { return barTop() + barH / 2.0f; }
    float trackX0() { return pad * 2.0f; }
    float trackX1() { return engine.width - pad * 2.0f; }
    float xAt(float t)  { return trackX0() + (t - viewStart) / viewLen * (trackX1() - trackX0()); }
    float tAt(double x) { return viewStart + glm::clamp(float(x - trackX0()) / (trackX1() - trackX0()), 0.0f, 1.0f) * viewLen; }
    bool contains(double x, double y) { return y > barTop() - 4.0f; }

    // Catmull-Rom spline through p1->p2 (the segment being played), using p0/p3 as
    // the neighboring keyframes to set the tangent at each end. Unlike a per-segment
    // ease (which only shapes one segment in isolation), this makes velocity
    // continuous ACROSS every keyframe — no ramp-to-zero-then-snap-back-to-speed
    // discontinuity at the keyframe itself, which is what plain segment easing had.
    // At the ends of the clip (no neighbor to borrow), the missing point is
    // mirrored off the segment itself so the curve doesn't overshoot at the edges.
    static glm::vec3 catmullRom(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, float t) {
        float t2 = t * t, t3 = t2 * t;
        return 0.5f * ((2.0f * p1) +
                       (-p0 + p2) * t +
                       (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                       (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
    }

    // --- mouse ---
    // grabbing anything in the timeline always returns to the saved animation:
    // back to animate mode, and unsaved pose edits / physics wreckage get
    // overwritten by applyPose on the next frame
    void grab() { currentMode = ANIMATE; }
    void mouseDown(double x, double y) {
        // dots first, then the playhead; clicking empty track does nothing (no click-to-jump)
        for (int i = 0; i < (int)keys.size(); i++)
            if (fabs(x - xAt(keys[i].time)) < 7.0 && fabs(y - trackY()) < 12.0) { dragKey = i; selKey = i; grab(); return; }
        selKey = -1; // clicked off the dots: deselect
        if (fabs(x - xAt(cursor)) < 8.0) { scrubbing = true; grab(); }
    }
    void mouseMove(double x, double y) {
        if (dragKey >= 0) {
            float t = tAt(x);
            keys[dragKey].time = t;
            // bubble the dragged key so the vector stays sorted while it moves past neighbors
            while (dragKey > 0 && keys[dragKey - 1].time > t) { std::swap(keys[dragKey - 1], keys[dragKey]); dragKey--; }
            while (dragKey + 1 < (int)keys.size() && keys[dragKey + 1].time < t) { std::swap(keys[dragKey + 1], keys[dragKey]); dragKey++; }
            selKey = dragKey;
        } else if (scrubbing) cursor = tAt(x);
    }
    void mouseUp() { scrubbing = false; dragKey = -1; }

    // --- keyframes ---
    glm::vec3 pelvisOffset = glm::vec3(0); // edited live in ANIMATE mode, baked into new keyframes
    glm::vec3 pelvisRot    = glm::vec3(0); // unwrapped per-axis angles (see Keyframe::pelvisRot)
    void addKey(Skeleton* e) {
        for (Keyframe& k : keys)
            if (fabs(k.time - cursor) < 0.05f) { // re-key on top of an existing dot = replace it
                k.angles.clear();
                for (Joint& j : e->joints) k.angles.push_back(j.targetAngle);
                k.pelvisOffset = pelvisOffset;
                k.pelvisRot = pelvisRot;
                std::cout << "Replaced keyframe at " << k.time << "s\n";
                return;
            }
        Keyframe k; k.time = cursor;
        for (Joint& j : e->joints) k.angles.push_back(j.targetAngle);
        k.pelvisOffset = pelvisOffset;
        k.pelvisRot = pelvisRot;
        bool hadNoView = keys.size() == 1; // going from 0/1 -> 2+ keyframes: pick an initial fit-to-view
        keys.push_back(k);
        std::sort(keys.begin(), keys.end(), [](const Keyframe& a, const Keyframe& b) { return a.time < b.time; });
        selKey = -1; // indices shifted
        if (hadNoView) fitView(); // don't clobber a manual zoom on every subsequent add
        std::cout << "Keyframe at " << cursor << "s (" << keys.size() << " total)\n";
    }
    void deleteSelected() {
        if (selKey < 0 || selKey >= (int)keys.size()) return;
        std::cout << "Deleted keyframe at " << keys[selKey].time << "s (" << keys.size() - 1 << " left)\n";
        keys.erase(keys.begin() + selKey);
        selKey = -1; dragKey = -1;
    }
    // --- copy/paste: CTRL+C grabs the selected dot, CTRL+V drops it at the playhead ---
    std::vector<glm::vec3> clipboard;
    glm::vec3 clipboardPelvisOffset = glm::vec3(0);
    glm::vec3 clipboardPelvisRot    = glm::vec3(0);
    void copyKey() {
        if (selKey < 0 || selKey >= (int)keys.size()) { std::cout << "No keyframe selected to copy\n"; return; }
        clipboard = keys[selKey].angles;
        clipboardPelvisOffset = keys[selKey].pelvisOffset;
        clipboardPelvisRot = keys[selKey].pelvisRot;
        std::cout << "Copied keyframe at " << keys[selKey].time << "s\n";
    }
    void pasteKey() {
        if (clipboard.empty()) { std::cout << "Nothing copied yet\n"; return; }
        for (Keyframe& k : keys)
            if (fabs(k.time - cursor) < 0.05f) { // pasting on top of a dot = overwrite it
                k.angles = clipboard;
                k.pelvisOffset = clipboardPelvisOffset;
                k.pelvisRot = clipboardPelvisRot;
                std::cout << "Pasted over keyframe at " << k.time << "s\n";
                return;
            }
        bool hadNoView = keys.size() == 1;
        keys.push_back({ cursor, clipboard, clipboardPelvisOffset, clipboardPelvisRot });
        std::sort(keys.begin(), keys.end(), [](const Keyframe& a, const Keyframe& b) { return a.time < b.time; });
        selKey = -1; // indices shifted
        if (hadNoView) fitView();
        std::cout << "Pasted keyframe at " << cursor << "s (" << keys.size() << " total)\n";
    }
    void play() {
        if (keys.size() < 2) { std::cout << "Need 2+ keyframes to play\n"; return; }
        if (!playing && (cursor < keys.front().time || cursor >= keys.back().time)) cursor = keys.front().time;
        playing = true;
        grab(); // playing always shows the saved animation, even from physics mode
    }
    void update(float dt) {
        if (!playing || scrubbing || keys.size() < 2) return;
        cursor += dt;
        if (cursor > keys.back().time) cursor = keys.front().time; // loop first -> last
    }
    void applyPose(Skeleton* e) {
        if (keys.empty()) return;
        for (Bone* b : e->bones) { b->vel = vec3(0); b->angVel = vec3(0); } // physics tests restart clean
        // FK only poses children off the pelvis, so after a physics fall the model
        // would otherwise stay tipped over/displaced — the keyframed pelvisOffset/
        // pelvisRot are what re-place the root, layered on top of the fixed home pose.
        if (keys.size() == 1) {
            for (int j = 0; j < 13; j++) e->joints[j].targetAngle = keys[0].angles[j];
            pelvisOffset = keys[0].pelvisOffset;
            pelvisRot = keys[0].pelvisRot;
            e->pelvis->pos = vec3(e->pos.x, 21.8f, e->pos.z) + pelvisOffset;
            e->pelvis->orient = AxisGizmo::pelvisQuat(pelvisRot);
            return;
        }
        float t = glm::clamp(cursor, keys.front().time, keys.back().time);
        int k = 0;
        while (k < (int)keys.size() - 2 && t >= keys[k + 1].time) k++;
        float blend = glm::clamp((t - keys[k].time) / glm::max(keys[k + 1].time - keys[k].time, 1e-6f), 0.0f, 1.0f);
        // Catmull-Rom through the whole keyframe sequence (see catmullRom() above) —
        // velocity is continuous across every keyframe, not just smoothed within one
        // segment, so there's no ramp-down-then-snap-back-to-speed at the dots.
        int kPrev = glm::max(k - 1, 0);
        int kNext = glm::min(k + 2, (int)keys.size() - 1);
        for (int j = 0; j < 13; j++)
            e->joints[j].targetAngle = catmullRom(keys[kPrev].angles[j], keys[k].angles[j],
                                                   keys[k + 1].angles[j], keys[kNext].angles[j], blend);
        pelvisOffset = catmullRom(keys[kPrev].pelvisOffset, keys[k].pelvisOffset,
                                   keys[k + 1].pelvisOffset, keys[kNext].pelvisOffset, blend);
        // pelvisRot is unwrapped (can exceed +-pi per axis), so this spline stays
        // monotonic across a continuous multi-revolution spin instead of folding
        // back to the short way the way a quaternion-recovered axis-angle would.
        pelvisRot = catmullRom(keys[kPrev].pelvisRot, keys[k].pelvisRot,
                                keys[k + 1].pelvisRot, keys[kNext].pelvisRot, blend);
        e->pelvis->pos = vec3(e->pos.x, 21.8f, e->pos.z) + pelvisOffset;
        e->pelvis->orient = AxisGizmo::pelvisQuat(pelvisRot);
    }

    // --- csv location: the binary can be run from the repo root or src/, so look
    // in both (and one level up); remember the folder so ENTER saves back there ---
    std::string dir = "";
    std::string find(const char* name) {
        for (const char* d : {"src/", "", "../"}) { // prefer src/: that's where train.py reads it
            std::ifstream f(std::string(d) + name);
            if (f.is_open()) { dir = d; return std::string(d) + name; }
        }
        return "";
    }

    // --- startup load: read the dense 30 Hz baked_motion.csv, but only import rows
    // flagged (trailing column) as user-placed keyframes, not the interpolated
    // in-between frames saveCSV also wrote ---
    bool load(Skeleton* e = nullptr) {
        std::ifstream f(find("baked_motion.csv"));
        if (!f.is_open()) return false;
        float dt = 1.0f / 30.0f;
        std::string line;
        int row = 0;
        glm::vec3 home = e ? vec3(e->pos.x, 21.8f, e->pos.z) : vec3(0, 21.8f, 0);
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<float> cells;
            while (std::getline(ss, cell, ',')) cells.push_back(std::stof(cell));
            if (cells.size() < 136) { row++; continue; } // short row: skip
            bool isKey = cells.size() >= 137 && cells[136] > 0.5f;
            row++;
            if (!isKey) continue;

            std::vector<glm::vec3> angles;
            for (int j = 0; j < 13; j++) {
                glm::quat q(cells[j*4], cells[j*4+1], cells[j*4+2], cells[j*4+3]);
                if (q.w < 0) q = -q;
                glm::vec3 v(q.x, q.y, q.z);
                float s = glm::length(v);
                angles.push_back(s < 1e-6f ? glm::vec3(0) : (2.0f * atan2(s, q.w)) * (v / s));
            }
            // pelvis (bones[6]) world position lives at link_p columns 52 + 6*3
            glm::vec3 pelvisWorld(cells[52 + 6*3], cells[52 + 6*3 + 1], cells[52 + 6*3 + 2]);
            // pelvis unwrapped per-axis rotation: trailing columns 141-143, written
            // directly (not derived from the quaternion at 137-140) — a single baked
            // quaternion can't distinguish "5 degrees" from "365 degrees" on some axis,
            // so the unwrapped angle has to be stored as its own value to round-trip
            // correctly. Absent in old files (pre-dates pelvis rotation) -> identity.
            glm::vec3 pelvisRotVal = cells.size() >= 144
                ? glm::vec3(cells[141], cells[142], cells[143]) : glm::vec3(0);
            keys.push_back({ (row - 1) * dt, angles, pelvisWorld - home, pelvisRotVal });
        }
        if (!keys.empty()) std::cout << "Loaded " << dir << "baked_motion.csv (" << keys.size() << " keyframes)\n";
        if (keys.empty()) return false;
        std::sort(keys.begin(), keys.end(), [](const Keyframe& a, const Keyframe& b) { return a.time < b.time; });
        // fit into the fixed-capacity bar: start at 0, squash proportionally if longer
        float t0 = keys.front().time;
        for (Keyframe& k : keys) k.time -= t0;
        if (keys.back().time > span) {
            float s = span / keys.back().time;
            for (Keyframe& k : keys) k.time *= s;
        }
        cursor = 0.0f;
        fitView(); // default view: zoomed to exactly first -> last keyframe
        return true;
    }

    // --- bake: first->last keyframe at 30 Hz, same CSV format as before ---
    void saveCSV(Skeleton* e) {
        if (keys.size() < 2) { std::cout << "Need 2+ keyframes to save\n"; return; }

        std::ofstream f(dir + "baked_motion.csv");
        float dt = 1.0f / 30.0f; // must match the policy control rate: train.py advances phase once per 30 Hz step
        std::vector<glm::vec3> prevAngles = keys[0].angles;

        // user keyframe times rarely land exactly on a 30 Hz grid sample, so a fixed
        // blend-tolerance check on the dense bake loop below would only ever catch
        // the very first frame. Instead, mark whichever grid row falls nearest each
        // keyframe's time (within half a frame) as that keyframe's row.
        std::vector<int> keyRow(keys.size(), -1);
        int totalRows = (int)std::round((keys.back().time - keys.front().time) / dt) + 1;
        for (size_t ki = 0; ki < keys.size(); ki++) {
            int row = (int)std::round((keys[ki].time - keys.front().time) / dt);
            keyRow[ki] = glm::clamp(row, 0, totalRows - 1);
        }

        int rowIdx = 0;
        for (float t = keys.front().time; t <= keys.back().time; t += dt, rowIdx++) {
            int k = 0;
            while (k < (int)keys.size() - 2 && t >= keys[k + 1].time) k++;
            float blend = glm::clamp((t - keys[k].time) / glm::max(keys[k + 1].time - keys[k].time, 1e-6f), 0.0f, 1.0f);
            // Catmull-Rom through the whole sequence — see catmullRom() and the
            // matching comment in applyPose(); keeps velocity continuous across
            // every keyframe instead of a linear ramp that resets speed at each dot.
            int kPrev = glm::max(k - 1, 0);
            int kNext = glm::min(k + 2, (int)keys.size() - 1);

            glm::vec3 com(0.0f); float tMass = 0.0f;

            // 1. Interpolate Angles
            for (int j = 0; j < 13; j++)
                e->joints[j].targetAngle = catmullRom(keys[kPrev].angles[j], keys[k].angles[j],
                                                       keys[k + 1].angles[j], keys[kNext].angles[j], blend);

            // 1b. Interpolate pelvis (root) translation + rotation. pelvisRot is
            // unwrapped per-axis (never re-derived from a quaternion), so this spline
            // stays monotonic across a >180 degree spin between two keyframes instead
            // of folding back to the short way — by the time it's converted to a
            // quaternion here, dense 30 Hz sampling means every frame-to-frame delta
            // is small, so this is a per-frame absolute quaternion exactly like any
            // other DeepMimic-style baked reference clip.
            vec3 pelvisRotSample = catmullRom(keys[kPrev].pelvisRot, keys[k].pelvisRot,
                                               keys[k + 1].pelvisRot, keys[kNext].pelvisRot, blend);
            e->pelvis->orient = AxisGizmo::pelvisQuat(pelvisRotSample);
            e->pelvis->pos = vec3(e->pos.x, 21.8f, e->pos.z)
                            + catmullRom(keys[kPrev].pelvisOffset, keys[k].pelvisOffset,
                                         keys[k + 1].pelvisOffset, keys[kNext].pelvisOffset, blend);

            // 2. Apply Forward Kinematics! (No physics solver used here)
            e->updateKinematics();
            // FK interpolation between two authored keyframes can dip a foot below
            // the floor even when neither keyframe does — never intentional, so
            // clamp (not snap: only pushes up, never pulls a clear pose down) every
            // baked frame the same way the live editor does while scrubbing/playing.
            e->clampAboveGround();

            // 3. Write Quats
            for (int j = 0; j < 13; j++) {
                float len = glm::length(e->joints[j].targetAngle);
                glm::quat q = len < 1e-6f ? glm::quat(1,0,0,0) : glm::angleAxis(len, e->joints[j].targetAngle / len);
                f << q.w << "," << q.x << "," << q.y << "," << q.z << ","; // w-first: train.py's qmul/qconj expect (w,x,y,z)
            }

            // 4. Write Link Positions & Calculate CoM
            for (int l = 0; l < 14; l++) {
                f << e->bones[l]->pos.x << "," << e->bones[l]->pos.y << "," << e->bones[l]->pos.z << ",";
                com += e->bones[l]->pos * e->bones[l]->mass;
                tMass += e->bones[l]->mass;
            }

            // 5. Write Angular Velocity (Math derivation, not physics!)
            for (int j = 0; j < 13; j++) {
                glm::vec3 av = (e->joints[j].targetAngle - prevAngles[j]) / dt;
                f << av.x << "," << av.y << "," << av.z << ",";
                prevAngles[j] = e->joints[j].targetAngle;
            }

            // 6. Write CoM
            glm::vec3 fCom = com / tMass;
            f << fCom.x << "," << fCom.y << "," << fCom.z << ",";

            // 7. Write keyframe flag + pelvis world orientation (quaternion, what
            // training actually reads via animate.py's link_p/joint_q — a plain
            // per-frame absolute quaternion, dense enough that every frame-to-frame
            // delta is small, exactly like any DeepMimic-style baked reference clip)
            // + pelvis unwrapped per-axis angle. The quaternion alone can't tell a
            // reload whether you meant 5 degrees or 365 degrees on some axis (that
            // winding info doesn't exist in a single absolute orientation), so the
            // unwrapped angle is also stored directly here — purely for mma.cpp's own
            // re-load, so re-opening the file restores the keyframes you actually
            // placed. animate.py/train.py only read the first 136 columns and never
            // see any of this.
            bool isKey = std::find(keyRow.begin(), keyRow.end(), rowIdx) != keyRow.end();
            f << (isKey ? 1 : 0) << ","
              << e->pelvis->orient.w << "," << e->pelvis->orient.x << ","
              << e->pelvis->orient.y << "," << e->pelvis->orient.z << ","
              << pelvisRotSample.x << "," << pelvisRotSample.y << "," << pelvisRotSample.z << "\n";
        }
        f.close();
        std::cout << "Bake complete! Wrote " << dir << "baked_motion.csv ("
                  << keys.back().time - keys.front().time << "s)\n";
    }

    // --- drawing (pixel-space ortho overlay, drawn last each frame) ---
    void rect(float x, float y, float w, float h, vec4 c) {
        float v[18] = { x, y, 0,  x + w, y, 0,  x + w, y + h, 0,
                        x, y, 0,  x + w, y + h, 0,  x, y + h, 0 };
        glUniform4f(engine.colorLoc, c.r, c.g, c.b, c.a);
        glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    void draw() {
        if (!VAO) {
            glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO);
            glBindVertexArray(VAO); glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
        }
        mat4 id(1.0f);
        mat4 proj = ortho(0.0f, (float)engine.width, (float)engine.height, 0.0f, -1.0f, 1.0f); // y down = cursor coords
        glUniformMatrix4fv(engine.projLoc, 1, GL_FALSE, value_ptr(proj));
        glUniformMatrix4fv(engine.viewLoc, 1, GL_FALSE, value_ptr(id));
        glUniformMatrix4fv(engine.modelLoc, 1, GL_FALSE, value_ptr(id));
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        rect(pad, barTop(), engine.width - 2 * pad, barH, vec4(0.10f, 0.10f, 0.13f, 0.92f)); // bar
        rect(trackX0(), trackY() - 1, trackX1() - trackX0(), 2, vec4(0.45f, 0.45f, 0.50f, 1)); // track
        for (int s = 0; s <= (int)span; s++) // second marks
            rect(xAt((float)s) - 1, trackY() - 7, 2, 14, vec4(0.30f, 0.30f, 0.36f, 1));
        for (int i = 0; i < (int)keys.size(); i++) // keyframe dots: white, yellow while selected/dragged
            rect(xAt(keys[i].time) - 4, trackY() - 4, 8, 8,
                 (i == dragKey || i == selKey) ? vec4(1.0f, 0.85f, 0.3f, 1) : vec4(1.0f, 1.0f, 1.0f, 1));
        rect(xAt(cursor) - 1, barTop() + 3, 2, barH - 6, // playhead: green = playing, red = paused
             playing ? vec4(0.3f, 1.0f, 0.4f, 1) : vec4(1.0f, 0.3f, 0.3f, 1));

        glEnable(GL_DEPTH_TEST);
    }
};
TimelineUI tl;

// ================= mouse routing ================= //
void onMouseButton(GLFWwindow* win, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        double x, y;
        glfwGetCursorPos(win, &x, &y);
        if (action == GLFW_PRESS && tl.contains(x, y)) { tl.mouseDown(x, y); return; }
        if (action == GLFW_PRESS && gizmo.axis >= 0 && !gizmo.useArrows && currentMode == ANIMATE) {
            gizmo.dragging = true; // this drag rotates the joint, not the camera
            camera.lastX = x; camera.lastY = y;
            return;
        }
        if (action == GLFW_RELEASE) { tl.mouseUp(); gizmo.dragging = false; } // fall through so the camera clears its drag too
    }
    camera.processMouseButton(button, action, win);
}
void onCursorMove(GLFWwindow* win, double x, double y) {
    if (tl.scrubbing || tl.dragKey >= 0) {
        tl.mouseMove(x, y);
        camera.lastX = x; camera.lastY = y;
        return;
    }
    if (gizmo.dragging && !envs.empty()) {
        Skeleton* e0 = envs[0];
        if (gizmo.jointIdx == (int)e0->joints.size()) gizmo.rotatePelvis(e0, tl.pelvisRot, float(x - camera.lastX));
        else gizmo.rotate(e0->joints[gizmo.jointIdx], float(x - camera.lastX));
        camera.lastX = x; camera.lastY = y;
        return;
    }
    camera.processMouseMove(x, y);
}

// ================= UDP ================= //
const int ACTION_DIM = 3 * 13;      // 3 axis per joint
const int STATE_DIM  = 2 + 14 * 13; // phase, root_height, then 14 links x [pos3 quat4 linvel3 angvel3]
struct Data {
    int sock, sendSock;
    sockaddr_in server, python;

    Data() {
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        sendSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

        server.sin_family = AF_INET;
        server.sin_port = htons(5005);
        server.sin_addr.s_addr = INADDR_ANY;

        python.sin_family = AF_INET;
        python.sin_port = htons(5006);
        inet_pton(AF_INET, "127.0.0.1", &python.sin_addr);

        bind(sock, (sockaddr*)&server, sizeof(server));
    }

    bool receiveData() {
        vector<float> recvBuffer(envs.size() * ACTION_DIM);
        int bytesRead = recv(sock, (char*)recvBuffer.data(), recvBuffer.size() * sizeof(float), MSG_DONTWAIT);
        if (bytesRead < (int)(2 * sizeof(float))) return false;

        if (recvBuffer[0] == -100.0f) {
            return true; // python is asking for state, no step
        } else if (recvBuffer[0] == -69.0f) {
            int idx = (int)recvBuffer[1];
            // TODO: reset to the reference pose at phase recvBuffer[2] once RSI is wired up;
            // for now this just re-inits to the default standing pose.
            if (idx >= 0 && idx < (int)envs.size()) envs[idx]->init();
            return false;
        } else if (bytesRead == (int)(envs.size() * ACTION_DIM * sizeof(float))) {
            int i = 0;
            for (Skeleton* env : envs) {
                for (int j = 0; j < 13; j++) {
                    env->joints[j].targetAngle = vec3(recvBuffer[i], recvBuffer[i + 1], recvBuffer[i + 2]);
                    i += 3;
                }
            }
            float dt = 1.0f / 30.0f; // must match train.py's control rate
            for (Skeleton* env : envs)
                for (int s = 0; s < 8; s++) env->step(dt / 8.0f);
            return true; // reply with the resulting state
        }
        return false;
    }
    void sendData() {
        vector<float> stateBuffer(envs.size() * STATE_DIM);
        int i = 0;
        for (Skeleton* env : envs) {
            vec3 root = env->pelvis->pos;
            stateBuffer[i++] = 0.0f; // phase: python owns the clock and overwrites this
            stateBuffer[i++] = root.y;
            for (Bone* b : env->bones) {
                stateBuffer[i++] = b->pos.x - root.x; // xz relative to root so drift is invisible
                stateBuffer[i++] = b->pos.y;          // y absolute so height (jumps) is visible
                stateBuffer[i++] = b->pos.z - root.z;
                stateBuffer[i++] = b->orient.w;
                stateBuffer[i++] = b->orient.x;
                stateBuffer[i++] = b->orient.y;
                stateBuffer[i++] = b->orient.z;
                stateBuffer[i++] = b->vel.x;
                stateBuffer[i++] = b->vel.y;
                stateBuffer[i++] = b->vel.z;
                stateBuffer[i++] = b->angVel.x;
                stateBuffer[i++] = b->angVel.y;
                stateBuffer[i++] = b->angVel.z;
            }
        }
        sendto(sendSock, (char*)stateBuffer.data(), i * sizeof(float), 0, (sockaddr*)&python, sizeof(python));
    }
};
Data udp;


void KeyControl(GLFWwindow* w) {
    static const char* jointNames[14] = { "waist", "spine", "neck", "shoulderR", "shoulderL",
                                          "elbowR", "elbowL", "hipR", "hipL", "kneeR", "kneeL",
                                          "ankleR", "ankleL", "pelvis" };
    static int lastPrintedIdx = -1;
    static bool upPrev = false, downPrev = false, spacePrev = false, enterPrev = false, mPrev = false;
    static bool xPrev = false, yPrev = false, zPrev = false, escPrev = false, delPrev = false;
    static bool cPrev = false, vPrev = false, tPrev = false, jPrev = false;
    static bool lPrev = false, kPrev = false, fPrev = false;

    if (envs.empty()) return;
    Skeleton* e = envs[0];
    int jointCount = (int)e->joints.size() + 1; // +1 for the "pelvis" pseudo-joint slot

    // --- MODE SWITCHING (M): animate <-> physics ---
    bool mNow = glfwGetKey(w, GLFW_KEY_M) == GLFW_PRESS;
    if (mNow && !mPrev) {
        currentMode = currentMode == ANIMATE ? PHYSICS : ANIMATE;
        std::cout << (currentMode == PHYSICS ? "[MODE] PHYSICS (ragdoll active)\n"
                                             : "[MODE] ANIMATE (pose the character)\n");
    }
    mPrev = mNow;

    // --- PLAYBACK (P = play, O = pause) ---
    if (glfwGetKey(w, GLFW_KEY_P) == GLFW_PRESS) tl.play();
    if (glfwGetKey(w, GLFW_KEY_O) == GLFW_PRESS) tl.playing = false;

    // --- TIMELINE ZOOM (- = zoom out, = = zoom in), anchored on the playhead ---
    float zoomRate = 1.03f; // per-frame multiplicative step while held, ~60fps
    if (glfwGetKey(w, GLFW_KEY_MINUS) == GLFW_PRESS) tl.zoom(zoomRate);
    if (glfwGetKey(w, GLFW_KEY_EQUAL) == GLFW_PRESS) tl.zoom(1.0f / zoomRate);

    if (currentMode != ANIMATE) return; // only allow editing in animate mode

    // --- JOINT SELECTION (UP/DOWN cycles) ---
    bool upNow = glfwGetKey(w, GLFW_KEY_UP) == GLFW_PRESS;
    bool downNow = glfwGetKey(w, GLFW_KEY_DOWN) == GLFW_PRESS;
    if (upNow && !upPrev) gizmo.jointIdx = (gizmo.jointIdx - 1 + jointCount) % jointCount;
    if (downNow && !downPrev) gizmo.jointIdx = (gizmo.jointIdx + 1) % jointCount;
    upPrev = upNow; downPrev = downNow;

    if (gizmo.jointIdx != lastPrintedIdx) {
        std::cout << "Selected joint: " << gizmo.jointIdx << " (" << jointNames[gizmo.jointIdx] << ")\n";
        lastPrintedIdx = gizmo.jointIdx;
    }

    // --- INPUT MODE (T): mouse click-drag <-> arrow keys ---
    bool tNow = glfwGetKey(w, GLFW_KEY_T) == GLFW_PRESS;
    if (tNow && !tPrev) {
        gizmo.useArrows = !gizmo.useArrows;
        gizmo.dragging = false;
        std::cout << (gizmo.useArrows ? "[INPUT] ARROWS (LEFT/RIGHT rotate)\n"
                                      : "[INPUT] MOUSE (click-drag to rotate)\n");
    }
    tPrev = tNow;

    // --- AXIS SELECT (X/Y/Z toggle, ESC releases) ---
    auto axisKey = [&](int key, bool& prev, int axis) {
        bool now = glfwGetKey(w, key) == GLFW_PRESS;
        if (now && !prev) {
            gizmo.toggle(axis);
            if (gizmo.axis < 0) std::cout << "Axis released\n";
            else std::cout << "Axis: " << "XYZ"[gizmo.axis]
                           << (gizmo.useArrows ? " (LEFT/RIGHT to rotate, ESC to release)\n"
                                               : " (click-drag to rotate, ESC to release)\n");
        }
        prev = now;
    };
    axisKey(GLFW_KEY_X, xPrev, 0);
    axisKey(GLFW_KEY_Y, yPrev, 1);
    axisKey(GLFW_KEY_Z, zPrev, 2);

    bool escNow = glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    if (escNow && !escPrev && gizmo.axis >= 0) { gizmo.release(); std::cout << "Axis released\n"; }
    escPrev = escNow;

    // --- ARROW-KEY ROTATION (LEFT/RIGHT while an axis is picked) ---
    if (gizmo.axis >= 0 && gizmo.useArrows) {
        bool isPelvis = gizmo.jointIdx == (int)e->joints.size();
        if (glfwGetKey(w, GLFW_KEY_LEFT)  == GLFW_PRESS) {
            if (isPelvis) gizmo.rotatePelvis(e, tl.pelvisRot, -3.0f);
            else gizmo.rotate(e->joints[gizmo.jointIdx], -3.0f);
        }
        if (glfwGetKey(w, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            if (isPelvis) gizmo.rotatePelvis(e, tl.pelvisRot, 3.0f);
            else gizmo.rotate(e->joints[gizmo.jointIdx], 3.0f);
        }
    }

    // --- PELVIS TRANSLATION (WASD = x/z, Q/E = y): moves the whole body's root ---
    // writes straight to e->pelvis->pos so it's visible immediately, and mirrors
    // into tl.pelvisOffset so SPACE (addKey) captures it into the keyframe
    {
        float speed = 8.0f * (1.0f / 60.0f); // units/sec at 60fps key-repeat polling
        vec3 d(0.0f);
        if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) d.x -= speed;
        if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) d.x += speed;
        if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) d.z -= speed;
        if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) d.z += speed;
        if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) d.y -= speed;
        if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) d.y += speed;
        if (d != vec3(0.0f)) {
            tl.pelvisOffset += d;
            e->pelvis->pos += d;
        }
    }

    // --- SNAP TO GROUND (F): one-shot, only when pressed (not held every frame) ---
    bool fNow = glfwGetKey(w, GLFW_KEY_F) == GLFW_PRESS;
    if (fNow && !fPrev) {
        float lowest = glm::min(e->footR->pos.y - e->footR->dims.y, e->footL->pos.y - e->footL->dims.y);
        tl.pelvisOffset.y -= lowest;
        e->snapToGround();
        std::cout << "[GROUND] snapped to floor (shifted " << -lowest << " in y)\n";
    }
    fPrev = fNow;

    // --- COPY/PASTE KEYFRAME (ctrl+c / ctrl+v) ---
    bool ctrl = glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    bool cNow = ctrl && glfwGetKey(w, GLFW_KEY_C) == GLFW_PRESS;
    bool vNow = ctrl && glfwGetKey(w, GLFW_KEY_V) == GLFW_PRESS;
    if (cNow && !cPrev) tl.copyKey();
    if (vNow && !vPrev) tl.pasteKey();
    cPrev = cNow; vPrev = vNow;

    // --- DELETE SELECTED KEYFRAME (delete/backspace) ---
    bool delNow = glfwGetKey(w, GLFW_KEY_DELETE) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_BACKSPACE) == GLFW_PRESS;
    if (delNow && !delPrev) tl.deleteSelected();
    delPrev = delNow;

    // --- ADD KEYFRAME (space, at the playhead) ---
    bool spaceNow = glfwGetKey(w, GLFW_KEY_SPACE) == GLFW_PRESS;
    if (spaceNow && !spacePrev) tl.addKey(e);
    spacePrev = spaceNow;

    // --- BAKE TO CSV (enter) ---
    bool enterNow = glfwGetKey(w, GLFW_KEY_ENTER) == GLFW_PRESS;
    if (enterNow && !enterPrev) tl.saveCSV(e);
    enterPrev = enterNow;

    // --- LOG BALANCE: CoM vs feet, current pose (L) ---
    bool lNow = glfwGetKey(w, GLFW_KEY_L) == GLFW_PRESS;
    if (lNow && !lPrev) {
        glm::vec3 com(0.0f); float tMass = 0.0f;
        for (Bone* b : e->bones) { com += b->pos * b->mass; tMass += b->mass; }
        com /= tMass;
        glm::vec3 feet = (e->bones[0]->pos + e->bones[1]->pos) * 0.5f;
        float dx = com.x - feet.x, dz = com.z - feet.z;
        float dist = std::sqrt(dx * dx + dz * dz);
        std::cout << "[BALANCE] CoM-feet offset (x,z): " << dx << ", " << dz
                  << " | dist: " << dist << " (lower is better, 0 = centered)\n";
    }
    lPrev = lNow;

    // --- DUMP STANCE ARRAYS (K): paste straight into mma.cpp's Skeleton::init() ---
    bool kNow = glfwGetKey(w, GLFW_KEY_K) == GLFW_PRESS;
    if (kNow && !kPrev) {
        static const char* boneNames[14] = { "footR", "footL", "calfR", "calfL", "thighR", "thighL",
                                              "pelvis", "abs", "chest", "armR", "armL", "forearmR", "forearmL", "head" };
        std::cout.precision(6);
        std::cout << std::fixed;
        std::cout << "\nconst float STANCE_POS[14][3] = {\n";
        for (int i = 0; i < 14; i++)
            std::cout << "    {" << e->bones[i]->pos.x << "f, " << e->bones[i]->pos.y << "f, " << e->bones[i]->pos.z
                       << "f},   // " << boneNames[i] << "\n";
        std::cout << "};\n";

        std::cout << "const float STANCE_QUAT[14][4] = { // w x y z, same bone order\n";
        for (int i = 0; i < 14; i++) {
            glm::quat q = e->bones[i]->orient;
            std::cout << "    {" << q.w << "f, " << q.x << "f, " << q.y << "f, " << q.z << "f},\n";
        }
        std::cout << "};\n";

        std::cout << "const float STANCE_TARGET[13][3] = { // PD targets (axis-angle), joint order = Skeleton::joints\n";
        for (size_t j = 0; j < e->joints.size(); j++) {
            glm::vec3 a = e->joints[j].targetAngle;
            std::cout << "    {" << a.x << "f, " << a.y << "f, " << a.z << "f},\n";
        }
        std::cout << "};\n\n";
        std::cout << "[STANCE DUMP] copy the three arrays above into mma.cpp's Skeleton struct\n";
    }
    kPrev = kNow;

    // --- PRINT SELECTED JOINT'S TARGET ANGLE (J) ---
    bool jNow = glfwGetKey(w, GLFW_KEY_J) == GLFW_PRESS;
    if (jNow && !jPrev) {
        glm::vec3 a = gizmo.jointIdx == (int)e->joints.size() ? tl.pelvisRot : e->joints[gizmo.jointIdx].targetAngle;
        std::cout << "[JOINT] " << gizmo.jointIdx << " (" << jointNames[gizmo.jointIdx]
                   << ") target angle (axis-angle): " << a.x << ", " << a.y << ", " << a.z << "\n";
    }
    jPrev = jNow;
}


// ================= main loop ================= //
int main() {
    std::cout <<
        "Controls:\n"
        "  M          animate <-> physics\n"
        "  UP/DOWN    select previous/next joint (cycles into \"pelvis\" at the end)\n"
        "  X/Y/Z      pick rotation axis (press again or ESC to release)\n"
        "  T          toggle rotation input: mouse click-drag <-> LEFT/RIGHT keys\n"
        "  WASD/Q/E   translate the pelvis (root) in x/z and y\n"
        "  F          snap the body to the ground (lowest foot -> floor)\n"
        "  J          print selected joint's target angle\n"
        "  SPACE      add/replace keyframe at the playhead\n"
        "  DELETE     delete the selected (yellow) keyframe\n"
        "  CTRL+C/V   copy selected keyframe / paste at the playhead\n"
        "  P / O      play / pause\n"
        "  - / =      zoom timeline out / in (around the playhead)\n"
        "  ENTER      bake baked_motion.csv (first -> last keyframe)\n"
        "  L          log CoM-vs-feet balance offset for the current pose\n"
        "  timeline   drag the playhead to scrub, drag dots to move keyframes\n";

    if (!envs.empty() && tl.load(envs[0])) // resume the previous animation if one was saved
        for (Skeleton* s : envs) { tl.applyPose(s); s->updateKinematics(); }

    float dt = 1.0f / 60.0f;
    while (!glfwWindowShouldClose(engine.window)) {
        if (udp.receiveData()) udp.sendData();

        engine.beginFrame();
        KeyControl(engine.window);
        grid.draw();

        tl.update(dt);
        for (Skeleton* skeleton : envs) {
            if (currentMode == PHYSICS) {
                for (int s = 0; s < 8; s++) skeleton->step(dt / 8.0f); // joint solver and torque
            } else { // ANIMATE: kinematic; the playhead drives the pose while playing/scrubbing/retiming
                bool interpolating = tl.playing || tl.scrubbing || tl.dragKey >= 0;
                if (interpolating) tl.applyPose(skeleton);
                skeleton->updateKinematics();
                // only the interpolated in-between frames get floor-clamped — a
                // pose you're deliberately editing at an exact keyframe (F to snap
                // manually) is left alone, since you're in full control there
                if (interpolating) skeleton->clampAboveGround();
            }
            skeleton->draw();
        }
        if (currentMode == ANIMATE && !envs.empty()) gizmo.draw(envs[0]);
        tl.draw(); // last: 2D overlay swaps the projection matrices

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
