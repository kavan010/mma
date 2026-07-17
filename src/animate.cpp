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
struct Keyframe { float time; std::vector<glm::vec3> angles; };
std::vector<Keyframe> timeline;
float timeOff = 0.0f;

// modes for making animation and testing
enum AppMode { PHYSICS, ANIMATE, PLAYBACK };
AppMode currentMode = ANIMATE; 


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
        glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int) {
            ((Camera*)glfwGetWindowUserPointer(win))->processMouseButton(button, action, win);
        });
        glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
            ((Camera*)glfwGetWindowUserPointer(win))->processMouseMove(x, y);
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
        chest    = new Bone(vec3(pos.x, 29.6f, pos.z),                    vec3(0.0f, 3.0f, 0.0f),  8.0f, 2);
        head     = new Bone(vec3(pos.x, 34.6f, pos.z),                    vec3(0.0f, 2.25f, 0.0f),  3.0f, 2);

        thighR   = new Bone(vec3(0, (hipY + kneeY) / 2, hipZ),    vec3(2.8f, 1.5f, 0.0f),  4.5f, 2);
        thighL   = new Bone(vec3(0, (hipY + kneeY) / 2, -hipZ),   vec3(2.8f, 1.5f, 0.0f),  4.5f, 2);
        calfR    = new Bone(vec3(0, (kneeY + ankleY) / 2, hipZ),  vec3(2.7f, 1.3f, 0.0f),  2.5f, 2);
        calfL    = new Bone(vec3(0, (kneeY + ankleY) / 2, -hipZ), vec3(2.7f, 1.3f, 0.0f),  2.5f, 2);
        footR    = new Bone(vec3(1.5f, 0.6f, hipZ),               vec3(2.7f, 0.6f, 1.0f),  1.0f, 1); // box half-extents
        footL    = new Bone(vec3(1.5f, 0.6f, -hipZ),              vec3(2.7f, 0.6f, 1.0f),  1.0f, 1);

        armR     = new Bone(vec3(0, shoulderY - 3.0f, shoulderZ), vec3(1.9f, 1.1f, 0.0f),  1.8f, 2);
        armL     = new Bone(vec3(0, shoulderY - 3.0f, -shoulderZ),vec3(1.9f, 1.1f, 0.0f),  1.8f, 2);
        forearmR = new Bone(vec3(0, shoulderY - 8.5f, shoulderZ), vec3(1.5f, 1.0f, 0.0f),  1.2f, 2);
        forearmL = new Bone(vec3(0, shoulderY - 8.5f, -shoulderZ),vec3(1.5f, 1.0f, 0.0f),  1.2f, 2);

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

        // FK only propagates outward from the pelvis, so it never moves the pelvis
        // itself — a crouch left the feet lifting instead of the hips dropping.
        // Shift the whole body so the lower foot rests on the floor, every frame.
        float lowest = glm::min(footR->pos.y - footR->dims.y, footL->pos.y - footL->dims.y);
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



bool loadBakedCSV(const char* path = "baked_motion.csv") {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    timeline.clear();
    float dt = 1.0f / 30.0f;
    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string cell;
        Keyframe k;
        k.time = timeline.size() * dt;
        for (int j = 0; j < 13; j++) {
            float qv[4];
            for (int c = 0; c < 4; c++) {
                if (!std::getline(ss, cell, ',')) return timeline.size() > 1;
                qv[c] = std::stof(cell);
            }
            glm::quat q(qv[0], qv[1], qv[2], qv[3]);
            if (q.w < 0) q = -q;
            glm::vec3 v(q.x, q.y, q.z);
            float s = glm::length(v);
            k.angles.push_back(s < 1e-6f ? glm::vec3(0) : (2.0f * atan2(s, q.w)) * (v / s));
        }
        timeline.push_back(k);
    }
    if (!timeline.empty()) timeOff = timeline.back().time + 0.2f;
    return timeline.size() > 1;
}
void KeyControl(GLFWwindow* w) {
    static int jointIdx = 0;
    static int lastPrintedIdx = -1;
    static bool upPrev = false, downPrev = false, spacePrev = false, enterPrev = false, mPrev = false;
    float delta = 0.001f;

    if (envs.empty()) return;
    Skeleton* e = envs[0];
    int jointCount = e->joints.size();

    // --- MODE SWITCHING (Press 'M') ---
    bool mNow = glfwGetKey(w, GLFW_KEY_M) == GLFW_PRESS;
    if (mNow && !mPrev) {
        currentMode = (AppMode)((currentMode + 1) % 3);
        if (currentMode == PHYSICS) std::cout << "[MODE] PHYSICS (Ragdoll active)\n";
        if (currentMode == ANIMATE) std::cout << "[MODE] ANIMATE (Pose the character)\n";
        if (currentMode == PLAYBACK) {
            std::cout << "[MODE] PLAYBACK (Watch animation)\n";
            if (timeline.size() < 2) { // nothing posed this session — fall back to the baked file
                if (loadBakedCSV()) std::cout << "Loaded baked_motion.csv (" << timeline.size() << " frames)\n";
                else std::cout << "No keyframes and no baked_motion.csv to load\n";
            }
        }
    }
    mPrev = mNow;

    if (currentMode != ANIMATE) return; // Only allow editing in Animate mode

    // --- 1. JOINT SELECTION ---
    bool upNow = glfwGetKey(w, GLFW_KEY_UP) == GLFW_PRESS;
    bool downNow = glfwGetKey(w, GLFW_KEY_DOWN) == GLFW_PRESS;
    if (upNow && !upPrev) jointIdx = (jointIdx + 1) % jointCount;
    if (downNow && !downPrev) jointIdx = (jointIdx - 1 + jointCount) % jointCount;
    upPrev = upNow; downPrev = downNow;

    if (jointIdx != lastPrintedIdx) {
        std::cout << "Selected joint: " << jointIdx << "\n";
        lastPrintedIdx = jointIdx;
    }

    // --- 2. ROTATE JOINT ---
    auto& a = e->joints[jointIdx].targetAngle;
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) a.x += delta;
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) a.x -= delta;
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) a.y += delta;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) a.y -= delta;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) a.z += delta;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) a.z -= delta;

    // --- 3. SAVE KEYFRAME (Spacebar) ---
    bool spaceNow = glfwGetKey(w, GLFW_KEY_SPACE) == GLFW_PRESS;
    if (spaceNow && !spacePrev) {
        Keyframe k; k.time = timeOff; timeOff += 0.2f; 
        for (int i = 0; i < jointCount; i++) k.angles.push_back(e->joints[i].targetAngle);
        timeline.push_back(k);
        std::cout << "Saved pose at " << k.time << "s. Total keyframes: " << timeline.size() << "\n";
    }
    spacePrev = spaceNow;

    // --- 4. BAKE TO CSV (Enter) ---
    bool enterNow = glfwGetKey(w, GLFW_KEY_ENTER) == GLFW_PRESS;
    if (enterNow && !enterPrev && timeline.size() > 1) {
        std::ofstream f("baked_motion.csv");
        float dt = 1.0f / 30.0f; // must match the policy control rate: train.py advances phase once per 30 Hz step
        std::vector<glm::vec3> prevAngles = timeline[0].angles;

        for (float t = 0; t <= timeline.back().time; t += dt) {
            int k = 0;
            while (k < timeline.size()-2 && t >= timeline[k+1].time) k++;
            float blend = (t - timeline[k].time) / (timeline[k+1].time - timeline[k].time);

            glm::vec3 com(0.0f); float tMass = 0.0f;

            // 1. Interpolate Angles
            for (int j = 0; j < 13; j++) {
                e->joints[j].targetAngle = glm::mix(timeline[k].angles[j], timeline[k+1].angles[j], blend);
            }
            
            // 2. Apply Forward Kinematics! (No physics solver used here)
            e->updateKinematics(); 

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
            f << fCom.x << "," << fCom.y << "," << fCom.z << "\n";
        }
        f.close();
        std::cout << "Bake complete! Wrote baked_motion.csv\n";
    }
    enterPrev = enterNow;
}


// ================= main loop ================= //
int main() {
    float dt = 1.0f / 60.0f;
    while (!glfwWindowShouldClose(engine.window)) {
        if (udp.receiveData()) udp.sendData();

        engine.beginFrame();
        KeyControl(engine.window);
        grid.draw();

        for (Skeleton* skeleton : envs) {
            skeleton->draw();
            if (currentMode == PHYSICS) {
                for (int s = 0; s < 8; s++) skeleton->step(dt / 8.0f); // joint solver and torque
            
            } else if (currentMode == ANIMATE) {
                skeleton->updateKinematics(); // no physics 
            
            } else if (currentMode == PLAYBACK && timeline.size() > 1) {    // play keyframes
                float t = fmod(glfwGetTime(), timeline.back().time);
                int k = 0;
                while (k < timeline.size()-2 && t >= timeline[k+1].time) k++;
                float blend = (t - timeline[k].time) / (timeline[k+1].time - timeline[k].time);
                
                for (int j = 0; j < 13; j++) {
                    skeleton->joints[j].targetAngle = glm::mix(timeline[k].angles[j], timeline[k+1].angles[j], blend);
                }
                skeleton->updateKinematics();
            }
        }

        

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
