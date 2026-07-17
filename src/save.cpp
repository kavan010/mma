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
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;

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




// ================= main loop ================= //
int main() {
    float dt = 1.0f / 60.0f;
    while (!glfwWindowShouldClose(engine.window)) {
        engine.beginFrame();
        grid.draw();

        for (Skeleton* skeleton : envs) {
            for (int s = 0; s < 8; s++) skeleton->step(dt / 8.0f); // substep for stability
            skeleton->draw();
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
