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
    double zoomSpeed = 110.0;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;

    vec3 position() const {
        float e = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        return vec3(radius * sin(e) * cos(azimuth), radius * cos(e), radius * sin(e) * sin(azimuth));
    }
    void processMouseMove(double x, double y) {
        if (dragging) {
            azimuth += float(x - lastX) * orbitSpeed;
            elevation = glm::clamp(elevation - float(y - lastY) * orbitSpeed, 0.01f, float(M_PI) - 0.01f);
        }
        lastX = x; lastY = y;
    }
    void processMouseButton(int button, int action, GLFWwindow* win) {
        if (button != GLFW_MOUSE_BUTTON_LEFT && button != GLFW_MOUSE_BUTTON_MIDDLE) return;
        if (action == GLFW_PRESS) { dragging = true; glfwGetCursorPos(win, &lastX, &lastY); }
        else if (action == GLFW_RELEASE) dragging = false;
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
    GLuint sphereVAO, sphereVBO;
    int sphereVertexCount;

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
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        shaderProgram = compileShader();
        modelLoc = glGetUniformLocation(shaderProgram, "model");
        viewLoc  = glGetUniformLocation(shaderProgram, "view");
        projLoc  = glGetUniformLocation(shaderProgram, "projection");
        colorLoc = glGetUniformLocation(shaderProgram, "objectColor");

        buildSphere();
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

    void buildSphere() {
        vector<float> vertices;
        float r = 1.0f;
        int stacks = 12, sectors = 16;
        for (int i = 0; i < stacks; ++i) {
            float t1 = (float)i / stacks * M_PI, t2 = (float)(i + 1) / stacks * M_PI;
            for (int j = 0; j < sectors; ++j) {
                float p1 = (float)j / sectors * 2 * M_PI, p2 = (float)(j + 1) / sectors * 2 * M_PI;
                auto pos = [&](float t, float p) { return vec3(r * sin(t) * cos(p), r * cos(t), r * sin(t) * sin(p)); };
                vec3 v1 = pos(t1, p1), v2 = pos(t1, p2), v3 = pos(t2, p1), v4 = pos(t2, p2);
                vertices.insert(vertices.end(), {v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z});
                vertices.insert(vertices.end(), {v2.x, v2.y, v2.z, v4.x, v4.y, v4.z, v3.x, v3.y, v3.z});
            }
        }
        sphereVertexCount = vertices.size() / 3;
        createVAO(sphereVAO, sphereVBO, vertices.data(), vertices.size());
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
        mat4 view = lookAt(camera.position(), vec3(0.0f), vec3(0, 1, 0));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, value_ptr(proj));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, value_ptr(view));
    }

    void drawSphere(vec3 pos, float radius, vec4 color) {
        mat4 model = translate(mat4(1.0f), pos) * scale(mat4(1.0f), vec3(radius));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, value_ptr(model));
        glUniform4fv(colorLoc, 1, value_ptr(color));
        glBindVertexArray(sphereVAO);
        glDrawArrays(GL_TRIANGLES, 0, sphereVertexCount);
    }
};
Engine engine;

// ================= grid ================= //
struct Grid {
    GLuint VAO, VBO;
    int vertexCount;

    Grid(float size = 500.0f, int divisions = 50) {
        vector<float> vertices;
        float step = size / divisions, half = size / 2.0f;
        for (int i = 0; i <= divisions; ++i) {
            float x = -half + i * step, z = -half + i * step;
            vertices.insert(vertices.end(), {x, 0, -half, x, 0, half});
            vertices.insert(vertices.end(), {-half, 0, z, half, 0, z});
        }
        vertexCount = vertices.size() / 3;
        engine.createVAO(VAO, VBO, vertices.data(), vertices.size());
    }

    void draw() {
        mat4 model = mat4(1.0f);
        glUniformMatrix4fv(engine.modelLoc, 1, GL_FALSE, value_ptr(model));
        glUniform4f(engine.colorLoc, 1.0f, 1.0f, 1.0f, 0.15f);
        glBindVertexArray(VAO);
        glDrawArrays(GL_LINES, 0, vertexCount);
    }
};
Grid grid;


struct Bone {
    vec3 pos = vec3(0), vel = vec3(0), angVel = vec3(0);
    quat orient = quat(1, 0, 0, 0);
    vec3 dims;
    float mass, invMass;
    mat3 invInertiaBody = mat3(1.0f);
    bool isEndEffector = false;
    vec3 color;
    GLuint VAO = 0, VBO = 0;
    int vertexCount = 0;

    Bone(vec3 p, vec3 d, float m, vec3 c = vec3(0.85f), bool endEffector = false)
        : pos(p), dims(d), mass(m), color(c), isEndEffector(endEffector) { invMass = 1.0f / m; }

    vec3 posRelativeTo(const Bone& root) const { return inverse(root.orient) * (pos - root.pos); }
    quat orientRelativeTo(const Bone& root) const { return inverse(root.orient) * orient; }

    void setInertia(vec3 I) {
        invInertiaBody = mat3(0.0f);
        invInertiaBody[0][0] = 1.0f / I.x; invInertiaBody[1][1] = 1.0f / I.y; invInertiaBody[2][2] = 1.0f / I.z;
    }

    void drawBox() {
        if (VAO == 0) {
            float x = 2 * dims.x, y = 2 * dims.y, z = 2 * dims.z;
            setInertia(mass * vec3(y * y + z * z, x * x + z * z, x * x + y * y) / 12.0f);

            vector<float> v;
            vec3 he = dims, n[6] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
            for (vec3 a : n) {
                vec3 u = vec3(a.y, a.z, a.x), w = cross(a, u), c = a * he;
                vec3 p1 = c + (u + w) * he, p2 = c + (u - w) * he, p3 = c - (u + w) * he, p4 = c - (u - w) * he;
                for (vec3 p : {p1, p2, p3, p1, p3, p4}) v.insert(v.end(), {p.x, p.y, p.z});
            }
            vertexCount = v.size() / 3;
            engine.createVAO(VAO, VBO, v.data(), v.size());
        }
        draw();
    }

    void drawCapsule() {
        if (VAO == 0) {
            float r = dims.x, h = dims.y, L = 2 * h;
            setInertia(vec3((mass / 12.0f) * (3 * r * r + L * L), 0.5f * mass * r * r, (mass / 12.0f) * (3 * r * r + L * L)));

            vector<float> v;
            int stacks = 8, sectors = 16;
            auto push = [&](vec3 p) { v.insert(v.end(), {p.x, p.y, p.z}); };
            for (int j = 0; j < sectors; j++) {
                float p1 = (float)j / sectors * 2 * M_PI, p2 = (float)(j + 1) / sectors * 2 * M_PI;
                vec3 a1(cos(p1) * r, h, sin(p1) * r), a2(cos(p2) * r, h, sin(p2) * r);
                vec3 b1(cos(p1) * r, -h, sin(p1) * r), b2(cos(p2) * r, -h, sin(p2) * r);
                push(a1); push(b1); push(a2); push(a2); push(b1); push(b2);
            }
            for (int cap = 0; cap < 2; cap++) {
                float dir = cap == 0 ? 1.0f : -1.0f, yoff = cap == 0 ? h : -h;
                for (int i = 0; i < stacks / 2; i++) {
                    float t1 = (float)i / stacks * M_PI, t2 = (float)(i + 1) / stacks * M_PI;
                    for (int j = 0; j < sectors; j++) {
                        float p1 = (float)j / sectors * 2 * M_PI, p2 = (float)(j + 1) / sectors * 2 * M_PI;
                        auto sp = [&](float t, float p) { return vec3(r * sin(t) * cos(p), yoff + dir * r * cos(t), r * sin(t) * sin(p)); };
                        vec3 v1 = sp(t1, p1), v2 = sp(t1, p2), v3 = sp(t2, p1), v4 = sp(t2, p2);
                        push(v1); push(v2); push(v3); push(v2); push(v4); push(v3);
                    }
                }
            }
            vertexCount = v.size() / 3;
            engine.createVAO(VAO, VBO, v.data(), v.size());
        }
        draw();
    }

    void draw() {
        mat4 model = translate(mat4(1.0f), pos) * mat4_cast(orient);
        glUniformMatrix4fv(engine.modelLoc, 1, GL_FALSE, value_ptr(model));
        glUniform4f(engine.colorLoc, color.r, color.g, color.b, 1.0f);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);
    }
};
struct Joint {

};

struct Skeleton {
    vector<Bone*> bones;
    vector<Joint> joints;
    vec3 startPos;
    int idx;

    Skeleton(vec3 p, int idx) : startPos(p), idx(idx) {}

    void step(float dt) {
        for (Bone* b : bones) {
            // Update bone physics here
        }
        for (Joint& j : joints) {
            // Solve joint constraints here
        }
    }
    void draw() {
        for (Bone* b : bones) {
            // Draw each bone here
        }
    }
};
vector<Skeleton*> envs {
    new Skeleton(vec3(0), 0)
};

Bone testBox(vec3(-2, 1, 0), vec3(5.5f, 2.5f, 7.5f), 1.0f, vec3(0.3f, 0.6f, 0.9f));
Bone testCapsule(vec3(0, 1, 0), vec3(2.3f, 5.5f, 0), 1.0f, vec3(0.9f, 0.4f, 0.3f));
Bone testSphere(vec3(2, 1, 0), vec3(4.4f, 0, 0), 1.0f, vec3(0.4f, 0.9f, 0.4f));

// ================= main loop ================= //
int main() {
    while (!glfwWindowShouldClose(engine.window)) {
        engine.beginFrame();
        grid.draw();
        testBox.drawBox();
        testCapsule.drawCapsule();
        testSphere.drawCapsule();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
