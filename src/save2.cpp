#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <list>
using namespace glm;
using namespace std;

// --- constants ----
const vec2 g(0.0f, -9.81f);

struct Engine {
    GLFWwindow* window;
    int WIDTH = 800, HEIGHT = 600;

    Engine () {
        if (!glfwInit()) {
            cerr << "Failed to initialize GLFW" << endl;
            exit(EXIT_FAILURE);
        }

        window = glfwCreateWindow(WIDTH, HEIGHT, "2D atom sim by kavan", nullptr, nullptr);
        if (!window) {
            cerr << "Failed to create GLFW window" << endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);
    }

    void run() {
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        double halfWidth = WIDTH / 2.0, halfHeight = HEIGHT / 2.0;
        glOrtho(-halfWidth, halfWidth, -halfHeight, halfHeight, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

    // ------ Callbacks ------
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
    }
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            
        }
    }
};
Engine engine;

// --- structs ---

struct Joint {
    vec2 pos;

    Joint(vec2 p) : pos(p) {}

    void draw(float radius) {
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(pos.x, pos.y);
        for (int i = 0; i <= 20; i++) {
            float a = (i / 20.0f) * 2.0f * glm::pi<float>();
            glVertex2f(pos.x + cos(a) * radius, pos.y + sin(a) * radius);
        }
        glEnd();
    }

};
struct Bone {
    Joint *j1, *j2;
    float length, mass;

    Bone(Joint* p1, Joint* p2, float len, float m) : j1(p1), j2(p2), length(len), mass(m) {}

    void solve() {
        if (!j1 || !j2) return;
        vec2 diff = j1->pos - j2->pos;
        float d = glm::length(diff);
        if (d == 0) return;
        float diff_factor = (length - d) / d;
        vec2 offset = diff * 0.5f * diff_factor;

        j1->pos += offset;
        j2->pos -= offset;
    }

void draw(float radius) {
    if (!j1 || !j2) return;

    vec2 p1 = j1->pos;
    vec2 p2 = j2->pos;

    vec2 dir = p2 - p1;
    float len = glm::length(dir);
    if (len == 0) return;

    vec2 dirN = dir / len;
    vec2 right(-dirN.y, dirN.x);

    // circle centers
    vec2 c1 = p1 + dirN * radius;
    vec2 c2 = p2 - dirN * radius;

    vec2 rOffset = right * radius;

    glColor4f(1,1,1,1);

    // rectangle body (touches circle centers)
    glBegin(GL_QUADS);
        glVertex2f((c2 + rOffset).x, (c2 + rOffset).y);
        glVertex2f((c2 - rOffset).x, (c2 - rOffset).y);
        glVertex2f((c1 - rOffset).x, (c1 - rOffset).y);
        glVertex2f((c1 + rOffset).x, (c1 + rOffset).y);
    glEnd();

    // circle 1
    glBegin(GL_TRIANGLE_FAN);
        glVertex2f(c1.x, c1.y);
        for (int i = 0; i <= 20; i++) {
            float a = (i / 20.0f) * 2.0f * glm::pi<float>();
            glVertex2f(c1.x + cos(a)*radius, c1.y + sin(a)*radius);
        }
    glEnd();

    // circle 2
    glBegin(GL_TRIANGLE_FAN);
        glVertex2f(c2.x, c2.y);
        for (int i = 0; i <= 20; i++) {
            float a = (i / 20.0f) * 2.0f * glm::pi<float>();
            glVertex2f(c2.x + cos(a)*radius, c2.y + sin(a)*radius);
        }
    glEnd();
}
};


list<Bone> bones;


int main () {
    glfwSetCursorPosCallback(engine.window, Engine::cursor_position_callback);
    glfwSetMouseButtonCallback(engine.window, Engine::mouse_button_callback);

    Joint shoulder = Joint(vec2(0, 150));
    Joint elbow = Joint(vec2(0, 0));
    Joint wrist = Joint(vec2(0, -150));
    bones.push_back(Bone(&shoulder, &elbow, 50.0f, 1.0f));
    bones.push_back(Bone(&elbow, &wrist,    50.0f, 1.0f));

    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();


        // ---- draw bones ----
        for (auto& b : bones) {
            b.draw(7.0f);
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
}
 