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
struct Bone {
    vec2 pos;           // center of mass in world space
    float angle;        // orientation (radians)
    
    float halfLength;   // half the bone length
    float radius;    // thickness / radius for collision/drawing

    Bone (vec2 p, float a, float l, float r) : pos(p), angle(a), halfLength(l), radius(r) {}

    // Compute the world-space point at local coordinate (x along bone, y perp)
    vec2 worldPoint(vec2 local) const {
        vec2 right = vec2(cos(angle), sin(angle));
        vec2 up = vec2(-sin(angle), cos(angle));
        return pos + right * local.x + up * local.y;
    }

    void draw() {
        // direction of the bone
        vec2 dir(cos(angle), sin(angle));

        // endpoints derived from center + orientation
        vec2 p1 = pos - dir * halfLength;
        vec2 p2 = pos + dir * halfLength;

        float len = glm::length(p2 - p1);
        if (len == 0) return;

        vec2 dirN = glm::normalize(p2 - p1);
        vec2 right(-dirN.y, dirN.x);

        // circle centers
        vec2 c1 = p1 + dirN * radius;
        vec2 c2 = p2 - dirN * radius;

        vec2 rOffset = right * radius;

        glColor4f(1,1,1,1);

        // rectangle body
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

struct Joint {
    Bone* A;
    Bone* B;
    vec2 anchorA_local;   // anchor in local coordinates of A
    vec2 anchorB_local;   // anchor in local coordinates of B

    Joint(Bone* a, Bone* b, vec2 anchorA, vec2 anchorB) : A(a), B(b), anchorA_local(anchorA), anchorB_local(anchorB) { }
};

void drawCircle(vec2 pos, float radius) {
    glColor3f(1,0, 0);
    glBegin(GL_TRIANGLE_FAN);
        glVertex2f(pos.x, pos.y);
        for (int i = 0; i <= 20; i++) {
            float a = (i / 20.0f) * 2.0f * glm::pi<float>();
            glVertex2f(pos.x + cos(a)*radius, pos.y + sin(a)*radius);
        }
    glEnd();
}

list<Bone*> bones;
list<Joint> joints;

int main () {
    glfwSetCursorPosCallback(engine.window, Engine::cursor_position_callback);
    glfwSetMouseButtonCallback(engine.window, Engine::mouse_button_callback);
                // center     angle length radius
    Bone* arm = new Bone(vec2(0, 50) ,  0.0f,   75.0f, 15.0f);
    Bone* forearm = new Bone(vec2(0, -50),  0.0f,   75.0f, 15.0f);
    Bone* finger1 = new Bone(vec2(150, 150), 45.0f,  15.0f, 5.0f);
    Bone* finger2 = new Bone(vec2(150, -50), 0.0f,   15.0f, 5.0f);
    Bone* finger3 = new Bone(vec2(150, 250), -45.0f, 15.0f, 5.0f);

    bones.push_back(arm);
    bones.push_back(forearm);
    bones.push_back(finger1);
    bones.push_back(finger2);
    bones.push_back(finger3);

            //  boneA     boneB    anchorA_local                  anchorB_local
    Joint elbow(arm, forearm, vec2(arm->halfLength,0), vec2(-forearm->halfLength,0));
    Joint finger1Joint(forearm, finger1, vec2(forearm->halfLength,0), vec2(-finger1->halfLength,0));
    Joint finger2Joint(forearm, finger2, vec2(forearm->halfLength,0), vec2(-finger2->halfLength,0));
    Joint finger3Joint(forearm, finger3, vec2(forearm->halfLength,0), vec2(-finger3->halfLength,0));


    joints.push_back(elbow);
    joints.push_back(finger1Joint);
    joints.push_back(finger2Joint);
    joints.push_back(finger3Joint);



    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();


        // ---- draw bones ----
        for (Bone* b : bones) {
            b->draw();
        }

        for (Joint& j : joints) {
            vec2 worldA = j.A->worldPoint(j.anchorA_local);
            vec2 worldB = j.B->worldPoint(j.anchorB_local);

            drawCircle(worldA, 5.0f);
            drawCircle(worldB, 5.0f);

            vec2 offsetB = worldA - worldB;

            j.B->pos += offsetB;
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
}
 