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

// --- constants ---
const vec2 GRAVITY(0.0f, -120.0f);

// --- engine ---
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
};
Engine engine;

struct Joint {
    vec2 pos, old_pos;

    Joint(vec2 p) : pos(p), old_pos(p) {}

    void update(float dt) {
        vec2 vel = pos - old_pos;
        old_pos = pos;
        pos += vel + GRAVITY * dt * dt;
    }
};

void draw_circle(vec2 center, float radius) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(center.x, center.y);
    for (int i = 0; i <= 20; i++) {
        float a = (i / 20.0f) * 2.0f * glm::pi<float>();
        glVertex2f(center.x + cos(a) * radius, center.y + sin(a) * radius);
    }
    glEnd();
}

struct Body {
    Joint *j1, *j2;
    float length;

    Body(Joint* p1, Joint* p2, float len) : j1(p1), j2(p2), length(len) {}

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
        if (glm::length(dir) == 0) return;
        vec2 right = normalize(vec2(-dir.y, dir.x));
        vec2 rOffset = right * radius;

        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        
        glBegin(GL_QUADS);
            glVertex2f((p2 + rOffset).x, (p2 + rOffset).y);
            glVertex2f((p2 - rOffset).x, (p2 - rOffset).y);
            glVertex2f((p1 - rOffset).x, (p1 - rOffset).y);
            glVertex2f((p1 + rOffset).x, (p1 + rOffset).y);
        glEnd();

        draw_circle(p1, radius);
        draw_circle(p2, radius);
    }
};

list<Joint> joints;
list<Body> bodies;
Joint* dragged_joint = nullptr;
vec2 mouse_pos;

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    mouse_pos.x = (float)xpos - (float)width / 2.0f;
    mouse_pos.y = (float)height / 2.0f - (float)ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            float closest_dist_sq = 20.0f * 20.0f; // Grab radius
            Joint* closest_joint = nullptr;
            for (auto& j : joints) {
                vec2 diff = mouse_pos - j.pos;
                float dist_sq = dot(diff, diff);
                if (dist_sq < closest_dist_sq) {
                    closest_dist_sq = dist_sq;
                    closest_joint = &j;
                }
            }
            dragged_joint = closest_joint;
        } else if (action == GLFW_RELEASE) {
            dragged_joint = nullptr;
        }
    }
}

void constrain_joints_to_screen(float width, float height) {
    float friction = 0.8f; 
    float buffer = 4.0f; 

    float half_w = width / 2.0f - buffer;
    float half_h = height / 2.0f - buffer;

    for (auto& j : joints) {
        vec2 vel = j.pos - j.old_pos;

        if (j.pos.x < -half_w) {
            j.pos.x = -half_w;
            j.old_pos.x = j.pos.x + vel.x * friction;
        }
        else if (j.pos.x > half_w) {
            j.pos.x = half_w;
            j.old_pos.x = j.pos.x + vel.x * friction;
        }

        if (j.pos.y < -half_h) {
            j.pos.y = -half_h;
            j.old_pos.y = j.pos.y + vel.y * friction;
        }
        else if (j.pos.y > half_h) {
            j.pos.y = half_h;
            j.old_pos.y = j.pos.y + vel.y * friction;
        }
    }
}

int main () {
    glfwSetCursorPosCallback(engine.window, cursor_position_callback);
    glfwSetMouseButtonCallback(engine.window, mouse_button_callback);

    // --- Create a structure ---
    // 4 double pendulums attached to a central joint
    joints.emplace_back(vec2(0, 250));
    Joint* center_joint = &joints.back();

    for (int i = 0; i < 4; ++i) {
        float angle = (float)i / 4.0f * (2.0f * glm::pi<float>());
        float l1 = 80.0f + i * 15.0f;
        float l2 = 70.0f - i * 10.0f;

        // Create the middle and end joints for this arm, starting slightly offset
        vec2 mid_pos = center_joint->pos + vec2(cos(angle), sin(angle)) * 2.0f;
        joints.emplace_back(mid_pos);
        Joint* mid_joint = &joints.back();

        vec2 end_pos = mid_joint->pos + vec2(cos(angle), sin(angle)) * 2.0f;
        joints.emplace_back(end_pos);
        Joint* end_joint = &joints.back();

        // Create the bodies (constraints)
        bodies.emplace_back(center_joint, mid_joint, l1);
        bodies.emplace_back(mid_joint, end_joint, l2);
    }
    
    float dt = 1.0f / 60.0f;
    int solver_iterations = 8;

    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();

        for (auto& j : joints) {
            j.update(dt);
        }
        
        constrain_joints_to_screen(engine.WIDTH, engine.HEIGHT);

        if (dragged_joint) {
            dragged_joint->pos = mouse_pos;
        }

        for (int i = 0; i < solver_iterations; ++i) {
            for (auto& b : bodies) {
                b.solve();
            }
             if (dragged_joint) {
                dragged_joint->pos = mouse_pos;
            }
        }

        for (auto& b : bodies) {
            b.draw(4.0f);
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
}
 