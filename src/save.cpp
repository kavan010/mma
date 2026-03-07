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
#include <vector>
using namespace glm;
using namespace std;

// --- constants ----
const vec2 g(0.0f, 0.0f);

vec2 mousePos;
bool mouseDown = false;

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
        mousePos = vec2(xpos - width/2.0, height/2.0 - ypos);
    }
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mouseDown = (action == GLFW_PRESS);
        }
    }
    float cross(vec2 a, vec2 b) {
        return a.x*b.y - a.y*b.x;
    }
};
Engine engine;

// --- structs ---
struct Bone {
    vec2 pos; float angle;
    float halfLength, radius;

    vec2 vel; // linear
    float angularVelocity, mass, inertia;
    float invMass, invInertia;

    Bone (vec2 p, float a, float l, float r, float m) : pos(p), angle(a), halfLength(l), radius(r), mass(m) { 
        vel = vec2(0.0f, 0.0f);
        angularVelocity = 0.0f;
        inertia = (1/12.0f) * m * halfLength*2 * halfLength*2; 
        invMass = 1.0f / m;
        invInertia = 1.0f / inertia;
    }

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


struct Skeleton {
    list<Bone*> bones;
    list<Joint> joints;

    Bone* body;
    Bone* head;
    Bone* hip;
    Bone* shoulderR;
    Bone* shoulderL;
    Bone* armR;
    Bone* armL;
    Bone* forearmR;
    Bone* forearmL;
    Bone* legR;
    Bone* legL;
    Bone* calfR;
    Bone* calfL;

    Skeleton() {
        init();
    }

    void init() {
                        // center     angle length radius
        body      = new Bone(vec2(0, 150),       3.14f/2.0f,   25.0f, 25.0f,    60.0f);
        head       = new Bone(vec2(0, 0),        0.0,          15.0f, 15.0f,    5.0f);
        hip       = new Bone(vec2(0, 0),         0.0,          15.0f, 15.0f,    40.0f);
        shoulderR     = new Bone(vec2(-150, 0),  0.0,          7.0f, 7.0f,      30.0f);
        shoulderL     = new Bone(vec2(150, 0),   0.0,          7.0f, 7.0f,      30.0f);
        armR      = new Bone(vec2(-150, 50),     3.14f/2.0f,   20.0f, 7.0f,     15.0f);
        armL      = new Bone(vec2(150, 50),      3.14f/2.0f,   20.0f, 7.0f,     15.0f);
        forearmR      = new Bone(vec2(-150, 50), 3.14f/2.0f,   20.0f, 7.0f,     10.0f);
        forearmL      = new Bone(vec2(150, 50),  3.14f/2.0f,   20.0f, 7.0f,     10.0f);
        legR      = new Bone(vec2(-150, 50),     3.14f/2.0f,   25.0f, 10.0f,    20.0f);
        legL      = new Bone(vec2(150, 50),      3.14f/2.0f,   25.0f, 10.0f,    20.0f);
        calfR      = new Bone(vec2(-150, 50),    3.14f/2.0f,   20.0f, 7.0f,     15.0f);
        calfL      = new Bone(vec2(150, 50),     3.14f/2.0f,   20.0f, 7.0f,     15.0f);
        

        bones.push_back(body);
        bones.push_back(head);
        bones.push_back(hip);
        bones.push_back(shoulderR);
        bones.push_back(shoulderL);
        bones.push_back(armR);
        bones.push_back(armL);
        bones.push_back(forearmR);
        bones.push_back(forearmL);
        bones.push_back(legR);
        bones.push_back(legL);
        bones.push_back(calfR);
        bones.push_back(calfL);

                //  boneA     boneB    anchorA_local                  anchorB_local
        Joint neck(body, head, vec2(body->halfLength, 0.0) , vec2(0.0, -head->halfLength));
        Joint j0(body, hip, vec2(-body->halfLength, 0.0) , vec2(0, hip->halfLength));
        Joint j1(body, shoulderR, vec2(body->halfLength*0.34, -body->radius*0.94) , vec2(0,0));
        Joint j2(body, shoulderL, vec2(body->halfLength*0.34, body->radius*0.94)  , vec2(0,0));
        Joint j3(shoulderR, armR, vec2(0, 0)  , vec2(armR->halfLength, -0));
        Joint j4(shoulderL, armL, vec2(0, 0)  , vec2(armL->halfLength, -0));
        Joint elbowR(armR, forearmR, vec2(-armR->halfLength, 0)  , vec2(forearmR->halfLength, 0));
        Joint elbowL(armL, forearmL, vec2(-armL->halfLength, 0)  , vec2(forearmL->halfLength, 0));
        Joint hipR(hip, legR, vec2(-hip->radius*0.71, -hip->radius*0.71)  , vec2(legR->halfLength, 0));
        Joint hipL(hip, legL, vec2(hip->radius*0.71,  -hip->radius*0.71)  , vec2(legL->halfLength, 0));
        Joint kneeR(legR, calfR, vec2(-legR->halfLength, 0)  , vec2(calfR->halfLength, 0));
        Joint kneeL(legL, calfL, vec2(-legL->halfLength, 0)  , vec2(calfL->halfLength, 0));

        joints.push_back(neck);
        joints.push_back(j0);
        joints.push_back(j1);
        joints.push_back(j2);
        joints.push_back(j3);
        joints.push_back(j4);
        joints.push_back(elbowR);
        joints.push_back(elbowL);
        joints.push_back(hipR);
        joints.push_back(hipL);
        joints.push_back(kneeR);
        joints.push_back(kneeL);

        for (Joint& j : joints) {
            vec2 worldA = j.A->worldPoint(j.anchorA_local);
            vec2 worldB = j.B->worldPoint(j.anchorB_local);

            drawCircle(worldA, 5.0f);
            drawCircle(worldB, 5.0f);

            vec2 error = worldA - worldB;

            j.B->pos += error;
        }
    }
};

int main () {
    Skeleton* sk = new Skeleton();

    glfwSetCursorPosCallback(engine.window,   Engine::cursor_position_callback);
    glfwSetMouseButtonCallback(engine.window, Engine::mouse_button_callback);

    sk->forearmL->vel = vec2(0.0f, 0.0f); // initial velocity to test

    Bone* dragBone = nullptr;
    vec2 dragOffset;

    float dt = 1.0f/60.0f;
    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();

        // --- Select drag bone ---
        if (mouseDown && !dragBone) {
            for (Bone* b : sk->bones) {
                if (length(mousePos - b->pos) < 30.0f) {
                    dragBone = b;
                    vec2 d = mousePos - b->pos;
                    dragOffset = vec2(d.x*cos(-b->angle) - d.y*sin(-b->angle), d.x*sin(-b->angle) + d.y*cos(-b->angle));
                    break;
                }
            }
        } else if (!mouseDown) dragBone = nullptr;

        vector<vec2> oldPos;
        vector<float> oldAng;

        // integrate motion first
        for (Bone* b : sk->bones) {
            oldPos.push_back(b->pos);
            oldAng.push_back(b->angle);

            // b->vel += vec2(0, -500.0f) * dt; // gravity
            if (b == dragBone) {
                vec2 wp = b->worldPoint(dragOffset);
                vec2 f = (mousePos - wp) * 500.0f; // spring force
                b->vel += f * b->invMass * dt;
                vec2 r = wp - b->pos;
                b->angularVelocity += (r.x*f.y - r.y*f.x) * b->invInertia * dt;
            }
            b->vel *= 0.99f; b->angularVelocity *= 0.99f; // damping

            b->pos += b->vel * dt;
            b->angle += b->angularVelocity * dt;
        }

        // solve joints several times
        for(int i=0;i<10;i++) {
            for (Joint& j : sk->joints) {
                // world anchors
                vec2 worldA = j.A->worldPoint(j.anchorA_local);
                vec2 worldB = j.B->worldPoint(j.anchorB_local);

                // distance from anchor to centre of mass (for torque)
                vec2 rA = worldA - j.A->pos;
                vec2 rB = worldB - j.B->pos;

                vec2 error = worldB - worldA;

                // Compute effective mass matrix K for the constraint
                float k11 = j.A->invMass     + j.B->invMass  + j.A->invInertia * rA.y * rA.y + j.B->invInertia * rB.y * rB.y;
                float k12 = -j.A->invInertia * rA.x * rA.y   - j.B->invInertia * rB.x * rB.y;
                float k22 = j.A->invMass     + j.B->invMass  + j.A->invInertia * rA.x * rA.x + j.B->invInertia * rB.x * rB.x;

                float det = k11 * k22 - k12 * k12;
                if (abs(det) < 1e-6f) continue;

                // Solve for impulse P
                vec2 P;
                P.x = (k22 * error.x - k12 * error.y) / det;
                P.y = (-k12 * error.x + k11 * error.y) / det;

                // Apply correction to position and angle (torque effect)
                j.A->pos += P * j.A->invMass;
                j.A->angle += j.A->invInertia * (rA.x * P.y - rA.y * P.x);
                j.B->pos -= P * j.B->invMass;
                j.B->angle -= j.B->invInertia * (rB.x * P.y - rB.y * P.x);
            }
        }

        int idx = 0;
        for (Bone* b : sk->bones) {
            b->vel = (b->pos - oldPos[idx]) / dt;
            b->angularVelocity = (b->angle - oldAng[idx]) / dt;
            idx++;
        }

        // --- PHASE 4: Draw ---
        for (Bone* b : sk->bones) {
            b->draw();
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
}
 