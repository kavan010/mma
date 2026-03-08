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
const vec2 g(0.0f, -98.0f);

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

// --- Globals for UI ---
int selectedJointIndex = 0;
struct Skeleton; // Forward declaration
Skeleton* sk_ptr = nullptr;


// --- structs ---
struct Bone {
    vec2 pos; float angle;
    float halfLength, radius;

    vec2 vel; // linear
    float angularVelocity, mass, inertia;
    float invMass, invInertia;

    vec2 oldPos; float oldAngle;

    Bone (vec2 p, float a, float l, float r, float m) : pos(p), angle(a), halfLength(l), radius(r), mass(m) { 
        vel = vec2(0.0f, 0.0f);
        angularVelocity = 0.0f;
        // Use a box approximation for inertia: I = 1/12 * m * (h^2 + w^2).
        // The simple rod formula (1/12 * m * h^2) is unstable for "fat" objects like the head
        // because it ignores the radius, making them too easy to spin.
        inertia = (1.0f/12.0f) * m * (pow(halfLength*2.0f, 2) + pow(radius*2.0f, 2));
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

    // --- PD Controller Properties ---
    float targetAngle = 0.0f;
    float stiffness = 0.0f; // k_p: How strongly it tries to reach the target angle.

    Joint(Bone* a, Bone* b, vec2 anchorA, vec2 anchorB) : A(a), B(b), anchorA_local(anchorA), anchorB_local(anchorB) {
        // Set initial target angle to the current relative angle so it's not floppy at the start
        targetAngle = B->angle - A->angle;
    }
};

void drawCircle(vec2 pos, float radius, vec3 color) {
    glColor3f(color.r, color.g, color.b);
    glBegin(GL_TRIANGLE_FAN);
        glVertex2f(pos.x, pos.y);
        for (int i = 0; i <= 20; i++) {
            float a = (i / 20.0f) * 2.0f * glm::pi<float>();
            glVertex2f(pos.x + cos(a)*radius, pos.y + sin(a)*radius);
        }
    glEnd();
}

// --- Physics ---
void checkBorderCollision(Bone* b) {
    float halfWidth = engine.WIDTH / 2.0f;
    float halfHeight = engine.HEIGHT / 2.0f;
    float floorLevel = -halfHeight + 200.0f;

    vec2 dir(cos(b->angle), sin(b->angle));
    vec2 offset = dir * b->halfLength;
    vec2 p1 = b->pos - offset;
    vec2 p2 = b->pos + offset;

    float minX = fmin(p1.x, p2.x) - b->radius;
    float maxX = fmax(p1.x, p2.x) + b->radius;
    float minY = fmin(p1.y, p2.y) - b->radius;
    float maxY = fmax(p1.y, p2.y) + b->radius;

    if (minX < -halfWidth) {
        float fix = -halfWidth - minX;
        b->pos.x += fix;
        // Kill velocity normal to the wall to prevent bouncing (inelastic collision)
        b->oldPos.x = b->pos.x;
    }
    if (maxX >  halfWidth) {
        float fix = halfWidth - maxX;
        b->pos.x += fix;
        // Kill velocity normal to the wall to prevent bouncing (inelastic collision)
        b->oldPos.x = b->pos.x;
    }
    if (minY < floorLevel) {
        float fix = floorLevel - minY;
        b->pos.y += fix;
        // Kill velocity normal to the wall to prevent bouncing (inelastic collision)
        b->oldPos.y = b->pos.y;
    }
    if (maxY > halfHeight) {
        float fix = halfHeight - maxY;
        b->pos.y += fix;
        // Kill velocity normal to the wall to prevent bouncing (inelastic collision)
        b->oldPos.y = b->pos.y;
    }
}


struct Skeleton {
    list<Bone*> bones;
    list<Joint> joints;

    Bone* body; Bone* head; Bone* hip;
    Bone* shoulderR; Bone* shoulderL;
    Bone* armR; Bone* armL;
    Bone* forearmR; Bone* forearmL;
    Bone* legR; Bone* legL;
    Bone* calfR; Bone* calfL;

    Skeleton() {
        init();
    }

    void init() {
                        // center     angle length radius
        body      = new Bone(vec2(0, 150),       3.14f/2.0f,   25.0f, 25.0f,    60.0f); // Torso
        head       = new Bone(vec2(0, 0),        0.0,          15.0f, 15.0f,    5.0f);  // Head
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

            vec2 error = worldA - worldB;

            j.B->pos += error;
        }
    }
};

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (!sk_ptr) return;

    if (key == GLFW_KEY_UP) {
        selectedJointIndex = (selectedJointIndex + 1) % sk_ptr->joints.size();
        auto it = sk_ptr->joints.begin();
        std::advance(it, selectedJointIndex);
        cout << "Joint " << selectedJointIndex << " Target: " << it->targetAngle << endl;
    }
    if (key == GLFW_KEY_DOWN) {
        selectedJointIndex = (selectedJointIndex > 0) ? selectedJointIndex - 1 : sk_ptr->joints.size() - 1;
        auto it = sk_ptr->joints.begin();
        std::advance(it, selectedJointIndex);
        cout << "Joint " << selectedJointIndex << " Target: " << it->targetAngle << endl;
    }

    auto it = sk_ptr->joints.begin();
    std::advance(it, selectedJointIndex);
    float angleIncrement = 0.1f; // radians

    if (key == GLFW_KEY_RIGHT) { it->targetAngle += angleIncrement; cout << "Target: " << it->targetAngle << endl; }
    if (key == GLFW_KEY_LEFT)  { it->targetAngle -= angleIncrement; cout << "Target: " << it->targetAngle << endl; }

    if (key == GLFW_KEY_W) {
        it->stiffness += 100000.0f;
        cout << "Stiffness: " << it->stiffness << endl;
    }
    if (key == GLFW_KEY_S) {
        it->stiffness -= 100000.0f;
        cout << "Stiffness: " << it->stiffness << endl;
    }
}

int main () {
    Skeleton* sk = new Skeleton();
    sk_ptr = sk; // Give global pointer access to the skeleton for the callback

    glfwSetCursorPosCallback(engine.window,   Engine::cursor_position_callback);
    glfwSetMouseButtonCallback(engine.window, Engine::mouse_button_callback);
    glfwSetKeyCallback(engine.window, key_callback);

    Bone* dragBone = nullptr;
    vec2 dragOffset;

    float dt = 0.032f;
    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();


        // -------- SKELETON MECHANICS --------
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

        // --- Joint Controls (PD Controller) ---
        for (Joint& j : sk->joints) {
            // 1. Calculate current relative angle between the two bones
            float currentAngle = j.B->angle - j.A->angle;

            // 2. Calculate the shortest angle difference (error)
            float error = j.targetAngle - currentAngle;

            // 4. Calculate torque using the PD formula
            float torque = (j.stiffness * error);

            // 5. Apply the torque to the bones
            j.A->angularVelocity -= torque * j.A->invInertia * dt;
            j.B->angularVelocity += torque * j.B->invInertia * dt;
        }

        // integrate motion
        for (Bone* b : sk->bones) {
            b->vel += g * dt; // gravity
            b->oldPos = b->pos;
            b->oldAngle = b->angle;
            
            b->vel*=0.99f; b->angularVelocity*=0.99f; // damping

            b->pos += b->vel * dt;
            checkBorderCollision(b);
            b->angle += b->angularVelocity * dt;
        }

        // solve for joints and momentum
        for (int iteration = 0; iteration < 8; iteration++) {
            // --- Mouse Constraint (Stable Drag) ---
            if (dragBone) {
                dragBone->pos = mousePos;
            }

            for (Joint& j : sk->joints) {
                // world anchors
                vec2 worldA = j.A->worldPoint(j.anchorA_local);
                vec2 worldB = j.B->worldPoint(j.anchorB_local);

                vec2 mid = (worldA + worldB) * 0.5f;

                // distance from anchor to centre of mass (for torque)
                vec2 rA = mid - j.A->pos;
                vec2 rB = mid - j.B->pos;

                // distance from anchor
                vec2 error = (worldB - worldA);

                // Compute effective mass matrix K for the constraint
                float k11 = j.A->invMass     + j.B->invMass  + j.A->invInertia * rA.y * rA.y + j.B->invInertia * rB.y * rB.y;
                float k12 = -j.A->invInertia * rA.x * rA.y   - j.B->invInertia * rB.x * rB.y;
                float k22 = j.A->invMass     + j.B->invMass  + j.A->invInertia * rA.x * rA.x + j.B->invInertia * rB.x * rB.x;

                float det = k11*k22 - k12*k12;
                if (abs(det) < 1e-6f) continue;

                // Solve for impulse P
                vec2 P{
                    (k22*error.x - k12*error.y)/det,
                    (-k12*error.x + k11*error.y)/det
                };

                // Apply correction to position and angle (torque effect)
                j.A->pos += P * j.A->invMass;
                j.A->angle += j.A->invInertia * (rA.x * P.y - rA.y * P.x);
                j.B->pos -= P * j.B->invMass;
                j.B->angle -= j.B->invInertia * (rB.x * P.y - rB.y * P.x);
            }
        }

        // Update velocities and draw
        for (Bone* b : sk->bones) {
            b->vel = (b->pos - b->oldPos) / dt;
            b->angularVelocity = (b->angle - b->oldAngle) / dt;
            b->draw();
        }

        // Draw UI for selected joint
        if (!sk->joints.empty()) {
            auto it = sk->joints.begin();
            std::advance(it, selectedJointIndex);
            vec2 jointPos = it->A->worldPoint(it->anchorA_local);
            
            // Draw red circle on selected joint
            drawCircle(jointPos, 8.0f, vec3(1,0,0));

            // Draw line indicating target angle
            float targetWorldAngle = it->A->angle + it->targetAngle;
            vec2 targetDir = vec2(cos(targetWorldAngle), sin(targetWorldAngle));
            glColor3f(0,1,0);
            glBegin(GL_LINES);
                glVertex2fv(value_ptr(jointPos));
                glVertex2fv(value_ptr(jointPos + targetDir * 30.0f));
            glEnd();
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
}
 