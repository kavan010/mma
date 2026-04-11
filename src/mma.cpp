#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <list>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using namespace glm; using namespace std;


// ------------------------ Engine & Constants ----------------------
vec2 g(0.0f, -980.6f);
vec2 mousePos; bool mouseDown = false;
struct Engine {
    GLFWwindow* window;
    int WIDTH = 800, HEIGHT = 600;

    Engine () {
        if (!glfwInit()) {
            cerr << "Failed to initialize GLFW" << endl; exit(EXIT_FAILURE);
        }

        window = glfwCreateWindow(WIDTH, HEIGHT, "RL is pain", nullptr, nullptr);
        if (!window) {
            cerr << "Failed to create GLFW window" << endl;
            glfwTerminate(); exit(EXIT_FAILURE);
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
        glOrtho(0.0, (double)WIDTH, 0.0, (double)HEIGHT, -1.0, 1.0);
    }

    // ------ Helper Functions ------
    float cross(vec2 a, vec2 b) {
        return a.x*b.y - a.y*b.x;
    }
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

    // ------ Callbacks ------
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        mousePos = vec2(xpos, height - ypos);
    }
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mouseDown = (action == GLFW_PRESS);
        }
    }
};
Engine engine;

// ------------------------ Bodies & Physics ------------------------
vec2 rotate(vec2 v, float a) {
    float c = cos(a), s = sin(a);
    return vec2(c*v.x - s*v.y, s*v.x + c*v.y);
}
float cross(vec2 a, vec2 b) {
    return a.x*b.y - a.y*b.x;
}
vec2 cross(float s, vec2 v) {
    return vec2(-s*v.y, s*v.x);
}

struct Bone {
    vec2 pos, vel;
    float angle, angVel;
    float halfLength, radius, mass, inertia, invMass, invInertia;
    vec2 force = vec2(0.0f);
    float torque = 0.0f;
    bool dragged = false;

    Bone(vec2 p,float a,float l,float r,float m) : pos(p), vel(0), angle(a), angVel(0), halfLength(l), radius(r), mass(m) {
        float L=l*2, W=r*2;
        inertia=(m*(L*L+W*W))/12.f;
        invMass=1.f/m;
        invInertia=1.f/inertia;
    }
    vec2 worldPoint(vec2 local) const {
        float c=cos(angle), s=sin(angle);
        return pos + vec2(c*local.x - s*local.y, s*local.x + c*local.y);
    }
    void draw() {
        vec2 dir(cos(angle),sin(angle));
        vec2 p1=pos-dir*halfLength, p2=pos+dir*halfLength;

        if(glm::length(p2-p1)*glm::length(p2-p1)==0) return;

        vec2 d=glm::normalize(p2-p1), r=vec2(-d.y,d.x)*radius;
        vec2 c1=p1+d*radius, c2=p2-d*radius;

        // Draw Rectangle Portion
        glColor4f(1,1,1,1);
        glBegin(GL_QUADS);
            glVertex2f((c2+r).x,(c2+r).y);
            glVertex2f((c2-r).x,(c2-r).y);
            glVertex2f((c1-r).x,(c1-r).y);
            glVertex2f((c1+r).x,(c1+r).y);
        glEnd();

        // Draw End Caps
        for(vec2 c:{c1,c2}){
            glBegin(GL_TRIANGLE_FAN);
            glVertex2f(c.x,c.y);
            for(int i=0;i<=20;i++){
                float a=i/20.f*2*glm::pi<float>();
                glVertex2f(c.x+cos(a)*radius,c.y+sin(a)*radius);
            }
            glEnd();
        }
    }
};
struct Joint {
    Bone* A, *B; // bones conected by this joint
    vec2 anchorA_local, anchorB_local; // local positions the joint connects to
    float maxTorque = 1e5f;

    // --- Controller Properties ---
    float targetAngle = 0;
    float stiffness = 1.0f;

    Joint(Bone* a, Bone* b, vec2 anchorA, vec2 anchorB, float targetAngle=0.0f, float stiffness=1.0f) : 
    A(a), B(b), anchorA_local(anchorA), anchorB_local(anchorB), targetAngle(targetAngle), stiffness(stiffness) { }

    void solve(float dt) {
        Bone* A = this->A; Bone* B = this->B;

        vec2 rA = rotate(anchorA_local, A->angle), rB = rotate(anchorB_local, B->angle);
        vec2 pA = A->pos + rA, pB = B->pos + rB;

        // position error
        vec2 C = pB - pA;
        // anchor velocities
        vec2 vA = A->vel + cross(A->angVel, rA);
        vec2 vB = B->vel + cross(B->angVel, rB);

        vec2 relVel = vB - vA;

        // --- Baumgarte stabilization ---
        float beta = 0.8f;
        vec2 bias = (beta / dt) * C;
        vec2 Cdot = relVel + bias;

        float k11 = A->invMass + B->invMass + A->invInertia*rA.y*rA.y + B->invInertia*rB.y*rB.y;
        float k12 = -A->invInertia*rA.x*rA.y - B->invInertia*rB.x*rB.y;
        float k21 = k12;
        float k22 = A->invMass + B->invMass + A->invInertia*rA.x*rA.x + B->invInertia*rB.x*rB.x;

        float det = k11*k22 - k12*k21;
        if(det == 0) return;

        float invDet = 1.0f / det;

        vec2 impulse;
        impulse.x = -( k22*Cdot.x - k12*Cdot.y) * invDet;
        impulse.y = -(-k21*Cdot.x + k11*Cdot.y) * invDet;

        A->vel -= impulse * A->invMass;
        B->vel += impulse * B->invMass;

        A->angVel -= cross(rA, impulse) * A->invInertia;
        B->angVel += cross(rB, impulse) * B->invInertia;
    }
    void applyTorque(float dt) {

        float angle   = B->angle - A->angle;
        float angVel  = B->angVel - A->angVel;
        float error   = targetAngle - angle;

        float k = stiffness * maxTorque;
        //float d = 2.0f * sqrt(k * (A->inertia + B->inertia));  // critical damping

        float torque  = k * error;

        // clamp to maxTorque so it cant explode
        torque = glm::clamp(torque, -maxTorque, maxTorque);

        // convert to impulse scaled by dt
        float impulse = torque * dt;

        A->angVel -= impulse * A->invInertia;
        B->angVel += impulse * B->invInertia;
    }
};

struct Skeleton {
    vector<Bone*> bones;
    vector<Joint> joints;

    Bone *rod1, *rod2, *ground;

    Skeleton(){ init(); }

    void init() {
        // ---- Bones ----
        cart     = new Bone({250,200},3.14f/2.0f,40,5,7);
        pole = new Bone({250,200},3.14f/2.0f,40,5,7);
        ground = new Bone({400,300},-3.14f/2.0f,2,1,70000);

        bones = {cart,pole,ground};

        // Joints
        joints.push_back(Joint(rod1,rod2,{rod1->halfLength,0},{rod2->halfLength,0},3.14,0.0f));


        // Initial constraint alignment
        for(auto& j : joints){
            vec2 err = j.A->worldPoint(j.anchorA_local) - j.B->worldPoint(j.anchorB_local);
            j.B->pos += err;
        }
    }

    void step(float dt) {
        // ---- Euler Integrate Gravity & Draw ----
        for (Bone* b : bones) {
            if (b != ground)
            b->vel += g * dt;
            else {
                b->vel=vec2(0.0f);
                b->angVel=0.0f;
            }
            b->vel    *= 1.0f / (1.0f + 0.7f * dt);
            b->angVel *= 1.0f / (1.0f + 0.7f * dt);
            b->draw();
        }
        // ---- Solve Joints & Apply Torque---
        for(int i=0;i<100;i++) {
            for (Joint& j : joints){
                j.solve(dt);
                j.applyTorque(dt);
            }
        }
        // ---- Euler Integrate ----
        for (Bone* b : bones) {
            if (b != ground)
                b->pos += b->vel * dt;\
                b->angVel = glm::clamp(b->angVel, -50.0f, 50.0f);
            if (b != ground)
            b->angle += b->angVel * dt;
            checkBorderCollision(b);
        }
    }
    void reset() {
        rod1->pos = {250, 200};   // add these
        rod1->vel = {0, 0};
        rod2->vel = {0, 0};
        rod1->angle = 3.14f/2.0f + ((rand() % 100) / 100.0f - 0.5f) * 0.2f;;
        rod1->angVel = 0;
        rod2->angle = 3*3.14f/2.0f + ((rand() % 100) / 100.0f - 0.5f) * 0.2f;
        rod2->angVel = 0;
        
        for (auto& j : joints) {
            j.targetAngle = 3.14f;
            vec2 err = j.A->worldPoint(j.anchorA_local) - j.B->worldPoint(j.anchorB_local);
            j.B->pos += err;
        }
    }
    void checkBorderCollision(Bone* b) {
        float restitution = 0.2f, slop = 0.01f, percent = 0.8f, friction = 0.8f;

        vec2 dir(cos(b->angle), sin(b->angle));
        vec2 offset = dir * b->halfLength;
        vec2 points[2] = { b->pos - offset, b->pos + offset };

        for (vec2 contact : points) {
            vec2 normal;
            float penetration = 0.0f;
            bool collided = false;

            if (contact.x < b->radius) {
                normal = vec2(1,0);
                penetration = b->radius - contact.x;
                collided = true;
            }
            else if (contact.x > engine.WIDTH - b->radius) {
                normal = vec2(-1,0);
                penetration = contact.x - (engine.WIDTH - b->radius);
                collided = true;
            }
            else if (contact.y < b->radius) {
                normal = vec2(0,1);
                penetration = b->radius - contact.y;
                collided = true;
            }
            else if (contact.y > engine.HEIGHT - b->radius) {
                normal = vec2(0,-1);
                penetration = contact.y - (engine.HEIGHT - b->radius);
                collided = true;
            }
            if (!collided) continue;

            // deal with the contacted point
            vec2 r = contact - b->pos;

            // velocity at contact
            vec2 vel = b->vel + cross(b->angVel, r);

            float velAlongNormal = dot(vel, normal);
            if (velAlongNormal > 0) continue;

            float rCrossN = cross(r, normal);
            float denom = b->invMass + (rCrossN * rCrossN) * b->invInertia;
            if (denom == 0) continue;

            float jn = -(1.0f + restitution) * velAlongNormal;
            jn /= denom;

            vec2 impulse = normal * jn;
            b->vel += impulse * b->invMass;
            b->angVel += cross(r, impulse) * b->invInertia;

            b->vel.x *= friction;
            b->vel.y *= 0.8f;

            // --- 3. Positional Correction ---
            vec2 correction = normal * percent * fmax(penetration - slop, 0.0f);
            b->pos += correction;
        }
    }
};


// ------------------------ File Connections ------------------------
struct Data {
    // Sockets
    WSADATA w;
    SOCKET sock, sendSock;
    sockaddr_in server, python;

    float recvBuffer[1], stateBuffer[6];
    bool primed = false;
    Data() {
        WSAStartup(MAKEWORD(2,2), &w);
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        sendSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

        server.sin_family = AF_INET;
        server.sin_port = htons(5005);
        server.sin_addr.S_un.S_addr = INADDR_ANY;

        python.sin_family = AF_INET;
        python.sin_port = htons(5006);
        inet_pton(AF_INET, "127.0.0.1", &python.sin_addr);

        bind(sock, (sockaddr*)&server, sizeof(server));
    }
    
    bool receiveData(Skeleton* sk) {
        int bytesRead = recv(sock, (char*)recvBuffer, sizeof(recvBuffer), 0);

        if (bytesRead != sizeof(recvBuffer)) {
            return false;
        } else if (recvBuffer[0] == -100){
            return true;
        } else if (recvBuffer[0] == -69){
            sk->reset();
        } else {
            sk->joints[0].targetAngle = recvBuffer[0]+3.14f;
        }

        return false; 
    }
    void sendData(Skeleton* sk) {
        int i = 0;

        float angle1   = sk->joints[0].B->angle - sk->joints[0].A->angle;
        float angle2   = sk->joints[1].B->angle - sk->joints[1].A->angle;
        stateBuffer[i++] =  sin(angle1);
        stateBuffer[i++] =  cos(angle1);
        stateBuffer[i++] =  sin(angle2);
        stateBuffer[i++] =  cos(angle2);
        stateBuffer[i++] =  sk->rod1->angVel;
        stateBuffer[i++] =  sk->rod2->angVel;

        sendto(sendSock, (char*)stateBuffer, i * sizeof(float), 0, (sockaddr*)&python, sizeof(python));
    }
};
Data dataManager;

// ------ Keyboard Callback ------
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        // Retrieve the Skeleton object associated with the window
        Skeleton* figure = static_cast<Skeleton*>(glfwGetWindowUserPointer(window));
        if (!figure) return;
        float angle_delta = 0.1f; // Small adjustment in radians
        if (key == GLFW_KEY_LEFT) {
            figure->joints[0].targetAngle -= angle_delta;
            cout << "Joint 0 targetAngle: " << figure->joints[0].targetAngle << endl;
        } else if (key == GLFW_KEY_RIGHT) {
            figure->joints[0].targetAngle += angle_delta;
            cout << "Joint 0 targetAngle: " << figure->joints[0].targetAngle << endl;
        }
    }
}
// ------------------------ MAIN ------------------------
int main() {
    Skeleton* figure = new Skeleton();
    Bone* dragBone=nullptr; vec2 dragOffset;
    glfwSetWindowUserPointer(engine.window, figure); // Store the figure pointer for callbacks
    glfwSetKeyCallback(engine.window, key_callback); // Register the new keyboard callback
    
    float dt = 1.0/60.0;
    double lastPrintTime = 0.0;
    glfwSwapBuffers(engine.window);
    while(!glfwWindowShouldClose(engine.window)) {
        engine.run();
        
        // ------ RECIEVE DATA FROM PYTHON -------
        //bool gotAction = dataManager.receiveData(figure);

        //if (gotAction) {
            figure->step(dt);
            // ------ SEND DATA TO PYTHON -------
            //dataManager.sendData(figure);
            glfwSwapBuffers(engine.window);
        //} 


        //glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    // Exit Program
    glfwTerminate(); return 0;
}