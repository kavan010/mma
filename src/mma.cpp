#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <list>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
using namespace glm; using namespace std;

// ------------------------ Engine & Constants ----------------------
const int NUM_ENV = 1, STATE_DIM = 10, ACTION_DIM = 2;
vec2 g(0.0f, -980.6f);
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
    vec2 rotate(vec2 v, float a) {
        float c = cos(a), s = sin(a);
        return vec2(c*v.x - s*v.y, s*v.x + c*v.y);
    }
    vec2 cross(float s, vec2 v) {
        return vec2(-s*v.y, s*v.x);
    }
};
Engine engine;


// ------------------------ Bodies & Physics ------------------------
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

        vec2 rA = engine.rotate(anchorA_local, A->angle), rB = engine.rotate(anchorB_local, B->angle);
        vec2 pA = A->pos + rA, pB = B->pos + rB;

        // position error
        vec2 C = pB - pA;
        // anchor velocities
        vec2 vA = A->vel + engine.cross(A->angVel, rA);
        vec2 vB = B->vel + engine.cross(B->angVel, rB);

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

        A->angVel -= engine.cross(rA, impulse) * A->invInertia;
        B->angVel += engine.cross(rB, impulse) * B->invInertia;
    }
    void applyTorque(float dt) {

        float angle   = B->angle - A->angle;
        float angVel  = B->angVel - A->angVel;
        float error   = targetAngle - angle;

        float k = stiffness * maxTorque;
        float d = 2.0f * sqrt(k * (A->inertia + B->inertia));  // critical damping

        float torque  = k * error - d * angVel;

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
    vec2 startPos;

    Bone *legR, *legL, *hip;

    Skeleton(vec2 p) : startPos(p) { init(p); }

    void init(vec2 p) {
        for(auto b : bones) delete b;
        joints.clear();
        // ---- Bones ----
        legR    = new Bone(p, 3.14f/3.0f,  25, 5, 7.0f);
        legL    = new Bone(p, 3.14f/1.0f, 25, 5, 7.0f);
        hip     = new Bone(p, 0.0f, 35, 18, 10.0f);

        bones = {legL,legR,hip};

        // Joints 
        joints.push_back(Joint(hip,legR, {-hip->halfLength*0.71f,-hip->radius*0.71f}, {legR->halfLength,0}, 0.35f));
        joints.push_back(Joint(hip,legL, { hip->halfLength*0.71f,-hip->radius*0.71f}, {legL->halfLength,0}, 3.14f-0.35f));


        // Initial constraint alignment
        for(auto& j : joints){
            vec2 err = j.A->worldPoint(j.anchorA_local) - j.B->worldPoint(j.anchorB_local);
            j.B->pos += err;
        }
    }

    void step(float dt) {
        // ---- Euler Integrate Gravity & Draw ----
        for (Bone* b : bones) {
            b->vel += g * dt;
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
            b->pos += b->vel * dt;\
            b->angVel = glm::clamp(b->angVel, -50.0f, 50.0f);
            b->angle += b->angVel * dt;
            checkBorderCollision(b);
        }
    }
    void reset() {
        init(startPos);
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
            vec2 vel = b->vel + engine.cross(b->angVel, r);

            float velAlongNormal = dot(vel, normal);
            if (velAlongNormal > 0) continue;

            float rCrossN = engine.cross(r, normal);
            float denom = b->invMass + (rCrossN * rCrossN) * b->invInertia;
            if (denom == 0) continue;

            float jn = -(1.0f + restitution) * velAlongNormal;
            jn /= denom;

            vec2 impulse = normal * jn;
            b->vel += impulse * b->invMass;
            b->angVel += engine.cross(r, impulse) * b->invInertia;

            b->vel.x *= friction;
            b->vel.y *= 0.8f;

            // --- 3. Positional Correction ---
            vec2 correction = normal * percent * fmax(penetration - slop, 0.0f);
            b->pos += correction;
        }
    }
};
vector<Skeleton*> envs {
    // new Skeleton(vec2(400,150)),
    new Skeleton(vec2(200,150)),
    // new Skeleton(vec2(600,150))
};


// ------------------------ File Connections ------------------------
struct Data {
    int sock, sendSock;
    sockaddr_in server, python;
    const static int revSize = NUM_ENV * ACTION_DIM;
    const static int sendSize = NUM_ENV * STATE_DIM;

    float recvBuffer[revSize], stateBuffer[sendSize];
    bool primed = false;
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
        int bytesRead = recv(sock, (char*)recvBuffer, sizeof(recvBuffer), 0);

        if (bytesRead < (int)(2 * sizeof(float))) return false;

        if (recvBuffer[0] == -100.0f) {
            return true; // python is asking for state
        } else if (recvBuffer[0] == -69.0f) {
            envs[(int)recvBuffer[1]]->reset();
            return false;
        } else if (bytesRead == (int)sizeof(recvBuffer)) {
            int i = 0;
            for (Skeleton* env : envs) {
                env->joints[0].targetAngle = recvBuffer[i++];
                env->joints[1].targetAngle = recvBuffer[i++];
            }
        }
        return false;
    }
    void sendData() {
        int i = 0;
        for (Skeleton* env : envs) {
            float relR = env->legR->angle - env->hip->angle;
            float relL = env->legL->angle - env->hip->angle;

            stateBuffer[i++] = sin(env->hip->angle);
            stateBuffer[i++] = cos(env->hip->angle);
            stateBuffer[i++] = env->hip->angVel;
            stateBuffer[i++] = sin(relR);
            stateBuffer[i++] = cos(relR);
            stateBuffer[i++] = env->legR->angVel;
            stateBuffer[i++] = sin(relL);
            stateBuffer[i++] = cos(relL);
            stateBuffer[i++] = env->legL->angVel;
            stateBuffer[i++] = env->hip->pos.y / 600.0f;
        }

        sendto(sendSock, (char*)stateBuffer, i * sizeof(float), 0, (sockaddr*)&python, sizeof(python));
    }
};
Data dataManager;

void tempKeyControl(GLFWwindow* w) {
    float delta_angle = 0.05f;

    // Control left leg (joints[1]) with left/right arrows
    if (glfwGetKey(w, GLFW_KEY_LEFT) == GLFW_PRESS) {
        for (auto* e : envs) e->joints[1].targetAngle -= delta_angle;
    } else if (glfwGetKey(w, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        for (auto* e : envs) e->joints[1].targetAngle += delta_angle;
    }
    // Control right leg (joints[0]) with up/down arrows
    if (glfwGetKey(w, GLFW_KEY_DOWN) == GLFW_PRESS) {
        for (auto* e : envs) e->joints[0].targetAngle -= delta_angle;
    } else if (glfwGetKey(w, GLFW_KEY_UP) == GLFW_PRESS) {
        for (auto* e : envs) e->joints[0].targetAngle += delta_angle;
    }
}

// ------------------------ MAIN ------------------------
int main() {
    
    float dt = 1.0/60.0;
    double lastPrintTime = 0.0;
    glfwSwapBuffers(engine.window);
    while(!glfwWindowShouldClose(engine.window)) {
        engine.run();

        tempKeyControl(engine.window);

        // ------ RECIEVE DATA FROM PYTHON -------
        bool gotAction = dataManager.receiveData();

        for (Skeleton* env : envs) {
            env->step(dt);
        }

        // ------ SEND DATA TO PYTHON -------
        if (gotAction) {
            dataManager.sendData();
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    // Exit Program
    glfwTerminate(); return 0;
}