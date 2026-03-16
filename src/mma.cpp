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


// ------------------------ Engine & Constants ------------------------
vec2 g(0.0f, -980.6f);
struct Engine {
    GLFWwindow* window;
    int WIDTH = 800, HEIGHT = 600;

    Engine () {
        if (!glfwInit()) {
            cerr << "Failed to initialize GLFW" << endl; exit(EXIT_FAILURE);
        }

        window = glfwCreateWindow(WIDTH, HEIGHT, "2D atom sim by kavan", nullptr, nullptr);
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

};
Engine engine;

vec2 mousePos; bool mouseDown = false;

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
    float targetAngle = 0.0f;
    float stiffness = 1.0f;

    Joint(Bone* a, Bone* b, vec2 anchorA, vec2 anchorB, float targetAngle=3.14f) : A(a), B(b), anchorA_local(anchorA), anchorB_local(anchorB), targetAngle(targetAngle) { }

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

    Bone *body,*head,*hip,*armR,*armL,*forearmR,*forearmL,*legR,*legL,*calfR,*calfL;

    Skeleton(){ init(); }

    void init() {
        // ---- Bones ----
        body     = new Bone({400,200},1.45786f,25,25,24.0);
        head     = new Bone({400,200},6.45816f,15,15,6.0);
        hip      = new Bone({400,200},6.35747f,15,15,10.0);

        armR     = new Bone({250,200},4.44994f,20,7,20.5);
        armL     = new Bone({550,200},5.48231f,20,7,20.5);

        forearmR = new Bone({250,200},4.44997f,20,7,10.5);
        forearmL = new Bone({550,200},5.48231f,20,7,10.5);

        legR     = new Bone({250,200},7.3253f,25,10,7.0);
        legL     = new Bone({550,200},8.36471f,25,10,7.0);

        calfR    = new Bone({250,200},7.32835f,20,7,3.5);
        calfL    = new Bone({550,200},8.36382f,20,7,3.5);

        bones = {body,head,hip,armR,armL,forearmR,forearmL,legR,legL,calfR,calfL};

        // Joints 
        joints.push_back(Joint(body,head,{body->halfLength,0},{0,-head->halfLength}, 5.0f));
        joints.push_back(Joint(body,hip,{-body->halfLength,0},{0,hip->halfLength}, 4.95f));

        joints.push_back(Joint(body,armR,{body->halfLength*0.34f,-body->radius*0.94f},{armR->halfLength,0},3.0f));
        joints.push_back(Joint(body,armL,{body->halfLength*0.34f, body->radius*0.94f},{armL->halfLength,0},4.0f));

        joints.push_back(Joint(armR,forearmR,{-armR->halfLength,0},{forearmR->halfLength,0},0.0f));
        joints.push_back(Joint(armL,forearmL,{-armL->halfLength,0},{forearmL->halfLength,0},0.0f));

        joints.push_back(Joint(hip,legR,{-hip->radius*0.71f,-hip->radius*0.71f},{legR->halfLength,0},1.0f));
        joints.push_back(Joint(hip,legL,{ hip->radius*0.71f,-hip->radius*0.71f},{legL->halfLength,0},2.0f));

        joints.push_back(Joint(legR,calfR,{-legR->halfLength,0},{calfR->halfLength,0},0.0f));
        joints.push_back(Joint(legL,calfL,{-legL->halfLength,0},{calfL->halfLength,0},0.0f));

        // Initial constraint alignment
        for(auto& j : joints){
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

    float recvBuffer[20], stateBuffer[20];
    int frameCount=0, skipAmount=5;
    float dt=1/60.f;

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
        u_long mode = 1;
        ioctlsocket(sock, FIONBIO, &mode);
    }
    void receiveData(Skeleton* sk) {
        if (frameCount % skipAmount == 0) {
            int bytesRead = recv(sock, (char*)recvBuffer, sizeof(recvBuffer), 0);
            if (bytesRead == sizeof(recvBuffer)) {
                for (int j = 0; j < sk->joints.size(); j++) {
                    sk->joints[j].targetAngle = fmod(recvBuffer[j], 2.0f * glm::pi<float>());
                    sk->joints[j].stiffness   = recvBuffer[j + 10];
                }
            }
        }
    }
    void sendData(Skeleton* sk) {
        if (frameCount % skipAmount == 0) {
            int i = 0;

            // 11 bones × 6 floats = 66
            // for (Bone* b : sk->bones) {
            //     stateBuffer[i++] = b->pos.x;
            //     stateBuffer[i++] = b->pos.y;
            //     stateBuffer[i++] = b->vel.x;
            //     stateBuffer[i++] = b->vel.y;
            //     stateBuffer[i++] = b->angle;
            //     stateBuffer[i++] = b->angVel;
            // }

            // 10 joints × 2 floats = 20
            for (Joint& jt : sk->joints) {
                stateBuffer[i++] = jt.targetAngle;
            }
            for (Joint& jt : sk->joints) {
                stateBuffer[i++] = jt.stiffness;
            }

            sendto(sendSock, (char*)stateBuffer, i * sizeof(float), 0, (sockaddr*)&python, sizeof(python));
        }
        frameCount++;
    }
};
Data dataManager;


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

// ------------------------ MAIN ------------------------
int main() {
    Skeleton* figure = new Skeleton();
    Bone* dragBone=nullptr; vec2 dragOffset;
    glfwSetCursorPosCallback(engine.window,   cursor_position_callback);
    glfwSetMouseButtonCallback(engine.window, mouse_button_callback);

    for (Joint& j : figure->joints) cout<<"Joint: targetAngle="<<j.targetAngle<<" stiffness="<<j.stiffness<<endl;

    float dt = 1.0/60.0;
    double lastPrintTime = 0.0;
    while(!glfwWindowShouldClose(engine.window)) {
        engine.run();

        // ------ RECIEVE DATA FROM PYTHON -------
        dataManager.receiveData(figure);



        // --- Select drag bone ---
        if (mouseDown && !dragBone) {
            for (Bone* b : figure->bones)
                if (length(mousePos - b->pos) < 30.f) {
                    dragBone = b;
                    vec2 d = mousePos - b->pos;
                    dragOffset = {d.x*cos(-b->angle)-d.y*sin(-b->angle),
                                d.x*sin(-b->angle)+d.y*cos(-b->angle)};
                    break;
                };
        } else if (!mouseDown) dragBone = nullptr;
        // ---- Euler Integrate Gravity & Draw ----
        for (Bone* b : figure->bones) {
            b->vel += g * dt;
            // air resistance, joint friction, yea you get it :D
            b->vel    *= 1.0f / (1.0f + 0.7f * dt);
            b->angVel *= 1.0f / (1.0f + 0.7f * dt);

            b->draw();

            if (dragBone == b) b->vel = (mousePos - b->pos) / dt;
        }
        // ---- Solve Joints & Apply Torque---
        for(int i=0;i<100;i++) {
            for (Joint& j : figure->joints){
                j.solve(dt);
                j.applyTorque(dt);
            }
        }
        // ---- Euler Integrate ----
        for (Bone* b : figure->bones) {
            b->pos += b->vel * dt;
            b->angle += b->angVel * dt;
            figure->checkBorderCollision(b);
        }



        // ------ SEND DATA TO PYTHON -------
        dataManager.sendData(figure);


        // ------ PRINT STATS -------
        if (glfwGetTime() - lastPrintTime > 5.0) {
            glfwSwapBuffers(engine.window);
            for (Bone* j : figure->bones)
                cout<<"Bone angle: "<<j->angle<<endl;
            
            cout<<"--------------------------------"<<endl;
            lastPrintTime = glfwGetTime();
        }

        
        
        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    // Exit Program
    glfwTerminate(); return 0;
}