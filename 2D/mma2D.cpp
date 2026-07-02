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
#include <ctime>
#include <unistd.h>
using namespace glm; using namespace std;

// ------------------------ Engine & Constants ----------------------
const int MAX_ENV = 20, STATE_DIM = 39, ACTION_DIM = 10;
int NUM_ENV = 20;
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
        error -= 2.0f * glm::pi<float>() * floorf((error + glm::pi<float>()) / (2.0f * glm::pi<float>()));

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
    int idx;
    int impulse_timer = 0;
    int next_impulse_delay = 90;
    float impulse_max = 0.0f;

    Bone *head, *body, *armL, *armR, *forearmL, *forearmR, *hip, *legR, *legL, *calfR, *calfL;

    Skeleton(vec2 p, int idx) : startPos(p), idx(idx) { init(p); }

    void init(vec2 p, float jitter = 0.4f) {
        //srand(time(0) + idx * 1000);
        auto rnd = [&](){ return ((rand() % 1000) / 1000.0f - 0.5f) * 2.0f * jitter; };

        impulse_timer = 0;
        vec2 offset = p - vec2(400.0f, 60.0f);
        for(auto b : bones) delete b;
        joints.clear();

        // ---- Bones with jittered initial angles ----
        head     = new Bone(vec2(400,200) + offset, 0.0f + rnd(),  12, 11,  5.5); 
        body     = new Bone(vec2(400,130) + offset, 1.50f + rnd(),  28, 15, 24.0); 
        hip      = new Bone(vec2(400,200) + offset, 6.36f + rnd(),  13, 13,  7.5); 

        armR     = new Bone(vec2(250,200) + offset, 4.45f + rnd(),  20,  7,  2.0); 
        armL     = new Bone(vec2(550,200) + offset, 5.48f + rnd(),  20,  7,  2.0);
        forearmR = new Bone(vec2(250,200) + offset, 4.45f + rnd(),  18,  6,  1.5);
        forearmL = new Bone(vec2(550,200) + offset, 5.48f + rnd(),  18,  6,  1.5);

        legR     = new Bone(vec2(250,200) + offset, 7.33f + rnd(),  28,  8,  9.5); 
        legL     = new Bone(vec2(550,200) + offset, 8.36f + rnd(),  28,  8,  9.5);
        calfR    = new Bone(vec2(250,200) + offset, 7.33f + rnd(),  24,  7,  3.5); 
        calfL    = new Bone(vec2(550,200) + offset, 8.36f + rnd(),  24,  7,  3.5);

        bones = {head,body,armL,armR,forearmL,forearmR,legL,legR,calfL,calfR,hip};

        // Joints with jittered target angles
        joints.push_back(Joint(body,head,{body->halfLength,0},{-head->halfLength,0}, 0.0f + rnd()));
        joints.push_back(Joint(body,hip,{-body->halfLength,0},{0,hip->halfLength}, 4.95f + rnd()));
        joints.push_back(Joint(body,armR,{body->halfLength*0.7f,-body->radius*0.94f},{armR->halfLength*0.95,0}, 3.0f + rnd()));
        joints.push_back(Joint(body,armL,{body->halfLength*0.7f, body->radius*0.94f},{armL->halfLength*0.95,0}, 4.0f + rnd()));
        joints.push_back(Joint(armR,forearmR,{-armR->halfLength,0},{forearmR->halfLength,0}, 0.0f + rnd()));
        joints.push_back(Joint(armL,forearmL,{-armL->halfLength,0},{forearmL->halfLength,0}, 0.0f + rnd()));
        joints.push_back(Joint(hip,legR,{-hip->radius*0.71f,-hip->radius*0.71f},{legR->halfLength,0}, 1.0f + rnd()));
        joints.push_back(Joint(hip,legL,{ hip->radius*0.71f,-hip->radius*0.71f},{legL->halfLength,0}, 2.0f + rnd()));
        joints.push_back(Joint(legR,calfR,{-legR->halfLength,0},{calfR->halfLength,0}, 0.0f + rnd()));
        joints.push_back(Joint(legL,calfL,{-legL->halfLength,0},{calfL->halfLength,0}, 0.0f + rnd()));

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
        for(int i=0;i<50;i++) {
            for (Joint& j : joints){
                j.solve(dt);
                j.applyTorque(dt);
            }
        }
        // ---- Euler Integrate ----
        for (Bone* b : bones) {
            b->pos += b->vel * dt;
            b->angVel = glm::clamp(b->angVel, -50.0f, 50.0f);
            b->vel    = glm::clamp(b->vel, vec2(-6000.0f), vec2(6000.0f));
            b->angle += b->angVel * dt;
            checkBorderCollision(b);
        }

        // magic force — set to 0.0f for training, nonzero to test friction
        body->vel.x += 0.0f * dt;

        // foot friction — only calf bottom endpoints, only when grounded
        for (Bone* b : {calfL, calfR}) {
            vec2 foot = b->pos - vec2(cos(b->angle), sin(b->angle)) * b->halfLength;
            if (foot.y < b->radius + 2.0f)
                b->vel.x *= 0.05f;
        }
        maybeApplyImpulse();
    }
    void maybeApplyImpulse() {
        if (impulse_max <= 0.0f) return;
        impulse_timer++;
        if (impulse_timer < next_impulse_delay) return;
        impulse_timer = 0;
        next_impulse_delay = 60 + rand() % 121;

        Bone* targets[] = { body, head, armL, armR, legL, legR };
        float weights[] = { 0.5f, 0.2f, 0.1f, 0.1f, 0.05f, 0.05f };

        float r = (float)rand() / RAND_MAX;
        float cum = 0.0f;
        Bone* target = body;
        for (int i = 0; i < 6; i++) {
            cum += weights[i];
            if (r < cum) { target = targets[i]; break; }
        }

        float dir = (rand() % 2 == 0) ? 1.0f : -1.0f;
        float mag = (0.5f + 0.5f * ((float)rand() / RAND_MAX)) * impulse_max;
        // mag is calibrated as "velocity the body would get if hit" (matches the original,
        // pre-normalization system exactly when target==body). Lighter bones still get
        // proportionally more velocity for the same momentum hit (real physics), same ratio
        // as before -- this is just a relabeling so impulse_max is comparable to old values.
        target->vel.x += dir * mag * body->mass / target->mass;
    }

    void reset() {
        init(startPos, 0.3f);
    }
    void checkBorderCollision(Bone* b) {
        float restitution = 0.2f, slop = 0.01f, percent = 0.8f, friction = 0.99f;

        vec2 dir(cos(b->angle), sin(b->angle));
        vec2 offset = dir * b->halfLength;
        vec2 points[2] = { b->pos - offset, b->pos + offset };

        for (vec2 contact : points) {
            vec2 normal;
            float penetration = 0.0f;
            bool collided = false;

            // if (contact.x < b->radius) {
            //     normal = vec2(1,0);
            //     penetration = b->radius - contact.x;
            //     collided = true;
            // }
            // else if (contact.x > engine.WIDTH - b->radius) {
            //     normal = vec2(-1,0);
            //     penetration = contact.x - (engine.WIDTH - b->radius);
            //     collided = true;
            // }
            // else if (contact.y < b->radius) {
            //     normal = vec2(0,1);
            //     penetration = b->radius - contact.y;
            //     collided = true;
            // }
            // else if (contact.y > engine.HEIGHT - b->radius) {
            //     normal = vec2(0,-1);
            //     penetration = contact.y - (engine.HEIGHT - b->radius);
            //     collided = true;
            // }
            if (contact.y < b->radius) {
                normal = vec2(0,1);
                penetration = b->radius - contact.y;
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


            // --- 3. Positional Correction ---
            vec2 correction = normal * percent * fmax(penetration - slop, 0.0f);
            b->pos += correction;
        }
    }
};
vector<Skeleton*> envs {
    new Skeleton(vec2(100,60), 0),
    new Skeleton(vec2(100,60), 1),
    new Skeleton(vec2(100,60), 2),
    new Skeleton(vec2(100,60), 3),
    new Skeleton(vec2(250,60), 4),
    new Skeleton(vec2(250,60), 5),
    new Skeleton(vec2(250,60), 6),
    new Skeleton(vec2(250,60), 7),
    new Skeleton(vec2(400,60), 8),
    new Skeleton(vec2(400,60), 9),
    new Skeleton(vec2(400,60), 10),
    new Skeleton(vec2(400,60), 11),
    new Skeleton(vec2(550,60), 12),
    new Skeleton(vec2(550,60), 13),
    new Skeleton(vec2(550,60), 14),
    new Skeleton(vec2(550,60), 15),
    new Skeleton(vec2(700,60), 16),
    new Skeleton(vec2(700,60), 17),
    new Skeleton(vec2(700,60), 18),
    new Skeleton(vec2(700,60), 19),
};


// ------------------------ Interactive Mass ------------------------
struct CircleMass {
    vec2 pos=vec2(400,300), vel=vec2(0), dragOff=vec2(0);
    float mass=0, radius=20;
    bool active=false, dragged=false;
    bool cWas=false, vWas=false, upWas=false, dnWas=false, mbWas=false;

    void collide(Bone* b) {
        vec2 d(cos(b->angle),sin(b->angle));
        vec2 p1=b->pos-d*b->halfLength, p2=b->pos+d*b->halfLength, seg=p2-p1;
        float l2=dot(seg,seg);
        vec2 cp=p1+(l2>0?glm::clamp(dot(pos-p1,seg)/l2,0.f,1.f):0.f)*seg;
        vec2 delta=pos-cp; float dist=glm::length(delta), minD=radius+b->radius;
        if(dist>=minD||dist<1e-6f) return;
        vec2 n=delta/dist; float pen=minD-dist;
        vec2 r=cp+n*b->radius-b->pos;
        float rvn=dot(vel-(b->vel+engine.cross(b->angVel,r)),n);
        if(rvn>0) return;
        float invM=mass>0?1.f/mass:0.f, rcn=engine.cross(r,n);
        float j=-1.2f*rvn/(b->invMass+rcn*rcn*b->invInertia+invM);
        b->vel-=n*j*b->invMass; b->angVel-=engine.cross(r,n*j)*b->invInertia;
        if(mass>0) vel+=n*j*invM;
        float corr=std::max(pen-0.01f,0.f)*0.8f;
        float tot=b->invMass+invM;
        if(tot>0){ b->pos-=n*corr*b->invMass/tot; pos+=n*corr*(mass>0?invM/tot:1.f); }
    }

    void update(GLFWwindow* w, float dt, vector<Skeleton*>& envs) {
        bool c=glfwGetKey(w,GLFW_KEY_C)==GLFW_PRESS;
        if(c&&!cWas){double mx,my;glfwGetCursorPos(w,&mx,&my);pos=vec2(mx,engine.HEIGHT-my);vel=vec2(0);mass=0;active=true;}
        cWas=c;
        bool v=glfwGetKey(w,GLFW_KEY_V)==GLFW_PRESS;
        if(v&&!vWas) active=false; vWas=v;
        if(!active) return;

        bool up=glfwGetKey(w,GLFW_KEY_UP)==GLFW_PRESS;
        if(up&&!upWas){mass+=2;cout<<"mass="<<mass<<"kg\n";} upWas=up;
        bool dn=glfwGetKey(w,GLFW_KEY_DOWN)==GLFW_PRESS;
        if(dn&&!dnWas){mass=std::max(0.f,mass-2);cout<<"mass="<<mass<<"kg\n";} dnWas=dn;

        double mx,my; glfwGetCursorPos(w,&mx,&my);
        vec2 mouse(mx,engine.HEIGHT-my);
        bool mb=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS;
        if(mb&&!mbWas&&glm::length(mouse-pos)<radius){dragged=true;dragOff=pos-mouse;}
        if(!mb) dragged=false; mbWas=mb;

        if(dragged){vel=(mouse+dragOff-pos)/dt;pos=mouse+dragOff;}
        else{
            if(mass>0) vel+=g*dt;
            vel*=1.f/(1.f+0.5f*dt);
            pos+=vel*dt;
            pos=glm::clamp(pos,vec2(radius),vec2(engine.WIDTH-radius,engine.HEIGHT-radius));
        }
        for(auto* sk:envs) for(Bone* b:sk->bones) collide(b);
        engine.drawCircle(pos,radius,vec3(1,.5f,0));
    }
} circleMass;


// ------------------------ File Connections ------------------------
struct Data {
    int sock, sendSock;
    sockaddr_in server, python;
    const static int revSize = MAX_ENV * ACTION_DIM;
    const static int sendSize = MAX_ENV * STATE_DIM;

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
        } else if (recvBuffer[0] == -150.0f) {
            glfwSwapBuffers(engine.window);
            return false;
        } else if (recvBuffer[0] == -69.0f) {
            envs[(int)recvBuffer[1]]->reset();
            return false;
        } else if (recvBuffer[0] == -68.0f) {
            for (Skeleton* env : envs) env->impulse_max = recvBuffer[1];
            return false;
        } else if (bytesRead == (int)(NUM_ENV * ACTION_DIM * sizeof(float))) {
            int i = 0;
            for (Skeleton* env : envs) {
                for (int j = 0; j < ACTION_DIM && j < env->joints.size(); j++) {
                    env->joints[j].targetAngle = recvBuffer[i++];
                }
            }
        }
        return false;
    }
    void sendData() {
        int i = 0;
        for (Skeleton* env : envs) {
            for (Bone* b : env->bones) {
                stateBuffer[i++] = sin(b->angle);
                stateBuffer[i++] = cos(b->angle);
                stateBuffer[i++] = b->angVel;
            }
            stateBuffer[i++] = env->hip->pos.y / 600.0f;
            stateBuffer[i++] = env->hip->pos.x / 800.0f;
            stateBuffer[i++] = env->hip->vel.y / 600.0f;
            stateBuffer[i++] = env->hip->vel.x / 800.0f;
            stateBuffer[i++] = (env->calfL->pos.y - fabsf(sin(env->calfL->angle)) * env->calfL->halfLength <= env->calfL->radius + 1.0f) ? 1.0f : 0.0f;
            stateBuffer[i++] = (env->calfR->pos.y - fabsf(sin(env->calfR->angle)) * env->calfR->halfLength <= env->calfR->radius + 1.0f) ? 1.0f : 0.0f;
        }

        sendto(sendSock, (char*)stateBuffer, i * sizeof(float), 0, (sockaddr*)&python, sizeof(python));
    }
};
Data dataManager;

void tempKeyControl(GLFWwindow* w) {
    static int jointIdx = 0;
    static bool upPressed = false, downPressed = false, yPressed = false, bPressed = false;
    static vector<Skeleton*> allEnvs = envs;
    bool b = glfwGetKey(w, GLFW_KEY_B) == GLFW_PRESS;
    if (b && !bPressed) { NUM_ENV = (NUM_ENV == 20) ? 1 : 20; envs = vector<Skeleton*>(allEnvs.begin(), allEnvs.begin() + NUM_ENV); cout << "numEnv=" << NUM_ENV << endl; }
    bPressed = b;
    extern int frameInterval;
    bool y = glfwGetKey(w, GLFW_KEY_Y) == GLFW_PRESS;
    if (y && !yPressed) { frameInterval = (frameInterval == 250) ? 1 : (frameInterval == 1) ? 2000 : 250; cout << "frameInterval=" << frameInterval << endl; }
    yPressed = y;
    float delta = 0.005f;
    int n = envs[0]->joints.size();
    // cout<<"target angle:" <<envs[0]->joints[jointIdx].targetAngle<<endl;

    // Cycle through joint indices (single press detection)
    bool up = glfwGetKey(w, GLFW_KEY_UP) == GLFW_PRESS;
    if (up && !upPressed && !circleMass.active) {
        jointIdx = (jointIdx + 1) % n;
        cout << "Selected Joint Index: " << jointIdx << endl;
    }
    upPressed = up;

    bool down = glfwGetKey(w, GLFW_KEY_DOWN) == GLFW_PRESS;
    if (down && !downPressed && !circleMass.active) {
        jointIdx = (jointIdx - 1 + n) % n;
        cout << "Selected Joint Index: " << jointIdx << endl;
    }
    downPressed = down;

    // Adjust target angle for selected joint (continuous hold)
    if (glfwGetKey(w, GLFW_KEY_LEFT) == GLFW_PRESS)
        for (auto* e : envs) e->joints[jointIdx].targetAngle -= delta;
    if (glfwGetKey(w, GLFW_KEY_RIGHT) == GLFW_PRESS)
        for (auto* e : envs) e->joints[jointIdx].targetAngle += delta;

    // 1-9, 0 = apply impulse toward cursor on selected joint (200 per key, 0=2000)
    static bool numPressed[10] = {};
    int numKeys[] = {GLFW_KEY_1,GLFW_KEY_2,GLFW_KEY_3,GLFW_KEY_4,GLFW_KEY_5,
                     GLFW_KEY_6,GLFW_KEY_7,GLFW_KEY_8,GLFW_KEY_9,GLFW_KEY_0};
    float mags[]  = {1500,2000,3000,4000,5000,6000,7000,8000,9000,10000};

    double cx, cy;
    glfwGetCursorPos(w, &cx, &cy);
    cy = engine.HEIGHT - cy;

    for (int k = 0; k < 10; k++) {
        bool pressed = glfwGetKey(w, numKeys[k]) == GLFW_PRESS;
        if (pressed && !numPressed[k]) {
            for (auto* e : envs) {
                Bone* b = e->joints[jointIdx].B;
                vec2 dir = vec2(cx, cy) - b->pos;
                float len = glm::length(dir);
                if (len > 0) dir /= len;
                b->vel += dir * mags[k];
            }
            cout << "Impulse " << mags[k] << " on joint " << jointIdx << endl;
        }
        numPressed[k] = pressed;
    }
}

// ------------------------ MAIN ------------------------
int timer = 0;
int frameInterval = 250;
int main() {
    srand(time(0) * 1234567891ULL ^ (uint64_t)clock());

    for (Skeleton* env : envs) env->impulse_max = 0.0f;

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
        circleMass.update(engine.window, dt, envs);

        // ------ SEND DATA TO PYTHON -------
        if (gotAction)
            dataManager.sendData();
        
        timer++;
        if (timer % frameInterval == 0)
            glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    // Exit Program
    glfwTerminate(); return 0;
}
