#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;

// --- variables --- 
float orbitDistance = 15.0f;

// --- engine ---
struct Engine {

    GLFWwindow* window;
    int WIDTH = 800, HEIGHT = 600;

    Engine () {
        // --- Init GLFW ---
        if (!glfwInit()) {
            cerr << "failed to init glfw, LOL";
            exit(EXIT_FAILURE);
        }

        // --- Create Window ---
        window = glfwCreateWindow(WIDTH, HEIGHT, "2D atom sim by kavan", nullptr, nullptr);
        if (!window) {
            cerr << "failed to create window, LOLOLOL";
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

        // set origin to centre
        double halfWidth = WIDTH / 2.0f, halfHeight = HEIGHT / 2.0f;
        glOrtho(-halfWidth, halfWidth, -halfHeight, halfHeight, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }


};
Engine engine;




// --- main ---
int main () {


    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
} 