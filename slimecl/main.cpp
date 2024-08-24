

#include "chrono"

#include "wrapper/opencl.hpp"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


#define MAP_WIDTH 1024
#define MAP_HEIGHT 1024
#define NUM_SLIMES 100000

GLFWwindow* window;

Device gpu;
Kernel k_decayTrails, k_updateSlimes;

Memory<float> positions, directions;
Memory<float>* trailMap, *nextTrailMap;
Memory<float> colouredTrail;

GLuint trailMapTexture;


float simDeltaTime = 0.1f; //time step to use each frame
float elapsedTime = 0.0f;

std::chrono::high_resolution_clock::time_point prevFrameEnd;
float prevFrameDuration;

struct SlimeSettings
{
	float slimeSpeed;
	int sensorRadius;
	float sensorAngle;
	float sensorTurnStrength;
	float directionRandomness;
} slimeSettings;

struct TrailSettings
{
	float blurRate;
	float decayRate;
	float r, g, b;
} trailSettings;


void drawMenu()
{
	ImGui::Begin("Settings");
	ImGui::SeparatorText("Simulation Settings");
	ImGui::DragFloat("Sim Delta Time", &simDeltaTime, 0.001f, 0.001f, 10.0f, "%.3f s", ImGuiSliderFlags_AlwaysClamp);

	ImGui::SeparatorText("Slime Settings");
	ImGui::SliderFloat("Speed", &slimeSettings.slimeSpeed, 0.0f, 20.0f, "%.3f pixels/s", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderInt("Sensor Box Radius", &slimeSettings.sensorRadius, 0, 7, "%d", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderAngle("Sensor Angle", &slimeSettings.sensorAngle, 10.0f, 90.0f, "%.0f deg", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Sensor Turn Strengh", &slimeSettings.sensorTurnStrength, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Direction Randomness", &slimeSettings.directionRandomness, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);

	ImGui::SeparatorText("Trail Settings");
	ImGui::SliderFloat("Blur Rate", &trailSettings.blurRate, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Decay Rate", &trailSettings.decayRate, 0.0f, 0.2f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
	float editColour[3] = { trailSettings.r, trailSettings.g, trailSettings.b };
	if (ImGui::ColorEdit3("Trail Colour", editColour))
	{
		trailSettings.r = editColour[0];
		trailSettings.g = editColour[1];
		trailSettings.b = editColour[2];
	}

	ImGui::End();
}

void drawTrails()
{	
	ImGui::Begin("Trail Map");
	ImGui::LabelText("", ("Frame time: " + std::to_string(prevFrameDuration) + "ms").c_str());
	ImGui::BeginChild("trailtexture");
	ImGui::Image((void*)(intptr_t)trailMapTexture, ImGui::GetWindowSize(), ImVec2(0, 1), ImVec2(1, 0));
	ImGui::EndChild();
	ImGui::End();
}

bool init()
{
	srand(std::chrono::system_clock::now().time_since_epoch().count());

	//set up GLFW and glad
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(1920, 1080, "SlimeCL", NULL, NULL);
	if (window == NULL)
	{
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate;
		return false;
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cerr << "Failed to initialize GLAD" << std::endl;
		return false;
	}

	//set up ImGUI
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init();
	ImGui::GetStyle().ScaleAllSizes(2.0f); //make things more readable at high screen res


	//set up OpenCL device, kernels and memory
	//gpu = Device(select_device_with_most_flops());
	gpu = Device(select_device_with_id(1));
	k_decayTrails = Kernel(gpu, MAP_WIDTH * MAP_HEIGHT, "decayTrails");
	k_updateSlimes = Kernel(gpu, NUM_SLIMES, "updateSlimes");

	positions = Memory<float>(gpu, NUM_SLIMES, 2);
	directions = Memory<float>(gpu, NUM_SLIMES, 2);
	trailMap = new Memory<float>(gpu, MAP_WIDTH * MAP_HEIGHT);
	nextTrailMap = new Memory<float>(gpu, MAP_WIDTH * MAP_HEIGHT);
	colouredTrail = Memory<float>(gpu, MAP_WIDTH * MAP_HEIGHT, 4);

	for (int i = 0; i < MAP_WIDTH * MAP_HEIGHT; i++)
	{
		(*trailMap)[i] = 0.0f;
		(*nextTrailMap)[i] = 0.0f;
	}

	(*trailMap).write_to_device();
	(*nextTrailMap).write_to_device();

	for (int i = 0; i < NUM_SLIMES; i++)
	{
		positions.x[i] = (float)(rand() % MAP_WIDTH);
		positions.y[i] = (float)(rand() % MAP_HEIGHT);

		float dx = (float)rand() / RAND_MAX - 0.5f;
		float dy = (float)rand() / RAND_MAX - 0.5f;
		float l = sqrtf(dx * dx + dy * dy);
		directions.x[i] = dx / l;
		directions.y[i] = dy / l;
	}

	positions.write_to_device();
	directions.write_to_device();

	slimeSettings.slimeSpeed = 5.0f; //number of pixels per 1 second of sim time
	slimeSettings.sensorRadius = 2; //size in pixels of sensor box width in each direction from centre (e.g. sensorRadius = 2, box is 5x5)
	slimeSettings.sensorAngle = pi * 0.25f; //angle from direction of slime to left/right sensors
	slimeSettings.sensorTurnStrength = 0.3f; //how strongly the slime turns towards the strongest sensor
	slimeSettings.directionRandomness = 0.1f; //how much randomness changes the slime's direction

	trailSettings.blurRate = 0.2f;
	trailSettings.decayRate = 0.005f;
	trailSettings.r = 0.2f;
	trailSettings.g = 0.6f;
	trailSettings.b = 1.0f;


	//set up texture for displaying trailMap
	glGenTextures(1, &trailMapTexture);
	glBindTexture(GL_TEXTURE_2D, trailMapTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	//frame timing
	prevFrameEnd = std::chrono::high_resolution_clock::now();
	prevFrameDuration = 0.0f;

	return true;
}

bool update()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	k_decayTrails.set_parameters(0, *trailMap, *nextTrailMap, colouredTrail, MAP_WIDTH, MAP_HEIGHT, simDeltaTime,
		trailSettings).run();
	
	k_updateSlimes.set_parameters(0, positions, directions, *trailMap, *nextTrailMap, MAP_WIDTH,
		MAP_HEIGHT, simDeltaTime, elapsedTime, slimeSettings).run();
	
	std::swap(trailMap, nextTrailMap);

	//copy updated trail back from device (to then send back to the device in the texture...)
	colouredTrail.read_from_device();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, MAP_WIDTH, MAP_HEIGHT, 0, GL_RGBA, GL_FLOAT, colouredTrail.data());
	
	drawMenu();
	drawTrails();

	elapsedTime += simDeltaTime;

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	std::chrono::high_resolution_clock::time_point currentFrameEnd = std::chrono::high_resolution_clock::now();
	prevFrameDuration = (currentFrameEnd - prevFrameEnd).count() / 1e6; //convert nanoseconds to milliseconds
	prevFrameEnd = currentFrameEnd;

	return true;
}

bool destroy()
{
	delete trailMap;
	delete nextTrailMap;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();

	return true;
}


int main()
{
	if (!init())
	{
		std::cout << "Error during initialisation, exiting" << std::endl;
		return -1;
	}

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		if (!update())
		{
			std::cout << "Error during update, exiting" << std::endl;
			break;
		}

		glfwSwapBuffers(window);
	}

	destroy();
	
	return 0;
}