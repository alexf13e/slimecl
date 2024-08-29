

#include "chrono"

#include "wrapper/opencl.hpp"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


GLFWwindow* window;

Device gpu;
Kernel k_decayTrails, k_updateSlimes;

Memory<float> positions, directions;
Memory<float>* trailMap, *nextTrailMap;
Memory<float> colouredTrail;
Memory<uint> randomSeeds;

GLuint trailMapTexture;

int mapWidth;
int mapHeight;
int numSlimes;
float simDeltaTime; //time step to use each frame
bool simRunning;

std::chrono::high_resolution_clock::time_point prevFrameEnd;
float prevFrameDuration;

struct SlimeSettings
{
	float slimeSpeed;
	int sensorRadius;
	float sensorAngle;
	float sensorTurnStrength;
	float directionRandomness;
	int depositWidth;
} slimeSettings;

struct TrailSettings
{
	float blurRate;
	float decayRate;
	float r, g, b;
} trailSettings;


bool initSim();
bool destroySim();

void drawMenu()
{
	ImGui::Begin("Settings");
	ImGui::SeparatorText("Simulation Settings");

	if (!simRunning)
	{
		if (ImGui::Button("Start Simulation"))
		{
			if (initSim()) simRunning = true;
		}

		if (ImGui::InputInt("Map Width", &mapWidth))
		{
			mapWidth = std::min(std::max(mapWidth, 10), 10000);
		}

		if (ImGui::InputInt("Map Height", &mapHeight))
		{
			mapHeight = std::min(std::max(mapHeight, 10), 10000);
		}

		if (ImGui::InputInt("Number of Slimes", &numSlimes))
		{
			numSlimes = std::min(std::max(numSlimes, 1), (int)1e6);
		}
	}
	else
	{
		if (ImGui::Button("Stop Simulation"))
		{
			destroySim();
		}

		ImGui::LabelText("Map Width", std::to_string(mapWidth).c_str());
		ImGui::LabelText("Map Height", std::to_string(mapHeight).c_str());
		ImGui::LabelText("NUmber of Slimes", std::to_string(numSlimes).c_str());
	}

	ImGui::DragFloat("Sim Delta Time", &simDeltaTime, 0.001f, 0.001f, 10.0f, "%.3f s", ImGuiSliderFlags_AlwaysClamp);

	ImGui::SeparatorText("Slime Settings");
	ImGui::SliderFloat("Speed", &slimeSettings.slimeSpeed, 0.0f, 20.0f, "%.3f pixels/s", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderInt("Sensor Box Radius", &slimeSettings.sensorRadius, 0, 7, "%d", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderAngle("Sensor Angle", &slimeSettings.sensorAngle, 10.0f, 90.0f, "%.0f deg", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Sensor Turn Strengh", &slimeSettings.sensorTurnStrength, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Direction Randomness", &slimeSettings.directionRandomness, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderInt("Deposit Width", &slimeSettings.depositWidth, 0, 5, "%d", ImGuiSliderFlags_AlwaysClamp);

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
	ImGui::BeginChild("trailimage");

	//make the image sit nicely in the window with the correct aspect ratio
	ImVec2 windowSize = ImGui::GetWindowSize();
	float windowHeightPerWidth = windowSize.y / windowSize.x;
	float mapHeightPerWidth = (float)mapHeight / mapWidth;
	ImVec2 displayedImageSize;
	if (windowHeightPerWidth > mapHeightPerWidth)
	{
		//window is more vertical than image, so fill horizontally
		displayedImageSize = { windowSize.x, windowSize.x * mapHeightPerWidth };
	}
	else
	{
		//window is more horizontal than image, so fill vertically
		displayedImageSize = { windowSize.y / mapHeightPerWidth, windowSize.y };
	}
	ImGui::Image((void*)(intptr_t)trailMapTexture, displayedImageSize, ImVec2(0, 1), ImVec2(1, 0));
	ImGui::EndChild();
	ImGui::End();
}

bool initOnce()
{
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


	//set default parameters
	//want to remember parameters between simulation resets, so only init once
	mapWidth = 1024;
	mapHeight = 1024;
	numSlimes = 1000;
	simDeltaTime = 0.1f; //time step to use each frame
	simRunning = false;

	slimeSettings.slimeSpeed = 5.0f; //number of pixels per 1 second of sim time
	slimeSettings.sensorRadius = 2; //size in pixels of sensor box width in each direction from centre (e.g. sensorRadius = 2, box is 5x5)
	slimeSettings.sensorAngle = pi * 0.25f; //angle from direction of slime to left/right sensors
	slimeSettings.sensorTurnStrength = 0.3f; //how strongly the slime turns towards the strongest sensor
	slimeSettings.directionRandomness = 0.1f; //how much randomness changes the slime's direction
	slimeSettings.depositWidth = 0; //how many pixels each side of the slime to leave a trail on

	trailSettings.blurRate = 0.2f;
	trailSettings.decayRate = 0.005f;
	trailSettings.r = 0.2f;
	trailSettings.g = 1.0f;
	trailSettings.b = 0.6f;


	//gpu = Device(select_device_with_most_flops());
	gpu = Device(select_device_with_id(1));

	//set up texture for displaying trailMap
	glGenTextures(1, &trailMapTexture);
	glBindTexture(GL_TEXTURE_2D, trailMapTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	return true;
}

bool initSim()
{
	int mapSize = mapWidth * mapHeight;
	positions = Memory<float>(gpu, numSlimes, 2);
	directions = Memory<float>(gpu, numSlimes, 2);
	trailMap = new Memory<float>(gpu, mapSize);
	nextTrailMap = new Memory<float>(gpu, mapSize);
	colouredTrail = Memory<float>(gpu, mapSize, 4);
	randomSeeds = Memory<uint>(gpu, numSlimes);

	k_decayTrails = Kernel(gpu, mapSize, "decayTrails");
	k_updateSlimes = Kernel(gpu, numSlimes, "updateSlimes");

	for (int i = 0; i < mapSize; i++)
	{
		(*trailMap)[i] = 0.0f;
		(*nextTrailMap)[i] = 0.0f;
	}

	(*trailMap).write_to_device();
	(*nextTrailMap).write_to_device();

	for (int i = 0; i < numSlimes; i++)
	{
		positions.x[i] = rand() % mapWidth;
		positions.y[i] = rand() % mapHeight;

		float dx = (float)rand() / RAND_MAX - 0.5f;
		float dy = (float)rand() / RAND_MAX - 0.5f;
		float l = sqrtf(dx * dx + dy * dy);
		directions.x[i] = dx / l;
		directions.y[i] = dy / l;

		randomSeeds[i] = rand();
	}

	positions.write_to_device();
	directions.write_to_device();
	randomSeeds.write_to_device();


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

	if (simRunning)
	{
		k_decayTrails.set_parameters(0, *trailMap, *nextTrailMap, colouredTrail, mapWidth, mapHeight, simDeltaTime,
			trailSettings).run();

		k_updateSlimes.set_parameters(0, positions, directions, *trailMap, *nextTrailMap, randomSeeds, mapWidth,
			mapHeight, simDeltaTime, slimeSettings, numSlimes).run();

		std::swap(trailMap, nextTrailMap);

		//copy updated trail back from device (to then send back to the device in the texture...)
		colouredTrail.read_from_device();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mapWidth, mapHeight, 0, GL_RGBA, GL_FLOAT, colouredTrail.data());
		
		drawTrails();
	}
	
	drawMenu();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	std::chrono::high_resolution_clock::time_point currentFrameEnd = std::chrono::high_resolution_clock::now();
	prevFrameDuration = (currentFrameEnd - prevFrameEnd).count() / 1e6; //convert nanoseconds to milliseconds
	prevFrameEnd = currentFrameEnd;

	return true;
}

bool destroySim()
{
	simRunning = false;
	delete trailMap;
	delete nextTrailMap;

	return true;
}

bool destroy()
{
	if (simRunning) destroySim();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();

	return true;
}

int main()
{
	srand(std::chrono::system_clock::now().time_since_epoch().count());

	if (!initOnce())
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