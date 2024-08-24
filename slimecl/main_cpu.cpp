
#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include "glm/glm.hpp"
#include "glm/gtx/rotate_vector.hpp"

#include "chrono"


#define MAPWIDTH 512
#define MAPHEIGHT 512

class SlimeCL : public olc::PixelGameEngine
{
public:
	SlimeCL()
	{
		sAppName = "Example";
	}

	std::vector<glm::vec2> positions, directions;
	float* trailMap, *nextTrailMap;

	float simTickTime = 0.1f;
	float slimeSpeed = 5.0f;
	int sensorRadius = 2; //size in pixels of sensor box width in each direction from centre (e.g. sensorRadius = 2, box is 5x5)
	int sensorSize = (1 + 2 * sensorRadius) * (1 + 2 * sensorRadius); // number of pixels sensor covers
	float sensorAngle = glm::pi<float>() * 0.25f;
	float sensorDist = 3.5f * sensorRadius;
	float sensorTurnStrength = 0.01f;
	float blurRate = 0.2f;
	float decayRate = 0.005f;
	float directionRandomness = 0.2f;

	void addSlime(const glm::vec2& pos)
	{
		positions.push_back(pos);
		float dx = (float)rand() / RAND_MAX - 0.5f;
		float dy = (float)rand() / RAND_MAX - 0.5f;
		directions.push_back(glm::normalize(glm::vec2(dx, dy)));
	}

	glm::vec2 wrapPos(glm::vec2 pos)
	{
		while(pos.x < 0) pos.x += MAPWIDTH;
		while(pos.x >= MAPWIDTH) pos.x -= MAPWIDTH;

		while (pos.y < 0) pos.y += MAPHEIGHT;
		while (pos.y >= MAPHEIGHT) pos.y -= MAPHEIGHT;

		return pos;
	}

	void wrap(int& x, const int& period)
	{
		while (x < 0) x += period;
		while (x >= period) x -= period;
	}

	void drawTrails()
	{
		for (int i = 0; i < MAPWIDTH * MAPHEIGHT; i++)
		{
			int x = i % MAPWIDTH;
			int y = i / MAPWIDTH;
			Draw(x, y, { (uint8_t)(trailMap[i] * 255), (uint8_t)(trailMap[i] * 255), (uint8_t)(trailMap[i] * 255) });
		}
	}

	void decayTrails()
	{
		glm::ivec2 s1;
		glm::ivec2 s2;
		glm::ivec2 s3;
		glm::ivec2 s4;
		glm::ivec2 s6;
		glm::ivec2 s7;
		glm::ivec2 s8;
		glm::ivec2 s9;
		int x, y;

		for (int mapPixelIndex = 0; mapPixelIndex < MAPWIDTH * MAPHEIGHT; mapPixelIndex++)
		{
			x = mapPixelIndex % MAPWIDTH;
			y = mapPixelIndex / MAPWIDTH;

			s1.x = x - 1; s1.y = y - 1;
			s2.x = x;	  s2.y = y - 1;
			s3.x = x + 1; s3.y = y - 1;

			s4.x = x - 1; s4.y = y;
			s6.x = x + 1; s6.y = y;

			s7.x = x - 1; s7.y = y + 1;
			s8.x = x;	  s8.y = y + 1;
			s9.x = x + 1; s9.y = y + 1;

			wrap(s1.x, MAPWIDTH); wrap(s1.y, MAPHEIGHT);
			wrap(s2.x, MAPWIDTH); wrap(s2.y, MAPHEIGHT);
			wrap(s3.x, MAPWIDTH); wrap(s3.y, MAPHEIGHT);
			wrap(s4.x, MAPWIDTH); wrap(s4.y, MAPHEIGHT);
			wrap(s6.x, MAPWIDTH); wrap(s6.y, MAPHEIGHT);
			wrap(s7.x, MAPWIDTH); wrap(s7.y, MAPHEIGHT);
			wrap(s8.x, MAPWIDTH); wrap(s8.y, MAPHEIGHT);
			wrap(s9.x, MAPWIDTH); wrap(s9.y, MAPHEIGHT);

			float average =
				trailMap[s1.y * MAPWIDTH + s1.x] +
				trailMap[s2.y * MAPWIDTH + s2.x] +
				trailMap[s3.y * MAPWIDTH + s3.x] +
				trailMap[s4.y * MAPWIDTH + s4.x] +
				trailMap[y * MAPWIDTH + x] +
				trailMap[s6.y * MAPWIDTH + s6.x] +
				trailMap[s7.y * MAPWIDTH + s7.x] +
				trailMap[s8.y * MAPWIDTH + s8.x] +
				trailMap[s9.y * MAPWIDTH + s9.x];

			average /= 9.0f;

			float current = trailMap[y * MAPWIDTH + x];
			float weight = blurRate * simTickTime;
			float weightedAverage = (1.0f - weight) * current + weight * average;
			float decayed = glm::max(0.0f, weightedAverage - decayRate * simTickTime);

			nextTrailMap[y * MAPWIDTH + x] = decayed;
		}
	}

	void updateSlimes()
	{
		for (int slimeIndex = 0; slimeIndex < positions.size(); slimeIndex++)
		{
			glm::vec2& pos = positions.at(slimeIndex);
			glm::vec2& dir = directions.at(slimeIndex);

			//find which sensor is the strongest
			float sensorStrength[3] = { 0, 0, 0 };
			for (int sensorIndex = 0; sensorIndex < 3; sensorIndex++)
			{
				glm::vec2 sensorDir = glm::rotateZ(glm::vec3(dir, 0.0f), sensorAngle * (sensorIndex - 1));
				glm::ivec2 sensorPos = pos + sensorDir * sensorDist;

				//DrawRect(sensorPos.x - sensorRadius, sensorPos.y - sensorRadius, 2 * sensorRadius, 2 * sensorRadius, olc::GREEN);

				for (int di = 0; di < sensorSize; di++)
				{
					int dx = di % (1 + 2 * sensorRadius) - sensorRadius;
					int dy = di / (1 + 2 * sensorRadius) - sensorRadius;
					int sx = sensorPos.x + dx;
					int sy = sensorPos.y + dy;
					wrap(sx, MAPWIDTH);
					wrap(sy, MAPHEIGHT);

					sensorStrength[sensorIndex] += trailMap[sy * MAPWIDTH + sx];
				}
			}

			//if all 3 are the same, default to 1 for straight ahead
			int strongest = sensorStrength[0] > sensorStrength[1] ? 0 : 1;
			strongest = sensorStrength[strongest] >= sensorStrength[2] ? strongest : 2;

			//turn towards strongest sensor, dont turn if straight ahead
			if (strongest != 1)
			{
				glm::vec2 sensorDir = glm::rotateZ(glm::vec3(dir, 0.0f), sensorAngle * (strongest - 1));
				dir = glm::normalize(sensorTurnStrength * sensorDir + (1.0f - sensorTurnStrength) * dir);
			}

			//turn a little towards a random direction
			float rdx = (float)rand() / RAND_MAX - 0.5f;
			float rdy = (float)rand() / RAND_MAX - 0.5f;
			glm::vec2 randDir = { rdx, rdy };
			dir = glm::normalize(directionRandomness * randDir + (1.0f - directionRandomness) * dir);


			//move along new direction
			pos += slimeSpeed * dir * simTickTime;
			pos = wrapPos(pos);

			nextTrailMap[(int)pos.y * MAPWIDTH + (int)pos.x] = 1.0f;

			//draw slimes
			DrawLine(pos.x, pos.y, pos.x + dir.x * 2.0f, pos.y + dir.y * 2.0f);
		}
	}

	bool OnUserCreate() override
	{
		srand(std::chrono::system_clock::now().time_since_epoch().count());

		trailMap = new float[MAPWIDTH * MAPHEIGHT];
		nextTrailMap = new float[MAPWIDTH * MAPHEIGHT];

		std::fill_n(trailMap, MAPWIDTH * MAPHEIGHT, 0.0f);
		std::fill_n(nextTrailMap, MAPWIDTH * MAPHEIGHT, 0.0f);

		for (int i = 0; i < 200; i++)
		{
			glm::vec2 pos = { rand() % MAPWIDTH, rand() % MAPHEIGHT };
			addSlime(pos);
		}

		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		Clear(olc::BLACK);

		//user input
		if (GetMouse(0).bReleased)
		{
			addSlime(glm::vec2(GetMouseX(), GetMouseY()));
		}

		decayTrails();
		updateSlimes();

		std::swap(trailMap, nextTrailMap);
		drawTrails();
		
		return true;
	}

	bool OnUserDestroy() override
	{
		delete trailMap;
		delete nextTrailMap;

		return true;
	}
};

//int main()
//{
//	SlimeCL demo;
//	if (demo.Construct(MAPWIDTH, MAPHEIGHT, 2, 2))
//		demo.Start();
//	return 0;
//}