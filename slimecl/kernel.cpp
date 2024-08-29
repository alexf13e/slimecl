
#include "wrapper/kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(

string opencl_c_container() {
return R( // ########################## begin of OpenCL C code ####################################################################

	typedef struct SlimeSettings
	{
		float slimeSpeed;
		int sensorRadius;
		float sensorAngle;
		float sensorTurnStrength;
		float directionRandomness;
		int depositWidth;
	};

	typedef struct TrailSettings
	{
		float blurRate;
		float decayRate;
		float r, g, b;
	};

	int wrap(int x, const int period)
	{
		while (x < 0) x += period;
		while (x >= period) x -= period;
		return x;
	}

	float2 wrapPos(float2 pos, int mapWidth, int mapHeight)
	{
		while (pos.x < 0) pos.x += mapWidth;
		while (pos.x > mapWidth) pos.x -= mapWidth;
		while (pos.y < 0) pos.y += mapHeight;
		while (pos.y > mapHeight) pos.y -= mapHeight;
		return pos;
	}

	uint randomInt(uint seed)
	{
		//https://www.cs.ubc.ca/~rbridson/docs/schechter-sca08-turbulence.pdf
		seed ^= 2747636419u;
		seed *= 2654435769u;
		seed ^= seed >> 16;
		seed *= 2654435769u;
		seed ^= seed >> 16;
		seed *= 2654435769u;
		return seed;
	}

	float random01(uint val)
	{
		//random float between 0 and 1
		return convert_float(val) / 4294967296;
	}

	float2 v_rotate(float2 vec, float angle)
	{
		return (float2)(vec.x * cos(angle) - vec.y * sin(angle), vec.x * sin(angle) + vec.y * cos(angle));
	}

	kernel void decayTrails(global float* trailMap, global float* nextTrailMap, global float4* colouredTrail,
		int mapWidth, int mapHeight, float simDeltaTime, struct TrailSettings trailSettings)
	{
		const uint mapPixelIndex = get_global_id(0);

		int x = mapPixelIndex % mapWidth;
		int y = mapPixelIndex / mapWidth;

		//3x3 box blur
		int2 s1 = (int2)(wrap(x - 1, mapWidth), wrap(y - 1, mapHeight));
		int2 s2 = (int2)(x,						wrap(y - 1, mapHeight));
		int2 s3 = (int2)(wrap(x + 1, mapWidth), wrap(y - 1, mapHeight));
		int2 s4 = (int2)(wrap(x - 1, mapWidth), y					  );
		int2 s6 = (int2)(wrap(x + 1, mapWidth), y					  );
		int2 s7 = (int2)(wrap(x - 1, mapWidth), wrap(y + 1, mapHeight));
		int2 s8 = (int2)(x,						wrap(y + 1, mapHeight));
		int2 s9 = (int2)(wrap(x + 1, mapWidth), wrap(y + 1, mapHeight));

		float average = (
			trailMap[s1.y * mapWidth + s1.x] +
			trailMap[s2.y * mapWidth + s2.x] +
			trailMap[s3.y * mapWidth + s3.x] +
			trailMap[s4.y * mapWidth + s4.x] +
			trailMap[mapPixelIndex] +
			trailMap[s6.y * mapWidth + s6.x] +
			trailMap[s7.y * mapWidth + s7.x] +
			trailMap[s8.y * mapWidth + s8.x] +
			trailMap[s9.y * mapWidth + s9.x]
		) / 9.0f;

		float current = trailMap[mapPixelIndex];
		float weight = trailSettings.blurRate * simDeltaTime;
		float weightedAverage = (1.0f - weight) * current + weight * average;
		float decayed = max(0.0f, weightedAverage - trailSettings.decayRate * simDeltaTime);

		nextTrailMap[mapPixelIndex] = decayed;
		colouredTrail[mapPixelIndex] = (float4)(decayed * trailSettings.r, decayed * trailSettings.g, decayed * trailSettings.b, 1.0f);
	}

	kernel void updateSlimes(global float2* positions, global float2* directions, global float* trailMap,
		global float* nextTrailMap, global uint* randomSeeds, int mapWidth, int mapHeight, float simDeltaTime,
		struct SlimeSettings slimeSettings, int numSlimes)
	{
		const uint slimeIndex = get_global_id(0);

		if (slimeIndex >= numSlimes)
		{
			//for some reason, at numslimes less than 128, extra slimes will be created up to 128, and start at 0,0 with
			//a random positive direction...
			//they are not present in the host buffer and the host memory object believes there to be the correct
			//specified number so
			return;
		}

		//find which sensor is the strongest
		float sensorStrength[3] = { 0, 0, 0 };
		int sensorSize = (1 + 2 * slimeSettings.sensorRadius) * (1 + 2 * slimeSettings.sensorRadius); //number of pixels in sensor
		for (int sensorIndex = 0; sensorIndex < 3; sensorIndex++)
		{
			float2 sensorDir = v_rotate(directions[slimeIndex], slimeSettings.sensorAngle * (sensorIndex - 1));
			int2 sensorPos = convert_int2(positions[slimeIndex] + sensorDir * 3.5f * slimeSettings.sensorRadius);

			for (int di = 0; di < sensorSize; di++)
			{
				//dx and dy range from -sensorRadius to +sensorRadius
				int dx = di % (1 + 2 * slimeSettings.sensorRadius) - slimeSettings.sensorRadius;
				int dy = di / (1 + 2 * slimeSettings.sensorRadius) - slimeSettings.sensorRadius;
				int sx = wrap(sensorPos.x + dx, mapWidth);
				int sy = wrap(sensorPos.y + dy, mapHeight);

				sensorStrength[sensorIndex] += trailMap[sy * mapWidth + sx];
			}
		}

		//if all 3 are the same, default to 1 for straight ahead
		int strongestIndex = sensorStrength[0] > sensorStrength[1] ? 0 : 1;
		strongestIndex = sensorStrength[strongestIndex] >= sensorStrength[2] ? strongestIndex : 2;

		//turn towards strongest sensor, dont turn if straight ahead
		if (strongestIndex != 1)
		{
			float2 strongestDir = v_rotate(directions[slimeIndex], slimeSettings.sensorAngle * (strongestIndex - 1));
			directions[slimeIndex] = normalize(slimeSettings.sensorTurnStrength * strongestDir +
				(1.0f - slimeSettings.sensorTurnStrength) * directions[slimeIndex]);
		}

		//turn a little towards a random direction
		randomSeeds[slimeIndex] = randomInt(randomSeeds[slimeIndex]);
		float rdx = random01(randomSeeds[slimeIndex]) - 0.5f;

		randomSeeds[slimeIndex] = randomInt(randomSeeds[slimeIndex]);
		float rdy = random01(randomSeeds[slimeIndex]) - 0.5f;

		float2 randDir = normalize((float2)(rdx, rdy));
		directions[slimeIndex] = normalize(slimeSettings.directionRandomness * randDir +
			(1.0f - slimeSettings.directionRandomness) * directions[slimeIndex]);

		//move along new direction
		positions[slimeIndex] += slimeSettings.slimeSpeed * directions[slimeIndex] * simDeltaTime;
		positions[slimeIndex] = wrapPos(positions[slimeIndex], mapWidth, mapHeight);

		//set trail map strength at position to 1
		nextTrailMap[(int)positions[slimeIndex].y * mapWidth + (int)positions[slimeIndex].x] = 1.0f;

		if (slimeSettings.depositWidth > 0)
		{
			//get direction perpendicular to slime forward
			float2 perp = (float2)(-directions[slimeIndex].y, directions[slimeIndex].x);
			for (float w = -slimeSettings.depositWidth; w <= slimeSettings.depositWidth; w++)
			{
				float2 depositPos = positions[slimeIndex] + w * perp;
				depositPos = wrapPos(depositPos, mapWidth, mapHeight);
				nextTrailMap[(int)depositPos.y * mapWidth + (int)depositPos.x] = 1.0f;
			}
		}
	}



);
} // ############################################################### end of OpenCL C code #####################################################################