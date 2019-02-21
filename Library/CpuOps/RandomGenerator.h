#pragma once

#include "General.h"
#include <cmath>

#define _MERSENNE_STATE_N 624
#define _MERSENNE_STATE_M 397

class UniformGenerator
{
private:
	unsigned __int64 the_initial_seed;
	int left;  /* = 1; */
	int seeded; /* = 0; */
	unsigned __int64 next;
	unsigned __int64 state[_MERSENNE_STATE_N];

	double normal_x;
	double normal_y;
	double normal_rho;
	bool normal_is_valid;

	void Reset();
	void NextState();
	unsigned __int64 RandomULong();

public:
	UniformGenerator();
	void SetSeed(unsigned __int64 newSeed);
	unsigned __int64 GetInitialSeed() { return the_initial_seed; }

	double NextUniform();

	double NextNormal(double mean, double stdv)
	{
		if (!normal_is_valid)
		{
			normal_x = NextUniform();
			normal_y = NextUniform();
			normal_rho = sqrt(-2 * log(1.0 - normal_y));
			normal_is_valid = true;
		}
		else
		{
			normal_is_valid = false;
		}

		if (normal_is_valid)
			return normal_rho * cos(2 * M_PI * normal_x) * stdv + mean;
		else
			return normal_rho * sin(2 * M_PI * normal_x) * stdv + mean;
	}
};
