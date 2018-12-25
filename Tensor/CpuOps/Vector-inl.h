#pragma once


template<typename T>
static INLINE_FUNC void Vector_add(T *y, const T *x, const T c, const __int64 n)
{
	__int64 i = 0;

	for (;i < n - 4; i += 4)
	{
		y[i] += c * x[i];
		y[i + 1] += c * x[i + 1];
		y[i + 2] += c * x[i + 2];
		y[i + 3] += c * x[i + 3];
	}

	for (; i < n; i++)
		y[i] += c * x[i];
}
