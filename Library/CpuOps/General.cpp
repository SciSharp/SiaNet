#include "General.h"
#include <string>

std::string lastError = "";

const char* TS_GetLastError()
{
	return lastError.c_str();
}

void TS_SetLastError(char* message)
{
	lastError = message;
}
