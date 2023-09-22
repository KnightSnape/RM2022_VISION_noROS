#include "Time.h"

void Time::init() 
{
    begin = std::chrono::steady_clock::now();
}

double Time::milliseconds() 
{
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - begin).count();
}

double Time::seconds() 
{
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - begin).count();
}

void Time::sleep(double milli) 
{
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(milli));
}