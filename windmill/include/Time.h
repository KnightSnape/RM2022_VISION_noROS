#ifndef TimeTra
#define TimeTra
#include <chrono>
#include <thread>

class Time 
{
public:
    std::chrono::steady_clock::time_point begin;

    void init();

    double milliseconds();
    double seconds();

    void sleep(double milli);

private:

};
#endif