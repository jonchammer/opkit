#ifndef TIMER_H

#include <chrono>
using namespace std;
using namespace std::chrono;

namespace athena
{
    
class Timer
{
public:
    Timer()
    {
        mStartTime = high_resolution_clock::now();
    }

    double getElapsedTimeSeconds()
    {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - mStartTime).count() / 1E6;
    }

    double getElapsedTimeMilliseconds()
    {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - mStartTime).count() / 1E3;
    }

private:
    time_point<system_clock, nanoseconds> mStartTime;
};

};

#endif
