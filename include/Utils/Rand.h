#ifndef RAND_H
#define RAND_H

#include <random>
#include <chrono>

namespace opkit
{

// General template
template <class T, class U = void>
class Rand
{};

// For integral types
template <class T>
class Rand <T, typename std::enable_if<std::is_integral<T>::value>::type>
{
public:
    Rand() :
        mGenerator(std::chrono::system_clock::now().time_since_epoch().count()),
        mDistribution(T{}, T{1.0}) {}

    Rand(const size_t seed) :
        mGenerator(seed),
        mDistribution(T{}, T{1.0}) {}

    T operator()(T min, T max)
    {
        return T(mDistribution(mGenerator) * (max - min) + min);
    }

private:
    std::default_random_engine mGenerator;
    std::uniform_real_distribution<double> mDistribution;
};

// For real types
template <class T>
class Rand <T, typename std::enable_if<!std::is_integral<T>::value>::type>
{
public:
    Rand() :
        mGenerator(std::chrono::system_clock::now().time_since_epoch().count()),
        mDistribution(T{}, T{1.0}) {}

    Rand(const size_t seed) :
        mGenerator(seed),
        mDistribution(T{}, T{1.0}) {}

    T operator()()
    {
        return mDistribution(mGenerator);
    }

    T operator()(T min, T max)
    {
        return mDistribution(mGenerator) * (max - min) + min;
    }

private:
    std::default_random_engine mGenerator;
    std::uniform_real_distribution<T> mDistribution;
};
}

#endif
