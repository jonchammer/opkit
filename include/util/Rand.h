#ifndef RAND_H
#define RAND_H

#include <random>
#include <chrono>

namespace tensorlib
{

// This class should be used to easily generate random numbers.
//
// NOTE: Instances of this class should not be qualified as 'const'. You will
// get some very unhelpful error messages if you do.
class Rand
{
public:

    // Create a new RNG with the given seed
    Rand(const size_t seed) : mSeed(seed), mGenerator(seed)
    {}

    // Create a new RNG that uses the current time as the seed
    Rand() :
        mSeed(getDefaultSeed()),
        mGenerator(mSeed)
    {}

    // Generate an integral type (e.g. int, long, short, size_t) within the
    // given range.
    template <class T>
    T nextInteger(T min, T max)
    {
        std::uniform_int_distribution<T> distribution (min, max);
        return distribution(mGenerator);
    }

    // Generate a real type (e.g. float, double) within the given range.
    template <class T>
    T nextReal(T min, T max)
    {
        std::uniform_real_distribution<T> distribution(min, max);
        return distribution(mGenerator);
    }

    // Generate a real number from the given normal distribution
    template <class T>
    T nextGaussian(const T mean, const T stdev)
    {
        std::normal_distribution<T> distribution (mean, stdev);
        return distribution(mGenerator);
    }

    // Choose an index weighted by the given vector of probabilities. The
    // elements of the probabilities vector do not necessarily have to sum to 1.
    template <class T>
    size_t nextCategorical(const vector<T>& probabilities)
    {
        std::discrete_distribution<size_t> distribution(
            probabilities.begin(), probabilities.end());
        return distribution(mGenerator);
    }

    // Returns a reference to the generator that is used internally for
    // random number generation.
    std::default_random_engine& getGenerator()
    {
        return mGenerator;
    }

    // Changes the seed used by this RNG.
    void setSeed(size_t seed)
    {
        mSeed = seed;
        mGenerator.seed(seed);
    }

    // Resets the RNG sequence, starting from the current seed.
    void reset()
    {
        mGenerator.seed(mSeed);
    }

    // Returns a seed based on the current time.
    static size_t getDefaultSeed()
    {
        return std::chrono::system_clock::now().time_since_epoch().count();
    }

private:
    size_t mSeed;
    std::default_random_engine mGenerator;
};

}

#endif
