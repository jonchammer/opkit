#ifndef RAND_H
#define RAND_H

#include <random>
#include <chrono>

namespace opkit
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
        mSeed(std::chrono::system_clock::now().time_since_epoch().count()),
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

    // Choose an index weighted by the given vector of probabilities.
    // NOTE: The elements of the vector must sum to 1.0.
    template <class T>
    size_t nextCategorical(const vector<T>& probabilities)
    {
        T val    = nextReal(T{0.0}, T{1.0});
    	size_t i = 0;

    	for (auto it = probabilities.begin(); it != probabilities.end(); ++it)
    	{
    		val -= *it;
    		if(val < 0)
    			return i;
    		i++;
    	}

        // Shouldn't happen
        return probabilities.size() - 1;
    }

    std::default_random_engine& getGenerator()
    {
        return mGenerator;
    }
    
private:
    size_t mSeed;
    std::default_random_engine mGenerator;
};

}

#endif
