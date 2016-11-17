/* 
 * File:   SimulatedAnnealing.h
 * Author: Jon C. Hammer
 *
 * Created on July 14, 2016, 4:37 PM
 */

#ifndef SIMULATEDANNEALING_H
#define SIMULATEDANNEALING_H

#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include "Trainer.h"
#include "ErrorFunction.h"
#include "Matrix.h"
using std::vector;

namespace athena
{

// This class implements a Simulated Annealing optimizer. Essentially, it 
// mirrors the physical process of annealing, in which materials are slowly 
// cooled in order to form crystalline structures. Simulated Annealing can be 
// appealing because it will (given the right choices of meta parameters) 
// converge to the globally optimal solution. This is because the algorithm will
// occasionally explore locally suboptimal paths in order to find a better 
// solution.
//
// The algorithm chooses a random neighbor from a normal distribution with an 
// variance that reduces over time. It then evaluates the performance of that 
// point and compares it to that of the starting point. If the new point is 
// "better", it is accepted, and the process is repeated. If the new point is 
// worse, there is some probability that it will be accepted based on the 
// current temperature of the system. When the temperature is high, worse values 
// are more likely to be accepted. As the temperature decreases, it becomes less 
// likely that poorer points will be chosen. When the temperature reaches its 
// minimum value, the algorithm terminates.
//
// This implementation also maintains the best value that has been seen so far.
// The optimal value will always be returned, even if the algorithm is currently
// searching a different part of the search space.
template <class T>
class SimulatedAnnealing : public Trainer<T>
{
public:
    
    // Default values for the various meta parameters
    constexpr static double DEFAULT_INITIAL_TEMPERATURE      = 1.0;
    constexpr static double DEFAULT_FINAL_TEMPERATURE        = 0.00001;
    constexpr static double DEFAULT_TEMPERATURE_COOLING_RATE = 0.9;
    constexpr static double DEFAULT_INITIAL_VARIANCE         = 1.0;
    constexpr static double DEFAULT_VARIANCE_COOLING_RATE    = 0.9;
    constexpr static int    DEFAULT_EQUILIBRIUM_ITERATIONS   = 1000;
        
    SimulatedAnnealing(ErrorFunction<T>* function) :
        // Superclass initialization
        Trainer(function),
    
        // Initialize meta parameters
        mInitialTemperature(DEFAULT_INITIAL_TEMPERATURE),
        mFinalTemperature(DEFAULT_FINAL_TEMPERATURE),
        mTemperatureCoolingRate(DEFAULT_TEMPERATURE_COOLING_RATE),
        mInitialVariance(DEFAULT_INITIAL_VARIANCE),
        mVarianceCoolingRate(DEFAULT_VARIANCE_COOLING_RATE),
        mNumEquilibriumIterations(DEFAULT_EQUILIBRIUM_ITERATIONS),
            
        // Initialize state information
        mTemperature(mInitialTemperature), 
        mCurrentVariance(mInitialVariance),
        mCurrentCost(std::numeric_limits<double>::max()),
            
        // Initialize optimal solution information
        mOptimalCost(std::numeric_limits<double>::max()),
            
        // Initialize random number generators
        mUniform(0.0, 1.0), 
        mNormal(0.0, 1.0) 
    {
        // Do nothing
    }
    
    void iterate(const Matrix& features, const Matrix& labels)
    {
        // Initialization
        vector<double>& params = Trainer<T>::function->getParameters();
        mCurrentCost           = function->evaluate(features, labels);
        mTemperature           = mInitialTemperature;
        mCurrentVariance       = mInitialVariance;
        
        vector<double> candidate(params.size());
        mOptimalSolution.resize(params.size());
        
        while (mTemperature > mFinalTemperature)
        {
            // Repeat the process until we find an equilibrium
            for (int i = 0; i < mNumEquilibriumIterations; ++i)
            {
                // Generate new point in neighborhood of current solution
                for (size_t j = 0; j < params.size(); ++j)
                    candidate[j] = params[j] + mCurrentVariance * mNormal(mRandGenerator);
                
                // Evaluate the new point
                params.swap(candidate);
                double proposedCost = Trainer<T>::function->evaluate(features, labels);

                // Save this solution if it is the best we've seen so far
                if (proposedCost < mOptimalCost)
                {
                    mOptimalCost = proposedCost;
                    std::copy(params.begin(), params.end(), mOptimalSolution.begin());
                }
                
                // Accept the new point
                if (getAcceptanceProbability(proposedCost) > mUniform(mRandGenerator))
                    mCurrentCost = proposedCost;
                
                // Reject the candidate (revert back to the previous solution)
                else params.swap(candidate);
            }

            // Decrease the temperature & variance for the next iteration
            mTemperature     *= mTemperatureCoolingRate;
            mCurrentVariance *= mVarianceCoolingRate;
            
            cout << "SSE: " << mCurrentCost << endl;
        }
        
        // Guarantee that the best solution we found is returned
        params.swap(mOptimalSolution);
        cout << "SSE: " << mOptimalCost << endl;
    }
    
    // Setters
    void setInitialTemperature(double temperature)   { mInitialTemperature       = temperature; }
    void setFinalTemperature(double temperature)     { mFinalTemperature         = temperature; }
    void setTemperatureCoolingRate(double rate)      { mTemperatureCoolingRate   = rate;        }
    void setInitialVariance(double variance)         { mInitialVariance          = variance;    }
    void setVarianceCoolingRate(double rate)         { mVarianceCoolingRate      = rate;        }
    void setNumEquilibriumIterations(int iterations) { mNumEquilibriumIterations = iterations;  }
    
private:
    
    // Meta parameters
    double mInitialTemperature;     // The initial temperature of the system.
    double mFinalTemperature;       // The temperature at which we stop. Close to 0.
    double mTemperatureCoolingRate; // The rate at which the temperature decreases after equilibrium is reached. [0, 1)
    double mInitialVariance;        // How far away to look for neighbors at the beginning of the algorithm.
    double mVarianceCoolingRate;    // The rate at which the variance is reduced as the algorithm proceeds.
    int mNumEquilibriumIterations;  // The number of iterations performed before equilibrium is reached.
    
    // State information
    double mTemperature;
    double mCurrentVariance;
    double mCurrentCost;

    // Information about the best solution found so far
    double mOptimalCost;
    vector<double> mOptimalSolution;
    
    // Random number creators
    std::default_random_engine mRandGenerator;        
    std::uniform_real_distribution<double> mUniform; 
    std::normal_distribution<> mNormal;
    
    // Helper functions
    double getAcceptanceProbability(double proposedCost)
    {
        // Better solutions are accepted unconditionally
        if (proposedCost < mCurrentCost) return 1.0;
        
        // Determine the if we should accept the solution or not
        else return exp((mCurrentCost - proposedCost) / mTemperature);
    }
};

};

#endif /* SIMULATEDANNEALING_H */

