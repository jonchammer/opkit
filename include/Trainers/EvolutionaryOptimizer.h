/* 
 * File:   EvolutionaryOptimization.h
 * Author: Jon C. Hammer
 *
 * Created on July 21, 2016, 5:11 PM
 */

#ifndef EVOLUTIONARYOPTIMIZATION_H
#define EVOLUTIONARYOPTIMIZATION_H

#include "ErrorFunction.h"
#include "Matrix.h"
#include "Trainer.h"

// Used to adjust the behavior of an EvolutionaryOptimizer object
// Default values will be set in the constructor.
struct EvolutionaryOptimizerParams
{
    int population_size;                 // The size of the entire population
    int rand_seed;                       // The seed to use for the random number generator
    double init_population_range;        // Initial values are in the [normal() * this value] range
    double tournament_rigged_chance;     // Probability that the winner of a tournament actually wins
    double mutation_deviation;           // Standard deviation for both mutations and catastrophic mutations
    double repopulate_extrapolate_range; // When extrapolation is used, the possible range of values
    double repopulate_chances[3];        // [0] - crossover, [1] - interpolation, [2] - extrapolation ([0-2] should sum to 1.0)
    double iterate_chances[3];           // [0] - tournament, [1] - single mutation, [2] - catastrophic mutation ([0-2] should sum to 1.0)

    EvolutionaryOptimizerParams(int populationSize)
    {
        population_size = populationSize;

        rand_seed                      = 42;
        init_population_range          = 10.0;
        tournament_rigged_chance       = 0.9;
        mutation_deviation             = 1.0;
        repopulate_extrapolate_range   = 2.0;

        repopulate_chances[0]          = 0.7;
        repopulate_chances[1]          = 0.2;
        repopulate_chances[2]          = 0.1;

        iterate_chances[0]             = 0.7;
        iterate_chances[1]             = 0.2;
        iterate_chances[2]             = 0.1;
    }
};

// This class implements an Evolutionary (or Genetic) optimizer. It maintains
// a population of candidate solutions to the problem and continually tries to
// improve a locally optimal solution by manipulating members of the population. 
// Several operations are implemented, and the probabilities that each will
// occur can be manipulated by the user. The operations include: tournament
// selection (in which two members 'fight' for the right to remain a part of the
// population), single trait mutations, and drastic mutations. Repopulation 
// options include crossover, interpolation, and extrapolation.
//
// This particular implementation guarantees that the best solution ever seen
// is returned to the user, which implies that the error measured after multiple
// iterations will monotonically decrease. However, to obtain the true optimum,
// careful examination of the meta-parameters will likely be necessary.
//
// Evolutionary optimization is well suited to problems in which derivative
// information is unavailable or expensive to calculate, but it tends to be
// slower than some other methods (e.g. Gradient Descent) in general.
class EvolutionaryOptimizer : public Trainer
{
public:
    
    // Constructors
    EvolutionaryOptimizer(ErrorFunction* function, EvolutionaryOptimizerParams& params);
    void iterate(const Matrix& features, const Matrix& labels);
 
private:
    // Members
    Matrix mPopulation;                    // The population of candidate solutions
    EvolutionaryOptimizerParams* mParams;  // The meta parameters to use for the simulation
    vector<double> mErrors;                // The fitness value of each member of the population
    vector<double> mInvErrors;             // 1.0 / mFitnesses[i] (used for run-time optimization)
    
    double mMinError;                      // The error of the best member of the population
    vector<double> mOptimalSolution;       // Our best estimate of the optimal solution
  
    // Random number creators
    std::default_random_engine mRandGenerator;        
    std::uniform_real_distribution<double> mUniform; 
    std::normal_distribution<> mNormal;
    
    // Functions

    // During repopulation, picks two suitable parents. Fitter parents are more 
    // likely to be chosen.
    void chooseParents(int& out1, int& out2);

    // Replace the vector at index with a new member (found using crossover, 
    // interpolation, or extrapolation)
    void repopulate(int index, const Matrix& features, const Matrix& labels);

    // Choose two vectors at random to fight to the death. The loser is repopulated.
    void tournament(const Matrix& features, const Matrix& labels);

    // Mutate a single given element of the given vector
    void mutateSingle(int row, int column, bool reevaluateFitness, 
        const Matrix& features, const Matrix& labels);

    // Mutates all elements of a random vector
    void mutateAll(const Matrix& features, const Matrix& labels);

    // Returns the best fitness (lowest error) in the entire population
    double getBestError();

    // Returns the average error over the entire population
    double getAverageError();
    
    // Updates information about the given population member (reevaluating its
    // fitness and comparing it to the optimal answer).
    void evaluateMember(int index, const Matrix& features, const Matrix& labels);
};

#endif /* EVOLUTIONARYOPTIMIZATION_H */

