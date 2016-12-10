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

namespace opkit
{
    
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
template <class T, class Model>
class EvolutionaryOptimizer : public Trainer<T, Model>
{
public:
    
    // Constructors
    EvolutionaryOptimizer(ErrorFunction<T, Model>* function, EvolutionaryOptimizerParams& params);
    void iterate(const Matrix<T>& features, const Matrix<T>& labels);
 
private:
    // Members
    Matrix<T> mPopulation;                 // The population of candidate solutions
    EvolutionaryOptimizerParams* mParams;  // The meta parameters to use for the simulation
    vector<T> mErrors;                     // The fitness value of each member of the population
    vector<T> mInvErrors;                  // 1.0 / mFitnesses[i] (used for run-time optimization)
    
    T mMinError;                      // The error of the best member of the population
    vector<T> mOptimalSolution;       // Our best estimate of the optimal solution
  
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
    void repopulate(int index, const Matrix<T>& features, const Matrix<T>& labels);

    // Choose two vectors at random to fight to the death. The loser is repopulated.
    void tournament(const Matrix<T>& features, const Matrix<T>& labels);

    // Mutate a single given element of the given vector
    void mutateSingle(int row, int column, bool reevaluateFitness, 
        const Matrix<T>& features, const Matrix<T>& labels);

    // Mutates all elements of a random vector
    void mutateAll(const Matrix<T>& features, const Matrix<T>& labels);

    // Returns the best fitness (lowest error) in the entire population
    T getBestError();

    // Returns the average error over the entire population
    T getAverageError();
    
    // Updates information about the given population member (reevaluating its
    // fitness and comparing it to the optimal answer).
    void evaluateMember(int index, const Matrix<T>& features, const Matrix<T>& labels);
};

template <class T, class Model>
EvolutionaryOptimizer<T, Model>::EvolutionaryOptimizer(ErrorFunction<T, Model>* function, 
    EvolutionaryOptimizerParams& params) : 
    Trainer<T, Model>(function), 
        
    // Initialize random number generators
    mUniform(0.0, 1.0), 
    mNormal(0.0, 1.0) 
{
    const size_t N = function->getNumParameters();
    
	// Initialize the member variables
    mRandGenerator.seed(params.rand_seed);
	mPopulation.setSize(params.population_size, N);
	mErrors.resize(params.population_size);
	mInvErrors.resize(params.population_size);
    mOptimalSolution.resize(N);

	// Fill the population with random vectors initially
	double range = params.init_population_range;
	for (int i = 1; i < params.population_size; ++i)
	{
		for (size_t j = 0; j < N; ++j)
			mPopulation[i][j] = range * mNormal(mRandGenerator);
	}
    // Store the current function parameters in the first population slot (in 
    // case the initialization parameters are bad)
    std::copy(function->getParameters().begin(), function->getParameters().end(), 
        mPopulation[0].begin());
    
	// Save the parameters for later use
	mParams = &params;
}

template <class T, class Model>
void EvolutionaryOptimizer<T, Model>::chooseParents(int& out1, int& out2)
{
	// Find the sum of the inverse errors
	double sum = std::accumulate(mInvErrors.begin(), mInvErrors.end(), 0.0);

	// Choose parent 1
	double rnd = mUniform(mRandGenerator) * sum;
	for (size_t i = 0; i < mPopulation.rows(); ++i)
	{
		if (rnd < mInvErrors[i])
		{
			out1 = i;
			break;
		}
		else rnd -= mInvErrors[i];
	}

	// Choose parent 2
	rnd = mUniform(mRandGenerator) * sum;
	for (size_t i = 0; i < mPopulation.rows(); ++i)
	{
		if (rnd < mInvErrors[i])
		{
			out2 = i;
			break;
		}
		else rnd -= mInvErrors[i];
	}
}

template <class T, class Model>
void EvolutionaryOptimizer<T, Model>::repopulate(int index, 
    const Matrix<T>& features, const Matrix<T>& labels)
{
	// Fitter parents are more likely to be chosen to repopulate
	int parent1 = -1, parent2 = -1;
	chooseParents(parent1, parent2);

	double chance = mUniform(mRandGenerator);

	// Do crossover
	if (chance < mParams->repopulate_chances[0])
	{
		// Choose a random value from each parent
		for (size_t j = 0; j < mPopulation.cols(); ++j)
		{
			mPopulation[index][j] = (mUniform(mRandGenerator) < 0.5) ?
				mPopulation[parent1][j] : mPopulation[parent2][j];
		}
	}

	// Do interpolation
	else if (chance < mParams->repopulate_chances[0] + 
        mParams->repopulate_chances[1])
	{
		for (size_t j = 0; j < mPopulation.cols(); ++j)
		{
			double w              = mUniform(mRandGenerator);
			T val                 = w * mPopulation[parent1][j] + 
                                    (1.0 - w) * mPopulation[parent2][j];
			mPopulation[index][j] = val;
		}
	}

	// Do extrapolation
	else
	{
		for (size_t j = 0; j < mPopulation.cols(); ++j)
		{
			double range = mParams->repopulate_extrapolate_range;

			double w              = mUniform(mRandGenerator) * range - range/2.0;
			T val                 = w * mPopulation[parent1][j] + 
                                    (1.0 - w) * mPopulation[parent2][j];
			mPopulation[index][j] = val;
		}
	}

	// Re-evaluate this member's fitness
	evaluateMember(index, features, labels);
}

template <class T, class Model>
void EvolutionaryOptimizer<T, Model>::tournament(const Matrix<T>& features, const Matrix<T>& labels)
{
	// Pick two candidates
	int c1 = (int)(mUniform(mRandGenerator) * mPopulation.rows());
	int c2 = (int)(mUniform(mRandGenerator) * mPopulation.rows());

	// Evaluate fitness of both
	T e1 = mErrors[c1];
	T e2 = mErrors[c2];

	int loser = -1;

	// The better will win usually
	if (mUniform(mRandGenerator) < mParams->tournament_rigged_chance)
		loser = (e1 < e2) ? c2 : c1;

	// The weaker will win occasionally
	else loser = (e1 < e2) ? c1 : c2;

	// Replace the loser
	repopulate(loser, features, labels);
}

template <class T, class Model>
void EvolutionaryOptimizer<T, Model>::mutateSingle(int row, int column, 
    bool reevaluateFitness, const Matrix<T>& features, const Matrix<T>& labels)
{
	// Perturb the element
	mPopulation[row][column] += mNormal(mRandGenerator) * 
        mParams->mutation_deviation;

	// Update the fitness value if necessary
	if (reevaluateFitness)
		evaluateMember(row, features, labels);
}

template <class T, class Model>
void EvolutionaryOptimizer<T, Model>::mutateAll(const Matrix<T>& features, 
    const Matrix<T>& labels)
{
	// Pick the subject for mutation
	int index = (int)(mUniform(mRandGenerator) * mPopulation.rows());

	// Perturb each column
	for (size_t i = 0; i < mPopulation.cols(); ++i)
		mutateSingle(index, i, false, features, labels);

	// Re-evaluate the fitness
	evaluateMember(index, features, labels);
}

template <class T, class Model>
T EvolutionaryOptimizer<T, Model>::getBestError()
{
	return *std::min_element(mErrors.begin(), mErrors.end());
}

template <class T, class Model>
T EvolutionaryOptimizer<T, Model>::getAverageError()
{
	T sum = std::accumulate(mErrors.begin(), mErrors.end(), 0.0);
	return sum / mPopulation.rows();
}

template <class T, class Model>
void EvolutionaryOptimizer<T, Model>::iterate(const Matrix<T>& features, const Matrix<T>& labels)
{
    // The first time this function is called, we calculate the fitness of each
    // member of the population
    static bool firstRun = true;
    
    if (firstRun)
    {
        mMinError = std::numeric_limits<T>::max();
        
        // Save the initial fitness (error) values for each member
        for (int i = 0; i < mParams->population_size; ++i)
            evaluateMember(i, features, labels);
        
        firstRun = false;
    }
    
    T currentMinError = mMinError;
    
	// Perform an update for each member of the population. (This distinction
    // is arbitrary. We could just as easily iterate 'N' times during one pass.)
	for (size_t i = 0; i < mPopulation.rows(); ++i)
	{
		T option = mUniform(mRandGenerator);

		// Tournament selection
		if (option < mParams->iterate_chances[0])
		{
			tournament(features, labels);
		}

		// Random mutation
		else if (option < 
            mParams->iterate_chances[0] + mParams->iterate_chances[1])
		{
			int row = (int)(mUniform(mRandGenerator) * mPopulation.rows());
			int col = (int)(mUniform(mRandGenerator) * mPopulation.cols());

			mutateSingle(row, col, true, features, labels);
		}

		// Catastrophic mutation
		else
		{
			mutateAll(features, labels);
		}
	}

    // If we made any improvements this iteration, save them in the function
    if (mMinError < currentMinError)
    {
        std::copy(mOptimalSolution.begin(), mOptimalSolution.end(), 
            Trainer<T, Model>::function->getParameters().begin());
    }
}

template <class T, class Model>
void EvolutionaryOptimizer<T, Model>::evaluateMember(int index, 
    const Matrix<T>& features, const Matrix<T>& labels)
{
    // Swap the desired parameters into the function
    vector<T>& origParams = Trainer<T, Model>::function->getParameters();
    origParams.swap(mPopulation[index]);

    // Do the evaluation with the desired parameters
    mErrors[index]    = Trainer<T, Model>::function->evaluate(features, labels);
    mInvErrors[index] = 1.0 / mErrors[index];

    // Put the original parameters back
    origParams.swap(mPopulation[index]);
    
    // Update our optimal answer if we've found a better solution.
    if (mErrors[index] < mMinError)
    {
        mMinError = mErrors[index];
        std::copy(mPopulation[index].begin(), mPopulation[index].end(), 
            mOptimalSolution.begin());
    }
} 

};
#endif /* EVOLUTIONARYOPTIMIZATION_H */

