#include <limits>
#include <numeric>
#include <algorithm>
#include "EvolutionaryOptimizer.h"

EvolutionaryOptimizer::EvolutionaryOptimizer(ErrorFunction* function, 
    EvolutionaryOptimizerParams& params) : 
    Trainer(function), 
        
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

void EvolutionaryOptimizer::chooseParents(int& out1, int& out2)
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

void EvolutionaryOptimizer::repopulate(int index, const Matrix& features, 
    const Matrix& labels)
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
			double val            = w * mPopulation[parent1][j] + 
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
			double val            = w * mPopulation[parent1][j] + 
                                    (1.0 - w) * mPopulation[parent2][j];
			mPopulation[index][j] = val;
		}
	}

	// Re-evaluate this member's fitness
	evaluateMember(index, features, labels);
}

void EvolutionaryOptimizer::tournament(const Matrix& features, 
    const Matrix& labels)
{
	// Pick two candidates
	int c1 = (int)(mUniform(mRandGenerator) * mPopulation.rows());
	int c2 = (int)(mUniform(mRandGenerator) * mPopulation.rows());

	// Evaluate fitness of both
	double e1 = mErrors[c1];
	double e2 = mErrors[c2];

	int loser = -1;

	// The better will win usually
	if (mUniform(mRandGenerator) < mParams->tournament_rigged_chance)
		loser = (e1 < e2) ? c2 : c1;

	// The weaker will win occasionally
	else loser = (e1 < e2) ? c1 : c2;

	// Replace the loser
	repopulate(loser, features, labels);
}

void EvolutionaryOptimizer::mutateSingle(int row, int column, 
    bool reevaluateFitness, const Matrix& features, const Matrix& labels)
{
	// Perturb the element
	mPopulation[row][column] += mNormal(mRandGenerator) * 
        mParams->mutation_deviation;

	// Update the fitness value if necessary
	if (reevaluateFitness)
		evaluateMember(row, features, labels);
}

void EvolutionaryOptimizer::mutateAll(const Matrix& features, 
    const Matrix& labels)
{
	// Pick the subject for mutation
	int index = (int)(mUniform(mRandGenerator) * mPopulation.rows());

	// Perturb each column
	for (size_t i = 0; i < mPopulation.cols(); ++i)
		mutateSingle(index, i, false, features, labels);

	// Re-evaluate the fitness
	evaluateMember(index, features, labels);
}

double EvolutionaryOptimizer::getBestError()
{
	return *std::min_element(mErrors.begin(), mErrors.end());
}

double EvolutionaryOptimizer::getAverageError()
{
	double sum = std::accumulate(mErrors.begin(), mErrors.end(), 0.0);
	return sum / mPopulation.rows();
}

void EvolutionaryOptimizer::iterate(const Matrix& features, const Matrix& labels)
{
    // The first time this function is called, we calculate the fitness of each
    // member of the population
    static bool firstRun = true;
    
    if (firstRun)
    {
        mMinError = std::numeric_limits<double>::max();
        
        // Save the initial fitness (error) values for each member
        for (int i = 0; i < mParams->population_size; ++i)
            evaluateMember(i, features, labels);
        
        firstRun = false;
    }
    
    double currentMinError = mMinError;
    
	// Perform an update for each member of the population. (This distinction
    // is arbitrary. We could just as easily iterate 'N' times during one pass.)
	for (size_t i = 0; i < mPopulation.rows(); ++i)
	{
		double option = mUniform(mRandGenerator);

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
            function->getParameters().begin());
    }
}

void EvolutionaryOptimizer::evaluateMember(int index, 
    const Matrix& features, const Matrix& labels)
{
    // Swap the desired parameters into the function
    vector<double>& origParams = function->getParameters();
    origParams.swap(mPopulation[index]);

    // Do the evaluation with the desired parameters
    mErrors[index]    = function->evaluate(features, labels);
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