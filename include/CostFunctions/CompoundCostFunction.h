#ifndef COMPOUND_COST_H
#define COMPOUND_COST_H

#include "CostFunction.h"
#include "Matrix.h"
#include "Acceleration.h"
#include <vector>

using std::vector;

namespace opkit
{

// A Compund Cost Function is an abstraction used to combine two or more simpler
// cost functions (e.g. SSE, CrossEntropy, or L1-L2 regularization). It is
// assumed that the costs from each function are independent and so can simply
// be added together to create the final cost. Derivative calculations work
// similarly. The gradients calculated by each individual cost function are
// summed to generate the total gradient.
template <class T, class Model>
class CompoundCostFunction : public CostFunction<T, Model>
{
public:

    CompoundCostFunction(Model& baseFunction) :
        CostFunction<T, Model>(baseFunction) {}

    ~CompoundCostFunction()
    {
        for (size_t i = 0; i < mCostFunctions.size(); ++i)
        {
            if (mFunctionOwnership[i])
            {
                delete mCostFunctions[i];
                mCostFunctions[i] = nullptr;
            }
        }
    }
    // Add a new cost function to this object.
    void addCostFunction(CostFunction<T, Model>* fn, bool ownsFunction = true)
    {
        mCostFunctions.push_back(fn);
        mFunctionOwnership.push_back(ownsFunction);
    }

    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        T sum{};
        for (CostFunction<T, Model>* fn : mCostFunctions)
            sum += fn->evaluate(features, labels);

        return sum;
    }

    void calculateGradientInputs(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        const size_t N = gradient.size();
        mLocalGradientInputs.resize(N);
        std::fill(gradient.begin(), gradient.end(), T{});

        // Calculate each individual gradient and sum them together to generate
        // the final gradient.
        for (CostFunction<T, Model>*& fn : mCostFunctions)
        {
            fn->calculateGradientInputs(features, labels, mLocalGradientInputs);
            vAdd(mLocalGradientInputs.data(), gradient.data(), N);
        }
    }

    void calculateGradientParameters(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        const size_t N = gradient.size();
        mLocalGradientParameters.resize(N);
        std::fill(gradient.begin(), gradient.end(), T{});

        // Calculate each individual gradient and sum them together to generate
        // the final gradient.
        for (CostFunction<T, Model>*& fn : mCostFunctions)
        {
            fn->calculateGradientParameters(features, labels, mLocalGradientParameters);
            vAdd(mLocalGradientParameters.data(), gradient.data(), N);
        }
    }

    void calculateHessianInputs(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
    {
        const size_t M = hessian.getRows();
        const size_t N = hessian.getCols();
        mLocalHessianInputs.resize(M, N);
        hessian.fill(T{});

        // Calculate each individual gradient and sum them together to generate
        // the final gradient.
        for (CostFunction<T, Model>*& fn : mCostFunctions)
        {
            fn->calculateHessianInputs(features, labels, mLocalHessianInputs);
            vAdd(mLocalHessianInputs.data(), hessian.data(), M * N);
        }
    }

    void calculateHessianParameters(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
    {
        const size_t M = hessian.getRows();
        const size_t N = hessian.getCols();
        mLocalHessianParameters.resize(M, N);
        hessian.fill(T{});

        // Calculate each individual gradient and sum them together to generate
        // the final gradient.
        for (CostFunction<T, Model>*& fn : mCostFunctions)
        {
            fn->calculateHessianParameters(features, labels, mLocalHessianParameters);
            vAdd(mLocalHessianParameters.data(), hessian.data(), M * N);
        }
    }

private:
    vector<CostFunction<T, Model>*> mCostFunctions;
    vector<bool> mFunctionOwnership;
    
    // Temporary storage space for the calculation methods
    vector<T> mLocalGradientParameters;
    vector<T> mLocalGradientInputs;
    Matrix<T> mLocalHessianParameters;
    Matrix<T> mLocalHessianInputs;
};

}

#endif
