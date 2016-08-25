/* 
 * File:   main.cpp
 * Author: Jon C. Hammer
 *
 * Created on July 9, 2016, 7:44 PM
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "PrettyPrinter.h"
#include "Matrix.h"
#include "Function.h"
#include "CommonFunctions.h"
#include "ConvNeuralNetwork.h"

#include "DataLoader.h"
#include "DataNormalizer.h"
#include "ModelPersistence.h"

#include "Trainer.h"
#include "HessianFreeOptimizer.h"
#include "GradientDescent.h"
#include "SSEFunction.h"

using namespace std;
using namespace std::chrono;

class Temp : public StandardFunction
{
public:
    Temp() : StandardFunction(2, 2, 4) {}
    
    void evaluate(const vector<double>& input, vector<double>& output)
    {
        output.resize(2);
        output[0] = mParameters[0] * input[0] + mParameters[1];
        output[1] = mParameters[2] * input[1] + mParameters[3];
    }
};

class Temp2 : public StandardFunction
{
public:
    Temp2() : StandardFunction(2, 2, 6) {}
    
    void evaluate(const vector<double>& input, vector<double>& output)
    {
        output.resize(2);
        output[0] = mParameters[0] * mParameters[0] * input[0] * input[0] + mParameters[1] * input[0] + mParameters[2];
        output[1] = mParameters[3] * mParameters[3] * input[1] * input[1] + mParameters[4] * input[1] + mParameters[5];
    }
       
    void calculateJacobianInputs(const vector<double>& x, Matrix& jacobian)
    {
        jacobian.setSize(2, 2);
        jacobian[0][0] = 2 * x[0] * mParameters[0] * mParameters[0] + mParameters[1];
        jacobian[0][1] = 0.0;
        jacobian[1][0] = 0.0;
        jacobian[1][1] = 2 * x[1] * mParameters[3] * mParameters[3] + mParameters[4];
    }
    
    void calculateJacobianParameters(const vector<double>& x, Matrix& jacobian)
    {
        jacobian.setSize(2, 6);
        jacobian.setAll(0.0);
        jacobian[0][0] = 2.0 * x[0] * x[0] * mParameters[0];
        jacobian[0][1] = x[0];
        jacobian[0][2] = 1;
        jacobian[1][3] = 2.0 * x[1] * x[1] * mParameters[3];
        jacobian[1][4] = x[1];
        jacobian[1][5] = 1;
    }
    
    void calculateHessianInputs(const vector<double>& /*x*/, const size_t outputIndex, Matrix& hessian)
    {
        hessian.setSize(2, 2);
        hessian.setAll(0.0);
        
        if (outputIndex == 0)
            hessian[0][0] = 2 * mParameters[0] * mParameters[0];
        else
            hessian[1][1] = 2 * mParameters[3] * mParameters[3];
    }
    
    void calculateHessianParameters(const vector<double>& x,
        const size_t outputIndex, Matrix& hessian)
    {
        hessian.setSize(6, 6);
        hessian.setAll(0.0);
        
        if (outputIndex == 0)
            hessian[0][0] = 2 * x[0] * x[0];
        else hessian[3][3] = 2 * x[1] * x[1];
    }
};


