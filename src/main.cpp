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

//int main()
//{
//    size_t inputWidth    = 5;
//    size_t inputHeight   = 5;
//    size_t inputChannels = 3;
//    size_t filterSize    = 3;
//    size_t numFilters    = 2;
//    size_t stride        = 1;
//    size_t padding       = 0;
//    
//    ConvLayer layer(inputWidth, inputHeight, inputChannels, filterSize, numFilters, stride, padding);
//    vector<double> params = 
//    {
//        0, 1, -1, 0, 0, 1, 0, 0, 1, 
//        0, 0, -1, 0, -1, -1, 1, 0, -1, 
//        0, 1, 0, 1, -1, -1, 0, -1, -1, 
//        1,
//        0, -1, 0, -1, -1, -1, -1, -1, 1,
//        -1, -1, 1, -1, -1, 1, 0, 1, -1, 
//        -1, 1, 1, 0, 0, 0, -1, 0, 1,
//        0
//    };
//    layer.assignStorage(&params, 0);
//    
//    vector<double> input =
//    {
//        1, 0, 0, 2, 1,
//        2, 2, 0, 0, 0,
//        0, 1, 0, 0, 1,
//        2, 2, 1, 2, 0,
//        0, 0, 1, 2, 1,
//        
//        0, 2, 0, 0, 1,
//        0, 1, 0, 2, 1,
//        2, 0, 0, 1, 1,
//        1, 1, 2, 0, 1,
//        0, 2, 2, 2, 2,
//        
//        1, 0, 0, 0, 1,
//        0, 0, 1, 1, 0,
//        2, 0, 0, 2, 0,
//        1, 0, 0, 1, 1,
//        2, 2, 0, 0, 0
//    };
//    
//    vector<double> output;
//    layer.feed(input, output);
//    
//    vector<double>& net = layer.getNet();
//    size_t i = 0;
//    for (size_t l = 0; l < numFilters; ++l)
//    {
//        for (size_t j = 0; j < layer.getOutputWidth(); ++j)
//        {
//            for (size_t k = 0; k < layer.getOutputHeight(); ++k)
//                cout << net[i++] << " ";
//            cout << endl;
//        }
//        cout << endl;
//    }
//    return 0;
//}

int main()
{
    const string filename = "../data/iris.arff";

    // Define the vectors that will store our data
    Matrix features, labels;

    // Fill in the arrays with the data from the file
    if (!loadArff(filename, features, labels, 1))
    {
        cout << "Unable to open file: " << filename << endl;
        return 1;
    }
    
    // Normalize the data
    scaleAllColumns(features, 0.0, 1.0);
    labels = convertColumnToOneHot(labels, 0);
    
    // Create the model
    NeuralNetwork* base = new NeuralNetwork();
//    FeedforwardLayer l1(features.cols(), labels.cols());
//    l1.setActivationFunction(scaledTanhActivation);
//    base->addLayer(&l1);
    FeedforwardLayer l1(features.cols(), 10);
    FeedforwardLayer l2(10, 10);
    FeedforwardLayer l3(10, labels.cols());
    
    l1.setActivationFunction(scaledTanhActivation);
    //l2.setActivationFunction(scaledTanhActivation);
    //l3.setActivationFunction(scaledTanhActivation);
    
    base->addLayer(&l1);
    base->addLayer(&l2);
    base->addLayer(&l3);
    
//    vector<int> dimensions = {(int) features.cols(), (int) labels.cols()};    
//    NeuralNetwork* base    = new NeuralNetwork(dimensions);
//    
////    for (size_t i = 0; i < base->getNumLayers(); ++i)
////        base->getLayer(i).setActivationFunction(sinActivation);
//    base->getOutputLayer().setActivationFunction(sinActivation);
    
    ErrorFunction* f       = new SSEFunction(*base);
    randomizeParameters(base->getParameters(), 0.0, 0.01);
    
//    Matrix hessian;
//    Matrix gradient;
//    f->calculateJacobianParameters(features, labels, gradient);
//    f->calculateHessianParameters(features, labels, hessian);
//    
//    printMatrix(gradient);
//    cout << endl;
//    printMatrix(hessian);
    
    // Create the trainer
    typedef GradientDescent Learner;
    Learner* trainer = new Learner(f);
    
    const int ITERATIONS = 50000;
    
    // Optimize the model
    for (int i = 0; i < ITERATIONS; ++i)
    {
        trainer->iterate(features, labels);
        cout << "SSE: " << f->evaluate(features, labels) << endl;
    }
    
    
    
    delete f;
    delete base;
    delete trainer;
    
    return 0;
}


