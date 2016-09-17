/* 
 * File:    activationTest.cpp
 * Author:  Jon C. Hammer
 * Purpose: This is a testbench for comparing different Neural Network 
 *          activation functions. The results are summarized in data/Activation 
 *          Function Tests.xlsx.
 * 
 * Created on August 24, 2016, 6:09 PM
 */

#include <iostream>
#include "Matrix.h"
#include "DataLoader.h"
#include "DataNormalizer.h"
#include "SSEFunction.h"
#include "GradientDescent.h"
#include "NeuralNetwork.h"

using namespace std;

//-----------------------------------------------//
// Testing parameters

// Use a 1, 2, or 3 layer Neural Network. Only one of the following should
// be uncommented.
//#define ONE_LAYER
//#define TWO_LAYER
#define THREE_LAYER

// Use the test activation in the first layer, all layers, or on the last
// layer. Only one of the following should be uncommented.
//#define FIRST_LAYER
//#define ALL_LAYERS
#define LAST_LAYER

// The number of iteration steps (epochs)
#define NUM_ITERATIONS 50000

// The activation function under test
#define TEST_ACTIVATION scaledTanhActivation
//-----------------------------------------------//

bool loadIris(Matrix& features, Matrix& labels)
{
    const string filename = "../data/iris.arff";

    // Fill in the arrays with the data from the file
    if (!loadArff(filename, features, labels, 1))
    {
        cout << "Unable to open file: " << filename << endl;
        return false;
    }
    
    // Normalize the data
    scaleAllColumns(features, 0.0, 1.0);
    labels = convertColumnToOneHot(labels, 0); 
    return true;
}

int main()
{
    // Load the data
    Matrix features, labels;
    if (!loadIris(features, labels))
        return 1;
    
    // Create the model
    NeuralNetwork base;

    // One layer Neural network
    #if defined ONE_LAYER

        FeedforwardLayer l1(features.cols(), labels.cols());
        base.addLayer(&l1);
        l1.setActivationFunction(TEST_ACTIVATION); 

    // Two layer Neural Network
    #elif defined TWO_LAYER

        FeedforwardLayer l1(features.cols(), 10);
        FeedforwardLayer l2(10, labels.cols());

        base.addLayer(&l1);
        base.addLayer(&l2);

        #if defined FIRST_LAYER
            l1.setActivationFunction(TEST_ACTIVATION);
        #elif defined ALL_LAYERS
            l1.setActivationFunction(TEST_ACTIVATION);
            l2.setActivationFunction(TEST_ACTIVATION);
        #else
            l2.setActivationFunction(TEST_ACTIVATION);
        #endif

    // Three layer Neural Network
    #else

        FeedforwardLayer l1(features.cols(), 10);
        FeedforwardLayer l2(10, 10);
        FeedforwardLayer l3(10, labels.cols());

        base.addLayer(&l1);
        base.addLayer(&l2);
        base.addLayer(&l3);

        #if defined FIRST_LAYER
            l1.setActivationFunction(TEST_ACTIVATION);
        #elif defined ALL_LAYERS
            l1.setActivationFunction(TEST_ACTIVATION);
            l2.setActivationFunction(TEST_ACTIVATION);
            l3.setActivationFunction(TEST_ACTIVATION);
        #else
            l3.setActivationFunction(TEST_ACTIVATION);
        #endif

    #endif
    
    ErrorFunction<NeuralNetwork>* f = new SSEFunction<NeuralNetwork>(base);
    randomizeParameters(base.getParameters(), 0.0, 0.01);
        
    cout << "Working..." << endl;
    
    // Optimize the model
    GradientDescent<NeuralNetwork> trainer(f);
    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        trainer.iterate(features, labels); 
    }
    cout << "SSE: " << f->evaluate(features, labels) << endl;
    
    delete f;    
    return 0;
}

