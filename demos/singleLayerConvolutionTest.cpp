/* 
 * File:    convolutionTest.cpp
 * Author:  Jon C. Hammer
 * Purpose: This tests the functionality (forward and backprop) of the
 *          convolutional neural network layers.
 * 
 * Created on August 24, 2016, 8:04 PM
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include "NeuralNetwork.h"
#include "PrettyPrinter.h"
#include "Tensor3D.h"
#include "ActivationFunction.h"
using namespace std;

// Tests forward prop by passing a known input through the network and comparing
// the result to a known output.
bool testForwardProp(NeuralNetwork& network, vector<double>& input, 
    vector<double>& trueOutput)
{
     // Run through the convolution to get output
    cout << "Testing Feedforward." << endl;
    vector<double> output;
    Convolutional2DLayer* layer  = (Convolutional2DLayer*) network.getLayer(0);
    size_t numFilters = layer->getNumFilters();
    
    network.evaluate(input, output);
    vector<double>& net = layer->getNet();
    
    // Compare the output (pre-activation) to the true output.
    size_t i = 0;
    for (size_t l = 0; l < numFilters; ++l)
    {
        for (size_t j = 0; j < layer->getOutputWidth(); ++j)
        {
            for (size_t k = 0; k < layer->getOutputHeight(); ++k)
            {
                if (abs(net[i] - trueOutput[i]) > 0.001)
                {
                    cout << "Forward Prop - FAIL" << endl;
                    
                    cout << "Network output (pre-activation):" << endl;
                    print3DTensor(net, layer->getOutputWidth(), 
                        layer->getOutputHeight(), numFilters);
                    
                    cout << "Expected:" << endl;
                    print3DTensor(trueOutput, layer->getOutputWidth(), 
                        layer->getOutputHeight(), numFilters);
                    return false;
                }
                ++i;
            }
        }
    }
    cout << "Forward Prop - PASS" << endl;
    return true;
}

// Test Backprop by calculating the Jacobian using backprop and by using
// finite differences. 
bool testBackProp(NeuralNetwork& network, vector<double>& input)
{    
    vector<double> test = 
    {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25,
        
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25,
        
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25,
    };
    std::copy(test.begin(), test.end(), input.begin());
    
    cout << "Testing Backprop." << endl;
    Matrix jacobian, jacobian2;
    network.calculateJacobianParameters(input, jacobian);
    network.Function::calculateJacobianParameters(input, jacobian2);
    
    cout << "Calculated Gradient:" << endl;
    printMatrix(jacobian);
    cout << "True Gradient:" << endl;
    printMatrix(jacobian2);
    
    for (size_t j = 0; j < jacobian.rows(); ++j)
    {
        for (size_t k = 0; k < jacobian.cols(); ++k)
        {
            if (abs(jacobian[j][k] - jacobian2[j][k]) > 0.001)
            {
                cout << "Backprop - FAIL" << endl;
                return false;
            }
        }
    }
    cout << "Backprop - PASS" << endl;
    return true;
}

// Test setup from: http://cs231n.github.io/convolutional-networks/
int main()
{
    // Define the metaparameters used for this test
    size_t inputWidth    = 5;
    size_t inputHeight   = 5;
    size_t inputChannels = 3;
    size_t filterSize    = 3;
    size_t numFilters    = 2;
    size_t stride        = 2;
    size_t padding       = 1;
    
    // Create a single convolutional layer
    Convolutional2DLayer layer(inputWidth, inputHeight, inputChannels, filterSize, 
        numFilters, stride, padding);
    layer.setActivationFunction(linearActivation);
    NeuralNetwork network;
    network.addLayer(&layer);
    
    vector<double> params = 
    {
        -1, 1, -1, 0, -1, 1, 1, -1, -1,
        0, -1, 1, 0, 1, -1, 0, -1, 1,
        0, 1, 0, 0, -1, 1, 0, 1, 0,
        1,
        -1, 1, 0, 0, 0, 0, -1, 1, -1,
        -1, 1, 0, 1, 1, -1, -1, 0, -1,
        0, 1, -1, 1, 1, 1, 0, 1, -1,
        0
    };
    std::copy(params.begin(), params.end(), network.getParameters().begin());
    
    // Test case
    vector<double> input =
    {
        2, 2, 2, 2, 0,
        0, 0, 2, 2, 1,
        2, 0, 1, 1, 2,
        1, 0, 1, 1, 0,
        2, 1, 0, 1, 2,
        
        2, 1, 0, 1, 1,
        2, 2, 1, 2, 0,
        2, 2, 1, 1, 0,
        2, 2, 1, 1, 1,
        0, 0, 0, 1, 0,
        
        1, 2, 2, 0, 0,
        2, 0, 0, 1, 0,
        1, 0, 2, 1, 2,
        0, 2, 1, 1, 1,
        2, 2, 2, 2, 1
    };
    
    vector<double> trueOutput =
    {
        5, -5, 3,
        -1, 0, -3,
        1, 2, -3, 
        
        4, -1, -1,
        2, 2, 0,
        5, 5, 4
    };
    
    if (!testForwardProp(network, input, trueOutput))
        return 1;
    else if (!testBackProp(network, input))
        return 2;
    
    return 0;
}