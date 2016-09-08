/* 
 * File:    multiLayerConvolutionBackpropTest.cpp
 * Author:  Jon C. Hammer
 * Purpose: This tests backprop with multiple convolutional neural network layers.
 * 
 * Created on September 1, 2016, 6:45 PM
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include "ConvNeuralNetwork.h"
#include "PrettyPrinter.h"
#include "Tensor3D.h"
#include "ActivationFunction.h"
using namespace std;

// Test Backprop by calculating the Jacobian using backprop and by using
// finite differences. 
bool testBackProp(NeuralNetwork& network, vector<double>& input)
{    
    cout << "Testing Backprop." << endl;
    Matrix jacobian, jacobian2;
    network.calculateJacobianParameters(input, jacobian);
    network.Function::calculateJacobianParameters(input, jacobian2);
    
    for (size_t j = 0; j < jacobian.rows(); ++j)
    {
        for (size_t k = 0; k < jacobian.cols(); ++k)
        {
            if (abs(jacobian[j][k] - jacobian2[j][k]) > 0.001)
            {
                cout << "Calculated Gradient:" << endl;
                printMatrix(jacobian, 2);
                cout << "True Gradient:" << endl;
                printMatrix(jacobian2, 2);
    
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
    // Create a network with two convolutional layers
    ConvLayer layer(5, 5, 3, 3, 2, 4, 1);
    layer.setActivationFunction(linearActivation);
    printf("Layer 1: (%zux%zux%zu) -> (%zux%zux%zu)\n",
        layer.getInputWidth(), layer.getInputHeight(), layer.getInputChannels(),
        layer.getOutputWidth(), layer.getOutputHeight(), layer.getNumFilters());

    ConvLayer layer2(3, 3, 2, 3, 2, 1, 0);
    layer.setActivationFunction(linearActivation);
    printf("Layer 2: (%zux%zux%zu) -> (%zux%zux%zu)\n",
        layer2.getInputWidth(), layer2.getInputHeight(), layer2.getInputChannels(),
        layer2.getOutputWidth(), layer2.getOutputHeight(), layer2.getNumFilters());
    
    NeuralNetwork network;
    network.addLayer(&layer);
    network.addLayer(&layer2);
    randomizeParameters(network.getParameters(), 0.0, 0.01);
    
    // Test case
//    vector<double> input =
//    {
//        2, 2, 2, 2, 0,
//        0, 0, 2, 2, 1,
//        2, 0, 1, 1, 2,
//        1, 0, 1, 1, 0,
//        2, 1, 0, 1, 2,
//        
//        2, 1, 0, 1, 1,
//        2, 2, 1, 2, 0,
//        2, 2, 1, 1, 0,
//        2, 2, 1, 1, 1,
//        0, 0, 0, 1, 0,
//        
//        1, 2, 2, 0, 0,
//        2, 0, 0, 1, 0,
//        1, 0, 2, 1, 2,
//        0, 2, 1, 1, 1,
//        2, 2, 2, 2, 1
//    };
        
//    vector<double> input = 
//    {
//        1, 2, 3, 4, 5,
//        6, 7, 8, 9, 10,
//        11, 12, 13, 14, 15, 
//        16, 17, 18, 19, 20, 
//        21, 22, 23, 24, 25,
//        
//        1, 2, 3, 4, 5,
//        6, 7, 8, 9, 10,
//        11, 12, 13, 14, 15, 
//        16, 17, 18, 19, 20, 
//        21, 22, 23, 24, 25,
//        
//        1, 2, 3, 4, 5,
//        6, 7, 8, 9, 10,
//        11, 12, 13, 14, 15, 
//        16, 17, 18, 19, 20, 
//        21, 22, 23, 24, 25,
//    };
    
    vector<double> input(layer.getInputWidth() * layer.getInputHeight() * layer.getInputChannels(), 0.0);
    randomizeParameters(input, 0.0, 2.0);
    
    if (!testBackProp(network, input))
        return 1;
    
    return 0;
}