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
#include "Layer.h"
#include "NeuralNetwork.h"
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

int main()
{
    // Create a network with three convolutional layers
    Convolutional2DLayer layer(20, 20, 3, 3, 10, 3, 2);
    layer.setActivationFunction(linearActivation);
    printf("Layer 1: (%zux%zux%zu) -> (%zux%zux%zu)\n",
        layer.getInputWidth(), layer.getInputHeight(), layer.getInputChannels(),
        layer.getOutputWidth(), layer.getOutputHeight(), layer.getNumFilters());

    Convolutional2DLayer layer2(8, 8, 10, 3, 5, 1, 0);
    layer.setActivationFunction(linearActivation);
    printf("Layer 2: (%zux%zux%zu) -> (%zux%zux%zu)\n",
        layer2.getInputWidth(), layer2.getInputHeight(), layer2.getInputChannels(),
        layer2.getOutputWidth(), layer2.getOutputHeight(), layer2.getNumFilters());
    
    Convolutional2DLayer layer3(6, 6, 5, 3, 5, 1, 0);
    layer.setActivationFunction(linearActivation);
    printf("Layer 3: (%zux%zux%zu) -> (%zux%zux%zu)\n",
        layer3.getInputWidth(), layer3.getInputHeight(), layer3.getInputChannels(),
        layer3.getOutputWidth(), layer3.getOutputHeight(), layer3.getNumFilters());
    
    NeuralNetwork network;
    network.addLayer(&layer);
    network.addLayer(&layer2);
    network.addLayer(&layer3);
    randomizeParameters(network.getParameters(), 0.0, 0.01);
    
    // Create a test input randomly
    size_t inputSize = layer.getInputWidth() * layer.getInputHeight() * layer.getInputChannels();
    vector<double> input(inputSize, 0.0);
    randomizeParameters(input, 0.0, 2.0);
    
    if (!testBackProp(network, input))
        return 1;
    
    return 0;
}