/* 
 * File:    heterogeneousLayerBackpropTest.cpp
 * Author:  Jon C. Hammer
 * Purpose: This tests backprop with multiple layers of different types
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
    layer.setActivationFunction(tanhActivation);
    printf("Layer 1: (%zux%zux%zu) -> (%zux%zux%zu)\n",
        layer.getInputWidth(), layer.getInputHeight(), layer.getInputChannels(),
        layer.getOutputWidth(), layer.getOutputHeight(), layer.getNumFilters());

    Convolutional2DLayer layer2(8, 8, 10, 3, 5, 1, 0);
    layer.setActivationFunction(tanhActivation);
    printf("Layer 2: (%zux%zux%zu) -> (%zux%zux%zu)\n",
        layer2.getInputWidth(), layer2.getInputHeight(), layer2.getInputChannels(),
        layer2.getOutputWidth(), layer2.getOutputHeight(), layer2.getNumFilters());
    
    FeedforwardLayer layer3(layer2.getOutputs(), 50);
    layer.setActivationFunction(reluActivation);
    printf("Layer 3: %zu -> %zu\n", layer3.getInputs(), layer3.getOutputs());
    
    FeedforwardLayer layer4(layer3.getOutputs(), 10);
    layer.setActivationFunction(sinActivation);
    printf("Layer 4: %zu -> %zu\n", layer4.getInputs(), layer4.getOutputs());
    
    NeuralNetwork network;
    network.addLayer(&layer);
    network.addLayer(&layer2);
    network.addLayer(&layer3);
    network.addLayer(&layer4);
    randomizeParameters(network.getParameters(), 0.0, 0.01);
    
    // Create a test input randomly
    size_t inputSize = layer.getInputWidth() * layer.getInputHeight() * layer.getInputChannels();
    vector<double> input(inputSize, 0.0);
    randomizeParameters(input, 0.0, 2.0);
    
    if (!testBackProp(network, input))
        return 1;
    
    return 0;
}