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
#include "opkit/opkit.h"

using namespace opkit;
using std::cout;
using std::endl;
using std::vector;
using Type = float;

// Test Backprop by calculating the Jacobian using backprop and by using
// finite differences.
bool testBackProp(NeuralNetwork<Type>& network, vector<Type>& input)
{
    cout << "Testing Backprop." << endl;
    Matrix<Type> jacobian, jacobian2;
    network.calculateJacobianInputs(input.data(), jacobian);
    network.Function::calculateJacobianInputs(input.data(), jacobian2);

    // Sanity check
    if (jacobian.getRows() != jacobian2.getRows() ||
        jacobian.getCols() != jacobian2.getCols())
    {
        cout << "Mismatched dimensions. " << endl;
        cout << "Backprop - FAIL" << endl;
        return false;
    }

    for (size_t j = 0; j < jacobian.getRows(); ++j)
    {
        for (size_t k = 0; k < jacobian.getCols(); ++k)
        {
            if (abs(jacobian(j, k) - jacobian2(j, k)) > 0.001)
            {
                // cout << "Calculated Gradient:" << endl;
                // printMatrix(cout, jacobian, 2);
                // cout << "True Gradient:" << endl;
                // printMatrix(cout, jacobian2, 2);

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
    Convolutional2DLayer<Type> layer(20, 20, 3, 3, 3, 10, 3, 3, 2, 2);
    Convolutional2DLayer<Type> layer2(8, 8, 10, 3, 3, 5, 1, 1, 0, 0);
    Convolutional2DLayer<Type> layer3(6, 6, 5, 3, 3, 5, 1, 1, 0, 0);

    NeuralNetwork<Type> network;
    network.addLayer(&layer, false);
    network.addLayer(&layer2, false);
    network.addLayer(&layer3, false);
    network.print(cout, "");

    Rand rand(42);
    network.initializeParameters(rand);

    // Create a test input randomly
    size_t inputSize = layer.getInputWidth() * layer.getInputHeight() * layer.getInputChannels();
    vector<Type> input(inputSize, 0.0);
    randomizeParameters(input, rand, 0.0, 2.0);

    if (!testBackProp(network, input))
        return 1;

    return 0;
}
