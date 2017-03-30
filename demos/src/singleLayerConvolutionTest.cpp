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
#include "opkit/opkit.h"

using namespace opkit;
using std::cout;
using std::endl;
using std::vector;
using Type = double;

// Tests forward prop by passing a known input through the network and comparing
// the result to a known output.
bool testForwardProp(NeuralNetwork<Type>& network, vector<Type>& input,
    vector<Type>& trueOutput)
{
     // Run through the convolution to get output
    cout << "Testing Feedforward." << endl;
    vector<Type> output(network.getOutputs());
    Convolutional2DLayer<Type>* layer  = (Convolutional2DLayer<Type>*) network.getLayer(0);
    size_t numChannels = layer->getOutputChannels();

    network.evaluate(input.data(), output.data());

    // Compare the output (pre-activation) to the true output.
    size_t i = 0;
    for (size_t l = 0; l < numChannels; ++l)
    {
        for (size_t j = 0; j < layer->getOutputWidth(); ++j)
        {
            for (size_t k = 0; k < layer->getOutputHeight(); ++k)
            {
                if (abs(output[i] - trueOutput[i]) > 0.001)
                {
                    cout << "Forward Prop - FAIL" << endl;

                    cout << "Network output:" << endl;
                    print3DTensor(cout, output, layer->getOutputWidth(),
                        layer->getOutputHeight(), numChannels);

                    cout << "Expected:" << endl;
                    print3DTensor(cout, trueOutput, layer->getOutputWidth(),
                        layer->getOutputHeight(), numChannels);
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
bool testBackProp(NeuralNetwork<Type>& network, vector<Type>& input)
{
    vector<Type> test =
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
    Matrix<Type> jacobian, jacobian2;
    network.calculateJacobianParameters(input.data(), jacobian);
    network.Function::calculateJacobianParameters(input.data(), jacobian2);

    // cout << "Calculated Gradient:" << endl;
    // printMatrix(cout, jacobian);
    // cout << "True Gradient:" << endl;
    // printMatrix(cout, jacobian2);

    for (size_t j = 0; j < jacobian.getRows(); ++j)
    {
        for (size_t k = 0; k < jacobian.getCols(); ++k)
        {
            if (abs(jacobian(j, k) - jacobian2(j, k)) > 0.1)
            {
                cout << j << " " << k << endl;
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
    Convolutional2DLayer<Type> layer(inputWidth, inputHeight, inputChannels,
        filterSize, filterSize, numFilters, padding, padding, stride, stride);
    NeuralNetwork<Type> network;
    network.addLayer(&layer, false);

    vector<Type> params =
    {
        // Weights - k1
        -1, 1, -1, 0, -1, 1, 1, -1, -1,
        0, -1, 1, 0, 1, -1, 0, -1, 1,
        0, 1, 0, 0, -1, 1, 0, 1, 0,

        // Weights - k2
        -1, 1, 0, 0, 0, 0, -1, 1, -1,
        -1, 1, 0, 1, 1, -1, -1, 0, -1,
        0, 1, -1, 1, 1, 1, 0, 1, -1,

        // Biases
        1, 0
    };
    std::copy(params.begin(), params.end(), network.getParameters().begin());

    // Test case
    vector<Type> input =
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

    vector<Type> trueOutput =
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
