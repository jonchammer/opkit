/*
 * File:    fullyConnectedBackpropTest.cpp
 * Author:  Jon C. Hammer
 * Purpose: This tests the fully-connected backprop implementation for
 *          neural networks.
 *
 * Created on August 25, 2016, 9:23 PM
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include "opkit/opkit.h"

using namespace opkit;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using Type = float;

// Test Backprop by calculating the Jacobian using backprop and by using
// finite differences.
bool testBackpropParameters(NeuralNetwork<Type>& network, vector<Type>& input)
{
    cout << "Testing Backprop." << endl;
    Matrix<Type> jacobian, jacobian2;
    network.calculateJacobianParameters(input.data(), jacobian);
    network.Function<Type>::calculateJacobianParameters(input.data(), jacobian2);

//    cout << "Exact:" << endl;
//    printMatrix(cout, jacobian, 8, 11);
//
//    cout << "Finite Differences:" << endl;
//    printMatrix(cout, jacobian2, 8, 11);

    if (jacobian.getRows() != jacobian2.getRows() || jacobian.getCols() != jacobian2.getCols())
    {
        cout << "Backprop (parameters) - FAIL" << endl;
        cout << "Dimensions don't match." << endl;
        return false;
    }
    for (size_t j = 0; j < jacobian2.getRows(); ++j)
    {
        for (size_t k = 0; k < jacobian2.getCols(); ++k)
        {
            if (std::abs(jacobian(j, k) - jacobian2(j, k)) > 0.001)
            {
                cout << "Backprop (parameters) - FAIL" << endl;
                return false;
            }
        }
    }
    cout << "Backprop (parameters) - PASS" << endl;


    return true;
}

// Test Backprop by calculating the Jacobian using backprop and by using
// finite differences.
bool testBackpropInputs(NeuralNetwork<Type>& network, vector<Type>& input)
{
    cout << "Testing Backprop." << endl;
    Matrix<Type> jacobian, jacobian2;
    network.calculateJacobianInputs(input.data(), jacobian);
    network.Function<Type>::calculateJacobianInputs(input.data(), jacobian2);

//    cout << "Exact:" << endl;
//    printMatrix(jacobian, 8, 11);
//
//    cout << "Finite Differences:" << endl;
//    printMatrix(jacobian2, 8, 11);

    if (jacobian.getRows() != jacobian2.getRows() || jacobian.getCols() != jacobian2.getCols())
    {
        cout << "Backprop (inputs) - FAIL" << endl;
        cout << "Dimensions don't match." << endl;
        return false;
    }

    for (size_t j = 0; j < jacobian2.getRows(); ++j)
    {
        for (size_t k = 0; k < jacobian2.getCols(); ++k)
        {
            if (std::abs(jacobian(j, k) - jacobian2(j, k)) > 0.001)
            {
                cout << "Backprop (inputs) - FAIL" << endl;
                return false;
            }
        }
    }
    cout << "Backprop (inputs) - PASS" << endl;


    return true;
}

int main()
{
    // Create a test network
    NeuralNetwork<Type> network;

    network.addLayer(new FullyConnectedLayer<Type>(3, 100));
    network.addLayer(new ActivationLayer<Type>(100, new tanhActivation<Type>()));
    network.addLayer(new FullyConnectedLayer<Type>(100, 50));
    network.addLayer(new ActivationLayer<Type>(50, new tanhActivation<Type>()));
    network.addLayer(new FullyConnectedLayer<Type>(50, 10));
    //network.addLayer(new ActivationLayer<Type>(2, new tanhActivation<Type>()));
    network.addLayer(new SoftmaxLayer<Type>(10));

    Rand rand(42);
    network.initializeParameters(rand);

    // Test case
    vector<Type> input =
    {
        2.0, 3.0, -2.0
    };

    if (!testBackpropParameters(network, input))
        return 1;
    if (!testBackpropInputs(network, input))
        return 2;

    return 0;
}
