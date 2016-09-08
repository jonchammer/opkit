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
#include "ConvNeuralNetwork.h"
#include "PrettyPrinter.h"
#include "Tensor3D.h"
using namespace std;

// Test Backprop by calculating the Jacobian using backprop and by using
// finite differences. 
bool testBackpropParameters(NeuralNetwork& network, vector<double>& input)
{    
    cout << "Testing Backprop." << endl;
    Matrix jacobian, jacobian2;
    network.calculateJacobianParameters(input, jacobian);
    network.Function::calculateJacobianParameters(input, jacobian2);
    
//    cout << "Exact:" << endl;
//    printMatrix(jacobian, 8, 11);
//    
//    cout << "Finite Differences:" << endl;
//    printMatrix(jacobian2, 8, 11);
    
    for (size_t j = 0; j < jacobian.rows(); ++j)
    {
        for (size_t k = 0; k < jacobian.cols(); ++k)
        {
            if ((jacobian[j][k] - jacobian2[j][k]) > 0.001)
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
bool testBackpropInputs(NeuralNetwork& network, vector<double>& input)
{    
    cout << "Testing Backprop." << endl;
    Matrix jacobian, jacobian2;
    network.calculateJacobianInputs(input, jacobian);
    network.Function::calculateJacobianInputs(input, jacobian2);
    
//    cout << "Exact:" << endl;
//    printMatrix(jacobian, 8, 11);
//    
//    cout << "Finite Differences:" << endl;
//    printMatrix(jacobian2, 8, 11);
    
    for (size_t j = 0; j < jacobian.rows(); ++j)
    {
        for (size_t k = 0; k < jacobian.cols(); ++k)
        {
            if ((jacobian[j][k] - jacobian2[j][k]) > 0.001)
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
    NeuralNetwork network;
    
    FeedforwardLayer l1(3, 100);
    FeedforwardLayer l2(100, 50);
    FeedforwardLayer l3(50, 2);
    
    network.addLayer(&l1);
    network.addLayer(&l2);
    network.addLayer(&l3);
    
    // Initialize the weights and biases
    randomizeParameters(network.getParameters(), 0.0, 0.01);
    
    // Test case
    vector<double> input =
    {
        2.0, 3.0, -2.0
    };
       
    if (!testBackpropParameters(network, input))
        return 1;
    else if (!testBackpropInputs(network, input))
        return 2;
    
    return 0;
}