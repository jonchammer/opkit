/*
 * File:   sseNeuralNetworkTest.cpp
 * Author: Jon C. Hammer
 * Purpose: This is a test to make sure that the template specialization of
 *          SSEFunction for NeuralNetworks is accurate.
 *
 * Created on September 17, 2016, 10:14 AM
 */

#include <cmath>
#include <iostream>
#include <vector>
#include "DataNormalizer.h"
#include "DataLoader.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include "SSEFunction.h"
#include "PrettyPrinter.h"
using namespace std;

int main()
{
    // Load the data
    cout << "Loading data..." << endl;
    Dataset<double> trainFeatures, trainLabels;
    loadArff("../data/iris.arff", trainFeatures, trainLabels, 1);
    cout << "Data loaded!" << endl;

    // Process the data
    cout << "Preprocessing the data..." << endl;
    scaleAllColumns(trainFeatures, -1.0, 1.0);
    trainLabels = convertColumnToOneHot(trainLabels, 0);
    cout << "Data ready!" << endl;

    // Create a testing network
    FeedforwardLayer layer1(trainFeatures.cols(), 10);
    FeedforwardLayer layer2(10, trainLabels.cols());

    NeuralNetwork network;
    network.addLayer(&layer1);
    network.addLayer(&layer2);
    randomizeParameters(network.getParameters(), 0.0, 0.001);

    // Create a trainer
    SSEFunction<NeuralNetwork> errorFunc(network);

    // Calculate the gradient with respect to the parameters using the template
    // specialization and using the finite differences approach.
    vector<double> gradient, gradient2;
    errorFunc.calculateGradientParameters(trainFeatures, trainLabels, gradient);
    errorFunc.ErrorFunction<NeuralNetwork>::calculateGradientParameters(trainFeatures, trainLabels, gradient2);

    for (size_t i = 0; i < gradient.size(); ++i)
    {
        if (abs(gradient[i] - gradient2[i]) > 0.01)
        {
            cout << "Gradient parameters: Fail!" << endl;

            cout << "SSE Version (size " << gradient.size() << ")" << endl;
            printVector(gradient, 4);

            cout << "Finite Differences Version (size " << gradient2.size() << ")" << endl;
            printVector(gradient2, 4);
            return 1;
        }
    }
    cout << "Gradient parameters: Pass!" << endl;

    errorFunc.calculateGradientInputs(trainFeatures, trainLabels, gradient);
    errorFunc.ErrorFunction<NeuralNetwork>::calculateGradientInputs(trainFeatures, trainLabels, gradient2);

    for (size_t i = 0; i < gradient.size(); ++i)
    {
        if (abs(gradient[i] - gradient2[i]) > 0.001)
        {
            cout << "Gradient inputs: Fail!" << endl;
            cout << abs(gradient[i] - gradient2[i]) << endl;

            cout << "SSE Version (size " << gradient.size() << ")" << endl;
            printVector(gradient, 8);

            cout << "Finite Differences Version (size " << gradient2.size() << ")" << endl;
            printVector(gradient2, 8);

            return 1;
        }
    }
    cout << "Gradient inputs: Pass!" << endl;

    return 0;
}
