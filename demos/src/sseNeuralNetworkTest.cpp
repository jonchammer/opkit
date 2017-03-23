/*
 * File:     neuralNetworkBatchTest.cpp
 * Author:   Jon C. Hammer
 * Purpose:  This is a test to make sure that the template specialization of
  *          SSEFunction for NeuralNetworks is accurate.
 *
 * Created on September 17, 2016, 10:14 AM
 */

#include <cmath>
#include <iostream>
#include <vector>
#include "opkit/opkit.h"
using namespace opkit;

int main()
{
    // Load the data
    cout << "Loading data..." << endl;
    Dataset<double> features, labels;
    Matrix<double> trainFeatures, trainLabels;
    if (!loadArff("/home/jhammer/data/iris.arff", features, labels, 1))
    {
        cout << "Unable to open file." << endl;
        return 1;
    }
    cout << "Data loaded!" << endl;

    // Process the data
    cout << "Preprocessing the data..." << endl;
    scaleAllColumns(features, -1.0, 1.0);
    labels = convertColumnToOneHot(labels, 0);
    features.toMatrix(trainFeatures);
    labels.toMatrix(trainLabels);
    cout << "Data ready!" << endl;

    // Create a testing network
    const size_t batchSize = trainFeatures.getRows();

    FullyConnectedLayer<double> layer1(trainFeatures.getCols(), 10);
    FullyConnectedLayer<double> layer2(10, trainLabels.getCols());
    SoftmaxLayer<double> layer3(trainLabels.getCols());

    NeuralNetwork<double> network(batchSize);
    network.addLayer(&layer1, false);
    network.addLayer(&layer2, false);
    network.addLayer(&layer3, false);

    Rand rand(42);
    network.initializeParameters(rand);

    // Create a trainer
    //SSEFunction<double, NeuralNetwork<double>> errorFunc(network);
    CrossEntropyFunction<double, NeuralNetwork<double>> errorFunc(network);

    // Calculate the gradient with respect to the parameters using the template
    // specialization and using the finite differences approach.
    vector<double> gradient(network.getNumParameters()), gradient2(network.getNumParameters());
    errorFunc.calculateGradientParameters(trainFeatures, trainLabels, gradient);
    errorFunc.CostFunction<double, NeuralNetwork<double>>::calculateGradientParameters(trainFeatures, trainLabels, gradient2);

    for (size_t i = 0; i < gradient.size(); ++i)
    {
        if (abs(gradient[i] - gradient2[i]) > 0.01)
        {
            cout << "Gradient parameters: Fail!" << endl;

            cout << "SSE Version (size " << gradient.size() << ")" << endl;
            printVector(cout, gradient, 4);

            cout << "Finite Differences Version (size " << gradient2.size() << ")" << endl;
            printVector(cout, gradient2, 4);
            return 1;
        }
    }
    cout << "Gradient parameters: Pass!" << endl;

    vector<double> gradientInputs(network.getInputs()), gradientInputs2(network.getInputs());
    errorFunc.calculateGradientInputs(trainFeatures, trainLabels, gradientInputs);
    errorFunc.CostFunction<double, NeuralNetwork<double>>::calculateGradientInputs(trainFeatures, trainLabels, gradientInputs2);

    for (size_t i = 0; i < gradientInputs.size(); ++i)
    {
        if (abs(gradientInputs[i] - gradientInputs2[i]) > 0.001)
        {
            cout << "Gradient inputs: Fail!" << endl;
            cout << abs(gradientInputs[i] - gradientInputs2[i]) << endl;

            cout << "SSE Version (size " << gradientInputs.size() << ")" << endl;
            printVector(cout, gradientInputs, 8);

            cout << "Finite Differences Version (size " << gradientInputs2.size() << ")" << endl;
            printVector(cout, gradientInputs2, 8);

            return 1;
        }
    }
    cout << "Gradient inputs: Pass!" << endl;

    return 0;
}
