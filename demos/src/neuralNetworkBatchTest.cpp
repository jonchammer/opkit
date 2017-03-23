/*
 * File:    neuralNetworkBatchTest.cpp
 * Author:  Jon C. Hammer
 * Purpose: This is a test to make sure that the batch gradient calculation for
 *          neural networks is accurate.
 *
 * Created on February 19, 6:18 PM
 */

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include "opkit/opkit.h"
using namespace opkit;

void print(const Matrix<double>& m)
{
    for (size_t i = 0; i < m.getRows(); ++i)
    {
        for (size_t j = 0; j < m.getCols(); ++j)
            cout << setw(12) << m(i, j) << " ";
        cout << endl;
    }
}

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
    const size_t inputs    = trainFeatures.getCols();
    const size_t outputs   = trainLabels.getCols();

    FullyConnectedLayer<double> layer1(inputs, outputs);
    NeuralNetwork<double> network(batchSize);
    network.addLayer(&layer1, false);

    Rand rand(42);
    network.initializeParameters(rand);

    // Create a trainer
    SSEFunction<double, NeuralNetwork<double>> errorFunc(network);

    // Print inputs
    cout << "X:" << endl;
    print(trainFeatures);

    // Print W^T
    Matrix<double> weights(network.getParameters().data(), outputs, inputs);
    Matrix<double> wT;
    wT += transpose(weights);
    cout << "W^T:" << endl;
    print(wT);

    // Print biases
    Matrix<double> biases(network.getParameters().data() + inputs * outputs, 1, outputs);
    Matrix<double> ones(batchSize, 1);
    ones.fill(1.0);
    Matrix<double> biasMatrix;
    biasMatrix += ones * biases;
    cout << "Biases:" << endl;
    print(biasMatrix);

    // Print mathematical result
    Matrix<double> res;
    res += (trainFeatures * wT) + biasMatrix;
    cout << "Truth:" << endl;
    print(res);

    // Print Y
    Matrix<double> predictions(batchSize, outputs);
    network.evaluateBatch(trainFeatures, predictions);
    cout << "Predictions Batch" << endl;
    print(predictions);

    // Check that Y matches res
    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < outputs; ++j)
        {
            if (std::abs(res(i, j) - predictions(i, j)) > 1E-5)
            {
                cout << "Row: " << i << endl;
                printVector(cout, res(i), outputs);
                printVector(cout, predictions(i), outputs);
            }
        }
    }
    cout << "Batch match!" << endl;

    // Check that we get the same results working one sample at a time
    vector<double> prediction(outputs);
    for (size_t row = 0; row < batchSize; ++row)
    {
        network.evaluate(trainFeatures(row), prediction.data());
        for (size_t col = 0; col < outputs; ++col)
        {
            if (std::abs(predictions(row, col) - prediction[col]) > 0.001)
            {
                cout << "ROW: " << row << endl;

                cout << "BATCH OUTPUT: " << endl;
                printVector(cout, predictions(row), outputs);

                cout << "SINGLE OUTPUT: " << endl;
                printVector(cout, prediction);

                break;
            }
        }
    }
    cout << "INDIVIDUAL: PASSED" << endl;
    return 0;
}
