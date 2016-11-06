/* 
 * File:   mnistConvolutionExample.cpp
 * Author: Jon C. Hammer
 *
 * Created on September 11, 2016, 9:21 AM
 */

#include <iostream>
#include "Layer.h"
#include "NeuralNetwork.h"
#include "StochasticGradientDescent.h"
#include "GradientDescent.h"
#include "RMSProp.h"
#include "SSEFunction.h"
#include "CategoricalErrorFunction.h"
#include "Matrix.h"
#include "DataNormalizer.h"
#include "ActivationFunction.h"
#include "PrettyPrinter.h"
using namespace std;

double variance(const Matrix& matrix, size_t column)
{
    // Calculate the variance
    double sum  = 0.0;
    double mean = matrix.columnMean(column);
    for (size_t i = 0; i < matrix.rows(); ++i)
    {
        double temp = matrix[i][column] - mean;
        sum += temp * temp;
    }
    
    return sum / (matrix.rows() - 1);
}

int main()
{
    // Load the data
    cout << "Loading data..." << endl;
    Matrix trainFeatures, trainLabels, testFeatures, testLabels;
    trainFeatures.loadARFF("../data/mnist_train_features_small.arff");
    trainLabels.loadARFF("../data/mnist_train_labels_small.arff");
    testFeatures.loadARFF("../data/mnist_test_features.arff");
    testLabels.loadARFF("../data/mnist_test_labels.arff");
    cout << "Data loaded!" << endl;
    
    // Process the data
    cout << "Preprocessing the data..." << endl;
    scaleAllColumns(trainFeatures, 0.0, 255.0, -1.0, 1.0);
    scaleAllColumns(testFeatures,  0.0, 255.0, -1.0, 1.0);
    normalizeVarianceAllColumns(trainFeatures);
    normalizeVarianceAllColumns(testFeatures);
    trainLabels = convertColumnToOneHot(trainLabels, 0);
    testLabels  = convertColumnToOneHot(testLabels, 0);
    cout << "Data ready!" << endl;
    
    // Create a testing network
    FeedforwardLayer layer1(784, 300);
    FeedforwardLayer layer2(300, 100);
    FeedforwardLayer layer3(100, 10);
    //FeedforwardLayer layer1(784, 300);
//    Convolutional2DLayer layer1(28, 28, 1, 3, 5, 3, 1);
//    layer1.setActivationFunction(reluActivation);
//    Convolutional2DLayer layer2(layer1.getOutputWidth(), layer1.getOutputHeight(), layer1.getNumFilters(), 3, 5, 3, 1);
//    layer2.setActivationFunction(reluActivation);
//    FeedforwardLayer layer3(layer2.getOutputs(), 10);
    //layer3.setActivationFunction(reluActivation);
    
    NeuralNetwork network;
    network.addLayer(&layer1);
    network.addLayer(&layer2);
    network.addLayer(&layer3);
    
    printf("%zu x %zu\n", layer1.getInputs(), layer1.getOutputs());
    printf("%zu x %zu\n", layer2.getInputs(), layer2.getOutputs());
    printf("%zu x %zu\n", layer3.getInputs(), layer3.getOutputs());
    cout << "# Parameters: " << network.getParameters().size() << endl;
    
    randomizeParameters(network.getParameters(), 0.0, 0.01);
    //layer1.normalizeKernels();
    //layer2.normalizeKernels();
    
    // Create a trainer
    SSEFunction<NeuralNetwork> errorFunc(network);
    CategoricalErrorFunction<NeuralNetwork> misclassifications(network);
    //GradientDescent<NeuralNetwork> trainer(&errorFunc);
    RMSProp<NeuralNetwork> trainer(&errorFunc);
    trainer.setLearningRate(1E-2);
    trainer.setDecay(0.0);
    trainer.setMomentum(0.0);
    
    printf("%5zu: %.0f\t%f\n", 
        0,
        misclassifications.evaluate(testFeatures, testLabels), 
        errorFunc.evaluate(testFeatures, testLabels));
    cout.flush();
    for (size_t i = 0; i < 1000; ++i)
    {
        trainer.iterate(trainFeatures, trainLabels);
        
        //printVector(network.getParameters(), 4);
        //layer1.normalizeKernels();
        //printVector(network.getParameters(), 4);
        
        printf("%5zu: %.0f\t%f\n", 
            (i+1),
            misclassifications.evaluate(testFeatures, testLabels), 
            errorFunc.evaluate(testFeatures, testLabels));
        cout.flush();
    }
    
    return 0;
}