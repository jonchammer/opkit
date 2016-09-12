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
#include "SSEFunction.h"
#include "CategoricalErrorFunction.h"
#include "Matrix.h"
#include "DataNormalizer.h"
#include "ActivationFunction.h"
using namespace std;

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
    scaleAllColumns(testFeatures, 0.0, 255.0, -1.0, 1.0);
    trainLabels = convertColumnToOneHot(trainLabels, 0);
    testLabels  = convertColumnToOneHot(testLabels, 0);
    cout << "Data ready!" << endl;
    
    // Create a testing network
    FeedforwardLayer layer1(784, 300);
    //Convolutional2DLayer layer1(28, 28, 1, 3, 10, 1, 0);
    layer1.setActivationFunction(reluActivation);
    FeedforwardLayer layer2(layer1.getOutputs(), 10);
    layer2.setActivationFunction(reluActivation);
    NeuralNetwork network;
    network.addLayer(&layer1);
    network.addLayer(&layer2);
    
    // Create a trainer
    SSEFunction errorFunc(network);
    CategoricalErrorFunction misclassifications(network);
    GradientDescent trainer(&errorFunc);
    //trainer.setLearningRate(1E-5);
    
    printf("%f\t%f\n", misclassifications.evaluate(testFeatures, testLabels), errorFunc.evaluate(testFeatures, testLabels));
    for (size_t i = 0; i < 100; ++i)
    {
        trainer.iterate(trainFeatures, trainLabels);
        printf("%f\t%f\n", misclassifications.evaluate(testFeatures, testLabels), errorFunc.evaluate(testFeatures, testLabels));
    }
    
    return 0;
}