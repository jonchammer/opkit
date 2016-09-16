/* 
 * File:   irisExample.cpp
 * Author: Jon C. Hammer
 *
 * Created on September 15, 2016, 2:12 PM
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
#include "DataLoader.h"
#include "ActivationFunction.h"
#include "Timer.h"
using namespace std;

int main()
{
    // Load the data
    cout << "Loading data..." << endl;
    Matrix trainFeatures, trainLabels;
    loadArff("../data/iris.arff", trainFeatures, trainLabels, 1);
    cout << "Data loaded!" << endl;
    
    // Process the data
    cout << "Preprocessing the data..." << endl;
    scaleAllColumns(trainFeatures, -1.0, 1.0);
    trainLabels = convertColumnToOneHot(trainLabels, 0);
    cout << "Data ready!" << endl;
    
    // Create a testing network
    FeedforwardLayer layer1(trainFeatures.cols(), 10);
    FeedforwardLayer layer2(10, 10);
    FeedforwardLayer layer3(10, trainLabels.cols());

    //Convolutional2DLayer layer1(28, 28, 1, 3, 10, 1, 0);
    //layer1.setActivationFunction(reluActivation);
    //FeedforwardLayer layer2(layer1.getOutputs(), 10);
    //layer2.setActivationFunction(reluActivation);

    NeuralNetwork network;
    network.addLayer(&layer1);
    network.addLayer(&layer2);
    network.addLayer(&layer3);
    randomizeParameters(network.getParameters(), 0.0, 0.001);    

    // Create a trainer
    SSEFunction errorFunc(network);
    CategoricalErrorFunction misclassifications(network);
    GradientDescent trainer(&errorFunc);
    trainer.setLearningRate(1E-4);
    
    //printf("%s\t%s\t%s\n", "Epoch", "Misclassifications", "SSE");
    //printf("%d\t%f\t%f\n", 0, misclassifications.evaluate(trainFeatures, trainLabels), 
    //    errorFunc.evaluate(trainFeatures, trainLabels));
    Timer timer;
    for (size_t i = 1; i <= 10000; ++i)
    {
        trainer.iterate(trainFeatures, trainLabels);
        //printf("%zu\t%f\t%f\n", i, misclassifications.evaluate(trainFeatures, trainLabels), 
        //    errorFunc.evaluate(trainFeatures, trainLabels));
    }
    cout << timer.getElapsedTimeSeconds() << endl;
    return 0;
}

