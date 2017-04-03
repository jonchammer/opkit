#include <iostream>
#include <vector>
#include "opkit/opkit.h"

using namespace opkit;
using std::cout;
using std::endl;
using std::vector;
using T = float;

bool checkJacobianParameters(NeuralNetwork<T>& network)
{
    cout << "TEST - Jacobian Parameters" << endl;

    // Generate test data
    Rand rand;
    vector<T> x(network.getInputs());
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = rand.nextReal(-1.0, 1.0);

    Matrix<T> jacobianParameters, jacobianParameters2;
    network.calculateJacobianParameters(x.data(), jacobianParameters);
    network.Function::calculateJacobianParameters(x.data(), jacobianParameters2);

    for (size_t i = 0; i < jacobianParameters2.getRows(); ++i)
    {
        for (size_t j = 0; j < jacobianParameters2.getCols(); ++j)
        {
            if (std::abs(jacobianParameters(i, j) - jacobianParameters2(i, j)) > 1E-3)
            {
                cout << "Jacobian Parameters - Test Failed." << endl;
                cout << "Exact: " << endl;
                printMatrix(cout, jacobianParameters);

                cout << "Approximate: " << endl;
                printMatrix(cout, jacobianParameters2);
                return false;
            }
        }
    }
    cout << "Jacobian Parameters - Test Passed." << endl;
    return true;
}

bool checkJacobianInputs(NeuralNetwork<T>& network)
{
    cout << "TEST - Jacobian Inputs" << endl;

    // Generate test data
    Rand rand;
    vector<T> x(network.getInputs());
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = rand.nextReal(-1.0, 1.0);

    Matrix<T> jacobianInputs, jacobianInputs2;
    network.calculateJacobianInputs(x.data(), jacobianInputs);
    network.Function::calculateJacobianInputs(x.data(), jacobianInputs2);

    for (size_t i = 0; i < jacobianInputs2.getRows(); ++i)
    {
        for (size_t j = 0; j < jacobianInputs2.getCols(); ++j)
        {
            if (std::abs(jacobianInputs(i, j) - jacobianInputs2(i, j)) > 1E-3)
            {
                cout << "Jacobian Inputs - Test Failed." << endl;
                cout << "Exact: " << endl;
                printMatrix(cout, jacobianInputs);

                cout << "Approximate: " << endl;
                printMatrix(cout, jacobianInputs2);
                return false;
            }
        }
    }
    cout << "Jacobian Inputs - Test Passed." << endl;
    return true;
}

bool checkErrorParameters(NeuralNetwork<T>& network, CostFunction<T, NeuralNetwork<T>>& costFunction)
{
    cout << "TEST - Error Parameters" << endl;

    const size_t batchSize = network.getMaxBatchSize();

    // Generate some training data
    Rand rand;
    Matrix<T> x(batchSize, network.getInputs());
    Matrix<T> y(batchSize, network.getOutputs());
    for (size_t row = 0; row < batchSize; ++row)
    {
        for (size_t col = 0; col < network.getInputs(); ++col)
            x(row, col) = rand.nextReal(-1.0, 1.0);
        y(row, rand.nextInteger(0, (int) network.getOutputs() - 1)) = 1.0;
    }

    vector<T> gradient(network.getNumParameters());
    vector<T> gradient2(network.getNumParameters());

    costFunction.calculateGradientParameters(x, y, gradient);
    costFunction.CostFunction::calculateGradientParameters(x, y, gradient2);

    for (size_t i = 0; i < gradient2.size(); ++i)
    {
        if (std::abs(gradient[i] - gradient2[i]) > 1E-2)
        {
            cout << "Gradient Error Parameters - Test Failed" << endl;
            cout << "Exact: " << endl;
            printVector(cout, gradient, 3);

            cout << "Approximate: " << endl;
            printVector(cout, gradient2, 3);
            return false;
        }
    }

    cout << "Gradient Error Parameters - Test Passed" << endl;
    return true;
}

bool checkErrorInputs(NeuralNetwork<T>& network, CostFunction<T, NeuralNetwork<T>>& costFunction)
{
    cout << "TEST - Error Inputs" << endl;
    const size_t batchSize = network.getMaxBatchSize();

    // Generate some training data
    Rand rand;
    Matrix<T> x(batchSize, network.getInputs());
    Matrix<T> y(batchSize, network.getOutputs());
    for (size_t row = 0; row < batchSize; ++row)
    {
        for (size_t col = 0; col < network.getInputs(); ++col)
            x(row, col) = rand.nextReal(-1.0, 1.0);
        y(row, rand.nextInteger(0, (int) network.getOutputs() - 1)) = 1.0;
    }

    vector<T> gradient(network.getInputs());
    vector<T> gradient2(network.getInputs());

    costFunction.calculateGradientInputs(x, y, gradient);
    costFunction.CostFunction::calculateGradientInputs(x, y, gradient2);

    for (size_t i = 0; i < gradient2.size(); ++i)
    {
        if (std::abs(gradient[i] - gradient2[i]) > 1E-2)
        {
            cout << "Gradient Error Inputs - Test Failed" << endl;
            cout << "Exact: " << endl;
            printVector(cout, gradient, 3);

            cout << "Approximate: " << endl;
            printVector(cout, gradient2, 3);
            return false;
        }
    }

    cout << "Gradient Error Inputs - Test Passed" << endl;
    return true;
}

void createNetwork(NeuralNetwork<T>& network)
{
    Rand rand(42);

    // ElasticDeformationLayer<T>* pre = new ElasticDeformationLayer<T>(
    //     10, 10, 1, 10, 10,
    //     ElasticDeformationLayer<T>::InterpolationScheme::CLAMPED_BILINEAR,
    //     42);
    // pre->setRotationRange(-10.0 * M_PI / 180.0, 10.0 * M_PI / 180.0);
    // pre->setScaleXRange(0.8, 1.0);
    // pre->setScaleYRange(0.8, 1.0);
    //
    // network.addLayer(pre);
    // network.addLayer(new FullyConnectedLayer<T>(100, 50));
    // network.addLayer(new ActivationLayer<T>(50, new tanhActivation<T>()));
    // network.addLayer(new FullyConnectedLayer<T>(50, 10));
    // network.addLayer(new SoftmaxLayer<T>(10));

    Convolutional1DLayer<T>* l1 = new Convolutional1DLayer<T>(100, 1, 3, 32, 1, 2);
    Convolutional1DLayer<T>* l2 = new Convolutional1DLayer<T>(
        l1->getOutputSize(), l1->getOutputChannels(), 5, 64, 0, 1);

    network.addLayer(l1);
    network.addLayer(new ActivationLayer<T>(l1->getOutputs(), new tanhActivation<T>()));
    network.addLayer(l2);
    network.addLayer(new SoftmaxLayer<T>(l2->getOutputs()));

    network.print(cout, "");
    network.initializeParameters(rand);
}

void createCostFunctions(
    vector<CostFunction<T, NeuralNetwork<T>>*>& costFunctions,
    NeuralNetwork<T>& network)
{
    // 1. SSE
    costFunctions.push_back(new SSEFunction<T, NeuralNetwork<T>>(network));

    // 2. Cross Entropy
    costFunctions.push_back(new CrossEntropyFunction<T, NeuralNetwork<T>>(network));

    // 3. Compound Cost
    CompoundCostFunction<T, NeuralNetwork<T>>* compound = new CompoundCostFunction<T, NeuralNetwork<T>>(network);
    costFunctions.push_back(compound);
    compound->addCostFunction(new SSEFunction<T, NeuralNetwork<T>>(network));
    compound->addCostFunction(new CrossEntropyFunction<T, NeuralNetwork<T>>(network));
    compound->addCostFunction(new L2Regularizer<T, NeuralNetwork<T>>(network));
}

int main()
{
    const size_t BATCH_SIZE = 10;

    // -----------------------------------------------------------------------//
    // Initialize the testing setup
    // -----------------------------------------------------------------------//

    NeuralNetwork<T> network(BATCH_SIZE);
    createNetwork(network);

    vector<CostFunction<T, NeuralNetwork<T>>*> costFunctions;
    createCostFunctions(costFunctions, network);

    // -----------------------------------------------------------------------//
    // Run the tests
    // -----------------------------------------------------------------------//

    // checkJacobianParameters(network); cout << endl;
    // checkJacobianInputs(network);     cout << endl;

    for (size_t i = 0; i < costFunctions.size(); ++i)
    {
        cout << "Cost Function: " << i << endl;
        checkErrorParameters(network, *costFunctions[i]); cout << endl;
        checkErrorInputs(network, *costFunctions[i]);     cout << endl;
    }

    return 0;
}
