#ifndef GRADIENT_VALIDATOR_H
#define GRADIENT_VALIDATOR_H

#include "graph/Graph.h"
#include "graph/Gradients.h"
#include "tensor/Tensor.h"

namespace opkit
{

// Calculates the gradients of the given graph with respect to each of the given
// targets both empirically and using automatic differentiation. Returns true
// when the gradients agree and false when they do not.
template <class T>
bool validate(Graph<T>& root, std::vector<Graph<T>>& targets, const double threshold = 1E-3)
{
    using namespace std;

    // Calculate the gradients using automatic differentiation
    std::unordered_set<std::string> names;
    for (Graph<T>& node : targets)
        names.insert(node.name());
    auto grads = gradients(root, names);

    // Calculate the gradient empirically and compare to the gradient found by
    // automatic differentiation.
    // d/dx ~= (f(x + dx) - f(x - dx)) / (2 * dx)
    const static T DELTA     = std::sqrt(std::numeric_limits<T>::epsilon());
    const static T INV_DENOM = T{0.5} / DELTA;
    for (Graph<T>& variable : targets)
    {
        Tensor<T>& value = ((Variable<T>&) variable.node()).value();

        // Declare a temporary tensor to hold the derivative
        const auto& shape = value.shape();
        Tensor<T> derivative(shape.begin(), shape.end());
        derivative.fill(T{});

        auto derivativeIt = derivative.begin();
        for (T& elem : value)
        {
            T orig       = elem;
            elem         = orig + DELTA;
            Tensor<T> f1 = root.evaluate(true);

            elem         = orig - DELTA;
            Tensor<T> f2 = root.evaluate(true);
            elem         = orig;

            // Calculate the derivative by subtracting the two estimates, dividing
            // by 2 * DELTA, and then adding all the entries together to get a
            // single scalar.
            *derivativeIt = T(reduceSum(multiply(Tensor<T>::fromScalar(INV_DENOM), sub(f1, f2))));
            ++derivativeIt;
        }

        if (grads.find(variable.name()) != grads.end())
        {
            //cout << grads.find(variable.name())->second << endl;

            Tensor<T> test       = grads.find(variable.name())->second.evaluate(true);
            Tensor<T> difference = sub(test, derivative);
            for (const auto& elem : difference)
            {
                if (std::abs(elem) > threshold)
                {
                    cout << "Mismatch detected for variable: " << variable.name() << endl;
                    cout << "Expected:"              << endl;
                    cout << to_string(derivative, 4) << endl;
                    cout << "Got: "                  << endl;
                    cout << to_string(test, 4)       << endl;

                    return false;
                }
            }
        }
    }

    return true;
}

}
#endif
