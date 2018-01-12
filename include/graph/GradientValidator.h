#ifndef GRADIENT_VALIDATOR_H
#define GRADIENT_VALIDATOR_H

#include "graph/core/GraphAPI.h"
#include "graph/Gradients.h"
#include "tensor/Tensor.h"



namespace opkit
{

// Calculates the gradients of the given graph with respect to each of the given
// targets both empirically and using automatic differentiation. Returns true
// when the gradients agree and false when they do not.
template <class T>
bool validate(Graph<T> root, std::vector<Graph<T>>& targets, const double threshold = 1E-3)
{
    using namespace std;

    // Calculate the gradients using automatic differentiation
    std::unordered_set<std::string> names;
    for (Graph<T>& node : targets)
        names.insert(node.name());
    auto grads = gradients(root, names);

    // Calculate the gradients using finite differences
    auto grads2 = empiricalGradients(root, names);

    // Compare the gradients produced by each method
    for (const std::string& name : names)
    {
        // Ignore variables that did not directly affect the root node
        if (grads.find(name) == grads.end()) continue;

        Tensor<T> automatic = grads[name]();
        Tensor<T> empirical = grads2[name];

        // Check for NaN cells
        for (T& elem : empirical)
        {
            if (std::isnan(elem))
            {
                cout << "Warning: NaN produced by empirical estimator." << endl;
                cout << "Empirical: " << endl;
                cout << empirical     << endl;
                cout << "Automatic: " << endl;
                cout << automatic     << endl << endl;
            }
        }

        Tensor<T> diff      = sub(automatic, empirical);
        diff.apply([](const T x) { return std::abs(x); });
        if (T(reduceMax(diff)) > threshold)
        {
            cout << "Mismatch detected for variable: " << name << endl;
            cout << "Empirical:"            << endl;
            cout << to_string(empirical, 4) << endl;
            cout << "Automatic: "           << endl;
            cout << to_string(automatic, 4) << endl;

            return false;
        }
    }

    return true;
}

}
#endif
