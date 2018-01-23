#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "graph/core/GraphAPI.h"
#include "graph/Gradients.h"
#include "graph/ops/GraphOps_all.h"

namespace opkit
{

template <class T>
Graph<T> gradientDescent(
    Graph<T> error,
    const std::unordered_set<std::string>& targets,
    Graph<T> lr)
{
    // Calculate the gradient graphs for each of the targets
    auto grads = gradients(error, targets);

    vector<Graph<T>> updateRules;
    for (auto& pair : grads)
    {
        // Locate the original node
        auto x = error.find(pair.first);
        ASSERT(x != nullptr, "Target " + pair.first + " not present in error graph");

        // Add the gradient descent logic
        updateRules.emplace_back(axpy(*x, pair.second, -lr));
    }

    // Tie all of the update rules together
    return list(updateRules);
}

// Helper that is a bit easier to use
template <class T, class U = T>
Graph<T> gradientDescent(
    Graph<T> error,
    std::unordered_set<std::string> targets = {},
    const U learningRate = 1E-3)
{
    auto lr = make_variable<T>("lr", Tensor<T>::fromScalar(learningRate));
    return gradientDescent(error, targets, lr);
}

template <class T>
Graph<T> gradientDescentMomentum(
    Graph<T> error,
    const std::unordered_set<std::string>& targets,
    Graph<T> lr,
    Graph<T> momentum,
    bool useNesterov)
{
    // Calculate the gradient graphs for each of the targets
    auto grads = gradients(error, targets);

    vector<Graph<T>> updateRules;
    for (auto& pair : grads)
    {
        // Locate the original node
        auto x = error.find(pair.first);
        ASSERT(x != nullptr, "Target " + pair.first + " not present in error graph");

        // Create a velocity vector per optimizable parameter
        Tensor<T>& value = ((Variable<T>&) x->node()).value();
        Tensor<T> init   = zeroesLike<T>(value);
        auto velocity    = make_variable<T>(x->name() + "_velocity", init);

        vector<Graph<T>> assignments;
        if (useNesterov)
        {
            // Update the velocity, apply the Nesterov step, then update the
            // parameters using the following equations:
            //
            // velocity = velocity * momentum + gradient
            // x       -= lr * gradient + (momentum * velocity)
            assignments.emplace_back(assign(velocity, velocity * momentum + pair.second));
            assignments.emplace_back(axpy(*x, pair.second + momentum * velocity, -lr));
        }
        else
        {
            assignments.emplace_back(assign(velocity, velocity * momentum + pair.second));
            assignments.emplace_back(axpy(*x, velocity, -lr));
        }

        // Tie the individual rules together using another list
        updateRules.emplace_back(list(assignments));
    }

    // Tie all of the update rules together
    auto res = list(updateRules);

    return list(updateRules);
}

// Helper that is a bit easier to use
template <class T, class U = T>
Graph<T> gradientDescentMomentum(
    Graph<T> error,
    const std::unordered_set<std::string> targets = {},
    const U learningRate = 1E-3,
    const U momentum = 1E-3,
    const bool useNesterov = false)
{
    auto lr  = make_variable<T>("lr",       Tensor<T>::fromScalar(learningRate));
    auto mom = make_variable<T>("momentum", Tensor<T>::fromScalar(momentum));
    return gradientDescentMomentum(error, targets, lr, mom, useNesterov);
}

template <class T>
Graph<T> adam(Graph<T> error,
    const std::unordered_set<std::string>& targets,
    Graph<T> learningRate,
    Graph<T> beta1,
    Graph<T> beta2,
    Graph<T> epsilon)
{
    // Calculate the gradient graphs for each of the targets
    auto grads = gradients(error, targets);

    vector<Graph<T>> updateRules;
    for (auto& pair : grads)
    {
        // Locate the original node
        auto x = error.find(pair.first);
        ASSERT(x != nullptr, "Target " + pair.first + " not present in error graph");
        ASSERT(x->type() == Graph<T>::Type::VAR, "Target " + pair.first + " is not a variable");

        // Create a mean and variance vector per optimizable parameter. We also
        // need a scalar for the current timestep
        const Tensor<T>& xVal     = (*x)();
        const Tensor<T>& beta1Val = beta1();
        const Tensor<T>& beta2Val = beta2();

        auto mean   = make_variable<T>(x->name() + "_mean",   zeroesLike<T>(xVal));
        auto var    = make_variable<T>(x->name() + "_var",    zeroesLike<T>(xVal));
        auto t      = make_variable<T>(x->name() + "_time",   Tensor<T>::fromScalar(0));

        vector<Graph<T>> assignments;

        // Update t
        assignments.emplace_back(addTo(t, make_constant<T>(1)));

        // Update the biased estimates for the mean and variance of the computed
        // gradient.
        assignments.emplace_back(assign(mean, beta1 * mean + (1 - beta1) * pair.second));
        assignments.emplace_back(assign(var,  beta2 * var  + (1 - beta2) * square(pair.second)));

        // Correct for the bias and compute a unique learning rate for each
        // parameter. Then descend the corrected gradient
        auto alpha = learningRate * sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t));
        assignments.emplace_back(subFrom(*x, alpha * mean / (sqrt(var) + epsilon)));

        // Tie the rules together using another list
        updateRules.emplace_back(list(assignments));
    }

    // Tie all of the update rules together
    return list(updateRules);
}

// Helper that is a bit easier to use
template <class T, class U = T>
Graph<T> adam(
    Graph<T> error,
    std::unordered_set<std::string> targets = {},
    const U learningRate                    = 1E-3,
    const U beta1                           = 0.9,
    const U beta2                           = 0.999,
    const U epsilon                         = std::sqrt(std::numeric_limits<T>::epsilon()))
{
    auto lr = make_variable<T>("lr",      Tensor<T>::fromScalar(learningRate));
    auto b1 = make_variable<T>("beta1",   Tensor<T>::fromScalar(beta1));
    auto b2 = make_variable<T>("beta2",   Tensor<T>::fromScalar(beta2));
    auto ep = make_variable<T>("epsilon", Tensor<T>::fromScalar(epsilon));

    return adam(error, targets, lr, b1, b2, ep);
}

template <class T>
Graph<T> rmsProp(Graph<T> error,
    const std::unordered_set<std::string>& targets,
    Graph<T> learningRate,
    Graph<T> decay,
    Graph<T> momentum,
    Graph<T> epsilon)
{
    // Calculate the gradient graphs for each of the targets
    auto grads = gradients(error, targets);

    vector<Graph<T>> updateRules;
    for (auto& pair : grads)
    {
        // Locate the original node
        auto x = error.find(pair.first);
        ASSERT(x != nullptr, "Target " + pair.first + " not present in error graph");
        ASSERT(x->type() == Graph<T>::Type::VAR, "Target " + pair.first + " is not a variable");

        // Create an RMS and velocity vector per optimizable parameter.
        const Tensor<T>& xValue = ((Variable<T>&) x->node()).value();
        auto rms = make_variable<T>(x->name() + "_rms",      onesLike<T>(xValue));
        auto mom = make_variable<T>(x->name() + "_momentum", onesLike<T>(xValue));

        vector<Graph<T>> ops;

        // RMS[i] = decay * RMS[i] + (1.0 - decay) * gradient^2
        // mom[i] = mom * momentum + learningRate * gradient / sqrt(RMS[i] + epsilon)
        // x     -= mom
        ops.emplace_back(assign(rms, decay * rms + (1 - decay) * square(pair.second)));
        ops.emplace_back(assign(mom, momentum * mom + learningRate * pair.second / sqrt(rms + epsilon)));
        ops.emplace_back(subFrom(*x, mom));

        // Tie the rules together using another list
        updateRules.emplace_back(list(ops));
    }

    // Tie all of the update rules together
    return list(updateRules);
}

// Helper that is a bit easier to use
template <class T, class U = T>
Graph<T> rmsProp(
    Graph<T> error,
    std::unordered_set<std::string> targets = {},
    const U learningRate                    = 1E-4,
    const U decay                           = 0.9,
    const U momentum                        = 1E-3,
    const U epsilon                         = std::sqrt(std::numeric_limits<T>::epsilon()))
{
    auto lr  = make_variable<T>("lr",       Tensor<T>::fromScalar(learningRate));
    auto dec = make_variable<T>("decay",    Tensor<T>::fromScalar(decay));
    auto mom = make_variable<T>("momentum", Tensor<T>::fromScalar(momentum));
    auto ep  = make_variable<T>("epsilon",  Tensor<T>::fromScalar(epsilon));

    return rmsProp(error, targets, lr, dec, mom, ep);
}

}
#endif
