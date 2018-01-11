#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "tensor/Tensor.h"
#include "graph/core/GraphAPI.h"
#include "graph/GraphSimplifier.h"
#include "graph/ops/GraphOps_all.h"

namespace opkit
{

// Calculates the gradients of the given graph with respect to each of the
// variables listed in 'targets'. The result is a map from the variable names to
// the corresponding graph that will calculate the gradient. If the targets set
// is empty, all gradients with respect to all variables will be returned.
template <class T>
std::unordered_map<std::string, Graph<T>> gradients(Graph<T> node,
    std::unordered_set<std::string> targets = {})
{
    // Reference to the singleton derivative map
    DerivativeMap<T>& derivativeMap = DerivativeMap<T>::instance();

    // Calculate all derivatives by walking the source graph in reverse using
    // a BFS. Save those gradients that are given in the 'targets' list to a
    // map that is returned to the user.
    std::unordered_map<std::string, Graph<T>> namesToGradients;
    std::vector<Graph<T>> grads;
    std::queue<Graph<T>> nodeQueue;
    std::queue<Graph<T>> gradQueue;
    nodeQueue.push(node);
    gradQueue.push(expand(make_constant<T>(1), shape(node)));

    while (!nodeQueue.empty())
    {
        // Remove the first element of each queue
        Graph<T> cur = nodeQueue.front();
        nodeQueue.pop();
        Graph<T> delta = gradQueue.front();
        gradQueue.pop();

        // When we see a variable, we append the gradient to the current graph
        // for that variable.
        if (cur.type() == Graph<T>::Type::VAR)
        {
            if (targets.empty() || targets.find(cur.name()) != targets.end())
            {
                auto it = namesToGradients.find(cur.name());
                if (it == namesToGradients.end())
                    namesToGradients.emplace(cur.name(), delta);

                else it->second = it->second + delta;
            }
        }

        // When we see a function, we calculate its gradient by calling the
        // corresponding derivative function. We then push that node's parents
        // and their current gradients into the queues so the process can
        // continue recursively.
        else if (cur.type() != Graph<T>::Type::CONSTANT)
        {
            // Calculate the new deltas. Ignore any nodes that are not
            // differentiable.
            grads.clear();
            if (derivativeMap.call(cur.name(), cur, delta, grads))
            {
                for (size_t i = 0; i < cur.getNumParents(); ++i)
                {
                    nodeQueue.push(cur.getParent(i));
                    gradQueue.push(grads[i]);
                }
            }
            // else std::cout << "Gradients.h - Skipping: " << cur.name() << std::endl;
        }
    }

    // Clean up the gradients before giving them to the user
    for (auto& pair : namesToGradients)
        pair.second = simplify(pair.second);
    return namesToGradients;
}
}

namespace detail
{
    template <class T>
    struct Epsilon{};

    template <>
    struct Epsilon<float>
    {
        constexpr static float value = 1E-2;
    };

    template <>
    struct Epsilon<double>
    {
        constexpr static double value = std::sqrt(std::numeric_limits<double>::epsilon());
    };
}

namespace opkit
{
// Calculates the gradients of the given graph with respect to each of the
// variables listed in 'targets' empirically using a finite difference
// approximation. The result is a map from the variable names to
// a corresponding tensor that will calculate the gradient. The targets set
// must not be empty.
template <class T>
std::unordered_map<std::string, Tensor<T>> empiricalGradients(Graph<T> node,
    std::unordered_set<std::string> targets)
{
    std::unordered_map<std::string, Tensor<T>> grads;

    // Calculate the gradient empirically using the finite difference
    // approximation:
    // d/dx ~= (f(x + dx) - f(x - dx)) / (2 * dx)
    const static T DELTA     = detail::Epsilon<T>::value;
    const static T INV_DENOM = T{0.5} / DELTA;
    for (const std::string& name : targets)
    {
        Graph<T>* ptr = node.find(name);
        if (ptr == nullptr) continue;

        Variable<T>& variable = (Variable<T>&) ptr->node();

        // Work with a copy of the original tensor because other nodes in the
        // graph might share its storage. (Changing an element of this variable
        // might affect other variables unintentionally).
        Tensor<T>& orig = variable.value();
        Tensor<T> value = orig.clone();
        ptr->assign(value);

        // Declare a temporary tensor to hold the derivative
        const SmallVector& shape = value.shape();
        Tensor<T> derivative(shape.begin(), shape.end());
        derivative.fill(T{});

        auto derivativeIt = derivative.begin();
        for (T& elem : value)
        {
            T orig = elem;

            elem = orig + DELTA;
            ptr->invalidate();
            Tensor<T> f1 = node().clone();

            elem = orig - DELTA;
            ptr->invalidate();
            Tensor<T> f2 = node().clone();

            elem = orig;

            // Calculate the derivative by subtracting the two estimates, dividing
            // by 2 * DELTA, and then adding all the entries together to get a
            // single scalar.
            *derivativeIt = T(reduceSum(multiply(Tensor<T>::fromScalar(INV_DENOM), sub(f1, f2))));
            ++derivativeIt;
        }

        // Replace the original value and save the gradient for this variable
        ptr->assign(orig);
        grads[name] = derivative;
    }

    return grads;
}

}
#endif
