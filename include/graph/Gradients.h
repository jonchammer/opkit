#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "tensor/Tensor.h"
#include "graph/Graph.h"
#include "graph/GraphSimplifier.h"
#include "graph/ops/GraphOps_all.h"

namespace opkit
{

// Calculates the gradients of the given graph with respect to each of the
// variables listed in 'targets'. The result is a map from the variable names to
// the corresponding graph that will calculate the gradient. If the targets set
// is empty, all gradients with respect to all variables will be returned.
template <class T>
std::unordered_map<std::string, Graph<T>> gradients(const Graph<T>& node,
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
    gradQueue.push(expand(make_constant<T>("1", Tensor<T>::fromScalar(1)), shape(node)));

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
        // corresponding derivative function. We then push that node's children
        // and their current gradients into the queues so the process can
        // continue recursively.
        else if (cur.type() != Graph<T>::Type::CONSTANT)
        {
            // Calculate the new deltas. Ignore any nodes that are not
            // differentiable.
            grads.clear();
            if (derivativeMap.call(cur.name(), cur, delta, grads))
            {
                for (size_t i = 0; i < cur.getNumChildren(); ++i)
                {
                    nodeQueue.push(cur.getChild(i));
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
#endif
