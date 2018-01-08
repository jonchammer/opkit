#ifndef GRAPH_SIMPLIFIER_H
#define GRAPH_SIMPLIFIER_H

#include "tensor/Tensor.h"
#include "tensor/TensorMath.h"
#include "graph/core/GraphAPI.h"
#include "graph/ops/GraphOps_all.h"

// This file contains a simplify() function that should be used to eliminate
// unnecessary elements from a Graph. For example, x * 0 = 0 for any x, so
// it is not necessary to perform the multiplication. Common usage is to call
// simplify immediately after calculating a gradient. For example:
//
// auto dydx = simplify(gradient(y, x));

namespace opkit
{

// Forward declarations
template <class T>
Graph<T> simplify(Graph<T> root);

// Performs simplification for all unary functions.
template <class T>
Graph<T> simplifyUnary(Graph<T> root)
{
    Graph<T> c1 = simplify(root.getParent(0));

    // Optionally, Simplify the root
    // ...

    // Only allocate a new graph node if we changed something.
    return (c1 == root.getParent(0)) ? root : copy_unary(root, c1);
}

template <class T>
bool simplifyMatrixMultiply(Graph<T> c1, Graph<T> c2, Graph<T> res)
{
    // 0 * x = 0
    // 1 * x = x
    if (c1.type() == Graph<T>::Type::CONSTANT)
    {
        const Constant<T>& constant = (const Constant<T>&) c1.node();

        if (constant.value().size() == 1 && T(constant.value()) == 0)
        {
            res = c1;
            return true;
        }

        else if (constant.value().size() == 1 && T(constant.value()) == 1)
        {
            res = c2;
            return true;
        }
    }

    // x * 0 = 0
    // x * 1 = x
    if (c2.type() == Graph<T>::Type::CONSTANT)
    {
        const Constant<T>& constant = (const Constant<T>&) c2.node();

        if (constant.value().size() == 1 && T(constant.value()) == 0)
        {
            res = c2;
            return true;
        }

        else if (constant.value().size() == 1 && T(constant.value()) == 1)
        {
            res = c1;
            return true;
        }
    }
    return false;
}

template <class T>
bool simplifyMultiplication(Graph<T> c1, Graph<T> c2, Graph<T> res)
{
    // 0 * x = 0
    // 1 * x = x
    if (c1.type() == Graph<T>::Type::CONSTANT)
    {
        const Constant<T>& constant = (const Constant<T>&) c1.node();

        if (constant.value().size() == 1 && T(constant.value()) == 0)
        {
            res = c1;
            return true;
        }

        else if (constant.value().size() == 1 && T(constant.value()) == 1)
        {
            res = c2;
            return true;
        }
    }

    // x * 0 = 0
    // x * 1 = x
    if (c2.type() == Graph<T>::Type::CONSTANT)
    {
        const Constant<T>& constant = (const Constant<T>&) c2.node();

        if (constant.value().size() == 1 && T(constant.value()) == 0)
        {
            res = c2;
            return true;
        }

        else if (constant.value().size() == 1 && T(constant.value()) == 1)
        {
            res = c1;
            return true;
        }
    }

    // expand(1, x) * y = expandIfSmaller(y, x)
    if (c1.name() == "expand")
    {
        Graph<T> expandC1 = c1.getParent(0);
        if (expandC1.type() == Graph<T>::Type::CONSTANT)
        {
            const Constant<T> constant = (const Constant<T>&) expandC1.node();
            if (constant.value().size() == 1 && T(constant.value()) == 1)
            {
                res = expandIfSmaller(c2, c1.getParent(1));
                return true;
            }
        }
    }

    // y * expand(1, x) = expandIfSmaller(y, x)
    if (c2.name() == "expand")
    {
        Graph<T> expandC2 = c2.getParent(0);
        if (expandC2.type() == Graph<T>::Type::CONSTANT)
        {
            const Constant<T> constant = (const Constant<T>&) expandC2.node();
            if (constant.value().size() == 1 && T(constant.value()) == 1)
            {
                res = expandIfSmaller(c1, c2.getParent(1));
                return true;
            }
        }
    }
    return false;
}

template <class T>
bool simplifyAddition(Graph<T> c1, Graph<T> c2, Graph<T> res)
{
    // 0 + x = 0
    if (c1.type() == Graph<T>::Type::CONSTANT)
    {
        const Constant<T>& constant = (const Constant<T>&) c1.node();
        if (constant.value().size() == 1 && T(constant.value()) == 0)
        {
            res = c2;
            return true;
        }
    }

    // x + 0 = 0
    if (c2.type() == Graph<T>::Type::CONSTANT)
    {
        const Constant<T>& constant = (const Constant<T>&) c2.node();
        if (constant.value().size() == 1 && T(constant.value()) == 0)
        {
            res = c1;
            return true;
        }
    }
    return false;
}

// Performs simplification for all binary functions.
template <class T>
Graph<T> simplifyBinary(Graph<T> root)
{
    Graph<T> c1 = simplify(root.getParent(0));
    Graph<T> c2 = simplify(root.getParent(1));

    // Apply any contextual simplifications
    if (root.name() == "matrixMultiply")
    {
        Graph<T> res;
        if (simplifyMatrixMultiply(c1, c2, res))
            return res;
    }
    // TODO: matrixMultiplyT1, matrixMultiplyT2

    else if (root.name() == "operator*")
    {
        Graph<T> res;
        if (simplifyMultiplication(c1, c2, res))
            return res;
    }
    // TODO: operator/

    else if (root.name() == "operator+")
    {
        Graph<T> res;
        if (simplifyAddition(c1, c2, res))
            return res;
    }
    // TODO: operator-, neg

    // Only allocate a new graph node if we changed something.
    return (c1 == root.getParent(0) && c2 == root.getParent(1)) ?
        root : copy_binary(root, c1, c2);
}

// Performs simplification for lists.
template <class T>
Graph<T> simplifyList(Graph<T> root)
{
    std::vector<Graph<T>> parents;
    for (size_t i = 0; i < root.getNumParents(); ++i)
        parents.emplace_back(simplify(root.getParent(i)));
    return make_list<T>(parents);
}

// Performs simplification for updates.
template <class T>
Graph<T> simplifyUpdate(Graph<T> root)
{
    Graph<T> simplifiedValue = simplify(root.getParent(1));
    return (simplifiedValue == root.getParent(1)) ? root :
        copy_update(root, root.getParent(0), simplifiedValue);
}

// Performs simplification for updates with arguments
template <class T>
Graph<T> simplifyUpdateArg(Graph<T> root)
{
    Graph<T> simplifiedValue = simplify(root.getParent(1));
    Graph<T> simplifiedArg   = simplify(root.getParent(2));

    if (simplifiedValue == root.getParent(1) && simplifiedArg == root.getParent(2))
        return root;
    else return copy_update(root, root.getParent(0), simplifiedValue, simplifiedArg);
}

// Performs simplification for all graphs.
template <class T>
Graph<T> simplify(Graph<T> root)
{
    switch (root.type())
    {
        case Graph<T>::Type::INVALID:    return root;
        case Graph<T>::Type::CONSTANT:   return root;
        case Graph<T>::Type::VAR:        return root;
        case Graph<T>::Type::UNARY:      return simplifyUnary(root);
        case Graph<T>::Type::BINARY_IN:
        case Graph<T>::Type::BINARY_OUT: return simplifyBinary(root);
        case Graph<T>::Type::LIST:       return simplifyList(root);
        case Graph<T>::Type::UPDATE:     return simplifyUpdate(root);
        case Graph<T>::Type::UPDATE_ARG: return simplifyUpdateArg(root);

        default:
            std::cerr << "simplify() - UNKNOWN GRAPH TYPE: " << root.type() << std::endl;
            return root;
    }
}

}

#endif
