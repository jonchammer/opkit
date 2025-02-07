#ifndef GRAPH_OPS_MATRIX_H
#define GRAPH_OPS_MATRIX_H

#include <functional>
#include "graph/core/GraphAPI.h"
#include "graph/DerivativeMap.h"
#include "graph/ops/GraphOps_core.h"
#include "tensor/TensorMath.h"

namespace opkit
{

// This file contains a set of shorthand functions to make manipulating graphs
// much easier to write.
//
// The shorthand functions are all optional. Their use is never required to use
// either the Math or the Graph APIs.

// -------------------------------------------------------------------------- //
// Forward declarations
template <class T> Graph<T> matrixMultiply(Graph<T>, Graph<T>);
template <class T> Graph<T> matrixMultiplyT1(Graph<T>, Graph<T>);
template <class T> Graph<T> matrixMultiplyT2(Graph<T>, Graph<T>);
template <class T> Graph<T> innerProduct(Graph<T>, Graph<T>);
template <class T> Graph<T> transpose(Graph<T>);
template <class T> Graph<T> l1Norm(Graph<T>);
template <class T> Graph<T> l2Norm(Graph<T>);

// -------------------------------------------------------------------------- //
// Forward declarations for the derivatives

#define FD_DERIV(name)                                                         \
template <class T>                                                             \
void name(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients);  \

FD_DERIV(dMatrixMultiply)
FD_DERIV(dMatrixMultiplyT1)
FD_DERIV(dMatrixMultiplyT2)
FD_DERIV(dInnerProduct)

#undef FD_DERIV

// -------------------------------------------------------------------------- //

// Shorthand for creating a binary function graph node
#define BINARY_OP(desiredName, derivFn, fn)                                    \
template <class T>                                                             \
Graph<T> desiredName(Graph<T> A, Graph<T> B)                                   \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](Graph<T> node, Graph<T> delta,                                      \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#desiredName, fn, A, B);                             \
}                                                                              \

BINARY_OP(matrixMultiply, dMatrixMultiply,
    [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return matrixMultiply(y, A, B);
});

BINARY_OP(matrixMultiplyT1, dMatrixMultiplyT1,
    [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return matrixMultiplyT1(y, A, B);
});

BINARY_OP(matrixMultiplyT2, dMatrixMultiplyT2,
    [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return matrixMultiplyT2(y, A, B);
});

BINARY_OP(innerProduct, dInnerProduct,
    [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return innerProduct(y, A, B);
});

#undef BINARY_OP

// -------------------------------------------------------------------------- //

template <class T>
void dMatrixMultiply(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getParent(0);
    Graph<T> right = node.getParent(1);

    // NOTE: Assumes 'left' and 'right' are NOT already transposed
    gradients.push_back(matrixMultiplyT2(delta, right));
    gradients.push_back(matrixMultiplyT1(left, delta));
}

template <class T>
void dMatrixMultiplyT1(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getParent(0);
    Graph<T> right = node.getParent(1);

    // NOTE: Assumes 'left' is transposed and 'right' is not
    gradients.push_back(matrixMultiplyT2(right, delta));
    gradients.push_back(matrixMultiply(left, delta));
}

template <class T>
void dMatrixMultiplyT2(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getParent(0);
    Graph<T> right = node.getParent(1);

    // NOTE: Assumes 'left' is not transposed and 'right' is
    gradients.push_back(matrixMultiply(delta, right));
    gradients.push_back(matrixMultiplyT1(delta, left));
}

template <class T>
void dInnerProduct(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getParent(0);
    Graph<T> right = node.getParent(1);

    gradients.push_back(right * expand(delta, shape(left)));
    gradients.push_back(left  * expand(delta, shape(right)));
}

// -------------------------------------------------------------------------- //

// Shorthand for creating a unary function graph node
#define UNARY_OP(desiredName, fn)                                              \
template <class T>                                                             \
Graph<T> desiredName(Graph<T> A)                                              \
{                                                                              \
    return make_unary<T>(#desiredName, fn, A);                                 \
}                                                                              \

UNARY_OP(transpose, [](const Tensor<T>& A) { return matrixTranspose(A); } );
UNARY_OP(l1Norm,    [](const Tensor<T>& A) { return l1Norm(A);          } );
UNARY_OP(l2Norm,    [](const Tensor<T>& A) { return l2Norm(A);          } );

#undef UNARY_OP

}

#endif
