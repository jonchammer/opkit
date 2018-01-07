#ifndef GRAPH_OPS_CORE_H
#define GRAPH_OPS_CORE_H

#include <functional>
#include "graph/Graph.h"
#include "graph/DerivativeMap.h"
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
template <class T> Graph<T> sqrt(Graph<T>);
template <class T> Graph<T> abs(Graph<T>);
template <class T> Graph<T> ceil(Graph<T>);
template <class T> Graph<T> floor(Graph<T>);
template <class T> Graph<T> square(Graph<T>);
template <class T> Graph<T> operator-(Graph<T>);

template <class T> Graph<T> operator+(Graph<T>, Graph<T>);
template <class T> Graph<T> operator-(Graph<T>, Graph<T>);
template <class T> Graph<T> operator*(Graph<T>, Graph<T>);
template <class T> Graph<T> operator/(Graph<T>, Graph<T>);
template <class T> Graph<T> max(Graph<T>, Graph<T>);
template <class T> Graph<T> min(Graph<T>, Graph<T>);

template <class T, class U> Graph<T> operator+(Graph<T>, U);
template <class T, class U> Graph<T> operator+(U, Graph<T>);
template <class T, class U> Graph<T> operator-(Graph<T>, U);
template <class T, class U> Graph<T> operator-(U, Graph<T>);
template <class T, class U> Graph<T> operator*(Graph<T>, U);
template <class T, class U> Graph<T> operator*(U, Graph<T>);
template <class T, class U> Graph<T> operator/(Graph<T>, U);
template <class T, class U> Graph<T> operator/(U, Graph<T>);
template <class T, class U> Graph<T> max(Graph<T>, U);
template <class T, class U> Graph<T> max(U, Graph<T>);
template <class T, class U> Graph<T> min(Graph<T>, U);
template <class T, class U> Graph<T> min(U, Graph<T>);

template <class T> Graph<T> reduceSumTo(Graph<T>, Graph<T>);
template <class T> Graph<T> reduceProductTo(Graph<T>, Graph<T>);
template <class T> Graph<T> reduceMinTo(Graph<T>, Graph<T>);
template <class T> Graph<T> reduceMaxTo(Graph<T>, Graph<T>);
template <class T> Graph<T> expand(Graph<T>, Graph<T>);
template <class T> Graph<T> expandIfSmaller(Graph<T>, Graph<T>);

template <class T> Graph<T> reduceSum(Graph<T>, Graph<T> axes);
template <class T> Graph<T> reduceProduct(Graph<T>, Graph<T> axes);
template <class T> Graph<T> reduceMin(Graph<T>, Graph<T> axes);
template <class T> Graph<T> reduceMax(Graph<T>, Graph<T> axes);
template <class T> Graph<T> reduceMean(Graph<T>, Graph<T> axes);

template <class T> Graph<T> shape(Graph<T>);
template <class T> Graph<T> size(Graph<T>);
template <class T> Graph<T> list(vector<Graph<T>>&);

template <class T> Graph<T> assign(Graph<T>, Graph<T>);
template <class T> Graph<T> addTo(Graph<T>, Graph<T>);
template <class T> Graph<T> subFrom(Graph<T>, Graph<T>);
template <class T> Graph<T> multBy(Graph<T>, Graph<T>);
template <class T> Graph<T> divBy(Graph<T>, Graph<T>);

template <class T> Graph<T> axpy(Graph<T>, Graph<T>, Graph<T>);
template <class T> Graph<T> scale(Graph<T>, Graph<T>);
template <class T, class U> Graph<T> clip(Graph<T>, const U min, const U max);

// -------------------------------------------------------------------------- //
// Forward declarations for the derivatives

#define FD_DERIV(name)                                                         \
template <class T>                                                             \
void name(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients);  \

FD_DERIV(dSqrt)
FD_DERIV(dAbs)
FD_DERIV(dCeil)
FD_DERIV(dFloor)
FD_DERIV(dSquare)
FD_DERIV(dNeg)
FD_DERIV(dAdd)
FD_DERIV(dSubtract)
FD_DERIV(dMultiply)
FD_DERIV(dDivide)
FD_DERIV(dMax)
FD_DERIV(dMin)
FD_DERIV(dReduceSum)
FD_DERIV(dReduceProduct)
FD_DERIV(dReduceMin)
FD_DERIV(dReduceMax)
FD_DERIV(dReduceMean)
FD_DERIV(dScalarProduct)

#undef FD_DERIV
// -------------------------------------------------------------------------- //

// Shorthand for creating a graph node that applies some function to every
// element of the incoming tensor.
#define ELEMENT_WISE_OP(desiredName, derivFn, fn)                              \
template <class T>                                                             \
Graph<T> desiredName(Graph<T> arg)                                            \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](Graph<T> node, Graph<T> delta,                                    \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_unary<T>(#desiredName, [](Tensor<T>& y, const Tensor<T>& x)    \
    {                                                                          \
        return elementwiseFunc(y, x, fn);                                      \
    }, arg);                                                                   \
}                                                                              \

ELEMENT_WISE_OP(sqrt,   dSqrt,   [](const T x) { return std::sqrt(x);        })
ELEMENT_WISE_OP(abs,    dAbs,    [](const T x) { return std::abs(x);         })
ELEMENT_WISE_OP(ceil,   dCeil,   [](const T x) { return std::ceil(x);        })
ELEMENT_WISE_OP(floor,  dFloor,  [](const T x) { return std::floor(x);       })
ELEMENT_WISE_OP(square, dSquare, [](const T x) { return x * x;               })

#undef ELEMENT_WISE_OP

template <class T>
Graph<T> operator-(Graph<T> arg)
{
    registerDerivative<T>("neg",
        [](Graph<T> node, Graph<T> delta,
        std::vector<Graph<T>>& gradients) {dNeg(node, delta, gradients);});

    return make_unary<T>("neg", [](Tensor<T>& y, const Tensor<T>& x)
    {
        return elementwiseFunc(y, x, [](const T val) { return -val; });
    }, arg);
}


// -------------------------------------------------------------------------- //

template <class T>
void dSqrt(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 / (2 * node)) * delta);
}

template <class T>
void dAbs(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((node.getChild(0) / node) * delta);
}

template <class T>
void dCeil(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    // Technically, the derivative doesn't exist for integers, but it's 0
    // everywhere else.
    gradients.push_back(make_constant<T>(0));
}

template <class T>
void dFloor(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    // Technically, the derivative doesn't exist for integers, but it's 0
    // everywhere else.
    gradients.push_back(make_constant<T>(0));
}

template <class T>
void dSquare(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    // dy/dx [a^2] = dy/da * 2a
    gradients.push_back(delta * (node.getChild(0) * 2));
}

template <class T>
void dNeg(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(-delta);
}

// -------------------------------------------------------------------------- //

// Shorthand for creating a binary function graph node
#define BINARY_OP(desiredName, derivFn, fn)                                    \
template <class T>                                                             \
Graph<T> desiredName(Graph<T> A, Graph<T> B)                                 \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](Graph<T> node, Graph<T> delta,                                    \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#desiredName, fn, A, B);                             \
}                                                                              \

BINARY_OP(operator+, dAdd, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return add(y, A, B);
});

BINARY_OP(operator-, dSubtract, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return sub(y, A, B);
});

BINARY_OP(operator*, dMultiply, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return multiply(y, A, B);
});

BINARY_OP(operator/, dDivide, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return divide(y, A, B);
});

BINARY_OP(max, dMax, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return max(y, A, B);
});

BINARY_OP(min, dMin, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return min(y, A, B);
});

#undef BINARY_OP
// -------------------------------------------------------------------------- //

template <class T>
void dAdd(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getChild(0);
    Graph<T> right = node.getChild(1);

    gradients.push_back(reduceSumTo(delta, shape(left)));
    gradients.push_back(reduceSumTo(delta, shape(right)));
}

template <class T>
void dSubtract(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getChild(0);
    Graph<T> right = node.getChild(1);

    gradients.push_back(reduceSumTo(delta, shape(left)));
    gradients.push_back(reduceSumTo(-delta, shape(right)));
}

template <class T>
void dMultiply(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getChild(0);
    Graph<T> right = node.getChild(1);

    gradients.push_back(reduceSumTo(right * delta, shape(left)));
    gradients.push_back(reduceSumTo(left * delta, shape(right)));
}

template <class T>
void dDivide(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getChild(0);
    Graph<T> right = node.getChild(1);

    gradients.push_back(reduceSumTo((1 / right) * delta, shape(left)));
    gradients.push_back(reduceSumTo((-left / square(right)) * delta, shape(right)));
}

template <class T>
void dMax(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getChild(0);
    Graph<T> right = node.getChild(1);

    // Create matrix with 1s wherever the max element appeared in both tensors.
    // When both tensors have the same value, we divide the gradient between the
    // two tensors equally.
    Graph<T> lIndicators = equal(left, node);
    Graph<T> rIndicators = equal(right, node);
    Graph<T> overlap     = 0.5 * equal(lIndicators, rIndicators);

    gradients.push_back(reduceSumTo((lIndicators - overlap) * delta, shape(left)));
    gradients.push_back(reduceSumTo((rIndicators - overlap) * delta, shape(right)));
}

template <class T>
void dMin(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> left  = node.getChild(0);
    Graph<T> right = node.getChild(1);

    // Create matrix with 1s wherever the max element appeared in both tensors.
    // When both tensors have the same value, we divide the gradient between the
    // two tensors equally.
    Graph<T> lIndicators = equal(left, node);
    Graph<T> rIndicators = equal(right, node);
    Graph<T> overlap     = 0.5 * equal(lIndicators, rIndicators);

    gradients.push_back(reduceSumTo((lIndicators - overlap) * delta, shape(left)));
    gradients.push_back(reduceSumTo((rIndicators - overlap) * delta, shape(right)));
}

// -------------------------------------------------------------------------- //
#define SCALAR_OP_RIGHT(fnName, fn, derivFn)                                   \
template <class T, class U>                                                    \
Graph<T> fnName(Graph<T> arg, U scalar)                                       \
{                                                                              \
    registerDerivative<T>(#fnName,                                             \
        [](Graph<T> node, Graph<T> delta,                                    \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#fnName, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)  \
    {                                                                          \
        return fn(y, A, B);                                                    \
    }, arg,                                                                    \
    make_constant<T>(scalar));                                                 \
}                                                                              \

#define SCALAR_OP_LEFT(fnName, fn, derivFn)                                    \
template <class T, class U>                                                    \
Graph<T> fnName(U scalar, Graph<T> arg)                                       \
{                                                                              \
    registerDerivative<T>(#fnName,                                             \
        [](Graph<T> node, Graph<T> delta,                                    \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#fnName, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)  \
    {                                                                          \
        return fn(y, A, B);                                                    \
    },                                                                         \
    make_constant<T>(scalar), arg);                                            \
}                                                                              \

SCALAR_OP_LEFT(operator+,  add,      dAdd);
SCALAR_OP_RIGHT(operator+, add,      dAdd);
SCALAR_OP_LEFT(operator-,  sub,      dSubtract);
SCALAR_OP_RIGHT(operator-, sub,      dSubtract);
SCALAR_OP_LEFT(operator*,  multiply, dMultiply);
SCALAR_OP_RIGHT(operator*, multiply, dMultiply);
SCALAR_OP_LEFT(operator/,  divide,   dDivide);
SCALAR_OP_RIGHT(operator/, divide,   dDivide);
SCALAR_OP_LEFT(max, max, dMax);
SCALAR_OP_RIGHT(max, max, dMax);
SCALAR_OP_LEFT(min, min, dMin);
SCALAR_OP_RIGHT(min, min, dMin);

#undef SCALAR_OP_LEFT
#undef SCALAR_OP_RIGHT

// -------------------------------------------------------------------------- //

#define CONTROL_OP(desiredName, fn)                                            \
template <class T>                                                             \
Graph<T> desiredName(Graph<T> A, Graph<T> B)                                 \
{                                                                              \
    return make_binary<T>(#desiredName, fn, A, B);                             \
}                                                                              \

CONTROL_OP(equal, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return equal(y, A, B);
});

CONTROL_OP(notEqual, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return notEqual(y, A, B);
});

CONTROL_OP(greater, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return greater(y, A, B);
});

CONTROL_OP(greaterEqual, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return greaterEqual(y, A, B);
});

CONTROL_OP(less, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return less(y, A, B);
});

CONTROL_OP(lessEqual, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    return lessEqual(y, A, B);
});

// -------------------------------------------------------------------------- //

#define SHAPE_OP(desiredName, fn)                                              \
template <class T>                                                             \
Graph<T> desiredName(Graph<T> A, Graph<T> shape)                             \
{                                                                              \
    return make_binary<T>(#desiredName, fn, A, shape);                         \
}                                                                              \

SHAPE_OP(reduceSumTo, [](Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& shape)
{
    return reduceSumTo(y, a, shape);
});
SHAPE_OP(reduceProductTo, [](Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& shape)
{
    return reduceProductTo(y, a, shape);
});
SHAPE_OP(reduceMinTo, [](Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& shape)
{
    return reduceMinTo(y, a, shape);
});
SHAPE_OP(reduceMaxTo, [](Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& shape)
{
    return reduceMaxTo(y, a, shape);
});

SHAPE_OP(expand, [](Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& shape)
{
    vector<T> shapeVec(shape.begin(), shape.end());
    Tensor<T> res = expand(a, shapeVec.begin(), shapeVec.end());
    res.copy(y);
});

SHAPE_OP(expandIfSmaller, [](Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& shape)
{
    vector<T> shapeVec(shape.begin(), shape.end());
    Tensor<T> res = expandIfSmaller(a, shapeVec.begin(), shapeVec.end());
    res.copy(y);
});

#undef SHAPE_OP

// -------------------------------------------------------------------------- //

// Shorthand for creating a reduction function graph node
#define REDUCE_OP(desiredName, derivFn, fn)                                    \
template <class T>                                                             \
Graph<T> desiredName(Graph<T> A, Graph<T> axes)                              \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](Graph<T> node, Graph<T> delta,                                    \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#desiredName, fn, A, axes);                          \
}                                                                              \

#define REDUCE_OP_SIMPLE(desiredName, derivFn, fn)                             \
template <class T>                                                             \
Graph<T> desiredName(Graph<T> A)                                              \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](Graph<T> node, Graph<T> delta,                                    \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_unary<T>(#desiredName, fn, A);                                 \
}                                                                              \

REDUCE_OP(reduceSum, dReduceSum, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    SmallVector axes(B.begin(), B.end());
    return reduceSum(y, A, axes);
});
REDUCE_OP_SIMPLE(reduceSum, dReduceSum, [](Tensor<T>& y, const Tensor<T>& x)
{
    return reduceSum(y, x);
});

REDUCE_OP(reduceProduct, dReduceProduct, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    SmallVector axes(B.begin(), B.end());
    return reduceProduct(y, A, axes);
});
REDUCE_OP_SIMPLE(reduceProduct, dReduceProduct, [](Tensor<T>& y, const Tensor<T>& x)
{
    return reduceProduct(y, x);
});

REDUCE_OP(reduceMin, dReduceMin, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    SmallVector axes(B.begin(), B.end());
    return reduceMin(y, A, axes);
});
REDUCE_OP_SIMPLE(reduceMin, dReduceMin, [](Tensor<T>& y, const Tensor<T>& x)
{
    return reduceMin(y, x);
});

REDUCE_OP(reduceMax, dReduceMax, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    SmallVector axes(B.begin(), B.end());
    return reduceMax(y, A, axes);
});
REDUCE_OP_SIMPLE(reduceMax, dReduceMax, [](Tensor<T>& y, const Tensor<T>& x)
{
    return reduceMax(y, x);
});

REDUCE_OP(reduceMean, dReduceMean, [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    SmallVector axes(B.begin(), B.end());
    return reduceMean(y, A, axes);
});
REDUCE_OP_SIMPLE(reduceMean, dReduceMean, [](Tensor<T>& y, const Tensor<T>& x)
{
    return reduceMean(y, x);
});

template <class T>
void dReduceSum(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(expand(delta, shape(node.getChild(0))));
    if (node.getNumChildren() == 2) gradients.push_back(make_constant<T>(0));
}

template <class T>
void dReduceProduct(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    // Broadcasting should make sure delta is the right shape
    gradients.push_back((node / node.getChild(0)) * delta);
    if (node.getNumChildren() == 2) gradients.push_back(make_constant<T>(0));
}

template <class T>
void dReduceMin(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    if (node.getNumChildren() == 1)
    {
        Graph<T> x         = node.getChild(0);
        Graph<T> indicators = equal(x, node);
        gradients.push_back((indicators / reduceSum(indicators)) * delta);
    }
    else
    {
        Graph<T> x    = node.getChild(0);
        Graph<T> axes = node.getChild(1);

        // Create matrix with 1s wherever the min element appeared, and
        // divide by the number of duplicates so all min elements share blame.
        Graph<T> indicators = equal(x, node);
        gradients.push_back((indicators / reduceSum(indicators, axes)) * delta);
        gradients.push_back(make_constant<T>(0));
    }
}

template <class T>
void dReduceMax(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    if (node.getNumChildren() == 1)
    {
        Graph<T> x         = node.getChild(0);
        Graph<T> indicators = equal(x, node);
        gradients.push_back((indicators / reduceSum(indicators)) * delta);
    }
    else
    {
        Graph<T> x    = node.getChild(0);
        Graph<T> axes = node.getChild(1);

        // Create matrix with 1s wherever the max element appeared, and
        // divide by the number of duplicates so all max elements share blame.
        Graph<T> indicators = equal(x, node);
        gradients.push_back((indicators / reduceSum(indicators, axes)) * delta);
        gradients.push_back(make_constant<T>(0));
    }
}

template <class T>
void dReduceMean(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> x     = node.getChild(0);
    Graph<T> factor = size(x) / size(node);
    gradients.push_back(expand(delta / factor, shape(x)));
    if (node.getNumChildren() == 2) gradients.push_back(make_constant<T>(0));
}

template <class T>
Graph<T> argmax(Graph<T> node, const size_t dimension)
{
    return make_binary<T>("argmax", [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& dim)
    {
        ASSERT(dim.size() == 1, "Dimension must be a single number.");
        ASSERT(T(dim) >= T{}, "Dimension must be non-negative.");

        return argmax(y, A, size_t(T(dim)));
    }, node, make_constant<T>("dim", Tensor<T>::fromScalar(dimension)));
}

template <class T>
Graph<T> argmin(Graph<T> node, const size_t dimension)
{
    return make_binary<T>("argmin", [](Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& dim)
    {
        ASSERT(dim.size() == 1, "Dimension must be a single number.");
        ASSERT(T(dim) >= T{}, "Dimension must be non-negative.");

        return argmin(y, A, size_t(T(dim)));
    }, node, make_constant<T>("dim", Tensor<T>::fromScalar(dimension)));
}

// -------------------------------------------------------------------------- //

template <class T>
Graph<T> shape(Graph<T> A)
{
    registerNonDifferentiable<T>("shape");
    return make_unary<T>("shape", [](Tensor<T>& y, const Tensor<T>& A)
    {
        const size_t rank = A.rank();
        y.resize({rank});

        Storage<T>& storage = y.storage();
        for (size_t i = 0; i < rank; ++i)
            storage[i] = T(A.shape(i));
    }, A);
}

template <class T>
Graph<T> shape(Graph<T> A, size_t index)
{
    registerNonDifferentiable<T>("shape");
    return make_unary<T>("shape", [index](Tensor<T>& y, const Tensor<T>& A)
    {
        y.resize({1});
        y = A.shape(index);
    }, A);
}

template <class T>
Graph<T> rank(Graph<T> A)
{
    registerNonDifferentiable<T>("rank");
    return make_unary<T>("rank", [](Tensor<T>& y, const Tensor<T>& A)
    {
        y.resize({1});
        y = A.rank();
    }, A);
}

template <class T>
Graph<T> size(Graph<T> A)
{
    registerNonDifferentiable<T>("size");
    return make_unary<T>("size", [](Tensor<T>& y, const Tensor<T>& A)
    {
        y.resize({1});
        y = A.size();
    }, A);
}

template <class T>
Graph<T> assign(Graph<T> target, Graph<T> value)
{
    registerNonDifferentiable<T>("=");
    return make_update<T>("=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        A = B;
    }, target, value);
}

template <class T>
Graph<T> list(vector<Graph<T>>& pieces)
{
    if (pieces.size() == 1)
        return pieces[0];
    else return make_list<T>(pieces);
}

template <class T>
Graph<T> addTo(Graph<T> target, Graph<T> value)
{
    registerNonDifferentiable<T>("+=");
    return make_update<T>("+=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        addTo(A, B);
    }, target, value);
}

template <class T>
Graph<T> subFrom(Graph<T> target, Graph<T> value)
{
    registerNonDifferentiable<T>("-=");
    return make_update<T>("-=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        subFrom(A, B);
    }, target, value);
}

template <class T>
Graph<T> multBy(Graph<T> target, Graph<T> value)
{
    registerNonDifferentiable<T>("*=");
    return make_update<T>("*=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        multBy(A, B);
    }, target, value);
}

template <class T>
Graph<T> divBy(Graph<T> target, Graph<T> value)
{
    registerNonDifferentiable<T>("/=");
    return make_update<T>("/=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        divBy(A, B);
    }, target, value);
}

template <class T>
Graph<T> axpy(Graph<T> y, Graph<T> x, Graph<T> alpha)
{
    registerNonDifferentiable<T>("axpy");
    return make_update<T>("axpy", [](Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C)
    {
        axpy(A, B, C);
    }, y, x, alpha);
}

template <class T>
Graph<T> scale(Graph<T> x, Graph<T> alpha)
{
    registerNonDifferentiable<T>("scale");
    return make_update<T>("scale", [](Tensor<T>& A, const Tensor<T>& B)
    {
        ASSERT(B.size() == 1, "B must be a scalar.");
        scale(A, T(B));
    }, x, alpha);
}

#undef UNARY_OP

template <class T, class U>
Graph<T> clip(Graph<T> x, const U min, const U max)
{
    registerNonDifferentiable<T>("clip");
    return make_unary<T>("clip", [min, max](Tensor<T>& y, const Tensor<T>& A)
    {
        return clip(y, A, min, max);
    }, x);
}

}

#endif
