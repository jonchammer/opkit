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
template <class T> Graph<T> sqrt(const Graph<T>&);
template <class T> Graph<T> abs(const Graph<T>&);
template <class T> Graph<T> ceil(const Graph<T>&);
template <class T> Graph<T> floor(const Graph<T>&);
template <class T> Graph<T> square(const Graph<T>&);
template <class T> Graph<T> operator-(const Graph<T>&);

template <class T> Graph<T> operator+(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> operator-(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> operator*(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> operator/(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> max(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> min(const Graph<T>&, const Graph<T>&);

template <class T, class U> Graph<T> operator+(const Graph<T>&, U);
template <class T, class U> Graph<T> operator+(U, const Graph<T>&);
template <class T, class U> Graph<T> operator-(const Graph<T>&, U);
template <class T, class U> Graph<T> operator-(U, const Graph<T>&);
template <class T, class U> Graph<T> operator*(const Graph<T>&, U);
template <class T, class U> Graph<T> operator*(U, const Graph<T>&);
template <class T, class U> Graph<T> operator/(const Graph<T>&, U);
template <class T, class U> Graph<T> operator/(U, const Graph<T>&);
template <class T, class U> Graph<T> max(const Graph<T>&, U);
template <class T, class U> Graph<T> max(U, const Graph<T>&);
template <class T, class U> Graph<T> min(const Graph<T>&, U);
template <class T, class U> Graph<T> min(U, const Graph<T>&);

template <class T> Graph<T> reduceSumTo(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> reduceProductTo(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> reduceMinTo(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> reduceMaxTo(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> expand(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> expandIfSmaller(const Graph<T>&, const Graph<T>&);

template <class T> Graph<T> reduceSum(const Graph<T>&, const Graph<T>& axes);
template <class T> Graph<T> reduceProduct(const Graph<T>&, const Graph<T>& axes);
template <class T> Graph<T> reduceMin(const Graph<T>&, const Graph<T>& axes);
template <class T> Graph<T> reduceMax(const Graph<T>&, const Graph<T>& axes);
template <class T> Graph<T> reduceMean(const Graph<T>&, const Graph<T>& axes);

template <class T> Graph<T> shape(const Graph<T>&);
template <class T> Graph<T> size(const Graph<T>&);
template <class T> Graph<T> list(const vector<Graph<T>>&);

template <class T> Graph<T> assign(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> addTo(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> subFrom(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> multBy(const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> divBy(const Graph<T>&, const Graph<T>&);

template <class T> Graph<T> axpy(const Graph<T>&, const Graph<T>&, const Graph<T>&);
template <class T> Graph<T> scale(const Graph<T>&, const Graph<T>&);
template <class T, class U> Graph<T> clip(const Graph<T>&, const U min, const U max);

// -------------------------------------------------------------------------- //
// Forward declarations for the derivatives

#define FD_DERIV(name)                                                         \
template <class T>                                                             \
void name(const Graph<T>& node, const Graph<T>& delta,                         \
    std::vector<Graph<T>>& gradients);                                         \

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
Graph<T> desiredName(const Graph<T>& arg)                                      \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](const Graph<T>& node, const Graph<T>& delta,                        \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_unary<T>(#desiredName, [](const Tensor<T>& A)                  \
    {                                                                          \
        return elementwiseFunc(A, fn);                                         \
    }, arg);                                                                   \
}                                                                              \

ELEMENT_WISE_OP(sqrt,   dSqrt,   [](const T x) { return std::sqrt(x);        })
ELEMENT_WISE_OP(abs,    dAbs,    [](const T x) { return std::abs(x);         })
ELEMENT_WISE_OP(ceil,   dCeil,   [](const T x) { return std::ceil(x);        })
ELEMENT_WISE_OP(floor,  dFloor,  [](const T x) { return std::floor(x);       })
ELEMENT_WISE_OP(square, dSquare, [](const T x) { return x * x;               })

#undef ELEMENT_WISE_OP

template <class T>
Graph<T> operator-(const Graph<T>& arg)
{
    registerDerivative<T>("neg",
        [](const Graph<T>& node, const Graph<T>& delta,
        std::vector<Graph<T>>& gradients) {dNeg(node, delta, gradients);});

    return make_unary<T>("neg", [](const Tensor<T>& A)
    {
        return elementwiseFunc(A, [](const T x) { return -x; });
    }, arg);
}


// -------------------------------------------------------------------------- //

template <class T>
void dSqrt(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 / (2 * node)) * delta);
}

template <class T>
void dAbs(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((node.getChild(0) / node) * delta);
}

template <class T>
void dCeil(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    // Technically, the derivative doesn't exist for integers, but it's 0
    // everywhere else.
    gradients.push_back(make_constant<T>("0"));
}

template <class T>
void dFloor(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    // Technically, the derivative doesn't exist for integers, but it's 0
    // everywhere else.
    gradients.push_back(make_constant<T>("0"));
}

template <class T>
void dSquare(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    // dy/dx [a^2] = dy/da * 2a
    gradients.push_back(delta * (node.getChild(0) * 2));
}

template <class T>
void dNeg(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(-delta);
}

// -------------------------------------------------------------------------- //

// Shorthand for creating a binary function graph node
#define BINARY_OP(desiredName, derivFn, fn)                                    \
template <class T>                                                             \
Graph<T> desiredName(const Graph<T>& A, const Graph<T>& B)                     \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](const Graph<T>& node, const Graph<T>& delta,                        \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#desiredName, fn, A, B);                             \
}                                                                              \

BINARY_OP(operator+, dAdd, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return add(A, B);
});

BINARY_OP(operator-, dSubtract, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return sub(A, B);
});

BINARY_OP(operator*, dMultiply, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return multiply(A, B);
});

BINARY_OP(operator/, dDivide, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return divide(A, B);
});

BINARY_OP(max, dMax, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return max(A, B);
});

BINARY_OP(min, dMin, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return min(A, B);
});

#undef BINARY_OP
// -------------------------------------------------------------------------- //

template <class T>
void dAdd(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    const Graph<T>& left  = node.getChild(0);
    const Graph<T>& right = node.getChild(1);

    gradients.push_back(reduceSumTo(delta, shape(left)));
    gradients.push_back(reduceSumTo(delta, shape(right)));
}

template <class T>
void dSubtract(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    const Graph<T>& left  = node.getChild(0);
    const Graph<T>& right = node.getChild(1);

    gradients.push_back(reduceSumTo(delta, shape(left)));
    gradients.push_back(reduceSumTo(-delta, shape(right)));
}

template <class T>
void dMultiply(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    const Graph<T>& left  = node.getChild(0);
    const Graph<T>& right = node.getChild(1);

    gradients.push_back(reduceSumTo(right * delta, shape(left)));
    gradients.push_back(reduceSumTo(left * delta, shape(right)));
}

template <class T>
void dDivide(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    const Graph<T>& left  = node.getChild(0);
    const Graph<T>& right = node.getChild(1);

    gradients.push_back(reduceSumTo((1 / right) * delta, shape(left)));
    gradients.push_back(reduceSumTo((-left / square(right)) * delta, shape(right)));
}

template <class T>
void dMax(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    const Graph<T>& left  = node.getChild(0);
    const Graph<T>& right = node.getChild(1);

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
void dMin(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    const Graph<T>& left  = node.getChild(0);
    const Graph<T>& right = node.getChild(1);

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
Graph<T> fnName(const Graph<T>& arg, U scalar)                                 \
{                                                                              \
    registerDerivative<T>(#fnName,                                             \
        [](const Graph<T>& node, const Graph<T>& delta,                        \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#fnName, [](const Tensor<T>& A, const Tensor<T>& B)  \
    {                                                                          \
        return fn(A, B);                                                       \
    }, arg,                                                                    \
    make_constant<T>(std::to_string(scalar), Tensor<T>::fromScalar(scalar)));  \
}                                                                              \

#define SCALAR_OP_LEFT(fnName, fn, derivFn)                                    \
template <class T, class U>                                                    \
Graph<T> fnName(U scalar, const Graph<T>& arg)                                 \
{                                                                              \
    registerDerivative<T>(#fnName,                                             \
        [](const Graph<T>& node, const Graph<T>& delta,                        \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#fnName, [](const Tensor<T>& A, const Tensor<T>& B)  \
    {                                                                          \
        return fn(A, B);                                                       \
    },                                                                         \
    make_constant<T>(std::to_string(scalar), Tensor<T>::fromScalar(scalar)),   \
    arg);                                                                      \
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
Graph<T> desiredName(const Graph<T>& A, const Graph<T>& B)                     \
{                                                                              \
    return make_binary<T>(#desiredName, fn, A, B);                             \
}                                                                              \

CONTROL_OP(equal, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return equal(A, B);
});

CONTROL_OP(notEqual, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return notEqual(A, B);
});

CONTROL_OP(greater, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return greater(A, B);
});

CONTROL_OP(greaterEqual, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return greaterEqual(A, B);
});

CONTROL_OP(less, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return less(A, B);
});

CONTROL_OP(lessEqual, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return lessEqual(A, B);
});

// -------------------------------------------------------------------------- //

#define SHAPE_OP(desiredName, fn)                                              \
template <class T>                                                             \
Graph<T> desiredName(const Graph<T>& A, const Graph<T>& shape)                 \
{                                                                              \
    return make_binary<T>(#desiredName, fn, A, shape);                         \
}                                                                              \

SHAPE_OP(reduceSumTo, [](const Tensor<T>& a, const Tensor<T>& shape)
{
    return reduceSumTo(a, shape);
});
SHAPE_OP(reduceProductTo, [](const Tensor<T>& a, const Tensor<T>& shape)
{
    return reduceProductTo(a, shape);
});
SHAPE_OP(reduceMinTo, [](const Tensor<T>& a, const Tensor<T>& shape)
{
    return reduceMinTo(a, shape);
});
SHAPE_OP(reduceMaxTo, [](const Tensor<T>& a, const Tensor<T>& shape)
{
    return reduceMaxTo(a, shape);
});

SHAPE_OP(expand, [](const Tensor<T>& a, const Tensor<T>& shape)
{
    vector<T> shapeVec(shape.begin(), shape.end());
    return expand(a, shapeVec.begin(), shapeVec.end());
});

SHAPE_OP(expandIfSmaller, [](const Tensor<T>& a, const Tensor<T>& shape)
{
    vector<T> shapeVec(shape.begin(), shape.end());
    return expandIfSmaller(a, shapeVec.begin(), shapeVec.end());
});

#undef SHAPE_OP

// -------------------------------------------------------------------------- //

// Shorthand for creating a unary function graph node
#define REDUCE_OP(desiredName, derivFn, fn)                                    \
template <class T>                                                             \
Graph<T> desiredName(const Graph<T>& A, const Graph<T>& axes)                  \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](const Graph<T>& node, const Graph<T>& delta,                        \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_binary<T>(#desiredName, fn, A, axes);                          \
}                                                                              \

#define REDUCE_OP_SIMPLE(desiredName, derivFn, fn)                             \
template <class T>                                                             \
Graph<T> desiredName(const Graph<T>& A)                                        \
{                                                                              \
    registerDerivative<T>(#desiredName,                                        \
        [](const Graph<T>& node, const Graph<T>& delta,                        \
        std::vector<Graph<T>>& gradients) {derivFn(node, delta, gradients);}); \
    return make_unary<T>(#desiredName, fn, A);                                 \
}                                                                              \

REDUCE_OP(reduceSum, dReduceSum, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return reduceSum(A, B);
});
REDUCE_OP_SIMPLE(reduceSum, dReduceSum, [](const Tensor<T>& A)
{
    return reduceSum(A);
});

REDUCE_OP(reduceProduct, dReduceProduct, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return reduceProduct(A, B);
});
REDUCE_OP_SIMPLE(reduceProduct, dReduceProduct, [](const Tensor<T>& A)
{
    return reduceProduct(A);
});

REDUCE_OP(reduceMin, dReduceMin, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return reduceMin(A, B);
});
REDUCE_OP_SIMPLE(reduceMin, dReduceMin, [](const Tensor<T>& A)
{
    return reduceMin(A);
});

REDUCE_OP(reduceMax, dReduceMax, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return reduceMax(A, B);
});
REDUCE_OP_SIMPLE(reduceMax, dReduceMax, [](const Tensor<T>& A)
{
    return reduceMax(A);
});

REDUCE_OP(reduceMean, dReduceMean, [](const Tensor<T>& A, const Tensor<T>& B)
{
    return reduceMean(A, B);
});
REDUCE_OP_SIMPLE(reduceMean, dReduceMean, [](const Tensor<T>& A)
{
    return reduceMean(A);
});

template <class T>
void dReduceSum(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(expand(delta, shape(node.getChild(0))));
    if (node.getNumChildren() == 2) gradients.push_back(make_constant<T>("0"));
}

template <class T>
void dReduceProduct(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    // Broadcasting should make sure delta is the right shape
    gradients.push_back((node / node.getChild(0)) * delta);
    if (node.getNumChildren() == 2) gradients.push_back(make_constant<T>("0"));
}

template <class T>
void dReduceMin(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    if (node.getNumChildren() == 1)
    {
        const Graph<T>& x   = node.getChild(0);
        Graph<T> indicators = equal(x, node);
        gradients.push_back((indicators / reduceSum(indicators)) * delta);
    }
    else
    {
        const Graph<T>& x    = node.getChild(0);
        const Graph<T>& axes = node.getChild(1);

        // Create matrix with 1s wherever the min element appeared, and
        // divide by the number of duplicates so all min elements share blame.
        Graph<T> indicators = equal(x, node);
        gradients.push_back((indicators / reduceSum(indicators, axes)) * delta);
        gradients.push_back(make_constant<T>("0"));
    }
}

template <class T>
void dReduceMax(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    if (node.getNumChildren() == 1)
    {
        const Graph<T>& x   = node.getChild(0);
        Graph<T> indicators = equal(x, node);
        gradients.push_back((indicators / reduceSum(indicators)) * delta);
    }
    else
    {
        const Graph<T>& x    = node.getChild(0);
        const Graph<T>& axes = node.getChild(1);

        // Create matrix with 1s wherever the max element appeared, and
        // divide by the number of duplicates so all max elements share blame.
        Graph<T> indicators = equal(x, node);
        gradients.push_back((indicators / reduceSum(indicators, axes)) * delta);
        gradients.push_back(make_constant<T>("0"));
    }
}

template <class T>
void dReduceMean(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    const Graph<T>& x = node.getChild(0);
    Graph<T> factor = size(x) / size(node);
    gradients.push_back(expand(delta / factor, shape(x)));
    if (node.getNumChildren() == 2) gradients.push_back(make_constant<T>("0"));
}

template <class T>
Graph<T> argmax(const Graph<T>& node, const size_t dimension)
{
    return make_binary<T>("argmax", [](const Tensor<T>& A, const Tensor<T>& dim)
    {
        ASSERT(dim.size() == 1, "Dimension must be a single number.");
        ASSERT(T(dim) >= T{}, "Dimension must be non-negative.");

        return argmax(A, size_t(T(dim)));
    }, node, make_constant<T>("d", Tensor<T>::fromScalar(dimension)));
}

template <class T>
Graph<T> argmin(const Graph<T>& node, const size_t dimension)
{
    return make_binary<T>("argmin", [](const Tensor<T>& A, const Tensor<T>& dim)
    {
        ASSERT(dim.size() == 1, "Dimension must be a single number.");
        ASSERT(T(dim) >= T{}, "Dimension must be non-negative.");

        return argmin(A, size_t(T(dim)));
    }, node, make_constant<T>("d", Tensor<T>::fromScalar(dimension)));
}

// -------------------------------------------------------------------------- //

template <class T>
Graph<T> shape(const Graph<T>& A)
{
    registerNonDifferentiable<T>("shape");
    return make_unary<T>("shape", [](const Tensor<T>& A)
    {
        const size_t rank = A.rank();
        Storage<T> storage(rank);
        for (size_t i = 0; i < rank; ++i)
            storage[i] = T(A.shape(i));

        return Tensor<T>(storage, {rank});
    }, A);
}

template <class T>
Graph<T> shape(const Graph<T>& A, size_t index)
{
    registerNonDifferentiable<T>("shape");
    return make_unary<T>("shape", [index](const Tensor<T>& A)
    {
        return Tensor<T>::fromScalar(A.shape(index));
    }, A);
}

template <class T>
Graph<T> rank(const Graph<T>& A)
{
    registerNonDifferentiable<T>("rank");
    return make_unary<T>("rank", [](const Tensor<T>& A)
    {
        return Tensor<T>::fromScalar(T(A.rank()));
    }, A);
}

template <class T>
Graph<T> size(const Graph<T>& A)
{
    registerNonDifferentiable<T>("size");
    return make_unary<T>("size", [](const Tensor<T>& A)
    {
        return Tensor<T>::fromScalar(A.size());
    }, A);
}

template <class T>
Graph<T> assign(const Graph<T>& target, const Graph<T>& value)
{
    registerNonDifferentiable<T>("=");
    return make_update<T>("=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        A = B;
    }, target, value);
}

template <class T>
Graph<T> list(const vector<Graph<T>>& pieces)
{
    if (pieces.size() == 1)
        return pieces[0];
    else return make_list<T>(pieces);
}

template <class T>
Graph<T> addTo(const Graph<T>& target, const Graph<T>& value)
{
    registerNonDifferentiable<T>("+=");
    return make_update<T>("+=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        addTo(A, B);
    }, target, value);
}

template <class T>
Graph<T> subFrom(const Graph<T>& target, const Graph<T>& value)
{
    registerNonDifferentiable<T>("-=");
    return make_update<T>("-=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        subFrom(A, B);
    }, target, value);
}

template <class T>
Graph<T> multBy(const Graph<T>& target, const Graph<T>& value)
{
    registerNonDifferentiable<T>("*=");
    return make_update<T>("*=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        multBy(A, B);
    }, target, value);
}

template <class T>
Graph<T> divBy(const Graph<T>& target, const Graph<T>& value)
{
    registerNonDifferentiable<T>("/=");
    return make_update<T>("/=", [](Tensor<T>& A, const Tensor<T>& B)
    {
        divBy(A, B);
    }, target, value);
}

template <class T>
Graph<T> axpy(const Graph<T>& y, const Graph<T>& x, const Graph<T>& alpha)
{
    registerNonDifferentiable<T>("axpy");
    return make_update<T>("axpy", [](Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C)
    {
        axpy(A, B, C);
    }, y, x, alpha);
}

template <class T>
Graph<T> scale(const Graph<T>& x, const Graph<T>& alpha)
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
Graph<T> clip(const Graph<T>& x, const U min, const U max)
{
    registerNonDifferentiable<T>("clip");
    return make_unary<T>("clip", [min, max](const Tensor<T>& A)
    {
        return clip(A, min, max);
    }, x);
}

}

#endif
