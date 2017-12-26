#ifndef GRAPH_OPS_FUNCTIONS_H
#define GRAPH_OPS_FUNCTIONS_H

#include <functional>
#include "graph/Graph.h"
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
template <class T> Graph<T> sin(const Graph<T>&);
template <class T> Graph<T> cos(const Graph<T>&);
template <class T> Graph<T> tan(const Graph<T>&);
template <class T> Graph<T> csc(const Graph<T>&);
template <class T> Graph<T> sec(const Graph<T>&);
template <class T> Graph<T> cot(const Graph<T>&);
template <class T> Graph<T> sinh(const Graph<T>&);
template <class T> Graph<T> cosh(const Graph<T>&);
template <class T> Graph<T> tanh(const Graph<T>&);
template <class T> Graph<T> asin(const Graph<T>&);
template <class T> Graph<T> acos(const Graph<T>&);
template <class T> Graph<T> atan(const Graph<T>&);
template <class T> Graph<T> asinh(const Graph<T>&);
template <class T> Graph<T> acosh(const Graph<T>&);
template <class T> Graph<T> atanh(const Graph<T>&);
template <class T> Graph<T> exp(const Graph<T>&);
template <class T> Graph<T> exp2(const Graph<T>&);
template <class T> Graph<T> log(const Graph<T>&);
template <class T> Graph<T> log10(const Graph<T>&);
template <class T> Graph<T> log2(const Graph<T>&);

// -------------------------------------------------------------------------- //
// Forward declarations for the derivatives

#define FD_DERIV(name)                                                         \
template <class T>                                                             \
void name(const Graph<T>& node, const Graph<T>& delta,                         \
    std::vector<Graph<T>>& gradients);                                         \

FD_DERIV(dSin)
FD_DERIV(dCos)
FD_DERIV(dTan)
FD_DERIV(dCsc)
FD_DERIV(dSec)
FD_DERIV(dCot)
FD_DERIV(dSinh)
FD_DERIV(dCosh)
FD_DERIV(dTanh)
FD_DERIV(dAsin)
FD_DERIV(dAcos)
FD_DERIV(dAtan)
FD_DERIV(dAsinh)
FD_DERIV(dAcosh)
FD_DERIV(dAtanh)
FD_DERIV(dExp)
FD_DERIV(dExp2)
FD_DERIV(dLog)
FD_DERIV(dLog2)
FD_DERIV(dLog10)

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

ELEMENT_WISE_OP(sin,    dSin,    [](const T x) { return std::sin(x);         })
ELEMENT_WISE_OP(cos,    dCos,    [](const T x) { return std::cos(x);         })
ELEMENT_WISE_OP(tan,    dTan,    [](const T x) { return std::tan(x);         })
ELEMENT_WISE_OP(csc,    dCsc,    [](const T x) { return T{1} / std::sin(x);  })
ELEMENT_WISE_OP(sec,    dSec,    [](const T x) { return T{1} / std::cos(x);  })
ELEMENT_WISE_OP(cot,    dCot,    [](const T x) { return T{1} / std::tan(x);  })
ELEMENT_WISE_OP(sinh,   dSinh,   [](const T x) { return std::sinh(x);        })
ELEMENT_WISE_OP(cosh,   dCosh,   [](const T x) { return std::cosh(x);        })
ELEMENT_WISE_OP(tanh,   dTanh,   [](const T x) { return std::tanh(x);        })
ELEMENT_WISE_OP(asin,   dAsin,   [](const T x) { return std::asin(x);        })
ELEMENT_WISE_OP(acos,   dAcos,   [](const T x) { return std::acos(x);        })
ELEMENT_WISE_OP(atan,   dAtan,   [](const T x) { return std::atan(x);        })
ELEMENT_WISE_OP(asinh,  dAsinh,  [](const T x) { return std::asinh(x);       })
ELEMENT_WISE_OP(acosh,  dAcosh,  [](const T x) { return std::acosh(x);       })
ELEMENT_WISE_OP(atanh,  dAtanh,  [](const T x) { return std::atanh(x);       })
ELEMENT_WISE_OP(exp,    dExp,    [](const T x) { return std::exp(x);         })
ELEMENT_WISE_OP(exp2,   dExp2,   [](const T x) { return std::exp2(x);        })
ELEMENT_WISE_OP(log,    dLog,    [](const T x) { return std::log(x);         })
ELEMENT_WISE_OP(log10,  dLog10,  [](const T x) { return std::log10(x);       })
ELEMENT_WISE_OP(log2,   dLog2,   [](const T x) { return std::log2(x);        })

#undef ELEMENT_WISE_OP

// -------------------------------------------------------------------------- //

template <class T>
void dSin(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(cos(node.getChild(0)) * delta);
}

template <class T>
void dCos(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(-sin(node.getChild(0)) * delta);
}

template <class T>
void dTan(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(square(sec(node.getChild(0))) * delta);
}

template <class T>
void dCsc(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    auto& c = node.getChild(0);
    gradients.push_back(csc(c) * cot(c) * delta);
}

template <class T>
void dSec(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    auto& c = node.getChild(0);
    gradients.push_back((sec(c) * tan(c)) * delta);
}

template <class T>
void dCot(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(square(csc(node.getChild(0))) * delta);
}

template <class T>
void dSinh(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(cosh(node.getChild(0)) * delta);
}

template <class T>
void dCosh(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(sinh(node.getChild(0)) * delta);
}

template <class T>
void dTanh(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 - square(node)) * delta);
}

template <class T>
void dAsin(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 / sqrt(1 - square(node.getChild(0)))) * delta);
}

template <class T>
void dAcos(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((-1 / sqrt(1 - square(node.getChild(0)))) * delta);
}

template <class T>
void dAtan(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 / (1 + square(node.getChild(0)))) * delta);
}

template <class T>
void dAsinh(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 / sqrt(square(node.getChild(0)) + 1)) * delta);
}

template <class T>
void dAcosh(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 / sqrt(square(node.getChild(0)) - 1)) * delta);
}

template <class T>
void dAtanh(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 / (1 - square(node.getChild(0)))) * delta);
}

template <class T>
void dExp(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back(node * delta);
}

template <class T>
void dExp2(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    // ln(2) ~= 0.69314...
    gradients.push_back((node * 0.6931471805599453) * delta);
}

template <class T>
void dLog(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back((1 / node.getChild(0)) * delta);
}

template <class T>
void dLog10(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    // ln(10) ~= 2.3025...
    gradients.push_back((1 / (node.getChild(0) * 2.3025850929940457)) * delta);
}

template <class T>
void dLog2(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    // ln(2) ~= 0.6931471805599453
    gradients.push_back((1 / (node.getChild(0) * 0.6931471805599453)) * delta);
}

}

#endif
