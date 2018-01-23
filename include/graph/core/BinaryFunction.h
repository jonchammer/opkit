#ifndef BINARY_FUNCTION_H
#define BINARY_FUNCTION_H

#include "tensor/Tensor.h"
#include "util/FunctionTraits.h"

namespace opkit
{

// A graph node that has exactly two single dependents (e.g. addition).
// Extra template arguments are used to reduce code duplication for two cases:
// 1) void (Tensor<T>& y, const Tensor<T>& x1, const Tensor<T>& x2)
// 2) Tensor<T> (const Tensor<T>& x1, const Tensor<T>& x2)
// Use either InPlaceBinaryFunction<T> or OutOfPlaceBinaryFunction<T> in code.
template <class T, class R, class... Args>
class BinaryFunction : public Node<T>
{
private:
    std::string mName;
    std::function<R(Args...)> mFunc;
    Graph<T> mDependent1;
    Graph<T> mDependent2;

    Tensor<T> mCachedResult;
    bool mHasCachedResult;

public:
    // Normal constructors
    BinaryFunction()                           = delete;
    BinaryFunction(const BinaryFunction& orig) = default;
    BinaryFunction(BinaryFunction&& orig)      = default;

    // Additional constructors
    template <class Func, class GraphType1, class GraphType2>
    BinaryFunction(const std::string& name, Func&& f,
        GraphType1&& dependent1, GraphType2&& dependent2) :

        mName(name),
        mFunc(std::forward<Func>(f)),
        mDependent1(std::forward<GraphType1>(dependent1)),
        mDependent2(std::forward<GraphType2>(dependent2)),
        mHasCachedResult(false)
    {}

    // Assignment operators
    BinaryFunction& operator=(const BinaryFunction& orig) = default;
    BinaryFunction& operator=(BinaryFunction&& orig)      = default;

    // Node class implementations
    void invalidate() override
    {
        mHasCachedResult = false;
    }

    const Tensor<T>& operator()() override
    {
        if (!mHasCachedResult)
        {
            // This function decides whether to use the in-place or out-of-place
            // function assignment
            callBinary(mCachedResult, mDependent1(), mDependent2(), mFunc);
            mHasCachedResult = true;
        }

        return mCachedResult;
    }

    const Graph<T>& getParent(const size_t index) const override
    {
        ASSERT(index < 2, "Binary functions have only two parents.");
        if (index == 0)
            return mDependent1;
        else return mDependent2;
    }
    Graph<T>& getParent(const size_t index) override
    {
        ASSERT(index < 2, "Binary functions have only two parents.");
        if (index == 0)
            return mDependent1;
        else return mDependent2;
    }
    constexpr size_t getNumParents() const override
    {
        return 2;
    }

    std::string name() const override
    {
        return mName;
    }

    std::ostream& print(std::ostream& out) const override
    {
        out << name() << "[ ";
        out << mDependent1;
        out << ", ";
        out << mDependent2;
        out << " ]";
        return out;
    }

    BinaryFunction* clone() const override
    {
        return new BinaryFunction(*this);
    }

    // Getters
    std::function<R(Args...)> getFunction() const
    {
        return mFunc;
    }

    Tensor<T>& cachedResult()
    {
        return mCachedResult;
    }
};

// Used for in-place function assignment
template <class T, class Func>
void callBinary(Tensor<T>& res, const Tensor<T>& arg1, const Tensor<T>& arg2, Func&& f,
    typename std::enable_if<
        std::is_void<
            typename function_traits<decltype(f)>::return_type
        >::value>::type* = 0)
{
    f(res, arg1, arg2);
}

// Used for out-of-place function assignment
template <class T, class Func>
void callBinary(Tensor<T>& res, const Tensor<T>& arg1, const Tensor<T>& arg2, Func&& f,
    typename std::enable_if<
        !std::is_void<
            typename function_traits<decltype(f)>::return_type
        >::value>::type* = 0)
{
    res = f(arg1, arg2);
}

// Binary functions that put their results in the given tensor
template <class T>
using InPlaceBinaryFunction =
    BinaryFunction<T, void, Tensor<T>&, const Tensor<T>&, const Tensor<T>&>;

// Binary functions that return a new tensor
template <class T>
using OutOfPlaceBinaryFunction =
    BinaryFunction<T, Tensor<T>, const Tensor<T>&, const Tensor<T>&>;


// Create a new binary function that uses the following signature:
// void (Tensor<T>& y, const Tensor<T>& x1, const Tensor<T>& x2);
template <class T, class Func>
Graph<T> make_binary(const std::string& name, Func&& func,
    Graph<T> dependent1, Graph<T> dependent2,
    typename std::enable_if<
        std::is_void<
            typename function_traits<decltype(func)>::return_type
        >::value
    >::type* = 0)
{
    Graph<T> res(new InPlaceBinaryFunction<T>(name, std::forward<Func>(func),
        dependent1, dependent2), Graph<T>::Type::BINARY_IN);

    dependent1.addChild(res);
    dependent2.addChild(res);
    return res;
}

// Create a new binary function that uses the following signature:
// Tensor<T> (const Tensor<T>& x1, const Tensor<T>& x2);
template <class T, class Func>
Graph<T> make_binary(const std::string& name, Func&& func,
    Graph<T> dependent1, Graph<T> dependent2,
    typename std::enable_if<
        !std::is_void<
            typename function_traits<decltype(func)>::return_type
        >::value
    >::type* = 0)
{
    Graph<T> res(new OutOfPlaceBinaryFunction<T>(name, std::forward<Func>(func),
        dependent1, dependent2), Graph<T>::Type::BINARY_OUT);

    dependent1.addChild(res);
    dependent2.addChild(res);
    return res;
}

// Creates a new graph node that performs the same task as 'binary', with
// different parent elements.
template <class T>
Graph<T> copy_binary(const Graph<T>& orig, Graph<T> parent1, Graph<T> parent2)
{
    if (orig.type() == Graph<T>::Type::BINARY_IN)
    {
        auto fn = ((InPlaceBinaryFunction<T>&)(orig.node())).getFunction();
        Graph<T> res(new InPlaceBinaryFunction<T>(orig.name(),
            fn, parent1, parent2), Graph<T>::Type::BINARY_IN);

        parent1.addChild(res);
        parent2.addChild(res);
        return res;
    }
    else if (orig.type() == Graph<T>::Type::BINARY_OUT)
    {
        auto fn = ((OutOfPlaceBinaryFunction<T>&)(orig.node())).getFunction();
        Graph<T> res(new OutOfPlaceBinaryFunction<T>(orig.name(),
            fn, parent1, parent2), Graph<T>::Type::BINARY_OUT);

        parent1.addChild(res);
        parent2.addChild(res);
        return res;
    }
}

}
#endif
