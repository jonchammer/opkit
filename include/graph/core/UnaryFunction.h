#ifndef UNARY_FUNCTION_H
#define UNARY_FUNCTION_H

#include "tensor/Tensor.h"

namespace opkit
{

// A graph node that has a single dependent (e.g. the trig functions)
template <class T>
class UnaryFunction : public Node<T>
{
private:
    std::string mName;
    std::function<void(Tensor<T>& y, const Tensor<T>& x)> mFunc;
    Graph<T> mDependent;

    Tensor<T> mCachedResult;
    bool mHasCachedResult;

public:
    // Normal constructors
    UnaryFunction()                          = delete;
    UnaryFunction(const UnaryFunction& orig) = default;
    UnaryFunction(UnaryFunction&& orig)      = default;

    // Additional constructors
    template <class GraphType, class Func>
    UnaryFunction(const std::string& name, Func&& f, GraphType&& dependent):
        mName(name),
        mFunc(std::forward<Func>(f)),
        mDependent(std::forward<GraphType>(dependent)),
        mHasCachedResult(false)
    {}

    // Assignment operators
    UnaryFunction& operator=(const UnaryFunction& orig) = default;
    UnaryFunction& operator=(UnaryFunction&& orig)      = default;

    // Node class implementations
    void invalidate() override
    {
        mHasCachedResult = false;
    }

    const Tensor<T>& operator()() override
    {
        if (!mHasCachedResult)
        {
            mFunc(mCachedResult, mDependent());
            mHasCachedResult = true;
        }

        return mCachedResult;
    }

    const Graph<T>& getParent(const size_t index) const override
    {
        ASSERT(index < 1, "Unary functions have only one parent.");
        return mDependent;
    }
    Graph<T>& getParent(const size_t index) override
    {
        ASSERT(index < 1, "Unary functions have only one parent.");
        return mDependent;
    }
    constexpr size_t getNumParents() const override
    {
        return 1;
    }

    std::string name() const override
    {
        return mName;
    }

    std::ostream& print(std::ostream& out) const override
    {
        out << name() << "[ ";
        out << mDependent;
        out << " ]";
        return out;
    }

    UnaryFunction* clone() const override
    {
        return new UnaryFunction(*this);
    }

    // Unary function-specific operations
    std::function<void(Tensor<T>& y, const Tensor<T>& x)>
    getFunction() const { return mFunc; }
};

template <class T, class Func>
Graph<T> make_unary(const std::string& name, Func&& func, Graph<T> dependent)
{
    Graph<T> res(new UnaryFunction<T>(name, std::forward<Func>(func),
        dependent), Graph<T>::Type::UNARY);

    dependent.addChild(res);
    return res;
}

// Creates a new graph node that performs the same task as 'unary', with a
// different parent element.
template <class T>
Graph<T> copy_unary(const Graph<T>& orig, Graph<T> parent)
{
    auto fn = ((UnaryFunction<T>&)(orig.node())).getFunction();
    Graph<T> res(new UnaryFunction<T>(orig.name(), fn, parent), Graph<T>::Type::UNARY);

    parent.addChild(res);
    return res;
}

}
#endif
