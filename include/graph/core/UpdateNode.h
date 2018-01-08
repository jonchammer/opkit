#ifndef UPDATE_NODE_H
#define UPDATE_NODE_H

#include "tensor/Tensor.h"

namespace opkit
{

// Update nodes are used to modify the values of variables using
// expressions like =, +=, -=, *=, /=, etc.
template <class T>
class UpdateNode : public Node<T>
{
private:
    std::string mName;
    std::function<void(Tensor<T>&, const Tensor<T>&)> mFunc;
    Graph<T> mTarget;
    Graph<T> mValue;

public:
    // Normal constructors
    UpdateNode()                       = delete;
    UpdateNode(const UpdateNode& orig) = default;
    UpdateNode(UpdateNode&& orig)      = default;

    // Additional constructors
    template <class Func, class GraphType1, class GraphType2>
    UpdateNode(const std::string& name, Func&& f, GraphType1&& target, GraphType2&& value) :
        mName(name),
        mFunc(std::forward<Func>(f)),
        mTarget(std::forward<GraphType1>(target)),
        mValue(std::forward<GraphType2>(value))
    {
        ASSERT(mTarget.type() == Graph<T>::Type::VAR,
            "Currently, only variables support updates.");
    }

    // Assignment operators
    UpdateNode& operator=(const UpdateNode<T>& orig) = default;
    UpdateNode& operator=(UpdateNode<T>&& orig)      = default;

    const Tensor<T>& operator()() override
    {
        Tensor<T>& value = ((Variable<T>&) mTarget.node()).value();
        mFunc(value, mValue());
        // mTarget.node().invalidateAll();
        return value;
    }

    const Graph<T>& getParent(const size_t index) const override
    {
        ASSERT(index < 2, "Updates only have two parents.");
        if (index == 0) return mTarget;
        else            return mValue;
    }
    Graph<T>& getParent(const size_t index) override
    {
        ASSERT(index < 2, "Updates only have two parents.");
        if (index == 0) return mTarget;
        else            return mValue;
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
        out << mTarget << " " << mName << " " << mValue;
        return out;
    }

    UpdateNode* clone() const override
    {
        return new UpdateNode(*this);
    }

    // Update-specific operations
    std::function<void(Tensor<T>&, const Tensor<T>&)> getFunction() const
    {
        return mFunc;
    }
};

// Update nodes are used to modify the values of variables using
// expressions like =, +=, -=, *=, /=, etc.
template <class T>
class UpdateNodeArg : public Node<T>
{
private:
    std::string mName;
    std::function<void(Tensor<T>&, const Tensor<T>&, const Tensor<T>&)> mFunc;
    Graph<T> mTarget;
    Graph<T> mValue;
    Graph<T> mArg;

public:
    // Normal constructors
    UpdateNodeArg()                          = delete;
    UpdateNodeArg(const UpdateNodeArg& orig) = default;
    UpdateNodeArg(UpdateNodeArg&& orig)      = default;

    // Additional constructors
    template <class Func, class TargetType, class ValueType, class ArgType>
    UpdateNodeArg(const std::string& name, Func&& f,
        TargetType&& target, ValueType&& value, ArgType&& arg) :

        mName(name),
        mFunc(std::forward<Func>(f)),
        mTarget(std::forward<TargetType>(target)),
        mValue(std::forward<ValueType>(value)),
        mArg(std::forward<ArgType>(arg))
    {
        ASSERT(mTarget.type() == Graph<T>::Type::VAR,
            "Currently, only variables support updates.");
    }

    // Assignment operators
    UpdateNodeArg& operator=(const UpdateNodeArg<T>& orig) = default;
    UpdateNodeArg& operator=(UpdateNodeArg<T>&& orig)      = default;

    const Tensor<T>& operator()() override
    {
        Tensor<T>& value = ((Variable<T>&) mTarget.node()).value();
        mFunc(value, mValue(), mArg());
        // mTarget.node().invalidateAll();
        return value;
    }

    const Graph<T>& getParent(const size_t index) const override
    {
        ASSERT(index < 3, "Updates with arguments only have three parents.");
        if      (index == 0) return mTarget;
        else if (index == 1) return mValue;
        else                 return mArg;
    }
    Graph<T>& getParent(const size_t index) override
    {
        ASSERT(index < 3, "Updates with arguments only have three parents.");
        if      (index == 0) return mTarget;
        else if (index == 1) return mValue;
        else                 return mArg;
    }
    constexpr size_t getNumParents() const override
    {
        return 3;
    }

    std::string name() const override
    {
        return mName;
    }

    std::ostream& print(std::ostream& out) const override
    {
        out << mTarget << " " << mName << " " << mValue << ", " << mArg;
        return out;
    }

    UpdateNodeArg* clone() const override
    {
        return new UpdateNodeArg(*this);
    }

    // Update-specific operations
    std::function<void(Tensor<T>&, const Tensor<T>&, const Tensor<T>&)> getFunction() const
    {
        return mFunc;
    }
};

template <class T, class Func>
Graph<T> make_update(const std::string& name, Func&& func,
    Graph<T> target, Graph<T> value)
{
    Graph<T> res(new UpdateNode<T>(name, std::forward<Func>(func),
        target, value), Graph<T>::Type::UPDATE);

    target.addChild(res);
    value.addChild(res);
    return res;
}

template <class T, class Func>
Graph<T> make_update(const std::string& name, Func&& func,
    Graph<T> target, Graph<T> value, Graph<T> arg)
{
    Graph<T> res(new UpdateNodeArg<T>(name, std::forward<Func>(func),
        target, value, arg), Graph<T>::Type::UPDATE_ARG);

    target.addChild(res);
    value.addChild(res);
    arg.addChild(res);
    return res;
}

// Creates a new graph node that performs the same task as this update rule,
// with different parent elements
template <class T>
Graph<T> copy_update(const Graph<T>& orig, Graph<T> target, Graph<T> value)
{
    auto fn = ((UpdateNode<T>&)(orig.node())).getFunction();
    Graph<T> res(new UpdateNode<T>(orig.name(), fn, target, value), Graph<T>::Type::UPDATE);

    target.addChild(res);
    value.addChild(res);
    return res;
}

template <class T>
Graph<T> copy_update(const Graph<T>& orig, Graph<T> target, Graph<T> value, Graph<T> arg)
{
    auto fn = ((UpdateNodeArg<T>&)(orig.node())).getFunction();
    Graph<T> res(new UpdateNodeArg<T>(orig.name(), fn, target, value, arg), Graph<T>::Type::UPDATE_ARG);

    target.addChild(res);
    value.addChild(res);
    arg.addChild(res);
    return res;
}

}
#endif
