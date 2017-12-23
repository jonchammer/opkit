#ifndef GRAPH_H
#define GRAPH_H

#include <functional>
#include <vector>
#include <memory>
#include <iostream>
#include "tensor/Tensor.h"
#include "util/ReferenceCount.h"

namespace opkit
{

// This file contains the definitions that make up a functional graph. The graph
// contains Nodes (e.g. Variables and Functions). Information progresses forward
// through the graph from the starting points (usually Variables) to the output.

// Forward declarations
template <class T>
struct Node;

template <class T>
class Constant;

template <class T>
struct Variable;

template <class T>
class UnaryFunction;

template <class T>
class BinaryFunction;

template <class T>
class ListNode;

template <class T>
class UpdateNode;

template <class T>
class UpdateNodeArg;

// Core graph class. This should be the entry point for users.
template <class T>
class Graph
{
public:
    enum Type {INVALID, CONSTANT, VAR, UNARY, BINARY, LIST, UPDATE, UPDATE_ARG};
private:
    RCPtr<Node<T>> mNode; // Pointer to the actual node.
    Type mType;           // Type of the actual node.

    // Construct a graph node with the proper data and type
    explicit Graph(Node<T>* ptr, Type type) : mNode(ptr), mType(type) {}

    // Allow only these functions to use the private constructor.
    template <class O, class TensorType>
    friend Graph<O> make_constant(const std::string& name, TensorType&& tensor);

    template <class O>
    friend Graph<O> make_constant(const std::string& name);

    template <class O, class TensorType>
    friend Graph<O> make_variable(const std::string& name, TensorType&& tensor);

    template <class O>
    friend Graph<O> make_variable(const std::string& name);

    template <class O, class Func, class GraphType>
    friend Graph<O> make_unary(const std::string& name, Func&& func, GraphType&& dependent);

    template <class O, class Func, class GraphType1, class GraphType2>
    friend Graph<O> make_binary(const std::string& name, Func&& func, GraphType1&& dependent1, GraphType2&& dependent2);

    template <class O, class VecType>
    friend Graph<O> make_list(VecType&& dependents);

    template <class O, class Func, class GraphType1, class GraphType2>
    friend Graph<O> make_update(const std::string& name, Func&& func, GraphType1&& target, GraphType2&& value);

    template <class O, class Func, class TargetType, class ValueType, class ArgType>
    friend Graph<O> make_update(const std::string& name, Func&& func, TargetType&& target, ValueType&& value, ArgType&& arg);

public:
    // Normal constructors
    Graph() :
        mNode(nullptr), mType(INVALID)
    {}

    Graph(const Graph<T>& orig) = default;
    Graph(Graph<T>&& orig) = default;

    // Assignment operators
    Graph<T>& operator=(const Graph<T>& rhs) = default;
    Graph<T>& operator=(Graph<T>&& rhs)      = default;

    // Creates a new graph node that performs the same task as 'unary', with a
    // different child element.
    Graph<T> copyUnary(const Graph<T>& child) const
    {
        auto fn = ((UnaryFunction<T>&)(*mNode)).getFunction();
        return Graph<T>(new UnaryFunction<T>(name(), fn, child), UNARY);
    }

    // Creates a new graph node that performs the same task as 'binary', with
    // different child elements.
    Graph<T> copyBinary(const Graph<T>& child1, const Graph<T>& child2) const
    {
        auto fn = ((BinaryFunction<T>&)(*mNode)).getFunction();
        return Graph<T>(new BinaryFunction<T>(name(), fn, child1, child2), BINARY);
    }

    // Creates a new graph node that performs the same task as this update rule,
    // with different child elements
    Graph<T> copyUpdate(const Graph<T>& target, const Graph<T>& value) const
    {
        auto fn = ((UpdateNode<T>&)(*mNode)).getFunction();
        return Graph<T>(new UpdateNode<T>(name(), fn, target, value), UPDATE);
    }

    Graph<T> copyUpdate(const Graph<T>& target, const Graph<T>& value, const Graph<T>& arg) const
    {
        auto fn = ((UpdateNodeArg<T>&)(*mNode)).getFunction();
        return Graph<T>(new UpdateNodeArg<T>(name(), fn, target, value, arg), UPDATE_ARG);
    }

    // Comparison operators
    bool operator==(const Graph<T>& other) const
    {
        return mNode == other.mNode;
    }
    bool operator!=(const Graph<T>& other) const
    {
        return !(*this == other);
    }

    const Graph<T>* find(const std::string& name) const
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        if (name == this->name()) return this;
        else
        {
            for (size_t i = 0; i < getNumChildren(); ++i)
            {
                const Graph<T>* res = getChild(i).find(name);
                if (res != nullptr) return res;
            }
            return nullptr;
        }
    }

    void clearCache()
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        mNode->clearCache();
    }

    // Evaluate this graph to obtain a result. When 'recalculate' is true, any
    // cached calculations will be discarded. Otherwise, caching will be used
    // as much as possible.
    Tensor<T> evaluate(const bool recalculate = false)
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        if (recalculate) mNode->clearCache();
        return mNode->evaluate();
    }

    void assign(const Tensor<T>& newValue)
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        mNode->assign(newValue);
    }

    // Each graph node has either 0, 1, or 2 children depending on its type.
    // This function allows one to traverse the graph backwards to find those
    // children.
    const Graph<T>& getChild(const size_t index) const
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        return mNode->getChild(index);
    }
    Graph<T>& getChild(const size_t index)
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        return mNode->getChild(index);
    }
    size_t getNumChildren() const
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        return mNode->getNumChildren();
    }

    // Returns the name for this node in the graph.
    std::string name() const
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        return mNode->name();
    }

    // Allows graphs to be printed in a human-readable format.
    friend std::ostream& operator<<(std::ostream& out, const Graph<T>& graph)
    {
        if (graph.mNode != nullptr)
            out << *graph.mNode;
        else out << "[Empty Graph]";
        return out;
    }

    // Graph-specific operations. Note that clients should avoid using these
    // as much as possible.
    Type type() const           { return mType;  }
    Node<T>& node()             { return *mNode; }
    const Node<T>& node() const { return *mNode; }
    Node<T>* ptr() const { return mNode.operator->(); }
};

template <class T, class TensorType>
Graph<T> make_constant(const std::string& name, TensorType&& tensor)
{
    return Graph<T>(new Constant<T>(name, std::forward<TensorType>(tensor)), Graph<T>::Type::CONSTANT);
}

template <class T>
Graph<T> make_constant(const std::string& name)
{
    return Graph<T>(new Constant<T>(name), Graph<T>::Type::CONSTANT);
}

template <class T, class TensorType>
Graph<T> make_variable(const std::string& name, TensorType&& tensor)
{
    return Graph<T>(new Variable<T>(name, std::forward<TensorType>(tensor)), Graph<T>::Type::VAR);
}

template <class T>
Graph<T> make_variable(const std::string& name)
{
    return Graph<T>(new Variable<T>(name), Graph<T>::Type::VAR);
}

template <class T, class Func, class GraphType>
Graph<T> make_unary(const std::string& name, Func&& func, GraphType&& dependent)
{
    return Graph<T>(new UnaryFunction<T>(name, std::forward<Func>(func),
        std::forward<GraphType>(dependent)), Graph<T>::Type::UNARY);
}

template <class T, class Func, class GraphType1, class GraphType2>
Graph<T> make_binary(const std::string& name, Func&& func, GraphType1&& dependent1, GraphType2&& dependent2)
{
    return Graph<T>(new BinaryFunction<T>(name, std::forward<Func>(func),
        std::forward<GraphType1>(dependent1), std::forward<GraphType2>(dependent2)),
        Graph<T>::Type::BINARY);
}

template <class T, class VecType>
Graph<T> make_list(VecType&& dependents)
{
    return Graph<T>(new ListNode<T>("list", std::forward<VecType>(dependents)), Graph<T>::Type::LIST);
}

template <class T, class Func, class GraphType1, class GraphType2>
Graph<T> make_update(const std::string& name, Func&& func, GraphType1&& target, GraphType2&& value)
{
    return Graph<T>(new UpdateNode<T>(name, std::forward<Func>(func),
        std::forward<GraphType1>(target), std::forward<GraphType2>(value)),
        Graph<T>::Type::UPDATE);
}

template <class T, class Func, class TargetType, class ValueType, class ArgType>
Graph<T> make_update(const std::string& name, Func&& func, TargetType&& target, ValueType&& value, ArgType&& arg)
{
    return Graph<T>(new UpdateNodeArg<T>(name, std::forward<Func>(func),
        std::forward<TargetType>(target), std::forward<ValueType>(value),
        std::forward<ArgType>(arg)),
        Graph<T>::Type::UPDATE_ARG);
}

// ----------------------------- Implementation ----------------------------- //

// Base class for all nodes in the graph.
template <class T>
struct Node : public RCObject
{
    virtual ~Node() {}

    // Evaluate the graph up to this point and return the result.
    virtual void clearCache() {}
    virtual Tensor<T> evaluate() = 0;

    // Certain node types support assignment of a new value
    virtual void assign(const Tensor<T>& newValue)
    {
        ASSERT(false, "This node type does not support assignment.");
    }

    // Get the nth child for this node. Some nodes may have 0 children.
    virtual const Graph<T>& getChild(const size_t index) const
    {
        ASSERT(false, "Component has no children.");
        throw std::exception();
    }

    virtual Graph<T>& getChild(const size_t index)
    {
        ASSERT(false, "Component has no children.");
        throw std::exception();
    }

    virtual size_t getNumChildren() const
    {
        return 0;
    }

    // Returns the name of this node. Used for printing and graph manipulation
    virtual std::string name() const = 0;

    // Allows child classes to be printed using <<
    virtual std::ostream& print(std::ostream&) const = 0;

    // Polymorphic print
    friend std::ostream& operator<<(std::ostream& out, const Node<T>& node)
    {
        return node.print(out);
    }

    // Returns a copy of this node.
    virtual Node* clone() const = 0;
};

// A graph node that acts as a placeholder for a single Tensor that cannot be
// modified later.
template <class T>
class Constant : public Node<T>
{
protected:
    Tensor<T> mValue;
    std::string mName;

public:
    // Normal constructors
    Constant()                        = delete;
    Constant(const Constant<T>& orig) = default;
    Constant(Constant<T>&& orig)      = default;

    // Construct and initialize simultaneously
    template <class TensorType>
    Constant(const std::string& name, TensorType&& val) :
        mValue(std::forward<TensorType>(val)), mName(name) {}
    Constant(const std::string& name) :
        mValue(zeroes<T>({1})), mName(name) {}

    // Assignment operators
    Constant& operator=(const Constant& orig) = default;
    Constant& operator=(Constant&& orig)      = default;

    // Node class implementations
    Tensor<T> evaluate() override
    {
        return mValue;
    }

    std::string name() const override
    {
        return mName;
    }

    std::ostream& print(std::ostream& out) const override
    {
        out << name();
        return out;
    }

    Constant* clone() const override
    {
        return new Constant(*this);
    }

    // Constant-specific functions
    const Tensor<T>& value() const { return mValue; }
};

// A graph node that acts as a placeholder for a single Tensor that can be
// modified later.
template <class T>
struct Variable : public Constant<T>
{
    // Normal constructors
    Variable()                        = delete;
    Variable(const Variable<T>& orig) = default;
    Variable(Variable<T>&& orig)      = default;

    // Construct and initialize simultaneously
    template <class TensorType>
    Variable(const std::string& name, TensorType&& val) :
        Constant<T>(name, std::forward<TensorType>(val)) {}
    Variable(const std::string name) :
        Constant<T>(name, zeroes<T>({1})) {}

    // Assignment operators
    Variable& operator=(const Variable& orig) = default;
    Variable& operator=(Variable&& orig)      = default;

    // Node class implementations
    void assign(const Tensor<T>& newValue) override
    {
        this->mValue = newValue;
    }

    Variable* clone() const override
    {
        return new Variable(*this);
    }

    // Variable-specific functions
    Tensor<T>& value() { return this->mValue; }
};

// A graph node that has a single dependent (e.g. the trig functions)
template <class T>
class UnaryFunction : public Node<T>
{
private:
    std::string mName;
    std::function<Tensor<T>(const Tensor<T>& x)> mFunc;
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
    void clearCache() override
    {
        mHasCachedResult = false;
        mDependent.clearCache();
    }

    Tensor<T> evaluate() override
    {
        if (!mHasCachedResult)
        {
            mCachedResult    = mFunc(mDependent.evaluate());
            mHasCachedResult = true;
        }

        return mCachedResult;
    }

    const Graph<T>& getChild(const size_t index) const override
    {
        ASSERT(index < 1, "Unary functions have only one child.");
        return mDependent;
    }
    Graph<T>& getChild(const size_t index) override
    {
        ASSERT(index < 1, "Unary functions have only one child.");
        return mDependent;
    }
    constexpr size_t getNumChildren() const override
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
    std::function<Tensor<T>(const Tensor<T>& x)> getFunction() const { return mFunc; }
};

// A graph node that has exactly two single dependents (e.g. addition)
template <class T>
class BinaryFunction : public Node<T>
{
private:
    std::string mName;
    std::function<Tensor<T>(const Tensor<T>& x1, const Tensor<T>& x2)> mFunc;
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
    BinaryFunction(const std::string& name, Func&& f, GraphType1&& dependent1, GraphType2&& dependent2):
        mName(name),
        mFunc(std::forward<Func>(f)),
        mDependent1(std::forward<GraphType1>(dependent1)),
        mDependent2(std::forward<GraphType2>(dependent2)),
        mHasCachedResult(false)
    {}

    // Assignment operators
    BinaryFunction& operator=(const BinaryFunction<T>& orig) = default;
    BinaryFunction& operator=(BinaryFunction<T>&& orig)      = default;

    // Node class implementations
    void clearCache() override
    {
        mHasCachedResult = false;
        mDependent1.clearCache();
        mDependent2.clearCache();
    }

    Tensor<T> evaluate() override
    {
        if (!mHasCachedResult)
        {
            mCachedResult    = mFunc(mDependent1.evaluate(), mDependent2.evaluate());
            mHasCachedResult = true;
        }

        return mCachedResult;
    }

    const Graph<T>& getChild(const size_t index) const override
    {
        ASSERT(index < 2, "Binary functions have only two children.");
        if (index == 0)
            return mDependent1;
        else return mDependent2;
    }
    Graph<T>& getChild(const size_t index) override
    {
        ASSERT(index < 2, "Binary functions have only two children.");
        if (index == 0)
            return mDependent1;
        else return mDependent2;
    }
    constexpr size_t getNumChildren() const override
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
    std::function<Tensor<T>(const Tensor<T>& x1, const Tensor<T>& x2)> getFunction() const
    {
        return mFunc;
    }
};

// A list is a simple node that stores a collection of other graph nodes. It
// has no logic of its own.
template <class T>
class ListNode : public Node<T>
{
private:
    std::string mName;
    std::vector<Graph<T>> mDependents;

public:
    // Normal constructors
    ListNode()                     = delete;
    ListNode(const ListNode& orig) = default;
    ListNode(ListNode&& orig)      = default;

    // Additional constructors
    template <class VecType>
    ListNode(const std::string& name, VecType&& dependents):
        mName(name),
        mDependents(std::forward<VecType>(dependents))
    {}

    // Assignment operators
    ListNode& operator=(const ListNode<T>& orig) = default;
    ListNode& operator=(ListNode<T>&& orig)      = default;

    // Node class implementations
    void clearCache() override
    {
        for (Graph<T>& dep : mDependents)
            dep.clearCache();
    }

    Tensor<T> evaluate() override
    {
        // Evaluate each of the list elements
        for (Graph<T>& dependent : mDependents)
            dependent.evaluate();

        // Return the result of the last one, which should be cached.
        return mDependents.back().evaluate();
    }

    const Graph<T>& getChild(const size_t index) const override
    {
        return mDependents[index];
    }
    Graph<T>& getChild(const size_t index) override
    {
        return mDependents[index];
    }
    constexpr size_t getNumChildren() const override
    {
        return mDependents.size();
    }

    std::string name() const override
    {
        return mName;
    }

    std::ostream& print(std::ostream& out) const override
    {
        out << name() << " = [ " << std::endl;
        for (size_t i = 0; i < getNumChildren(); ++i)
            out << (i+1) << ": " << mDependents[i] << std::endl;
        out << "]";
        return out;
    }

    ListNode* clone() const override
    {
        return new ListNode(*this);
    }
};

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

    // Node class implementations
    void clearCache() override
    {
        mValue.clearCache();
    }

    Tensor<T> evaluate() override
    {
        Tensor<T>& value = ((Variable<T>&) mTarget.node()).value();
        mFunc(value, mValue.evaluate());
        return value;
    }

    const Graph<T>& getChild(const size_t index) const override
    {
        ASSERT(index < 2, "Updates only have two children.");
        if (index == 0) return mTarget;
        else            return mValue;
    }
    Graph<T>& getChild(const size_t index) override
    {
        ASSERT(index < 2, "Updates only have two children.");
        if (index == 0) return mTarget;
        else            return mValue;
    }
    constexpr size_t getNumChildren() const override
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
    UpdateNodeArg(const std::string& name, Func&& f, TargetType&& target, ValueType&& value, ArgType&& arg) :
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

    // Node class implementations
    void clearCache() override
    {
        mValue.clearCache();
        mArg.clearCache();
    }

    Tensor<T> evaluate() override
    {
        Tensor<T>& value = ((Variable<T>&) mTarget.node()).value();
        mFunc(value, mValue.evaluate(), mArg.evaluate());
        return value;
    }

    const Graph<T>& getChild(const size_t index) const override
    {
        ASSERT(index < 3, "Updates with arguments only have three children.");
        if      (index == 0) return mTarget;
        else if (index == 1) return mValue;
        else                 return mArg;
    }
    Graph<T>& getChild(const size_t index) override
    {
        ASSERT(index < 3, "Updates with arguments only have three children.");
        if      (index == 0) return mTarget;
        else if (index == 1) return mValue;
        else                 return mArg;
    }
    constexpr size_t getNumChildren() const override
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

// namespace helper
// {
//     template <int... Is>
//     struct index {};
//
//     template <int N, int... Is>
//     struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};
//
//     template <int... Is>
//     struct gen_seq<0, Is...> : index<Is...> {};
// }
//
// template <size_t n, typename... T>
// typename std::enable_if<(n >= sizeof...(T))>::type
//     print_tuple(std::ostream&, const std::tuple<T...>&)
// {}
//
// template <size_t n, typename... T>
// typename std::enable_if<(n < sizeof...(T))>::type
//     print_tuple(std::ostream& os, const std::tuple<T...>& tup)
// {
//     if (n != 0)
//         os << ", ";
//     os << std::get<n>(tup);
//     print_tuple<n+1>(os, tup);
// }
//
// template <typename... T>
// std::ostream& operator<<(std::ostream& os, const std::tuple<T...>& tup)
// {
//     print_tuple<0>(os, tup);
//     return os;
// }
//
//
//
// // Forward declarations
// template <class T>
// class Node;
//
// // Core graph class. This should be the entry point for users.
// //
// // Most of the actual work is done in the Node class. The Graph class is mostly
// // a wrapper that handles the reference counting, which allows clients to create
// // and share graphs without having to worry about whether the underlying objects
// // are stored on the stack or the heap.
// template <class T>
// class Graph
// {
// private:
//     RCPtr<Node<T>> mNode; // Pointer to the actual node.
//
// public:
//     // Used for meta-programming.
//     using OutputType = T;
//
//     // Normal constructors
//     Graph()                  = delete;
//     Graph(const Graph& orig) = default;
//     Graph(Graph&& orig)      = default;
//
//     // Construct a normal node
//     template <class Func, class... Args>
//     Graph(const std::string& name, Func&& f, Args&&... children) :
//         mNode(new Node<T, Args...>(name, std::forward<Func>(f),
//             std::forward<Args>(children)...))
//     {}
//
//     // Construct a variable node
//     Graph(const std::string& name, Tensor<O>&& out) :
//         mNode(new Node<T>(name, std::forward<Tensor<T>>(out)))
//     {}
//
//     Graph(const std::string& name, const Tensor<T>& out) :
//         mNode(new Node<T>(name, out))
//     {}
//
//     // Assignment operators
//     Graph& operator=(const Graph& rhs) = default;
//     Graph& operator=(Graph&& rhs)      = default;
//
//     // Comparison operators
//     bool operator==(const Graph& other) const
//     {
//         return mNode == other.mNode;
//     }
//     bool operator!=(const Graph& other) const
//     {
//         return !(*this == other);
//     }
//
//     // Allows graphs to be printed in a human-readable format.
//     friend std::ostream& operator<<(std::ostream& out,
//         const Graph<T>& graph)
//     {
//         out << *graph.mNode;
//         return out;
//     }
//
//     // Graph-specific operations. Note that clients should avoid using these
//     // as much as possible.
//     Node<T>& node()             { return *mNode; }
//     const Node<T>& node() const { return *mNode; }
//
//     // --------------- Interface exported from the Node class --------------- //
//     Tensor<T> evaluate(const bool recalculate = false) const
//     {
//         return mNode->evaluate(recalculate);
//     }
//
//     auto& children()
//     {
//         return mNode->children();
//     }
//
//     const auto& children() const
//     {
//         return mNode->children();
//     }
//
//     size_t numChildren() const
//     {
//         return mNode->numChildren();
//     }
//
//     std::string name() const
//     {
//         return mNode->name();
//     }
//
//     void assignOutput(Tensor<T>&& out)
//     {
//         mNode->assignOutput(std::forward<Tensor<T>>(out));
//     }
// };
//
// // Helper function to make creating Graphs easier. Some of the template types
// // can be inferred from the usage, which saves typing.
// template <class T, class Func, class... Args>
// Graph<T> make_graph(const string& name, Func&& f, Args&&... args)
// {
//     return Graph<T>(name, std::forward<Func>(f), std::forward<Args>(args)...);
// }
//
// template <class T>
// Graph<T> make_variable(const string& name, Tensor<T>&& value)
// {
//     return Graph<T>(name, std::forward<Tensor<T>>(value));
// }
//
// template <class T>
// Graph<T> make_variable(const string& name, const Tensor<T>& value)
// {
//     return Graph<T>(name, value);
// }
//
// // 'T' is type. E.g. "double", "int", etc.
// template <class T>
// class Node : public RCObject
// {
// private:
//     // The name of this node. All nodes must have identifying names.
//     std::string mName;
//
//     // The function to be performed for this part of the graph.
//     // The declaration looks complicated, but it's really just saying the
//     // function should look like this:
//     // Tensor<T> function(const Tensor<I1>& x, const Tensor<I2>& y, ...)
//     std::function<Tensor<T>(const Tensor<typename
//         std::remove_reference<ChildTypes>::type::OutputType>&...)> mFunc;
//
//     // Store the nodes we depend on. Each of these should also be a Node with
//     // an associated output type and zero or more input types.
//     std::vector< Graph<T> > > mChildren;
//
//     // Each node caches its final result to avoid excessive recalcuations, but
//     // it can also be set directly (e.g. to represent a variable).
//     Tensor<T> mOutput;
//     bool mHasOutput;
//     bool mIsVariable;
//
//     // Helper for evaluate().
//     template <class... Args, int... Indices>
//     Tensor<T> func(std::tuple<Args...>& tup, helper::index<Indices...>,
//         const bool recalculate)
//     {
//         // Variables will never be evaluated.
//         // Non-variables will be evaluated if
//         //   1. The recalculate flag has been set.
//         //   2. This node has never been evaluated.
//         if (!mIsVariable && (recalculate || !mHasOutput))
//         {
//             mOutput    = mFunc(std::get<Indices>(tup).evaluate(recalculate)...);
//             mHasOutput = true;
//         }
//
//         return mOutput;
//     }
//
//     // Helper for evaluate().
//     template <class... Args>
//     Tensor<O> func(std::tuple<Args...>& tup, const bool recalculate)
//     {
//         return func(tup, helper::gen_seq<sizeof...(Args)>{}, recalculate);
//     }
//
// public:
//
//     Node() = delete;
//
//     // Create a graph node given a function and all the children. Compilation
//     // will fail if 'f' does not fit the format for 'mFunc' or if 'children'
//     // does not match the format for 'mChildren'.
//     template <class Func, class... Args>
//     Node(const std::string& name, Func&& f, Args&&... children) :
//         mName(name),
//         mFunc(std::forward<Func>(f)),
//         mChildren(std::forward<Args>(children)...),
//         mHasOutput(false),
//         mIsVariable(false)
//     {}
//
//     // Create a variable Node
//     Node(const std::string& name, Tensor<T>&& out) :
//         mName(name),
//         mOutput(std::forward<Tensor<T>>(out)),
//         mHasOutput(true),
//         mIsVariable(true)
//     {}
//
//     // Create a variable Node
//     Node(const std::string& name, const Tensor<T>& out) :
//         mName(name),
//         mOutput(out),
//         mHasOutput(true),
//         mIsVariable(true)
//     {}
//
//     Tensor<T> evaluate(const bool recalculate = false)
//     {
//         if (!mIsVariable && (recalculate || !mHasOutput))
//         {
//             mOutput    = mFunc(std::get<Indices>(tup).evaluate(recalculate)...);
//             mHasOutput = true;
//         }
//
//         return mOutput;
//     }
//
//     size_t numChildren() const
//     {
//         return mChildren.size();
//     }
//
//     auto& children()
//     {
//         return mChildren;
//     }
//     const auto& children() const
//     {
//         return mChildren;
//     }
//
//     std::string name() const
//     {
//         return mName;
//     }
//
//     friend std::ostream& operator<<(std::ostream& out, Node<T>& node)
//     {
//         out << node.name();
//         if (numChildren() > 0)
//         {
//             out << "[ ";
//             for (auto& child : mChildren)
//                 out << child << ", ";
//             out << " ]";
//         }
//         return out;
//     }
//
//     Node<T>* clone() const
//     {
//         return new Node<T>(*this);
//     }
//
//     void assignOutput(Tensor<T>&& out)
//     {
//         mOutput    = out;
//         mHasOutput = true;
//     }
// };
}

#endif
