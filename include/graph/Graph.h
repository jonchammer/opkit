#ifndef GRAPH_H
#define GRAPH_H

#include <functional>
#include <vector>
#include <memory>
#include <iostream>
#include "tensor/Tensor.h"
#include "tensor/TensorOps.h"
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

public:

    // Construct a graph node with the proper data and type
    // Should only be used for low-level routines (e.g. make_constant())
    Graph(Node<T>* ptr, Type type) : mNode(ptr), mType(type) {}

    // Normal constructors
    Graph() : mNode(nullptr), mType(INVALID) {}
    Graph(const Graph<T>& orig) = default;
    Graph(Graph<T>&& orig)      = default;

    // Assignment operators
    Graph<T>& operator=(const Graph<T>& rhs) = default;
    Graph<T>& operator=(Graph<T>&& rhs)      = default;

    // Comparison operators
    bool operator==(const Graph<T>& other) const
    {
        return mNode == other.mNode;
    }
    bool operator!=(const Graph<T>& other) const
    {
        return !(*this == other);
    }

    // Add a new node dependency. 'child' relies on the output of this node.
    void addChild(const Graph<T>& child)
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        mNode->addChild(child);
    }

    // Many graph nodes cache their most recent calculations to improve
    // performance. This function invalidates the cache of this graph node as
    // well as the caches of any node that depend on this one (children).
    void invalidate()
    {
        static vector<Graph<T>*> stack;
        stack.push_back(this);

        while (!stack.empty())
        {
            Graph<T>* cur = stack.back();
            stack.pop_back();

            cur->node().invalidate();
            for (size_t i = 0; i < cur->getNumChildren(); ++i)
                stack.push_back(&cur->getChild(i));
        }
    }

    // Locate a node with the given name in the dependencies for this node
    // (parent nodes only). If a parent cannot be found, nullptr is returned.
    const Graph<T>* find(const std::string& name) const
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        if (name == this->name()) return this;
        else
        {
            for (size_t i = 0; i < getNumParents(); ++i)
            {
                const Graph<T>* res = getParent(i).find(name);
                if (res != nullptr) return res;
            }
            return nullptr;
        }
    }

    // Evaluate this graph to obtain a result.
    const Tensor<T>& operator()()
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        return mNode->operator()();
    }

    // For variables only - Assign a new value to this node. Note that this
    // operation invalidates part of the graph. Any node that depends on this
    // one will have its cache cleared.
    void assign(const Tensor<T>& newValue)
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        mNode->assign(newValue);
        invalidate();
    }

    // Each graph node has 0 or more parents (dependencies) depending on its
    // type. This allows one to traverse the graph backwards to find those
    // parents.
    const Graph<T>& getParent(const size_t index) const
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        return mNode->getParent(index);
    }
    Graph<T>& getParent(const size_t index)
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        return mNode->getParent(index);
    }
    size_t getNumParents() const
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        return mNode->getNumParents();
    }

    // Each graph node has either 0 or more children (dependents) depending on
    // its type. This function allows one to traverse the graph forwards to
    // find those children.
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
    Type type()           const { return mType;              }
    Node<T>& node()             { return *mNode;             }
    const Node<T>& node() const { return *mNode;             }
    Node<T>* ptr()        const { return mNode.operator->(); }
};

// ----------------------------- Implementation ----------------------------- //

// Base class for all nodes in the graph.
template <class T>
struct Node : public RCObject
{

// ---------------------- Begin Invalidation framework ---------------------- //
private:
    std::vector<Graph<T>> children;

public:

    void addChild(const Graph<T>& child)
    {
        children.push_back(child);
    }

    // Get the nth child for this node. Some nodes may have 0 children.
    const Graph<T>& getChild(const size_t index) const
    {
        return children[index];
    }

    Graph<T>& getChild(const size_t index)
    {
        return children[index];
    }

    size_t getNumChildren() const
    {
        return children.size();
    }

// ----------------------- End Invalidation framework ----------------------- //

    virtual ~Node() {}

    virtual void invalidate() {}

    // Evaluate the graph up to this point and return the result.
    virtual const Tensor<T>& operator()() = 0;

    // Certain node types support assignment of a new value
    virtual void assign(const Tensor<T>& newValue)
    {
        ASSERT(false, "This node type does not support assignment.");
    }

    // Get the nth parent for this node. Some nodes may have 0 parents.
    virtual const Graph<T>& getParent(const size_t index) const
    {
        ASSERT(false, "Component has no parents.");
        throw std::exception();
    }

    virtual Graph<T>& getParent(const size_t index)
    {
        ASSERT(false, "Component has no parents.");
        throw std::exception();
    }

    virtual size_t getNumParents() const
    {
        return 0;
    }

    // Returns the name of this node. Used for printing and graph manipulation
    virtual std::string name() const = 0;

    // Allows subclasses to be printed using <<
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
    const Tensor<T>& operator()() override
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

// A graph node that has exactly two single dependents (e.g. addition)
template <class T>
class BinaryFunction : public Node<T>
{
private:
    std::string mName;
    std::function<void(Tensor<T>& y, const Tensor<T>& x1, const Tensor<T>& x2)> mFunc;
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
    BinaryFunction& operator=(const BinaryFunction<T>& orig) = default;
    BinaryFunction& operator=(BinaryFunction<T>&& orig)      = default;

    // Node class implementations
    void invalidate() override
    {
        mHasCachedResult = false;
    }

    const Tensor<T>& operator()() override
    {
        if (!mHasCachedResult)
        {
            mFunc(mCachedResult, mDependent1(), mDependent2());
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
    std::function<void(Tensor<T>& y, const Tensor<T>& x1, const Tensor<T>& x2)>
    getFunction() const
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
    const Tensor<T>& operator()() override
    {
        // Evaluate each of the list elements
        for (Graph<T>& dependent : mDependents)
            dependent();

        // Return the result of the last one, which should be cached.
        return mDependents.back()();
    }

    const Graph<T>& getParent(const size_t index) const override
    {
        return mDependents[index];
    }
    Graph<T>& getParent(const size_t index) override
    {
        return mDependents[index];
    }
    constexpr size_t getNumParents() const override
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
        for (size_t i = 0; i < getNumParents(); ++i)
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

}
#endif
