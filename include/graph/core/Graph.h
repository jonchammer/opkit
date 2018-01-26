#ifndef GRAPH_H
#define GRAPH_H

#include <functional>
#include <vector>
#include <iostream>
#include "tensor/Tensor.h"
#include "util/ReferenceCount.h"
#include "util/TinySet.h"

namespace opkit
{

// This file contains the implementation of the Graph class. This is used to
// represent nodes in the computational graph and will be the primary access
// point for users (rather than dealing directly with the Node class or any
// of its subclasses).

// Forward declarations
template <class T>
struct Node;

// Core graph class. This should be the entry point for users.
template <class T>
class Graph
{
public:
    enum Type {INVALID, CONSTANT, VAR, UNARY, BINARY_IN,
        BINARY_OUT, LIST, UPDATE, UPDATE_ARG, COMPONENT};
private:
    RCPtr<Node<T>> mNode; // Pointer to the actual node.
    Type mType;           // Type of the actual node.

public:

    // Construct a graph node with the proper data and type
    // Should only be used for low-level routines (e.g. make_constant())
    Graph(Node<T>* nodePtr, Type type) : mNode(nodePtr), mType(type) {}

    // Normal constructors
    Graph() : mNode(nullptr, true), mType(INVALID) {}
    Graph(const Graph<T>& orig, bool weak = false) :
        mNode(orig.ptr(), weak), mType(orig.mType) {}
    Graph(Graph<T>&& orig) = default;

    ~Graph()
    {
        // Storing children nodes creates cycles in the dependency graph that
        // prevent reference counting from being able to clean up the memory
        // allocated by the nodes. We have to manually break those cycles in
        // order to prevent memory leaks. Here we search all parent nodes for
        // references to ourself and eliminate those references.
        if (mNode != nullptr && getNumParents() > 0 &&
            mNode->getRefCount() <= 1)
            mNode->destroyReferences();
    }

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
    void addChild(Graph<T> child)
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        mNode->addChild(child);
    }

    // Removes any connection to the underlying node (used as an implementation
    // detail.)
    void reset()
    {
        mNode = nullptr;
        mType = INVALID;
    }

    // Many graph nodes cache their most recent calculations to improve
    // performance. This function invalidates the cache of this graph node as
    // well as the caches of any node that depend on this one (children).
    void invalidate()
    {
        const static int DEFAULT_CAPACITY = 128;
        static vector<Graph<T>*> stack(DEFAULT_CAPACITY);
        static TinySet<Node<T>*> seen(DEFAULT_CAPACITY);
        seen.clear();
        stack.push_back(this);

        while (!stack.empty())
        {
            Graph<T>* cur = stack.back();
            stack.pop_back();
            if (cur == nullptr || cur->type() == INVALID ||
                seen.search(cur->ptr())) continue;

            seen.insert(cur->ptr());
            cur->node().invalidate();
            for (size_t i = 0; i < cur->getNumChildren(); ++i)
                stack.push_back(&cur->getChild(i));
        }
    }

    // Locate a node with the given name in the dependencies for this node
    // (parent nodes only). If a parent cannot be found, nullptr is returned.
    Graph<T>* find(const std::string& name)
    {
        ASSERT(mNode != nullptr, "Empty graph nodes cannot be used.");
        if (name == this->name()) return this;
        else
        {
            for (size_t i = 0; i < getNumParents(); ++i)
            {
                Graph<T>* res = getParent(i).find(name);
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
        return (mNode != nullptr) ? mNode->getNumParents() : 0;
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
        return (mNode != nullptr) ? mNode->getNumChildren() : 0;
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

}
#endif
