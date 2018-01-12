#ifndef NODE_H
#define NODE_H

#include "tensor/Tensor.h"

namespace opkit
{

// Base class for all nodes in the graph.
template <class T>
struct Node : public RCObject
{

// ---------------------- Begin Invalidation framework ---------------------- //
private:
    std::vector<Graph<T>> children;

public:

    void addChild(Graph<T> child)
    {
        // Add a weak reference to the child to avoid cycles in the dependency
        // graph.
        children.emplace_back(child, true);
    }

    void destroyReferences()
    {
        // Search each parent node for children that reference us. If we find
        // any, eliminate those references so no one is left with an invalid
        // reference after we are destroyed.
        for (size_t i = 0; i < getNumParents(); ++i)
        {
            Graph<T>& parent = getParent(i);
            Node<T>& parentNode = parent.node();
            for (Graph<T>& child : parentNode.children)
            {
                if (child.ptr() == this)
                    child.reset();
            }
        }
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

}
#endif
