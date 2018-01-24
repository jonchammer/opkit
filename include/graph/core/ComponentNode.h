#ifndef COMPONENT_NODE_H
#define COMPONENT_NODE_H

namespace opkit
{

// A graph node that wraps an entire subgraph, making it easier to visualize
// large components.
template <class T>
class ComponentNode : public Node<T>
{
private:
    std::string mName;
    Graph<T> mSubgraph;

public:
    // Normal constructors
    ComponentNode()                           = delete;
    ComponentNode(const ComponentNode& other) = default;
    ComponentNode(ComponentNode&& other)      = default;

    // Additional Constructors
    template <class GraphType>
    ComponentNode(const std::string& name, GraphType&& subgraph):
        mName(name),
        mSubgraph(std::forward<GraphType>(subgraph))
    {}

    // Assignment operators
    ComponentNode& operator=(const ComponentNode& orig) = default;
    ComponentNode& operator=(ComponentNode&& orig)      = default;

    const Tensor<T>& operator()() override
    {
        return mSubgraph();
    }

    const Graph<T>& getParent(const size_t index) const override
    {
        ASSERT(index < 1, "Components have only one parent.");
        return mSubgraph;
    }

    Graph<T>& getParent(const size_t index) override
    {
        ASSERT(index < 1, "Components have only one parent.");
        return mSubgraph;
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
        out << mName << "[...]";
        return out;
    }

    ComponentNode* clone() const override
    {
        return new ComponentNode(*this);
    }
};

template <class T>
Graph<T> make_component(const std::string& name, Graph<T> subgraph)
{
    Graph<T> res(new ComponentNode<T>(name, subgraph), Graph<T>::Type::COMPONENT);
    subgraph.addChild(res);
    return res;
}

}
#endif
