#ifndef LIST_NODE_H
#define LIST_NODE_H

#include "tensor/Tensor.h"

namespace opkit
{

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
        // Evaluate each of the first N - 1 list elements
        for (size_t i = 0; i < mDependents.size() - 1; ++i)
            mDependents[i]();

        // Return the result of the last one
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

template <class T, class VecType>
Graph<T> make_list(VecType&& dependents)
{
    Graph<T> res(new ListNode<T>("list", std::forward<VecType>(dependents)),
        Graph<T>::Type::LIST);

    for (auto& elem : dependents)
        elem.addChild(res);
    return res;
}

}
#endif
