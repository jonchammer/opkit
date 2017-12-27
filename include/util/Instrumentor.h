#ifndef INSTRUMENTOR_H
#define INSTRUMENTOR_H

// This file contains an instrumentor that is used to count how often functions
// are called. It also allows the user to keep track of the frequency of
// arbitrary properties (like function arguments).
//
// The public interface consists of the macros INSTRUMENT(), which counts the
// number of times the host function is called, and INSTRUMENT_PROPERTY(), which
// counts the number of times a property of a method has been provided.
//
// Work is only done when NDEBUG is NOT defined. When NDEBUG is enabled, the
// instrumentor will be optimized out by the compiler, incurring no additional
// runtime.
#ifndef NDEBUG

    #include <unordered_map>
    #include <map>
    #include <iostream>

    namespace opkit
    {

    using std::unordered_map;
    using std::map;
    using std::string;

    class Instrumentor
    {
    private:
        unordered_map<string, size_t> mMethodCounts;
        unordered_map<string, map<string, size_t>> mPropertyCounts;

        Instrumentor() {}

    public:
        static Instrumentor& instance()
        {
            static Instrumentor inst;
            return inst;
        }

        void update(const string& methodName)
        {
            mMethodCounts[methodName]++;
        }

        void updateProperty(const string& methodName, const string& propertyName)
        {
            mPropertyCounts[methodName][propertyName]++;
        }

        void print() const
        {
            // Calculate the ideal widths
            int width = 5;
            for (const auto& pair : mMethodCounts)
            {
                if (mPropertyCounts.find(pair.first) != mPropertyCounts.end())
                {
                    for (const auto& property : mPropertyCounts.at(pair.first))
                    {
                        if (property.first.length() > width)
                            width = property.first.length();
                    }
                }
            }

            // Print everything
            printf("\n-------------------- Instrumentation Results --------------------\n");
            map<string, size_t> sorted(mMethodCounts.begin(), mMethodCounts.end());
            for (const auto& pair : sorted)
            {
                printf("+ %s\n", pair.first.c_str(), pair.second);
                printf("  - %-*s => %zu\n", width, "Total", pair.second);
                if (mPropertyCounts.find(pair.first) != mPropertyCounts.end())
                {
                    for (const auto& property : mPropertyCounts.at(pair.first))
                    {
                        printf("  - %-*s => %zu\n", width,
                            property.first.c_str(), property.second);
                    }
                }
            }
            printf("-----------------------------------------------------------------\n\n");
        }
    };

    // Given a string representing the name of the current function, returns a
    // simpler version that is better for printing with the Instrumentor class.
    std::string simpleFunctionName(const std::string& fullName)
    {
        std::string name(fullName);
        size_t leftParen = name.find_first_of('(');

        // __PRETTY_FUNCTION__ - replace arguments with ... and remove any
        // template-specific information
        if (leftParen != name.npos)
        {
            size_t rightParen = name.find_last_of(')');
            name.erase(rightParen + 1);
            name.replace(leftParen + 1, rightParen - leftParen - 1, "...");
        }

        // __FUNCTION__ or __func__ - add empty arguments
        else name.append("(...)");

        return name;
    }
}
#endif
#endif

// Define the macros INSTRUMENT() and INSTRUMENT_PROPERTY() as a public interface
// to the Instrumentor class. Clients should avoid using the class directly, but
// they are not prevented from doing so if necessary.
#ifndef NDEBUG

#define INSTRUMENT()                                                           \
    opkit::Instrumentor::instance().update(                                    \
        simpleFunctionName(__PRETTY_FUNCTION__));                              \

#define INSTRUMENT_PROPERTY(property)                                          \
    opkit::Instrumentor::instance().updateProperty(                            \
        simpleFunctionName(__PRETTY_FUNCTION__), (property));                  \

// Define the macros anyway, but they will be optimized away, since they have
// no actual content when NDEBUG is enabled.
#else
    #define INSTRUMENT()
    #define INSTRUMENT_PROPERTY(property)
#endif
