// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef Dataset_H
#define Dataset_H

#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include "Matrix.h"
#include "Error.h"

using std::string;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::unordered_map;

namespace opkit
{

#define UNKNOWN_VALUE -1e308

// This stores a Dataset, A.K.A. data set, A.K.A. table. Each element is
// represented as a double value. Nominal values are represented using their
// corresponding zero-indexed enumeration value. For convenience,
// the Dataset also stores some meta-data which describes the columns (or attributes)
// in the Dataset. You can access the elements in the Dataset using square
// brackets. (Row comes first. Column comes second. Both are zero-indexed.)
// For example:
//
// Dataset m;
// m.setSize(3, 2);
// m[0][0] = 1.0;
// m[0][1] = 1.5;
// m[1][0] = 2.3;
// m[1][1] = 3.5;
// m[2][0] = 0.0;
// m[2][1] = 1234.567;
//
template <class T>
class Dataset
{
private:
    // Data
    vector< vector<T> > m_data; // Dataset elements

    // Meta-data
    string m_filename; // the name of the file
    vector<string> m_attr_name; // the name of each attribute (or column)
    vector< unordered_map<string, size_t> > m_str_to_enum; // value to enumeration
    vector< unordered_map<size_t, string> > m_enum_to_str; // enumeration to value

public:
    /// Creates a 0x0 Dataset. (Next, to give this Dataset some dimensions, you should call:
    ///    loadARFF,
    ///    setSize,
    ///    addColumn, or
    ///    copyMetaData
    Dataset() {}

    /// Destructor
    ~Dataset() {}

    /// Loads the Dataset from an ARFF file
    void loadARFF(string filename);

    /// Saves the Dataset to an ARFF file
    void saveARFF(string filename) const;

    /// Makes a rows x columns Dataset of *ALL CONTINUOUS VALUES*.
    /// This method wipes out any data currently in the Dataset. It also
    /// wipes out any meta-data.
    void setSize(size_t rows, size_t cols);

    /// Clears this Dataset and copies the meta-data from that Dataset.
    /// In other words, it makes a zero-row Dataset with the same number
    /// of columns as "that" Dataset. You will need to call newRow or newRows
    /// to give the Dataset some rows.
    void copyMetaData(const Dataset& that);

    /// Adds a column to this Dataset with the specified number of values. (Use 0 for
    /// a continuous attribute.) This method also sets the number of rows to 0, so
    /// you will need to call newRow or newRows when you are done adding columns.
    void newColumn(size_t vals = 0);

    /// Adds one new row to this Dataset. Returns a reference to the new row.
    vector<T>& newRow();

    /// Adds 'n' new rows to this Dataset. (These rows are not initialized.)
    void newRows(size_t n);

    /// Returns the number of rows in the Dataset
    size_t rows() const { return m_data.size(); }

    /// Returns the number of columns (or attributes) in the Dataset
    size_t cols() const { return m_attr_name.size(); }

    /// Returns the name of the specified attribute
    const string& attrName(size_t col) const { return m_attr_name[col]; }

    /// Returns the name of the specified value
    const string& attrValue(size_t attr, size_t val) const;

    /// Returns a reference to the specified row
    vector<T>& row(size_t index) { return m_data[index]; }

    /// Returns a const reference to the specified row
    const vector<T>& row(size_t index) const {return m_data[index]; }

    /// Returns a reference to the specified row
    vector<T>& operator [](size_t index) { return m_data[index]; }

    /// Returns a reference to the specified row
    const vector<T>& operator [](size_t index) const { return m_data[index]; }

    /// Returns the number of values associated with the specified attribute (or column)
    /// 0=continuous, 2=binary, 3=trinary, etc.
    size_t valueCount(size_t attr) const { return m_enum_to_str[attr].size(); }

    /// Returns the mean of the elements in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
    T columnMean(size_t col) const;

    /// Returns the min elements in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
    T columnMin(size_t col) const;

    /// Returns the min elements in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
    T columnMax(size_t col) const;

    /// Returns the most common value in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
    T mostCommonValue(size_t col) const;

    /// Copies the specified rectangular portion of that Dataset, and adds it to the bottom of this Dataset.
    /// (If colCount does not match the number of columns in this Dataset, then this Dataset will be cleared first.)
    void copyPart(const Dataset& that, size_t rowBegin, size_t colBegin, size_t rowCount, size_t colCount);

    /// Sets every elements in the Dataset to the specified value.
    void setAll(T val);

    /// Throws an exception if that has a different number of columns than
    /// this, or if one of its columns has a different number of values.
    void checkCompatibility(const Dataset& that) const;

    // Convert this dataset to an equivalent matrix
    void toMatrix(Matrix<T>& dest) const;

    // Take the given column and turn it into a categorical column
    void convertColumnToCategorical(const size_t col);

    // Convert the given column to a 1-hot representation
    void convertColumnToOneHot(const size_t col);
};

template <class T>
void Dataset<T>::setSize(size_t rows, size_t cols)
{
    // Make space for the data
    m_data.resize(rows);
    for(size_t i = 0; i < rows; i++)
        m_data[i].resize(cols);

    // Set the meta-data
    m_filename = "";
    m_attr_name.resize(cols);
    m_str_to_enum.resize(cols);
    m_enum_to_str.resize(cols);
    for(size_t i = 0; i < cols; i++)
    {
        m_str_to_enum[i].clear();
        m_enum_to_str[i].clear();
    }
}

template <class T>
void Dataset<T>::copyMetaData(const Dataset<T>& that)
{
    m_data.clear();
    m_attr_name = that.m_attr_name;
    m_str_to_enum = that.m_str_to_enum;
    m_enum_to_str = that.m_enum_to_str;
}

template <class T>
void Dataset<T>::newColumn(size_t vals)
{
    m_data.clear();
    size_t c = cols();
    string name = "col_";
    name += std::to_string(c);
    m_attr_name.push_back(name);
    unordered_map <string, size_t> temp_str_to_enum;
    unordered_map <size_t, string> temp_enum_to_str;
    for(size_t i = 0; i < vals; i++)
    {
        string sVal = "val_";
        sVal += std::to_string(i);
        temp_str_to_enum[sVal] = i;
        temp_enum_to_str[i] = sVal;
    }
    m_str_to_enum.push_back(temp_str_to_enum);
    m_enum_to_str.push_back(temp_enum_to_str);
}

template <class T>
std::vector<T>& Dataset<T>::newRow()
{
    size_t c = cols();
    if(c == 0)
        throw Ex("You must add some columns before you add any rows.");
    size_t rc = rows();
    m_data.resize(rc + 1);
    std::vector<T>& newrow = m_data[rc];
    newrow.resize(c);
    return newrow;
}

template <class T>
void Dataset<T>::newRows(size_t n)
{
    size_t c = cols();
    if (c == 0)
        throw Ex("You must add some columns before you add any rows.");
    size_t rc = rows();
    m_data.resize(rc + n);
    for (size_t i = 0; i < n; ++i)
        m_data[rc + i].resize(c);
}

template <class T>
T Dataset<T>::columnMean(size_t col) const
{
    T sum = 0.0;
    size_t count = 0;
    typename std::vector< std::vector<T> >::const_iterator it;
    for(it = m_data.begin(); it != m_data.end(); it++)
    {
        T val = (*it)[col];
        if(val != UNKNOWN_VALUE)
        {
            sum += val;
            count++;
        }
    }
    return sum / count;
}

template <class T>
T Dataset<T>::columnMin(size_t col) const
{
    T m = 1e300;
    typename std::vector< std::vector<T> >::const_iterator it;
    for(it = m_data.begin(); it != m_data.end(); it++)
    {
        T val = (*it)[col];
        if(val != UNKNOWN_VALUE)
            m = std::min(m, val);
    }
    return m;
}

template <class T>
T Dataset<T>::columnMax(size_t col) const
{
    T m = -1e300;
    typename std::vector< std::vector<T> >::const_iterator it;
    for(it = m_data.begin(); it != m_data.end(); it++)
    {
        T val = (*it)[col];
        if(val != UNKNOWN_VALUE)
            m = std::max(m, val);
    }
    return m;
}

template <class T>
T Dataset<T>::mostCommonValue(size_t col) const
{
    unordered_map<T, size_t> counts;
    typename vector< vector<T> >::const_iterator it;
    for(it = m_data.begin(); it != m_data.end(); it++)
    {
        T val = (*it)[col];
        if(val != UNKNOWN_VALUE)
        {
            typename unordered_map<T, size_t>::iterator pair = counts.find(val);
            if(pair == counts.end())
                counts[val] = 1;
            else
                pair->second++;
        }
    }
    size_t valueCount = 0;
    T value = 0;
    for(auto i = counts.begin(); i != counts.end(); i++)
    {
        if(i->second > valueCount)
        {
            value = i->first;
            valueCount = i->second;
        }
    }
    return value;
}

template <class T>
void Dataset<T>::copyPart(const Dataset<T>& that, size_t rowBegin, size_t colBegin, size_t rowCount, size_t colCount)
{
    if(rowBegin + rowCount > that.rows() || colBegin + colCount > that.cols())
        throw Ex("out of range");

    // Copy the specified region of meta-data
    if(cols() != colCount)
        setSize(0, colCount);
    for(size_t i = 0; i < colCount; i++)
    {
        m_attr_name[i] = that.m_attr_name[colBegin + i];
        m_str_to_enum[i] = that.m_str_to_enum[colBegin + i];
        m_enum_to_str[i] = that.m_enum_to_str[colBegin + i];
    }

    // Copy the specified region of data
    size_t rowsBefore = m_data.size();
    m_data.resize(rowsBefore + rowCount);
    for(size_t i = 0; i < rowCount; i++)
    {
        typename vector<T>::const_iterator itIn = that[rowBegin + i].begin() + colBegin;
        m_data[rowsBefore + i].resize(colCount);
        typename vector<T>::iterator itOut = m_data[rowsBefore + i].begin();
        for(size_t j = 0; j < colCount; j++)
            *itOut++ = *itIn++;
    }
}

string toLower(string strToConvert)
{
    //change each element of the string to lower case
    for(size_t i = 0; i < strToConvert.length(); i++)
        strToConvert[i] = tolower(strToConvert[i]);
    return strToConvert;//return the converted string
}

template <class T>
void Dataset<T>::saveARFF(string filename) const
{
    std::ofstream s;
    s.exceptions(std::ios::failbit | std::ios::badbit);
    try
    {
        s.open(filename.c_str(), std::ios::binary);
    }
    catch(const std::exception&)
    {
        throw Ex("Error creating file: ", filename);
    }
    s.precision(10);
    s << "@RELATION " << m_filename << "\n";
    for(size_t i = 0; i < m_attr_name.size(); i++)
    {
        s << "@ATTRIBUTE " << m_attr_name[i];
        if(m_attr_name[i].size() == 0)
            s << "x";
        size_t vals = valueCount(i);
        if(vals == 0)
            s << " REAL\n";
        else
        {
            s << " {";
            for(size_t j = 0; j < vals; j++)
            {
                s << attrValue(i, j);
                if(j + 1 < vals)
                    s << ",";
            }
            s << "}\n";
        }
    }
    s << "@DATA\n";
    for(size_t i = 0; i < rows(); i++)
    {
        const std::vector<T>& r = (*this)[i];
        for(size_t j = 0; j < cols(); j++)
        {
            if(r[j] == UNKNOWN_VALUE)
                s << "?";
            else
            {
                size_t vals = valueCount(j);
                if(vals == 0)
                    s << std::to_string(r[j]);
                else
                {
                    size_t val = r[j];
                    if(val >= vals)
                        throw Ex("value out of range");
                    s << attrValue(j, val);
                }
            }
            if(j + 1 < cols())
                s << ",";
        }
        s << "\n";
    }
}

template <class T>
void Dataset<T>::loadARFF(string fileName)
{
    size_t lineNum = 0;
    string line;                 //line of input from the arff file
    ifstream inputFile;          //input stream
    unordered_map <string, size_t> tempMap;   //temp map for int->string map (attrInts)
    unordered_map <size_t, string> tempMapS;  //temp map for string->int map (attrString)
    size_t attrCount = 0;           //Count number of attributes

    inputFile.open ( fileName.c_str() );
    if ( !inputFile )
        throw Ex ( "failed to open the file: ", fileName );
    while ( !inputFile.eof() && inputFile )
    {
        //Iterate through each line of the file
        getline ( inputFile, line );
        lineNum++;

        if ( toLower ( line ).find ( "@relation" ) == 0 )
            m_filename = line.substr ( line.find_first_of ( " " ) );
        else if ( toLower ( line ).find ( "@attribute" ) == 0 )
        {
            line = line.substr ( line.find_first_of ( " \t" ) + 1 );
            string attrName = line.substr ( 0, line.find_first_of ( " \t" ) );
            m_attr_name.push_back ( attrName );
            line = line.substr ( attrName.size() );
            string value = line.substr ( line.find_first_not_of ( " \t" ) );
            tempMap.clear();
            tempMapS.clear();

            //If the attribute is nominal
            if ( value.find_first_of ( "{" ) == 0 )
            {
                int firstComma;
                int firstSpace;
                int firstLetter;
                value = value.substr ( 1, value.find_last_of ( "}" ) - 1 );
                size_t valCount = 0;
                string tempValue;

                //Parse the attributes--push onto the maps
                while ( ( firstComma = value.find_first_of ( "," ) ) > -1 )
                {
                    firstLetter = value.find_first_not_of ( " \t," );

                    value = value.substr ( firstLetter );
                    firstComma = value.find_first_of ( "," );
                    firstSpace = value.find_first_of ( " \t" );
                    tempMapS[valCount] = value.substr ( 0, firstComma );
                    string valName = value.substr ( 0, firstComma );
                    valName = valName.substr ( 0, valName.find_last_not_of(" \t") + 1);
                    tempMap[valName] = valCount++;
                    firstComma = ( firstComma < firstSpace &&
                        firstSpace < ( firstComma + 2 ) ) ? firstSpace :
                        firstComma;
                    value = value.substr ( firstComma + 1 );
                }

                //Push final attribute onto the maps
                firstLetter = value.find_first_not_of ( " \t," );
                value = value.substr ( firstLetter );
                string valName = value.substr ( 0, value.find_last_not_of(" \t") + 1);
                tempMapS[valCount] = valName;
                tempMap[valName] = valCount++;
                m_str_to_enum.push_back ( tempMap );
                m_enum_to_str.push_back ( tempMapS );
            }
            else
            {
                //The attribute is continuous
                m_str_to_enum.push_back ( tempMap );
                m_enum_to_str.push_back ( tempMapS );
            }
            attrCount++;
        }
        else if ( toLower ( line ).find ( "@data" ) == 0 )
        {
            vector<T> temp; //Holds each line of data
            temp.resize(attrCount);
            m_data.clear();
            while ( !inputFile.eof() )
            {
                getline ( inputFile, line );
                lineNum++;
                if(line.length() == 0 || line[0] == '%' || line[0] == '\n' || line[0] == '\r')
                    continue;
                size_t pos = 0;
                for ( size_t i = 0; i < attrCount; i++ )
                {
                    size_t vals = valueCount ( i );
                    size_t valStart = line.find_first_not_of ( " \t", pos );
                    if(valStart == string::npos)
                        throw Ex("Expected more elements on line ", std::to_string(lineNum));
                    size_t valEnd = line.find_first_of ( ",\n\r", valStart );
                    string val;
                    if(valEnd == string::npos)
                    {
                        if(i + 1 == attrCount)
                            val = line.substr( valStart );
                        else
                            throw Ex("Expected more elements on line ", std::to_string(lineNum));
                    }
                    else
                        val = line.substr ( valStart, valEnd - valStart );
                    pos = valEnd + 1;
                    if ( vals > 0 ) //if the attribute is nominal...
                    {
                        if ( val == "?" )
                            temp[i] = UNKNOWN_VALUE;
                        else
                        {
                            auto it = m_str_to_enum[i].find ( val );
                            if(it == m_str_to_enum[i].end())
                                throw Ex("Unrecognized enumeration value, \"", val, "\" on line ", std::to_string(lineNum), ", attr ", std::to_string(i));
                            temp[i] = m_str_to_enum[i][val];
                        }
                    }
                    else
                    {
                        // The attribute is continuous
                        if ( val == "?" )
                            temp[i] = UNKNOWN_VALUE;
                        else
                            temp[i] = atof( val.c_str() );
                    }
                }
                m_data.push_back ( temp );
            }
        }
    }
}

template <class T>
const std::string& Dataset<T>::attrValue(size_t attr, size_t val) const
{
    auto it = m_enum_to_str[attr].find(val);
    if(it == m_enum_to_str[attr].end())
        throw Ex("no name");
    return it->second;
}

template <class T>
void Dataset<T>::setAll(T val)
{
    size_t c = cols();
    typename std::vector< std::vector<T> >::iterator it;
    for(it = m_data.begin(); it != m_data.end(); it++)
        it->assign(c, val);
}

template <class T>
void Dataset<T>::checkCompatibility(const Dataset<T>& that) const
{
    size_t c = cols();
    if(that.cols() != c)
        throw Ex("Matrices have different number of columns");
    for(size_t i = 0; i < c; i++)
    {
        if(valueCount(i) != that.valueCount(i))
            throw Ex("Column ", std::to_string(i), " has mis-matching number of values");
    }
}

template <class T>
void Dataset<T>::toMatrix(Matrix<T>& dest) const
{
    dest.resize(rows(), cols());
    for (size_t i = 0; i < rows(); ++i)
    {
        for (size_t j = 0; j < cols(); ++j)
            dest(i, j) = m_data[i][j];
    }
}

template <class T>
void Dataset<T>::convertColumnToCategorical(const size_t col)
{
    // Column is already categorical. Nothing to do.
    if (valueCount(col) != 0) return;

    // Search all rows to identify the unique categorical values
    std::unordered_map<T, size_t> values;
    size_t counter = 0;
    for (size_t row = 0; row < rows(); ++row)
    {
        if (values.find(m_data[row][col]) == values.end())
        {
            values[m_data[row][col]] = counter;
            ++counter;
        }
    }

    // Replace the original values with their indices [0, N - 1]
    for (auto& row : m_data)
        row[col] = T(values[row[col]]);

    // Update the meta-data
    for (auto pair : values)
    {
        m_enum_to_str[col][pair.second] = std::to_string(pair.first);
        m_str_to_enum[col][std::to_string(pair.first)] = pair.second;
    }
}

template <class T>
void Dataset<T>::convertColumnToOneHot(const size_t col)
{
    // Continuous column - convert to categorical first
    if (valueCount(col) == 0) convertColumnToCategorical(col);

    const size_t N = valueCount(col);
    for (size_t r = 0; r < rows(); ++r)
    {
        // Add additional columns for each row
        vector<T>& row = m_data[r];
        row.insert(row.begin() + col + 1, N - 1, T{});

        // Convert to one-hot
        size_t value     = (size_t) row[col];
        row[col]         = T{};
        row[col + value] = T{1};
    }

    // Update the metadata accordingly
    m_attr_name.insert(m_attr_name.begin() + col + 1, N - 1, "");
    m_str_to_enum.insert(m_str_to_enum.begin() + col + 1, N - 1, std::unordered_map<string, size_t>());
    m_enum_to_str.insert(m_enum_to_str.begin() + col + 1, N - 1, std::unordered_map<size_t, string>());

    m_str_to_enum[col].clear();
    m_enum_to_str[col].clear();
}

};

#endif // Dataset_H
