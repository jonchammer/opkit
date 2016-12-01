/* 
 * File:   DataLoader.h
 * Author: Jon C. Hammer
 *
 * Created on July 20, 2016, 3:38 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <vector>
#include "Matrix.h"
#include "Error.h"

using std::vector;
using std::ifstream;
using std::string;
using std::cout;
using std::endl;

namespace opkit
{
    
// Loads data from the given .ARFF file into 'features' and 'labels'.
// The data will be split based on the last 'numLabels' columns.
bool loadArff(const string& filename, Matrix& features, 
    Matrix& labels, int numLabels)
{
    Matrix temp;
    try
    {
        temp.loadARFF(filename);
        
        features.setSize(0, temp.cols() - numLabels);
        labels.setSize(0, numLabels);

        features.copyPart(temp, 0, 0, temp.rows(), temp.cols() - numLabels);
        labels.copyPart(temp, 0, temp.cols() - numLabels, temp.rows(), numLabels);
        
        return true;
    }
    catch (Ex e)
    {
        return false;
    }
}

// Loads data from a plain text file into 'features' and 'labels'.
// The number of feature dimensions and the number of label dimensions
// must be given ahead of time, as well any delimiters that separate
// sample entries. (Spaces are the default delimiters.)
bool loadText(const string& filename, Matrix& features, Matrix& labels, 
    const int numFeatures, const int numLabels, const string& delimiters = " ")
{
    // Open the file
    ifstream din;
    din.open(filename.c_str());
    if (!din)
    {
        cout << "Unable to open file." << endl;
        return false;
    }
    
    // Make sure the matrices have the proper number of columns
    features.setSize(0, numFeatures);
    labels.setSize(0, numLabels);
    
    // Save this. We'll need it later
    const char* delimitersC = delimiters.c_str();
    
    string line;
    while (getline(din, line))
    {
        // Add a new row to both matrices
        vector<double>& featureRow = features.newRow();
        vector<double>& labelRow   = labels.newRow();
        
        // Split the line into pieces based on the delimiters
        char* lineC = (char*) line.c_str();
       
        double val = atof(strtok(lineC, delimitersC));
        for (int i = 0; i < numFeatures; ++i)
        {
            featureRow[i] = val;
            val = atof(strtok(NULL, delimitersC));
        }
        
        for (int i = 0; i < numLabels; ++i)
        {
            labelRow[i] = val;
            val = atof(strtok(NULL, delimitersC));
        }
    }
    
    din.close();
    return true;
}

void print(const Matrix& features, const Matrix& labels)
{
    cout << std::fixed << std::showpoint << std::setprecision(6);
        
    for (size_t row = 0; row < features.rows(); ++row)
    {
        const vector<double>& feature = features.row(row);
        const vector<double>& label   = labels.row(row);
        
        cout << "[ ";
        for (size_t col = 0; col < feature.size(); ++col)
            cout << std::setw(10) << feature[col] << " ";

        cout << "][ ";

        for (size_t col = 0; col < label.size(); ++col)
            cout << std::setw(10) << label[col] << " ";

        cout << "]" << std::endl;
    }
}

};

#endif /* DATALOADER_H */

