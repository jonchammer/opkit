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
using std::ofstream;
using std::string;
using std::cout;
using std::endl;

namespace opkit
{
    
// Loads data from the given .ARFF file into 'features' and 'labels'.
// The data will be split based on the last 'numLabels' columns.
template <class T>
bool loadArff(const string& filename, Matrix<T>& features, 
    Matrix<T>& labels, int numLabels)
{
    Matrix<T> temp;
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
template <class T>
bool loadText(const string& filename, Matrix<T>& features, Matrix<T>& labels, 
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
        vector<T>& featureRow = features.newRow();
        vector<T>& labelRow   = labels.newRow();
        
        // Split the line into pieces based on the delimiters
        char* lineC = (char*) line.c_str();
       
        T val = atof(strtok(lineC, delimitersC));
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

// Saves the data in the given features and labels matrices into a single RAW data
// file. Note that this is only useful for continuous attributes, since categorical
// attributes are not supported.
template <class T>
bool saveDataRaw(const string& filename, const Matrix<T>& features, const Matrix<T>& labels)
{
	ofstream dout;
	dout.open(filename.c_str(), std::ios::binary);
	if (!dout)
	{
		cout << "Unable to open file: " << filename << endl;
		return false;
	} 

	const size_t N = features.rows();
	const size_t M = features.cols();
	const size_t C = labels.cols();

	// Write the metadata
	dout.write((char*) &N, sizeof(size_t));
	dout.write((char*) &M, sizeof(size_t));
	dout.write((char*) &C, sizeof(size_t));

	// Write the actual data
	for (size_t i = 0; i < N; ++i)
	{
		const vector<T>& featureRow = features.row(i);
		const vector<T>& labelRow   = labels.row(i);

		dout.write((char*) &featureRow[0], sizeof(T) * M);
		dout.write((char*) &labelRow[0],   sizeof(T) * C);
	}

	dout.close();
	return true;
}

// Loads the data from a single RAW data file into the given features and labels matrices.
// Note that this is only useful for continuous attributes, since categorical attributes 
// are not supported.
template <class T>
bool loadDataRaw(const string& filename, Matrix<T>& features, Matrix<T>& labels)
{
	ifstream din;
	din.open(filename.c_str(), std::ios::binary);
	if (!din)
	{
		cout << "Unable to open file: " << filename << endl;
		return false;
	} 

	size_t N, M, C;

	// Read the metadata
	din.read((char*) &N, sizeof(size_t));
	din.read((char*) &M, sizeof(size_t));
	din.read((char*) &C, sizeof(size_t));

	features.setSize(N, M);
	labels.setSize(N, C);

	// Read the actual data into the matrices
	for (size_t i = 0; i < N; ++i)
	{
		const vector<T>& featureRow = features.row(i);
		const vector<T>& labelRow   = labels.row(i);

		din.read((char*) &featureRow[0], sizeof(T) * M);
		din.read((char*) &labelRow[0],   sizeof(T) * C);
	}

	din.close();
	return true;
}

template <class T>
void print(const Matrix<T>& features, const Matrix<T>& labels)
{
    cout << std::fixed << std::showpoint << std::setprecision(6);
        
    for (size_t row = 0; row < features.rows(); ++row)
    {
        const vector<T>& feature = features.row(row);
        const vector<T>& label   = labels.row(row);
        
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

