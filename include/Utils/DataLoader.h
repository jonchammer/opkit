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
#include <algorithm>
#include <sstream>
#include "Dataset.h"
#include "Matrix.h"
#include "Error.h"

using std::vector;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;
using std::cout;
using std::endl;

namespace opkit
{

// Loads data from the given .ARFF file into 'features' and 'labels'.
// The data will be split based on the last 'numLabels' columns.
template <class T>
bool loadArff(const string& filename, Dataset<T>& features,
    Dataset<T>& labels, int numLabels)
{
    Dataset<T> temp;
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

// Loads data from the given .ARFF file into 'features' and 'labels'.
// The data will be split based on the last 'numLabels' columns. Since CSV files
// don't natively encode the datatype (either categorical or continous), we
// assume all columns are continuous. Use the convertColumnToCategorical()
// function in Dataset to convert the columns that really should be categorical.
template <class T>
bool loadCSV(const string& filename, Dataset<T>& features, Dataset<T>& labels,
    int numLabels)
{
    ifstream din(filename);
    if (!din) return false;

    // Determine the number of columns
    string line;
    if (getline(din, line))
    {
        int numCols = std::count(line.begin(), line.end(), ',') + 1;
        features.setSize(0, numCols - numLabels);
        labels.setSize(0, numLabels);

        // Rewind to the beginning of the file
        din.seekg(0);

        // Read the data
        string token;
        while (din.peek() != EOF)
        {
            vector<T>& featureRow = features.newRow();
            vector<T>& labelRow   = labels.newRow();
            for (int i = 0; i < features.cols(); ++i)
            {
                getline(din, token, ',');
                featureRow[i] = stof(token);
            }
            for (int i = 0; i < numLabels - 1; ++i)
            {
                getline(din, token, ',');
                labelRow[i] = stof(token);
            }
            getline(din, token);
            labelRow[numLabels - 1] = stof(token);
        }
    }

    return true;
}

// Loads data from a plain text file into 'features' and 'labels'.
// The number of feature dimensions and the number of label dimensions
// must be given ahead of time, as well any delimiters that separate
// sample entries. (Spaces are the default delimiters.)
template <class T>
bool loadText(const string& filename, Dataset<T>& features, Dataset<T>& labels,
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
bool saveDataRaw(const string& filename, const Dataset<T>& features, const Dataset<T>& labels)
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

	const size_t N = features.getRows();
	const size_t M = features.getCols();
	const size_t C = labels.getCols();

	// Write the metadata
	dout.write((char*) &N, sizeof(size_t));
	dout.write((char*) &M, sizeof(size_t));
	dout.write((char*) &C, sizeof(size_t));

	// Write the actual data
	for (size_t i = 0; i < N; ++i)
	{
		dout.write((char*) features(i), sizeof(T) * M);
		dout.write((char*) labels(i),   sizeof(T) * C);
	}

	dout.close();
	return true;
}

// Loads the data from a single RAW data file into the given features and labels matrices.
// Note that this is only useful for continuous attributes, since categorical attributes
// are not supported.
template <class T>
bool loadDataRaw(const string& filename, Dataset<T>& features, Dataset<T>& labels)
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

	features.resize(N, M);
	labels.resize(N, C);

	// Read the actual data into the matrices
	for (size_t i = 0; i < N; ++i)
	{
		din.read((char*) features(i), sizeof(T) * M);
		din.read((char*) labels(i),   sizeof(T) * C);
	}

	din.close();
	return true;
}

template <class T>
void print(const Dataset<T>& features, const Dataset<T>& labels)
{
    cout << std::fixed << std::showpoint << std::setprecision(6);

    for (size_t row = 0; row < 5 /*features.rows()*/; ++row)
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
