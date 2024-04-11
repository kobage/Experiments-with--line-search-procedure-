#pragma once
#include<random>
#include<string>
#include<fstream>
#include<numeric>
#include<vector>
#include<array>
#include<chrono>
#include<iostream>
#include<Eigen/Dense>
using namespace Eigen;
using namespace std;

class two_layer_mnist
{
public:
	std::string name{};
	long int n{};
	size_t iterations{ 1000 };
	double objective_function{};
	long int n0{};				//  the size of the sample, i.e. the width of the input layer
	long int n1{};				//  the width of the hidden layer
	long int n2{};				//  the width of the output layer 

	size_t n_train{};
	size_t n_test{};
	Vector<uint8_t, Dynamic> labels_train;
	Vector<uint8_t, Dynamic> labels_test{};
	Vector<double, Dynamic> b1;
	Vector<double, Dynamic> b2;
	Vector<double, Dynamic> gr_b1;  //here and below, gr_??? means gradient of ???
	Vector<double, Dynamic> gr_b2;

	size_t label{};				//corresponds to a given sample

	Matrix <double, Dynamic, Dynamic> X_train{};
	Matrix <double, Dynamic, Dynamic> X_test{};
	Matrix <double, Dynamic, Dynamic> a1;
	Matrix <double, Dynamic, Dynamic> a2;
	Matrix <double, Dynamic, Dynamic> z1;
	Matrix <double, Dynamic, Dynamic> z2;
	Matrix <double, Dynamic, Dynamic> gr_z1;
	Matrix <double, Dynamic, Dynamic> gr_z2;
	Matrix <double, Dynamic, Dynamic> w1{};
	Matrix <double, Dynamic, Dynamic> w2{};
	Matrix <double, Dynamic, Dynamic> w2T{};
	Matrix <double, Dynamic, Dynamic> gr_w1;
	Matrix <double, Dynamic, Dynamic> gr_w2;

	double learning_rate{};				//learning rate
	double accuracy{};

	two_layer_mnist
	(
		std::string pathToTrainData,
		std::string pathToTrainLabels,
		std::string pathToTestnData,
		std::string pathToTestLabels
	)
	{
		name = "mnist";
		n0 = 784;
		n1 = 10;
		n2 = 10;
		n = n1 * n0 + n1 + n2 * n1 + n2;
		learning_rate = 0.0000017;
		n_train = 60000;
		n_test = 10000;
		
		labels_train.resize(n_train);
		labels_test.resize(n_test);
		b1.resize(n1);
		gr_b1.resize(n1);
		b2.resize(n2);
		gr_b2.resize(n2);

		X_train.resize(n0, n_train);
		X_test.resize(n0, n_test);
		
		//Filling "X_train"
		std::ifstream ifs("X", std::ios::in | std::ios::binary);
		uint8_t* q = new uint8_t[784];
		for (int i = 0; i < 60000; ++i) {
			ifs.read((char*)q, 784);
			for (int j = 0; j < 784; ++j) {
				X_train(j,i) = q[j] / 255.0;
			}
		}
		delete[] q;
	
		//Filling "labels_train"
		{
			double tmp;
			ifstream ifs{ "y_train.txt" };
			for (int i = 0; i < 60000; ++i)
			{
				ifs >> tmp;
				labels_train[i] = uint8_t(tmp);
			}
		}
		//Filling "X_tests"
		{
			std::string image_file = pathToTestnData;
			std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
			char p[4];
			ifs.read(p, 4); ifs.read(p, 4); ifs.read(p, 4); ifs.read(p, 4);
			//as we already have parameter values, so just read with zero effect
			char* q = new char[784];
			for (int i = 0; i < 10000; ++i) {
				ifs.read(q, 784);
				for (int j = 0; j < 784; ++j) {
					X_test(j,i) = q[j] / 255.0;
				}
			}
			delete[] q;
		}
		//Filling "labels_test"
		{
			std::string image_file = pathToTestLabels;
			std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
			char p[4];
			ifs.read(p, 4); ifs.read(p, 4);
			for (int i = 0; i < 10000; ++i) {
				ifs.read(p, 1);
				int label = p[0];
				labels_test[i] = label;
			}
		}

		a1.resize(n1, n_train);
		z1.resize(n1, n_train);
		gr_z1.resize(n1, n_train);
		a2.resize(n2, n_train);
		z2.resize(n2, n_train);
		gr_z2.resize(n2, n_train);

		w1.resize(n1, n0);
		gr_w1.resize(n1, n0);
		w2.resize(n2, n1);
		gr_w2.resize(n2, n1);
	}

	std::string about() const { return "Problem: " + name + ", " + getSizes(); }
	std::string virtual getName() const { return name; }
	string  getSizes() const { return to_string(n0) + "X" + to_string(n1) + "X" + to_string(n2); };
	long int getSize() const { return n; }

	double forward_propagation()
	{
		double fx(0.0);
		accuracy = 0.;
		size_t maxInd{};
		double mx{};

		z1 = (w1 * X_train).colwise() + b1;	
		a1 = z1.array().cwiseMax(0).pow(2);
		z2 = (w2 *a1).colwise() + b2;
		 
		auto X = z2.array().exp();
		auto sm = X.colwise().sum();
		a2 = X.rowwise() / sm.array();

		for (size_t i = 0; i < n_train; ++i)
		{
			//maxindex at i
			label = labels_train[i];
			fx += -log(a2(label, i));
			maxInd = 0;
			mx = a2(0, i);
			for (size_t j{ 1 }; j < n2; ++j)
				if (a2(j, i) > mx)
				{
					mx = a2(j, i);
					maxInd = j;
				}
			if (maxInd == label)
				++accuracy;
		}
		accuracy /= 600;
		return fx;
	}
	
	void backward_propagation()
	{
		gr_z2 = a2;
		for (size_t i = 0; i < n_train; ++i)
			gr_z2(labels_train[i],i) -= 1;

		gr_w2 = gr_z2*a1.transpose();
		gr_b2 = gr_z2.array().rowwise().sum();
		gr_z1 = (w2.transpose() * gr_z2).array() * (z1.array().cwiseMax(0) * 2);
		gr_w1 = gr_z1 * X_train.transpose();
		gr_b1 = gr_z1.array().rowwise().sum();
	}

	void update_params()
	{
		w1 -= learning_rate * gr_w1;
		b1 -= learning_rate * gr_b1;
		w2 -= learning_rate * gr_w2;
		b2 -= learning_rate * gr_b2;
	}

	void gradient_descent()
	{
		initialize();
		for (size_t i{ 0 }; i < iterations; ++i)
		{
			objective_function = forward_propagation();
			backward_propagation();
			update_params();
			//	if (i % 10 == 0)
			cout << i << ":  " << objective_function << "  " << accuracy << endl;
		}
	}

	void initialize()
	{
		{
			ifstream ifs{ "W1.txt" };
			for (int i = 0; i < n1; ++i)
				for (size_t j = 0; j < n0; ++j)
					ifs >> w1(i,j);
		}
		{
			ifstream ifs{ "W2.txt" };
			for (int i = 0; i < n2; ++i)
				for (size_t j = 0; j < n1; ++j)
					ifs >> w2(i,j);
		}
		{
			ifstream ifs{ "B1.txt" };
			for (int i = 0; i < n1; ++i)
				ifs >> b1[i];
		}
		{
			ifstream ifs{ "B2.txt" };
			for (int i = 0; i < n2; ++i)
				ifs >> b2[i];
		}
	}
};

void gradient_descent()
{
	path dataPath = findPathTo("Data");
	two_layer_mnist* p = new two_layer_mnist
	(
		dataPath.string() + "/mnist/train-images.idx3-ubyte",
		dataPath.string() + "/mnist/train-labels.idx1-ubyte",
		dataPath.string() + "/mnist/t10k-images.idx3-ubyte",
		dataPath.string() + "/mnist/t10k-labels.idx1-ubyte"
	);


	p->gradient_descent();
}