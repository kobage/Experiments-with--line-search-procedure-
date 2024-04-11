#pragma once
#include<random>
#include<string>
#include<fstream>
#include"auxiliaryFunctions.h"
#include<numeric>
#include<vector>
#include<array>
#include<chrono>
using namespace std;

//35:   class CURLY
//219:  class one_layer_log_regr 
//391:  one_layer_log_regr* IrisLoader
//483:  one_layer_log_regr* mnistLoader
//589:  class SMG_QuadraticPenalties
//817: class SMG_CubicPenalties
//1060: class memCubPenSMG_Eigen 

class composite_problem
{
public:
	//fields
	long int ValGradEvaluations{};
	long int ValueEvaluations{};
	long int dirDerEvaluations{};
	long int gradEvaluations{};
	//methods
	double virtual valGrad(double*, double*) = 0;
	double virtual memoised_val_dirDer(double, double*, double*, double&) = 0;
	void virtual memoised_grad(double, double*, double*, double*) = 0;
	void virtual  initialize(double*) = 0;
	std::string virtual about() const = 0;
	std::string virtual getName() const = 0;
	long int virtual getSize() const = 0;
	bool virtual stoppingCondition(double*) const = 0;
	void virtual setMomoisation(const bool) = 0;
};

class CURLY : public composite_problem
{
	size_t k{};
	double fx{};
	double cube{};
	int i{}, j{};
	double q{};
	double* Ax{};
	double* Ad{};
public:
	double EPS{};
	long int n{};
	CURLY() {}
	CURLY(double const tolerance, size_t kValue, long int nValue) : EPS{ tolerance }, k{ kValue }, n{ nValue }
	{
		Ax = new double[n];
		Ad = new double[n];
	}
	~CURLY()
	{
		delete[] Ad;
		delete[] Ax;
	}
	std::string  aboutSize()  const { return "size: " + std::to_string(n); }
	std::string about() const { return "Problem: CURLY,  size: " + aboutSize(); }
	std::string virtual getName() const { return "CURL"; }
	long int  getSize() const { return n; };
	void setMomoisation(const bool b) { memoizable = b; };
	bool memoizable{ true };

	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		fx = 0.;
		for (int i = 0; i < n; i++)
			g[i] = 0;

		q = 0.0;
		for (int j = 0; j <= k; j++)
			q += x[j];
		if (memoizable)
			Ax[0] = q;
		fx += pow(q, 4.0) - 20 * q * q - 0.1 * q;

		for (int j = 0; j <= k; j++)
			g[j] += 4 * pow(q, 3.0) - 40 * q - 0.1;

		for (i = 1; i < n - k; i++)
		{
			q = q - x[i - 1] + x[i + k];
			if (memoizable)
				Ax[i] = q;
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;
			cube = 4 * t0 * t1 - 40 * t0 - 0.1;
			j = i;
			g[j] += cube; ++j;
			for (; j <= i + k; )
			{
				g[j] += cube; ++j;
				g[j] += cube; ++j;
				g[j] += cube; ++j;
				g[j] += cube; ++j;
				g[j] += cube; ++j;
			}
		}
		for (; i < n; i++)
		{
			q -= x[i - 1];
			if (memoizable)
				Ax[i] = q;
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;
			cube = 4 * t0 * t1 - 40 * t0 - 0.1;
			for (int j = i; j <= n - 1; j++)
				g[j] += cube;
		}
		return fx;
	}

	//calculates directional derivative dd at the point x0-td;
	double memoised_val_dirDer
	(
		double t,
		double* x,
		double* d,			//direction
		double& dd  			//directional derivative
	)
	{
		++dirDerEvaluations;
		fx = 0.;
		if (memoizable)
		{
			Ad[0] = 0;
			for (int j = 0; j <= k; j++)
				Ad[0] += d[j];

			for (i = 1; i < n - k; ++i)
				Ad[i] = Ad[i - 1] - d[i - 1] + d[i + k];

			for (; i < n; i++)
				Ad[i] = Ad[i - 1] - d[i - 1];
			memoizable = false;
		}

		double q{};
		double q2{};
		double q3{};
		double q4{};
		double y{};
		dd = 0;

		for (i = 0; i < n; i++)
		{
			q = Ax[i] - t * Ad[i];
			q2 = q * q;
			q3 = q * q2;
			q4 = q2 * q2;
			fx += q4 - 20 * q2 - 0.1 * q;
			y = 4 * q3 - 40 * q - 0.1;
			dd -= y * Ad[i];
		}
		return fx;
	}
	void memoised_grad
	(
		double t,
		double* x,
		double* d,
		double* g
	)
	{
		++gradEvaluations;
		for (int i = 0; i < n; i++)
			g[i] = 0;

		double q{};
		double q2{};
		double q3{};
		double y{};

		for (i = 0; i < n - k; i++)
		{
			Ax[i] = q = Ax[i] - t * Ad[i];
			q2 = q * q;
			q3 = q * q2;
			y = 4 * q3 - 40 * q - 0.1;
			for (j = i; j <= i + k; ++j)
				g[j] += y;
		}
		for (; i < n; i++)
		{
			Ax[i] = q = Ax[i] - t * Ad[i];
			q2 = q * q;
			q3 = q * q2;
			y = 4 * q3 - 40 * q - 0.1;
			for (int j = i; j < n; j++)
				g[j] += y;
		}
		memoizable = true;
	}

	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 0.0001 / (n + 1);
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-------------------------------------    
double sigmoid(double x)
{
	return 1. / (1. + exp(-x));
}

class one_layer_log_regr : public composite_problem
{
public:
	double EPS{};
	long int n{};
	size_t n_neurons{};
	size_t n_train{};
	size_t n_test{};
	double** X_train{};
	double** X_test{};
	double* p{};					// np.matmul(X_train, z)) - y
	double* q{};					// it is used in directional derivative calculations - - - ???? ----????  --- ???
	double* Xx{};					// it is used in the memoised directional derivative calculations
	double* Xd{};					// it is used in the memoised directional derivative calculations
	uint8_t* labels_train{};
	uint8_t* labels_test{};
	uint8_t* y{};
	std::string name{};
	bool memoizable{ true };
	size_t neurNumb{};

	std::string  aboutSize()  const { return "size: " + std::to_string(n); }
	std::string about() const { return "Problem: " + name + ", " + aboutSize(); }
	std::string virtual getName() const { return name; }
	long int  getSize() const { return n; };
	void setMomoisation(const bool b) { memoizable = b; };

	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx(0.0);
		double tmp{};
		double* z{};

		for (size_t i = 0; i < n_train; ++i)
		{
			z = X_train[i];
			tmp = std::inner_product(z, z + n, x, double{});

			if (memoizable)
				Xx[i] = tmp;

			if (0 == y[i])
				fx += log(sigmoid(-tmp));
			else
				fx += log(sigmoid(tmp));
			p[i] = sigmoid(tmp) - y[i];
		}

		for (size_t i = 0; i < n; ++i)
			g[i] = 0.;

		for (size_t j = 0; j < n_train; ++j)
		{
			tmp = p[j];
			for (size_t i = 0; i < n; ++i)
				g[i] += X_train[j][i] * tmp;
		}
		return -fx;
	}

	//calculates directional derivative dd at the point x0-td;
	double memoised_val_dirDer
	(
		double t,
		double* x,
		double* d,			//direction
		double& dd			//directional derivative
	)
	{
		++dirDerEvaluations;
		double fx(0.0);
		double tmp{};
		double* z{};
	
		if (memoizable)
		{
			for (size_t i = 0; i < n_train; ++i)
			{
				z = X_train[i];
				Xd[i] = std::inner_product(z, z + n, d, double{});
			}
			memoizable = false;
		}

		for (size_t i = 0; i < n_train; ++i)
		{
			tmp = Xx[i] - t * Xd[i];

			if (0 == y[i])
				fx += log(sigmoid(-tmp));
			else
				fx += log(sigmoid(tmp));

			p[i] = sigmoid(tmp) - y[i];
		}

		dd = -std::inner_product(p, p + n_train, Xd, double{});
		return -fx;
	}
	void memoised_grad
	(
		double t,
		double* x,
		double* d,
		double* g
	)
	{
		++gradEvaluations;
		double tmp{};
		double* z{};

		for (size_t i = 0; i < n_train; ++i)
			Xx[i] = tmp = Xx[i] - t * Xd[i];

		for (size_t i = 0; i < n; ++i)
				g[i] = 0.;
		for (size_t j = 0; j < n_train; ++j)
		{
			tmp = p[j];
			for (size_t i = 0; i < n; ++i)
				g[i] += X_train[j][i] * tmp;
		}
		memoizable = true;
	}

	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = .0;
	}
	bool stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
	void set_y(size_t currentNeuron)
	{
		neurNumb = currentNeuron;
		for (size_t j = 0; j < n_train; ++j)
		{
			if (labels_train[j] == currentNeuron) y[j] = 1;
			else y[j] = 0;
		}
	}

	~one_layer_log_regr()
	{
		delete[] p;
		delete[] q;
		delete[] Xx;
		delete[] Xd;
		delete[] y;

		for (int i = 0; i < n_train; ++i)
		{
			delete[] X_train[i];
		}
		delete[] X_train;

		for (int i = 0; i < n_test; ++i)
		{
			delete[] X_test[i];
		}
		delete[] X_test;
		delete[] labels_train;
		delete[] labels_test;
	}
};

// Iris Loadig 
one_layer_log_regr* IrisLoader(double const tolerance, size_t neuron, std::string s)
{
	one_layer_log_regr* a = new one_layer_log_regr();
	a->EPS = tolerance;
	size_t nTrains{ 120 };
	size_t nTests{ 30 };
	size_t nNeurons{ 3 };
	a->n = 5;
	a->p = new double[nTrains];
	a->q = new double[nTrains];
	a->Xx = new double[nTrains];
	a->Xd = new double[nTrains];
	a->y = new uint8_t[nTrains];
	a->name = "Iris";
	a->X_train = new double* [nTrains];
	for (int i = 0; i < nTrains; ++i)
	{
		a->X_train[i] = new double[5];
	}
	a->X_test = new double* [nTests];
	for (int i = 0; i < nTests; ++i)
	{
		a->X_test[i] = new double[5];
	}

	a->labels_train = new uint8_t[nTrains];
	a->labels_test = new uint8_t[nTests];

	a->n_train = nTrains;
	a->n_neurons = nNeurons;
	a->n_test = nTests;
	a->neurNumb = neuron;

	std::array<size_t, 150> inds;
	for (size_t i = 0; i < 150; ++i)
		inds[i] = i;

	// obtain a time-based seed:
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(inds.begin(), inds.end(), std::default_random_engine(seed));

	std::ifstream ifs{ s };
	double d{};
	char c{};
	std::string str{};

	//fill	
	for (size_t i = 0; i < 150; ++i)
	{
		if (inds[i] < a->n_train)
		{

			for (size_t j = 0; j < 4; ++j)
			{
				ifs >> d >> c;
				a->X_train[inds[i]][j] = d;
			}
			a->X_train[inds[i]][4] = 1.;
			ifs >> str;
			if (str == "setosa")
				a->labels_train[inds[i]] = 0;
			else
				if (str == "versicolor")
					a->labels_train[inds[i]] = 1;
				else
					a->labels_train[inds[i]] = 2;
		}
		else
		{
			for (size_t j = 0; j < 4; ++j)
			{
				ifs >> d >> c;
				a->X_test[inds[i] - a->n_train][j] = d;
			}
			a->X_test[inds[i] - a->n_train][4] = 1.;

			ifs >> str;
			if (str == "setosa")
				a->labels_test[inds[i] - a->n_train] = 0;
			else
				if (str == "versicolor")
					a->labels_test[inds[i] - a->n_train] = 1;
				else
					a->labels_test[inds[i] - a->n_train] = 2;
		}
	}
	//Only after data are gathered
	a->set_y(neuron);
	return a;
}

// mnist Loader
one_layer_log_regr* mnistLoader
(
	double const tolerance, size_t neuron,
	std::string pathToTrainData,
	std::string pathToTrainLabels,
	std::string pathToTestnData,
	std::string pathToTestLabels
)
{
	one_layer_log_regr* a = new one_layer_log_regr();

	size_t nTrains{ 60000 };
	size_t nTests{ 10000 };
	size_t nNeurons{ 10 };
	a->EPS = tolerance;
	a->n = 785;
	a->p = new double[nTrains];
	a->q = new double[nTrains];
	a->Xx = new double[nTrains];
	a->Xd = new double[nTrains];
	a->y = new uint8_t[nTrains];
	a->name = "mnist";
	a->X_train = new double* [nTrains];
	for (int i = 0; i < nTrains; ++i)
	{
		a->X_train[i] = new double[785];
	}
	a->X_test = new double* [nTests];
	for (int i = 0; i < nTests; ++i)
	{
		a->X_test[i] = new double[785];
	}

	a->labels_train = new uint8_t[nTrains];
	a->labels_test = new uint8_t[nTests];

	a->n_train = nTrains;
	a->n_neurons = nNeurons;
	a->n_test = nTests;
	a->neurNumb = neuron;

	//In blocks, to reuse notion "ifs"
	//Fill X_train
	{
		std::string image_file = pathToTrainData;
		std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
		char p[4];
		ifs.read(p, 4); ifs.read(p, 4); ifs.read(p, 4); ifs.read(p, 4);
		//as we already have parameter values, so just read with zero effect
		char* q = new char[784];
		for (int i = 0; i < 60000; ++i) {
			ifs.read(q, 784);
			for (int j = 0; j < 784; ++j) {
				a->X_train[i][j] = q[j] / 255.0;
			}
			a->X_train[i][784] = 1.;
		}
		delete[] q;
	}
	//Filling "labels_train"
	{
		std::string image_file = pathToTrainLabels;
		std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
		char p[4];
		ifs.read(p, 4); ifs.read(p, 4);
		for (int i = 0; i < 60000; ++i) {
			ifs.read(p, 1);
			int label = p[0];
			a->labels_train[i] = label;   //directly p[0]? later?
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
				a->X_test[i][j] = q[j] / 255.0;
			}
			a->X_test[i][784] = 1.;
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
			a->labels_test[i] = label;   //directly p[0]? later?
		}
	}
	//Only after data are gathered
	a->set_y(neuron);
	return a;
}

//---Symmetric matrix game with quadratic penalty functio
class SMG_QuadraticPenalties : public composite_problem
{
	double Dev{};
	int i{}, j{};
	vector<vector<double>> A;
	double* Ax{};
	double* Ad{};
	double dTd{};
public:
	double EPS{};
	long int n{};
	SMG_QuadraticPenalties() {}
	SMG_QuadraticPenalties(double const tolerance, long int size)
	{
		EPS = tolerance;
		n = size;
		Ax = new double[n];
		Ad = new double[n];
	}
	~SMG_QuadraticPenalties()
	{
		delete[] Ad;
		delete[] Ax;
	}
	long int virtual getSize()  const { return n; }

	std::string  aboutSize()  const { return "size: " + std::to_string(n); }
	std::string about() const { return "Problem: SMG, quardatic penalties, " + aboutSize(); }
	std::string virtual getName() const { return "Problem:SMG, trquardatic penalties"; }
	void setMomoisation(const bool b) { memoizable = b; };
	bool memoizable{ true };

	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.;
		Dev = 0.;
		double prod{};
		double sum{};
		double* p;
		for (i = 0; i < n; i++)
		{
			g[i] = 0;
			sum += x[i];
		}
		for (i = 0; i < n; i++)
		{
			p = A[i].data();
			prod = std::inner_product(p, p + n, x, double{});
			if (memoizable)
				Ax[i] = prod;

			if (prod > 0.)
			{
				if (Dev < prod) Dev = prod;
				fx += prod * prod;
				for (j = 0; j < n; j++)
					g[j] += prod * p[j];
			}
		}
		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		fx += (sum - 1) * (sum - 1);
		for (i = 0; i < n; i++)
			g[i] += (sum - 1);

		for (i = 0; i < n; i++)
		{
			if (x[i] < 0.)
			{
				if (-x[i] > Dev)
					Dev = -x[i];
				prod = x[i];
				fx += prod * prod;
				g[i] += prod;
			}
		}

		return fx / 2;
	}

	//calculates directional derivative dd at the point x0-td;
	double memoised_val_dirDer
	(
		double t,
		double* x,
		double* d,			//direction
		double& dd  //,			//directional derivative
	//	bool& isMemoised		//if false, components for the inner function should be prepared
	)
	{
		++dirDerEvaluations;
		double fx = 0.;
		dd = 0;
		double y{};
		double sum{};
		double* p;

		if (memoizable)
		{
			dTd = 0.;
			for (i = 0; i < n; i++)
			{
				p = A[i].data();
				Ad[i] = std::inner_product(p, p + n, d, double{});
				dTd += d[i];
			}
			memoizable = false;
		}
		for (i = 0; i < n; i++)
		{
			y = Ax[i] - t * Ad[i];
			if (y > 0.)
			{
				fx += y * y;
				dd -= y * Ad[i];
			}
		}
		for (i = 0; i < n; i++)
		{
			y = x[i] - t * d[i];
			sum += y;
			if (y < 0.)
			{
				fx += y * y;
				dd -= y * d[i];
			}
		}
		y = sum - 1;
		fx += y * y;
		dd -= y * dTd;
		//	for (i = 0; i < n; i++)
		//		dd -= y * d[i];

		return fx / 2;
	}
	void memoised_grad
	(
		double t,
		double* x,
		double* d,
		double* g
	)
	{
		++gradEvaluations;
		Dev = 0.;
		double y{};
		double sum{};
		double* p;

		for (i = 0; i < n; i++)
			g[i] = 0;

		for (i = 0; i < n; i++)
		{
			p = A[i].data();
			Ax[i] = y = Ax[i] - t * Ad[i];
			if (y > 0.)
			{
				if (Dev < y) Dev = y;
				for (j = 0; j < n; j++)
					g[j] += y * p[j];
			}
		}

		for (i = 0; i < n; i++)
		{
			y = x[i] - t * d[i];
			sum += y;
			if (y < 0.)
			{
				if (-y > Dev)
					Dev = -y;
				g[i] += y;
			}
		}

		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		for (i = 0; i < n; i++)
			g[i] += (sum - 1);

		memoizable = true;
	}
	void printMatrix()
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
				cout << A[i][j] << "  ";
			cout << endl;
		}
		cout << endl << endl;
	}

	void initialize(double* x)
	{
		double const TMP = sqrt(1. / n);
		for (i = 0; i < n; ++i)
		{
			x[i] = TMP;
		}
		{
			std::default_random_engine dre;
			std::uniform_real_distribution<double> urdi(-1, 1);
			//std::normal_distribution<double> urdi(-1,1);
			//std::cauchy_distribution<double> urdi(-1, 1);
			A.resize(n);
			for (int i = 0; i < n; i++)
			{
				A[i].resize(n);
				for (int j = 0; j < i; j++)
					A[i][j] = urdi(dre);
				A[i][i] = 0;
			}
			for (int i = 0; i < n; i++)
				for (int j = i + 1; j < n; j++)
					A[i][j] = (-1.0 * A[j][i]);
		}
	}

	bool  stoppingCondition(double* g) const
	{
		return (Dev < EPS);
	}
};

//---Symmetric matrix game with cubic penalty function
class SMG_CubicPenalties : public composite_problem
{
	double Dev{};
	int i{}, j{};
	vector<vector<double>> A;
	double* Ax{};
	double* Ad{};
	double dTd{};
public:
	double EPS{};
	long int n{};
	SMG_CubicPenalties() {}
	SMG_CubicPenalties(double const tolerance, long int size)
	{
		EPS = tolerance;
		n = size;
		Ax = new double[n];
		Ad = new double[n];
	}
	~SMG_CubicPenalties()
	{
		delete[] Ad;
		delete[] Ax;
	}
	long int virtual getSize()  const { return n; }

	std::string  aboutSize()  const { return "size: " + std::to_string(n); }
	std::string about() const { return "Problem: SMG, cubic penalties,  " + aboutSize(); }
	std::string virtual getName() const { return "Problem:SMG, cubic penalties"; }
	void setMomoisation(const bool b) { memoizable = b; };
	bool memoizable{ true };

	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.;
		Dev = 0.;
		double prod{};
		double sum{};
		double* p;
		double qu{}, sq{};
		int sign{};

		for (i = 0; i < n; i++)
		{
			g[i] = 0;
			sum += x[i];
		}
		for (i = 0; i < n; i++)
		{
			p = A[i].data();
			prod = std::inner_product(p, p + n, x, double{});
			if (memoizable)
				Ax[i] = prod;

			if (prod > 0.)
			{
				if (Dev < prod) Dev = prod;
				qu = prod * prod;
				fx += qu * prod;
				for (j = 0; j < n; j++)
					g[j] += qu * p[j];
			}
		}
		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		sign = (sum < 1) ? (-1) : 1;
		sq = (sum - 1) * (sum - 1);
		fx += sq * (sum - 1) * sign;
		prod = sq * sign;
		for (j = 0; j < n; j++)
			g[j] += prod;

		for (i = 0; i < n; i++)
		{
			if (x[i] < 0.)
			{
				if (-x[i] > Dev)
					Dev = -x[i];
				qu = x[i];
				sq = qu * qu;
				fx -= sq * qu;
				g[i] -= sq;
			}
		}

		return fx / 3;
	}

	//calculates directional derivative dd at the point x0-td;
	double memoised_val_dirDer
	(
		double t,
		double* x,
		double* d,			//direction
		double& dd  //,			//directional derivative
	//	bool& isMemoised		//if false, components for the inner function should be prepared
	)
	{
		++dirDerEvaluations;
		double fx = 0.;
		dd = 0;
		double y{};
		double sum{};
		double* p;
		int sign{};

		if (memoizable)
		{
			dTd = 0.;
			for (i = 0; i < n; i++)
			{
				p = A[i].data();
				Ad[i] = std::inner_product(p, p + n, d, double{});
				dTd += d[i];
			}
			memoizable = false;
		}

		for (i = 0; i < n; i++)
		{
			y = Ax[i] - t * Ad[i];
			if (y > 0.)
			{
				fx += y * y * y;
				dd -= y * y * Ad[i];
			}
		}

		for (i = 0; i < n; i++)
		{
			y = x[i] - t * d[i];
			sum += y;
			if (y < 0.)
			{
				fx -= y * y * y;
				dd += y * y * d[i];
			}
		}
		y = sum - 1;
		sign = (sum < 1) ? (-1) : 1;
		fx += y * y * y * sign;
		dd -= y * y * sign * dTd;

		//	for (i = 0; i < n; i++)
		//		dd -= y * y * sign * d[i];

		return fx / 3;
	}
	void memoised_grad
	(
		double t,
		double* x,
		double* d,
		double* g
	)
	{
		++gradEvaluations;
		Dev = 0.;
		double y{};
		double sum{};
		double* p;
		int sign{};

		for (i = 0; i < n; i++)
			g[i] = 0;

		for (i = 0; i < n; i++)
		{
			p = A[i].data();
			Ax[i] = y = Ax[i] - t * Ad[i];
			if (y > 0.)
			{
				if (Dev < y) Dev = y;
				for (j = 0; j < n; j++)
					g[j] += y * y * p[j];
			}
		}

		for (i = 0; i < n; i++)
		{
			y = x[i] - t * d[i];
			sum += y;
			if (y < 0.)
			{
				if (-y > Dev)
					Dev = -y;
				g[i] -= y * y;
			}
		}

		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		sign = (sum < 1) ? (-1) : 1;
		y = (sum - 1) * (sum - 1) * sign;
		for (j = 0; j < n; j++)
			g[j] += y;
		memoizable = true;
	}
	void printMatrix()
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
				cout << A[i][j] << "  ";
			cout << endl;
		}
		cout << endl << endl;
	}

	void initialize(double* x)
	{
		double const TMP = sqrt(1. / n);
		for (i = 0; i < n; ++i)
		{
			x[i] = TMP;
		}
		{
			std::default_random_engine dre;
			std::uniform_real_distribution<double> urdi(-1, 1);
			//std::normal_distribution<double> urdi(-1,1);
			//std::cauchy_distribution<double> urdi(-1, 1);
			A.resize(n);
			for (int i = 0; i < n; i++)
			{
				A[i].resize(n);
				for (int j = 0; j < i; j++)
					A[i][j] = urdi(dre);
				A[i][i] = 0;
			}
			for (int i = 0; i < n; i++)
				for (int j = i + 1; j < n; j++)
					A[i][j] = (-1.0 * A[j][i]);
		}
	}

	bool  stoppingCondition(double* g) const
	{
		return (Dev < EPS);
	}
};

//---Symmetric matrix game with cubic penalty function,  using Eigen 
class memCubPenSMG_Eigen : public composite_problem
{
	double Dev{};
	int i{}, j{};
	Matrix <double, Dynamic, Dynamic> A;
	Vector<double, Dynamic> x_vec;		//to store a copy of "x"
	Vector<double, Dynamic> g_vec;		//to store a copy of "g"
	Vector<double, Dynamic> d_vec;		//to store a copy of "d"
	Vector<double, Dynamic> tmp_vec;
	Vector<double, Dynamic> Ax;
	Vector<double, Dynamic> Ad;
	double d_sum{};
public:
	double EPS{};
	long int n{};
	memCubPenSMG_Eigen() {}
	memCubPenSMG_Eigen(double const tolerance, long int size)
	{
		EPS = tolerance;
		n = size;
		A.resize(n, n);
		x_vec.resize(n);
		g_vec.resize(n);
		d_vec.resize(n);
		tmp_vec.resize(n);
		Ax.resize(n);
		Ad.resize(n);
	}

	long int virtual getSize()  const { return n; }

	std::string  aboutSize()  const { return "size: " + std::to_string(n); }
	std::string about() const { return "Problem: SMG,  with cubic penalty function, using Eigen  " + aboutSize(); }
	std::string virtual getName() const { return "Problem:SMG, cubic penalties and Eigen"; }
	void setMomoisation(const bool b) { memoizable = b; };
	bool memoizable{ true };

	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.;
		Dev = 0.;
		double prod{};
		double sum{};
		double qu{}, sq{};
		int sign{};

		std::memmove(x_vec.data(), x, n * sizeof(double));
		sum = x_vec.sum();

		Ax = A * x_vec;		
		double tmp = Ax.maxCoeff();
		if (tmp > Dev)
			Dev = tmp;
		fx = Ax.array().cwiseMax(0).pow(3).sum();
		g_vec = (Ax.array().cwiseMax(0).pow(2));
		g_vec = g_vec.transpose() * A;

		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		sign = (sum < 1) ? (-1) : 1;
		sq = (sum - 1) * (sum - 1);
		fx += sq * (sum - 1) * sign;
		prod = sq * sign;
		for (j = 0; j < n; j++)
			g_vec(j) += prod;
	
		for (i = 0; i < n; i++)
		{
			if (x[i] < 0.)
			{
				if (-x[i] > Dev)
					Dev = -x[i];
				qu = x[i];
				sq = qu * qu;
				fx -= sq * qu;
				g_vec(i) -= sq;
			}
		}

		std::memmove(g, g_vec.data(), n * sizeof(double));
		return fx / 3;
	}

	//calculates directional derivative dd at the point x0 - td;
	double memoised_val_dirDer
	(
		double t,
		double* x,
		double* d,			//direction
		double& dd    //,			//directional derivative
	//	bool& isMemoised		//if false, components for the inner function should be prepared
	)
	{
		++dirDerEvaluations;
		double fx = 0.;
		dd = 0;
		double y{};
		double sum{};
		int sign{};

		std::memmove(x_vec.data(), x, n * sizeof(double));
		std::memmove(d_vec.data(), d, n * sizeof(double));

		if (memoizable)
		{
			d_sum = d_vec.sum();
			Ad = A * d_vec;
			memoizable = false;
		}
		
		tmp_vec = (Ax - t * Ad).array().cwiseMax(0);
		for (i = 0; i < n; i++)
		{
			y = tmp_vec(i);
			if (y > 0.)
			{
				fx += y * y * y;
				dd -= y * y * Ad[i];
			}
		}
		
		for (i = 0; i < n; i++)
		{
			y = x[i] - t * d[i];
			sum += y;
			if (y < 0.)
			{
				fx -= y * y * y;
				dd += y * y * d[i];
			}
		}

		y = sum - 1;
		sign = (sum < 1) ? (-1) : 1;
		fx += y * y * y * sign;
		dd -= y * y * sign * d_sum;

		return fx / 3;
	}
	void memoised_grad
	(
		double t,
		double* x,
		double* d,
		double* g
	)
	{
		++gradEvaluations;
		Dev = 0.;
		double y{};
		double sum{};
		int sign{};

		std::memmove(x_vec.data(), x, n * sizeof(double));
		std::memmove(d_vec.data(), d, n * sizeof(double));
		
		Ax = Ax - t * Ad;
		g_vec = (Ax.array().cwiseMax(0).pow(2));
		g_vec = g_vec.transpose() * A;

		for (i = 0; i < n; i++)
		{
			y = x[i] - t * d[i];
			sum += y;
			if (y < 0.)
			{
				if (-y > Dev)
					Dev = -y;
				g_vec(i) -= y * y;
			}
		}
	

		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		sign = (sum < 1) ? (-1) : 1;
		y = (sum - 1) * (sum - 1) * sign;
		for (j = 0; j < n; j++)
			g_vec(j) += y;

		std::memmove(g, g_vec.data(), n * sizeof(double));
		memoizable = true;
	}

	void initialize(double* x)
	{
		double const TMP = sqrt(1. / n);
		for (i = 0; i < n; ++i)
		{
			x[i] = TMP;
		}
		{
			std::default_random_engine dre;
			std::uniform_real_distribution<double> urdi(-1, 1);
			//std::normal_distribution<double> urdi(-1,1);
			//std::cauchy_distribution<double> urdi(-1, 1);
	
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < i; j++)
					A(i,j) = urdi(dre);
				A(i,i) = 0;
			}
			for (int i = 0; i < n; i++)
				for (int j = i + 1; j < n; j++)
					A(i,j) = - A(j, i);
		}
	}

	bool  stoppingCondition(double* g) const
	{
		return (Dev < EPS);
	}
};
