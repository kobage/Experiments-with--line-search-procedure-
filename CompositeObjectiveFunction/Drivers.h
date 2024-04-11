#pragma once
#include"Problems.h"
#include"compositeProblems.h"
#include<vector>
#include<chrono>
#include<algorithm>
#include"cm_namespace.h"
#include"L_BFGS.h"
#include"MHB.h"
using namespace std;

vector<unique_ptr<problem>> problems_container;

void makeDoubleTestsVector(void)
{
	problems_container.emplace_back(make_unique<ARWHEAD>(1e-6));
	problems_container.emplace_back(make_unique<BDQRTIC>(1e-6));
	problems_container.emplace_back(make_unique<BROYDN7D>(1e-6));
	problems_container.emplace_back(make_unique<BRYBND>(1e-6));
	problems_container.emplace_back(make_unique<CHAINWOO>(1e-6));
	//	problems_container.emplace_back(make_unique<COSINE>(1e-6));
	problems_container.emplace_back(make_unique<CRAGGLVY>(1e-6));
	problems_container.emplace_back(make_unique<CURLY10>(1e-6));
	problems_container.emplace_back(make_unique<CURLY20>(1e-6));
	problems_container.emplace_back(make_unique<CURLY30>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANA>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANB>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANC>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAAND>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANE>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANF>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANG>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANH>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANI>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANJ>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANK>(1e-6));
	problems_container.emplace_back(make_unique<DIXMAANL>(1e-6));
	problems_container.emplace_back(make_unique<DIXON3DQ>(1e-6));
	problems_container.emplace_back(make_unique<DQDRTIC>(1e-6));
	problems_container.emplace_back(make_unique<DQRTIC>(1e-6));
	problems_container.emplace_back(make_unique<EDENSCH>(1e-6));
	//	problems_container.emplace_back(make_unique<EG2>(1e-6));
	problems_container.emplace_back(make_unique<ENGVAL1>(1e-6));
	problems_container.emplace_back(make_unique<EXTROSNB>(1e-6));
	problems_container.emplace_back(make_unique<FLETCHR>(1e-6));
	problems_container.emplace_back(make_unique<FREUROTH>(1e-6));
	//	problems_container.emplace_back(make_unique<GENHUMPS>(1e-6));
	problems_container.emplace_back(make_unique<GENROSE>(1e-6));
	problems_container.emplace_back(make_unique<LIARWDH>(1e-6));
	problems_container.emplace_back(make_unique<MOREBV>(1e-6));
	//	problems_container.emplace_back(make_unique<NONCVXU2>(1e-6));
	problems_container.emplace_back(make_unique<NONDIA>(1e-6));
	problems_container.emplace_back(make_unique<NONDQUAR>(1e-6));
	problems_container.emplace_back(make_unique<PENALTY1>(1e-6));
	//	problems_container.emplace_back(make_unique<PENALTY2>(1e-6));
	problems_container.emplace_back(make_unique<POWER>(1e-6));
	problems_container.emplace_back(make_unique<SROSENBR>(1e-6));
	problems_container.emplace_back(make_unique<TRIDIA>(1e-6));
	problems_container.emplace_back(make_unique<Woods>(1e-6));
	problems_container.emplace_back(make_unique<quadrPenalSMG>(1e-6));
	problems_container.emplace_back(make_unique<CubicPenalSMG>(1e-6));
	problems_container.emplace_back(make_unique<CubPenSMG_Eigen>(1e-6));
}

#include"lineSearch.h"
using namespace cm;

void runUCONTests()
{
	int repNumber(1);
	makeDoubleTestsVector();

	vector<unique_ptr<problem>>& v = problems_container;
	std::cout << "Test Problems: " << std::endl;
	for (size_t i = 0; i < v.size(); ++i)
		std::cout << i + 1 << ":   Problem:  " << v[i]->getName() << ",  size: " << v[i]->getSize() << endl;

	std::cout << "Enter test's index between  "
		<< 1 << "  and  " << v.size() << std::endl;
	int i;
	cin >> i;
	if (1 > i || i > v.size())
	{
		std::cout << "Wrong input!" << std::endl;
		return;
	}
	--i;
	vector<_int64> repetitions(repNumber);
	freopen("results.txt", "w", stdout);


	std::chrono::high_resolution_clock::time_point  st;  //
	std::chrono::high_resolution_clock::duration  diff;  //
	for (int j = 0; j < repNumber; j++)
	{
		st = chrono::high_resolution_clock::now();

		//	cm::L_BFGS solver{ v[i].get() , 50 }; 
		cm::L_BFGS solver{ v[i].get() };			//default, with: the number of BFGS corrections saved = 11
		//	cm::Daniel solver{ v[i].get() };
		//	cm::MHB solver{ v[i].get() };

		cm::lineSearch lnSrch;

		v[i]->initialize(x0);

		solver(lnSrch, v[i]);

		diff = chrono::high_resolution_clock::now() - st;

		repetitions[j] = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();

		if (j == repNumber - 1)
		{
			std::sort(repetitions.begin(), repetitions.end());
			std::cout << i + 1 << ". Problem:  " << v[i]->getName() << ",  size: " << v[i]->getSize() << std::endl;
			solver.printStats(v[i]);
			std::cout << "   Average time:  " << repetitions[repNumber / 2] << " microseconds"
				<< ", Gradient inf norm: " << infNorm(g0, v[i]->getSize()) << std::endl;
			std::cout << std::endl;
			printVector(x0, "x", n);
		}
	}
}

void runAllUCONTests()
{
	int repNumber(1);
	makeDoubleTestsVector();
	vector<unique_ptr<problem>>& v = problems_container;
	 
	vector<_int64> repetitions(repNumber);
	freopen("results.txt", "w", stdout);

	std::chrono::high_resolution_clock::time_point  st;  //
	std::chrono::high_resolution_clock::duration  diff;  //

	for (int i{ 0 }; i < v.size(); ++i)
	{
		for (int j = 0; j < repNumber; j++)
		{
			st = chrono::high_resolution_clock::now();
			cm::lineSearch lnSrch;

			//	cm::L_BFGS solver{ v[i].get() , 20 }; 
			cm::L_BFGS solver{ v[i].get() };			//default, with: the number of BFGS corrections saved = 11

			v[i]->initialize(x0);

			solver(lnSrch, v[i]);

			diff = chrono::high_resolution_clock::now() - st;

			repetitions[j] = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();

			if (j == repNumber - 1)
			{
				std::sort(repetitions.begin(), repetitions.end());
				std::cout << i + 1 << ". Problem:  " << v[i]->getName() << ",  size: " << v[i]->getSize() << std::endl;
				solver.printStats(v[i]);
				std::cout << "   Average time:  " << repetitions[repNumber / 2] << " microseconds"
					<< ", Gradient inf norm: " << infNorm(g0, v[i]->getSize()) << std::endl;
				std::cout << std::endl;
				//	printVector(x0, "x", n);
			}
		}
	}
}

vector<composite_problem*> comp_probl_container;

void makeDoubleCompTestsVector(void)
{
	const int SIZE{ 200 };
	comp_probl_container.emplace_back(new CURLY(1e-6, 30, 500));
/*	*/
	path dataPath = findPathTo("Data");

	one_layer_log_regr* a = IrisLoader(1e-0, 0, dataPath.string() + "/Iris/Iris.txt");
	comp_probl_container.emplace_back(a);

	a = mnistLoader
	(
		1e-0, 0,		//neuron 0 - by default
		dataPath.string() + "/mnist/train-images.idx3-ubyte",
		dataPath.string() + "/mnist/train-labels.idx1-ubyte",
		dataPath.string() + "/mnist/t10k-images.idx3-ubyte",
		dataPath.string() + "/mnist/t10k-labels.idx1-ubyte"
	);
	comp_probl_container.emplace_back(a);

	comp_probl_container.emplace_back(new SMG_QuadraticPenalties(1e-6, SIZE));
	comp_probl_container.emplace_back(new SMG_CubicPenalties(1e-6, SIZE));
	comp_probl_container.emplace_back(new memCubPenSMG_Eigen(1e-6, SIZE));
}

void run_comp_probl_tests()
{
	int repNumber(1);
	makeDoubleCompTestsVector();
	vector<composite_problem*>& v = comp_probl_container;

	std::chrono::high_resolution_clock::time_point  st;  // 
	std::chrono::high_resolution_clock::duration  diff;  //

	while (true)
	{
		std::cout << std::endl << "Test Problems: " << std::endl;
		for (size_t i = 0; i < v.size(); ++i)
		{
			std::cout << i + 1 << ".  " << v[i]->about() << std::endl;
			v[i]->setMomoisation(false);
		}

		std::cout << std::endl << "To exit, do wrong input, or" << std::endl
			<< "enter test's index between  "
			<< 1 << "  and  " << v.size() << std::endl;
		int i;
		cin >> i;

		if (1 > i || i > v.size())
		{
			std::cout << "Wrong input!" << std::endl;
			return;
		}
		--i;
	
		//Set neuron in mnist
		if (v[i]->getName() == "mnist")
		{
			std::cout << "Enter neuron's index between "
				<< 0 << "  and  " << 9 << ":   - - - ";

			int neuronIndex{};
			cin >> neuronIndex;
			if (0 > i || i > 9)
			{
				std::cout << "Wrong input!" << std::endl;
				return;
			}
			static_cast<one_layer_log_regr*>(v[i])->set_y(neuronIndex);
		}

		//Set neuron in Iris
		if (v[i]->getName() == "Iris")
		{
			int neuronIndex{};
			std::cout << "Enter neuron's index between "
				<< 0 << "  and  " << 2 << ":   - - - ";
			cin >> neuronIndex;

			if (0 > i || i > 2)
			{
				std::cout << "Wrong input!" << std::endl;
				return;
			}
			static_cast<one_layer_log_regr*>(v[i])->set_y(neuronIndex);
		}

		vector<_int64> repetitions(repNumber);

		//	freopen("results.txt", "w", stdout);
		for (int j = 0; j < repNumber; j++)
		{
			st = chrono::high_resolution_clock::now();
			//	cm::L_BFGS solver{ v[i], 50 }; 
			cm::L_BFGS solver{ v[i]};			//default, with: the number of BFGS corrections saved = 11
			cm::lineSearch lnSrch;
			v[i]->initialize(x0);

			solver(lnSrch, v[i]);

			diff = chrono::high_resolution_clock::now() - st;

			repetitions[j] = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();

			if (j == repNumber - 1)
			{
				std::sort(repetitions.begin(), repetitions.end());
				std::cout << i + 1 << ".  " << v[i]->about() << std::endl;
				solver.printStats(v[i]);
				std::cout << "   Composite problem without memoization, average time:  " << repetitions[repNumber / 2] << " microseconds"
					<< ", Gradient inf norm: " << infNorm(g0, v[i]->getSize()) << std::endl;
				std::cout << std::endl;
				printVector(x0, "x", n);
			}
		}
	}
	for (size_t i = 0; i < v.size(); ++i)
		delete v[i];
}

#include"memLineSearch.h"

void run_mem_comp_probl_tests()
{
	int repNumber(1);
	makeDoubleCompTestsVector();
	vector<composite_problem*>& v = comp_probl_container;

	std::chrono::high_resolution_clock::time_point  st;
	std::chrono::high_resolution_clock::duration  diff;

	while (true)
	{
		std::cout << std::endl << "Test Problems: " << std::endl;
		for (size_t i = 0; i < v.size(); ++i)
			std::cout << i + 1 << ".  " << v[i]->about() << std::endl;

		std::cout << std::endl << "To exit, do wrong input, or" << std::endl
			<< "enter test's index between  "
			<< 1 << "  and  " << v.size() << std::endl;
		int i;
		cin >> i;

		if (1 > i || i > v.size())
		{
			std::cout << "Wrong input!" << std::endl;
			return;
		}
		--i;

		//Set neuron in mnist
		if (v[i]->getName() == "mnist")
		{
			cout << "Enter neuron's index between "
				<< 0 << "  and  " << 9 << ":   - - - ";

			int neuronIndex{};
			cin >> neuronIndex;
			if (0 > neuronIndex || neuronIndex > 9)
			{
				std::cout << "Wrong input!" << std::endl;
				return;
			}
			static_cast<one_layer_log_regr*>(v[i])->set_y(neuronIndex);
		}

		//Set neuron in Iris
		if (v[i]->getName() == "Iris")
		{
			cout << "Enter neuron's index between "
				<< 0 << "  and  " << 2 << ":   - - - ";

			int neuronIndex{};
			cin >> neuronIndex;
			if (0 > neuronIndex || neuronIndex > 2)
			{
				std::cout << "Wrong input!" << std::endl;
				return;
			}
			static_cast<one_layer_log_regr*>(v[i])->set_y(neuronIndex);
		}

		vector<_int64> repetitions(repNumber);

		//	freopen("results.txt", "w", stdout);

		for (int j = 0; j < repNumber; j++)
		{
			st = chrono::high_resolution_clock::now();

			//	cm::L_BFGS solver{ v[i].get() , 50 }; 			
			cm::L_BFGS solver{ v[i] };		//default, with: the number of BFGS corrections saved = 11
			cm::memLineSearch lnSrch;
			v[i]->initialize(x0);

			solver(lnSrch, v[i]);

			diff = chrono::high_resolution_clock::now() - st;

			repetitions[j] = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();

			if (j == repNumber - 1)
			{
				std::sort(repetitions.begin(), repetitions.end());
				std::cout << i + 1 << ".  " << v[i]->about() << std::endl;
				solver.printStats(v[i]);
				if (v[i]->dirDerEvaluations != 0)
					cout << ",  dirDerEvaluations  " << v[i]->dirDerEvaluations;
				if (v[i]->gradEvaluations != 0)
					cout << ",  gradEvaluations  " << v[i]->gradEvaluations << endl;
				std::cout << "   Composite problem with memoization, average time:  " << repetitions[repNumber / 2] << " microseconds"
					<< ", Gradient inf norm: " << infNorm(g0, v[i]->getSize()) << std::endl;
				std::cout << std::endl;
				printVector(x0, "x", n);
			}
		}
	}

	for (size_t i = 0; i < v.size(); ++i)
		delete v[i];
}