#pragma once			
#include<iostream>
#include "auxiliaryFunctions.h" 
#include"Problems.h"
using namespace std;

namespace cm		//collection of minimizers
{
	int n;
	int mVALUE;				//a default value for parameter that determines the number of BFGS corrections saved, usually 11
	double f0(0.0);			//objective function's value at x0 
	double f1(0.0);
	double alpha{ 0.001 };
	double* x0, * dir, * tmp;
	double* g1, * g0;
	double* x1;
	long int LineSearchCounter{};

	//line searches
	class lineSearch;
	class memLineSearch;

	//Solvers
	class L_BFGS;
	class MHB;
	class steepest;
};

