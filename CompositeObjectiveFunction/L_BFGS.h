#pragma once
#include"cm_namespace.h"

class cm::L_BFGS
{
	double** s, ** gamma;		//BFGS corrections to be saved 
	double* alpha_2loop, * ro_2loop;
	void L_BFGS_two_loop_recursion(int m);
public:

	L_BFGS(const auto p, int m = 11)
	{
		n = p->getSize();
		mVALUE = m;
		dir = new double[n];
		x0 = new double[n];
		x1 = new double[n];
		g1 = new double[n];
		g0 = new double[n];
		s = new double* [m];
		gamma = new double* [m];
		for (int i = 0; i < m; ++i)
		{
			gamma[i] = new double[n];
			s[i] = new double[n];
		}
		alpha_2loop = new double[m];
		ro_2loop = new double[m];
	}

	~L_BFGS()
	{
		delete[]  dir;
		delete[]  x0;
		delete[]  x1;
		delete[]  g0;
		delete[]  g1;
		for (int i = 0; i < mVALUE; ++i)
		{
			delete[]  gamma[i];
			delete[]  s[i];
		}
		delete[]  s;
		delete[]  gamma;

		delete[] alpha_2loop;
		delete[] ro_2loop;
	}

	void operator()(auto&, const auto& ppr) noexcept;

	void printStats(const auto& ppr)
	{
		std::cout << "   Algorithm L-BFGS;  Objective function: " << f0 << ",  Value-Grad evaluations:  "
			<< ppr->ValGradEvaluations << ",  Line searches:  " << LineSearchCounter << endl;
	}
};

void cm::L_BFGS::L_BFGS_two_loop_recursion(int m)
{
	vecCopy(g0, dir, n);
	//First loop
	for (int i = m - 1; i > -1; --i)
	{
		alpha_2loop[i] = ro_2loop[i] * vecProd(dir, s[i], n);
		for (int j = 0; j < n; ++j)
			dir[j] -= alpha_2loop[i] * gamma[i][j];
	}

	double coeff = 1 / (vecProd(gamma[m - 1], gamma[m - 1], n) * ro_2loop[m - 1]);
	for (int j = 0; j < n; ++j)
		dir[j] *= coeff;

	//Second loop
	double beta;
	for (int i = 0; i < m; ++i)
	{
		beta = ro_2loop[i] * vecProd(dir, gamma[i], n);
		for (int j = 0; j < n; ++j)
			dir[j] += (alpha_2loop[i] - beta) * s[i][j];
	}
}

void cm::L_BFGS::operator()(auto& lnSrch, const auto& ppr) noexcept
{
	LineSearchCounter = 0;
	f0 = ppr->valGrad(x0, g0);
	vecCopy(g0, dir, n);
	lnSrch(ppr);    f0 = f1;
	++LineSearchCounter;

	for (int i = 0; i < n; ++i)
	{
		gamma[0][i] = g1[i] - g0[i];
		s[0][i] = x1[i] - x0[i];
	}
	ro_2loop[0] = 1 / vecProd(gamma[0], s[0], n);
	swap(x0, x1);
	swap(g0, g1);

	int j = 1;
	while (j < mVALUE)
	{
		L_BFGS_two_loop_recursion(j);
		lnSrch(ppr);      f0 = f1;
		++LineSearchCounter;
		for (int i = 0; i < n; ++i)
		{
			gamma[j][i] = g1[i] - g0[i];
			s[j][i] = x1[i] - x0[i];
		}
		ro_2loop[j] = 1 / vecProd(gamma[j], s[j], n);
		swap(x0, x1);		  
		swap(g0, g1);				
		++j;
	}

	while (!ppr->stoppingCondition(g0))
	{
		L_BFGS_two_loop_recursion(mVALUE);
		lnSrch(ppr);      f0 = f1;
		++LineSearchCounter;

		tmp = gamma[0];
		for (int i = 0; i < mVALUE - 1; ++i)
			gamma[i] = gamma[i + 1];
		gamma[mVALUE - 1] = tmp;
		tmp = s[0];
		for (int i = 0; i < mVALUE - 1; ++i)
			s[i] = s[i + 1];
		s[mVALUE - 1] = tmp;

		for (int i = 0; i < mVALUE - 1; ++i)
			ro_2loop[i] = ro_2loop[i + 1];

		for (int i = 0; i < n; ++i)
		{
			gamma[mVALUE - 1][i] = g1[i] - g0[i];
			s[mVALUE - 1][i] = x1[i] - x0[i];
		}
		ro_2loop[mVALUE - 1] = 1 / vecProd(gamma[mVALUE - 1], s[mVALUE - 1], n);
		swap(x0, x1);				  
		swap(g0, g1);				
	}
}