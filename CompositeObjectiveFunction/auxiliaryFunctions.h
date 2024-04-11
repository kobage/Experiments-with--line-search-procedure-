#pragma once
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

using namespace std;

inline double infNorm
(
	double* x, /* vector */
	int     k /* length of vector */
)
{
	int i, n5;
	double t = 0.;
	n5 = k % 5;

	for (i = 0; i < n5; i++) if (t < fabs(x[i])) t = fabs(x[i]);
	for (; i < k; i += 5)
	{
		if (t < fabs(x[i])) t = fabs(x[i]);
		if (t < fabs(x[i + 1])) t = fabs(x[i + 1]);
		if (t < fabs(x[i + 2])) t = fabs(x[i + 2]);
		if (t < fabs(x[i + 3])) t = fabs(x[i + 3]);
		if (t < fabs(x[i + 4])) t = fabs(x[i + 4]);
	}
	return (t);
}

inline void vec2norm(double* ans, double* gr, const int k)
{
	int i, n5;
	*ans = 0.;
	n5 = k % 5;
	for (i = 0; i < n5; i++) *ans += gr[i] * gr[i];
	for (; i < k; i += 5)
	{
		*ans += gr[i] * gr[i] + gr[i + 1] * gr[i + 1] + gr[i + 2] * gr[i + 2]
			+ gr[i + 3] * gr[i + 3] + gr[i + 4] * gr[i + 4];
	}
	*ans = sqrt(*ans);
}

inline double vecProd(double* x, double* y, const int k)
{
	int i, n5;
	double sum(0.0);
	if (k <= 0) return sum;
	n5 = k % 5;
	for (i = 0; i < n5; i++) sum += x[i] * y[i];
	for (; i < k; i += 5)
	{
		sum += x[i] * y[i] + x[i + 1] * y[i + 1] + x[i + 2] * y[i + 2]
			+ x[i + 3] * y[i + 3] + x[i + 4] * y[i + 4];
	}
	return sum;
}

inline void vecCopy(double* x, double* y, const int n)
{
	int i, n5;
	if (n <= 0) return;
	n5 = n % 5;
	for (i = 0; i < n5; i++)  y[i] = x[i];
	while (i < n)
	{
		y[i] = x[i];
		++i;
		y[i] = x[i];
		++i;
		y[i] = x[i];
		++i;
		y[i] = x[i];
		++i;
		y[i] = x[i];
		++i;
	}
}

void fillRandomMatrix(double** M, int k)
{
	std::default_random_engine dre;
	std::uniform_real_distribution<double> urdi(-1, 1);
	//std::normal_distribution<double> urdi(-1,1);
	//std::cauchy_distribution<double> urdi(-1, 1);

	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < i; j++)
			M[i][j] = urdi(dre);
		M[i][i] = 0;
	}
	for (int i = 0; i < k; i++)
		for (int j = i + 1; j < k; j++)
			M[i][j] = (-1.0 * M[j][i]);
}


int digits(long int k)
{
	if (k <= 0) return -1;
	int c = 0;
	while (k /= 10)
		++c;
	return c;
}

void printVector(double* vec, std::string s, long int n)  noexcept
{
	long int quarter = n / 4;
	int sp = digits(n - 1) + 1;

	for (size_t i = 0; i < quarter; ++i)
	{
		std::cout << std::left << s << std::left << '[' << std::left << std::setw(sp) << i << std::left << "]="
			<< std::left << std::setw(21) << vec[i];
		std::cout << std::left << s << std::left << '[' << std::left << std::setw(sp) << i + quarter << std::left << "]="
			<< std::left << std::setw(21) << vec[i + quarter];
		std::cout << std::left << s << std::left << '[' << std::left << std::setw(sp) << i + 2 * quarter << std::left << "]="
			<< std::left << std::setw(21) << vec[i + 2 * quarter];
		std::cout << std::left << s << std::left << '[' << std::left << std::setw(sp) << i + 3 * quarter << std::left << "]="
			<< std::left << std::setw(21) << vec[i + 3 * quarter];
		std::cout << std::endl;
	}
	for (int i = 4 * quarter; i < n; i++)
		std::cout << std::left << s << std::left << '[' << std::left << std::setw(sp) << i << std::left << "]="
		<< std::left << std::setw(21) << vec[i];
	std::cout << std::endl;
}

void printMatrix(double** m, size_t rows, size_t columns)
{
	for (int i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
			cout << m[i][j] << "  ";
		cout << endl;
	}
	cout << endl;
}


#include <filesystem>
using namespace std::filesystem;

path findPathTo(string folder)
{
	path curPath = current_path();
	path tmp{};
	vector<path> subPaths;
	for (auto e : curPath)
	{
		tmp /= e;
		subPaths.push_back(tmp);
	}

	reverse(subPaths.begin(), subPaths.end());

	path found{};
	for (auto e : subPaths)
	{
		for (auto& p : filesystem::directory_iterator(e))
		{
			if (p.path().string().substr(p.path().string().size() - 4) == "Data")
			{
				found = p.path();
				break;
			}
		}
		if (found.string().size() > 3)
			break;
	}
	return found;
}



