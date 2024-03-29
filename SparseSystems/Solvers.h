#pragma once
#include <iostream>
#include <Eigen/SparseCore>
#include <vector>
#include<fstream>
#include<string>
#include<sstream>
#include <filesystem>
#include <regex>
using namespace Eigen;
#include <Eigen/IterativeLinearSolvers>
using namespace std;
using namespace std::filesystem;

//3 Auxiliary functions
string findPath()
{
	path pathToMatrices = current_path() /= "Matrices";
	for (auto& p : std::filesystem::directory_iterator(pathToMatrices))
		std::cout << p.path().stem() << '\n';

	std::cout << "Select matrix by typing first few symbols " << std::endl;
	string s;
	std::cin >> s;
	s = '^' + s;
	std::smatch m;
	std::regex e(s);

	std::filesystem::directory_iterator iter(pathToMatrices), end;
	while (iter != end)
	{
		s = iter->path().stem().string();
		if (std::regex_search(s, m, e))
		{
			s = iter->path().string();
			break;
		}
		++iter;
	}
	if (iter == end)
	{
		std::cout << "Incorrect input!" << std::endl;
	}
	return s;
}

void fillMatrix(string s, SparseMatrix<double, RowMajor>& m)
{
	/* open an existing file for reading */
	FILE* infile = fopen(s.c_str(), "r");

	/* declare a file pointer */
	char* buffer;
	long numbytes;

	/* if the file does not exist */
	if (infile == NULL)
		cout << "the file does not exist!" << endl;

	/* Get the number of bytes */
	fseek(infile, 0L, SEEK_END);
	numbytes = ftell(infile);

	/* reset the file position indicator to
	the beginning of the file */
	fseek(infile, 0L, SEEK_SET);

	/* grab sufficient memory for the
	buffer to hold the text */
	buffer = (char*)malloc(numbytes * sizeof(char));

	/* memory error */
	if (buffer == NULL)
		cout << "memory error!" << endl;

	/* copy all the text into the buffer */
	fread(buffer, sizeof(char), numbytes, infile);
	fclose(infile);

	// Ignore comment section
	size_t pos = 0;
	char* data = buffer;
	while (data[pos] == '%')
	{
		++pos;
		while (data[pos] != '\n')
			++pos;
		data += (pos + 1);
		pos = 0;
	}

	int i{};
	int j{};
	double v{};

	istringstream iss{ string(data) };
	iss >> i >> j >> v;
	m.resize(i, j);
	typedef Triplet<double, int> T;
	vector<T> entries;

	T tmp;
	while (iss >> i >> j >> v)
	{
		entries.push_back(T(i - 1, j - 1, v));
		if (i != j)
			entries.push_back(T(j - 1, i - 1, v));
	}

	m.setFromTriplets(entries.begin(), entries.end());
	free(buffer);
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


SparseMatrix<double, RowMajor> m;
Vector<double, Dynamic> b;


void conjugate_gradient_Eigen()  noexcept
{
	//prepare data
	string filename = findPath();
	fillMatrix(filename, m);
	b.resize(m.rows());
	b.setOnes();

	//solve
	std::chrono::high_resolution_clock::time_point  st;
	std::chrono::high_resolution_clock::duration  diff;

	st = chrono::high_resolution_clock::now();
	ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
	cg.compute(m);
	Vector<double, Dynamic> x = cg.solve(b);
	diff = chrono::high_resolution_clock::now() - st;

	//print results to file
	freopen("results.txt", "w", stdout);
	cout << "Problem - " << filename << endl;
	std::cout << "#iterations:     " << cg.iterations() << std::endl;
	std::cout << "estimated error: " << cg.error() << std::endl;
	std::cout << "tolerance: " << cg.tolerance() << std::endl;
	cout << "CPU time: " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << endl;
	printVector(x.data(), "x", b.size());
}

void SteepestDescent_2d(double tolerance) noexcept
{
	//prepare data
	string filename = findPath();
	fillMatrix(filename, m);
	b.resize(m.rows());
	b.setOnes();

	//solve
	std::chrono::high_resolution_clock::time_point  st;
	std::chrono::high_resolution_clock::duration  diff;
	st = chrono::high_resolution_clock::now();
	Vector<double, Dynamic> x;
	Vector<double, Dynamic> r;
	Vector<double, Dynamic> d;
	Vector<double, Dynamic> Ar;
	Vector<double, Dynamic> Ad;
	x.resize(m.rows());
	r.resize(m.rows());
	d.resize(m.rows());
	Ar.resize(m.rows());
	Ad.resize(m.rows());

	//elements of 2*2 maqtrix
	//to be used in 2d search using 2d steepest descent
	Matrix2d A_2d;
	Vector2d x_2;
	Vector2d r_2;
	Vector2d q_2;

	double new_b1{};
	double new_b2{};

	//solving
	r = d = b;
	x.setZero();

	Ad = m * d;
	double alpha = d.dot(r) / d.dot(Ad);
	x += alpha * d;
	r -= alpha * Ad;

	while ( r.dot(r) >= tolerance * tolerance)
	{
		Ar = m * r;
		A_2d(0, 0) = Ar.dot(r);  
		A_2d(1, 0) = A_2d(0, 1) = Ar.dot(d);
		A_2d(1, 1) = Ad.dot(d);

		new_b1 = -x.dot(Ar) + b.dot(r);
		new_b2 = -x.dot(Ad) + b.dot(d);

		{
			r_2[0] = new_b1;
			r_2[1] = new_b2;
			x_2.setZero();

			for (size_t ii{ 0 }; ii < 100; ++ii)
			{
				q_2 = A_2d * r_2;
				double alpha_2 = r_2.dot(r_2) / r_2.dot(q_2);
				x_2 += alpha_2 * r_2;
				r_2 -= alpha_2 * q_2;
			}
		}

		d = x_2[0] * r + x_2[1] * d;
		Ad = x_2(0) * Ar + x_2[1] * Ad;
		x += d;
		r -= Ad;
	}

	diff = chrono::high_resolution_clock::now() - st;

	//print results to file
	freopen("results.txt", "w", stdout);
	cout << "Problem - " << filename << endl;
	cout << "CPU time: " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << endl;
	printVector(x.data(), "x", b.size());
}


void cg_usual(double tolerance) noexcept
{
	//prepare data
	string filename = findPath();
	fillMatrix(filename, m);
	b.resize(m.rows());
	b.setOnes();

	//solve
	std::chrono::high_resolution_clock::time_point  st;
	std::chrono::high_resolution_clock::duration  diff;
	st = chrono::high_resolution_clock::now();
	Vector<double, Dynamic> x;
	Vector<double, Dynamic> r;
	Vector<double, Dynamic> d;
	Vector<double, Dynamic> q;
	x.resize(m.rows());
	r.resize(m.rows());
	d.resize(m.rows());
	q.resize(m.rows());

	//solving
	r = d = b;
	x.setZero();

	double deltaNew{ r.dot(r) };

	while (deltaNew >= tolerance * tolerance)
	{
		q = m *d;
		double alpha = deltaNew / d.dot(q);
		x += alpha * d;
		r -= alpha * q;
		 
		double deltaOld{ deltaNew };
		deltaNew = r.dot(r);
		double beta { deltaNew / deltaOld };
		d = r + beta * d;
	}

	diff = chrono::high_resolution_clock::now() - st;

	//print results to file
	freopen("results.txt", "w", stdout);
	cout << "Problem - " << filename << endl;
	cout << "CPU time: " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << endl;
	printVector(x.data(), "x", b.size());
}