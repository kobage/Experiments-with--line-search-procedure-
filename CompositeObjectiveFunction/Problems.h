#pragma once
#include<string>
#include<iomanip>
#include<iostream>
#include"auxiliaryFunctions.h"

//------------- Two abstract base classes  ----------
class problem
{
public:
	//fields
	long int ValGradEvaluations{};
	long int ValueEvaluations{};
	//methods
	double virtual valGrad(double*, double*) = 0;
	//	double virtual value(double*) = 0;
	void virtual  initialize(double*) = 0;
	long int virtual getSize() const = 0;
	std::string virtual getName() const = 0;
	bool virtual stoppingCondition(double*) const = 0;
};

//-----------------------  1  -----------------------
class ARWHEAD : public problem
{
public:
	double EPS{};
	long int n{};
	ARWHEAD() {}
	ARWHEAD(double const tolerance) { EPS = tolerance; n = 10000; }
	long int virtual getSize() const { return n; }
	std::string getName() const { return "ARWHEAD"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		double group1(0.0);
		g[n - 1] = 0;
		for (int i = 0; i < n - 1; i++)
		{
			group1 = x[i] * x[i] + x[n - 1] * x[n - 1];
			fx += group1 * group1 + 3.0 - 4.0 * x[i];
			g[i] = -4.0 + 4.0 * group1 * x[i];
			g[n - 1] += 4.0 * group1 * x[n - 1];
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double group1(0.0);
		for (int i = 0; i < n - 1; i++)
		{
			group1 = x[i] * x[i] + x[n - 1] * x[n - 1];
			fx += group1 * group1 + 3.0 - 4.0 * x[i];
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 1.0;
	}

	bool virtual stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};
//-----------------------  2  -----------------------
class BDQRTIC : public problem
{
public:
	double EPS{};
	long int n{};
	BDQRTIC() {}
	BDQRTIC(double const tolerance) { EPS = tolerance; n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "BDQRTIC"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx(0.0);
		double group1(0.0), group2(0.0);

		for (int i = 0; i < n; i++)
			g[i] = 0;
		double current_squared[4] = { 0.,pow(x[0], 2),pow(x[1], 2),pow(x[2], 2) };
		double last(pow(x[n - 1], 2));

		for (int i = 0; i < n - 4; i++)
		{
			current_squared[0] = current_squared[1];
			current_squared[1] = current_squared[2];
			current_squared[2] = current_squared[3];
			current_squared[3] = x[i + 3] * x[i + 3];
			group1 = 3.0 - 4.0 * x[i];
			group2 = current_squared[0] + 2 * current_squared[1] + 3 * current_squared[2]
				+ 4 * current_squared[3] + 5 * last;
			fx += group1 * group1 + group2 * group2;
			g[i] += -8.0 * group1 + 4.0 * group2 * x[i];
			g[i + 1] += 8.0 * group2 * x[i + 1];
			g[i + 2] += 12.0 * group2 * x[i + 2];
			g[i + 3] += 16.0 * group2 * x[i + 3];
			g[n - 1] += 20.0 * group2 * x[n - 1];
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx(0.0);
		double group1(0.0), group2(0.0);
		double current_squared[4] = { 0.,pow(x[0], 2),pow(x[1], 2),pow(x[2], 2) };
		double last(pow(x[n - 1], 2));

		for (int i = 0; i < n - 4; i++)
		{
			current_squared[0] = current_squared[1];
			current_squared[1] = current_squared[2];
			current_squared[2] = current_squared[3];
			current_squared[3] = x[i + 3] * x[i + 3];
			group1 = 3.0 - 4.0 * x[i];
			group2 = current_squared[0] + 2 * current_squared[1] + 3 * current_squared[2]
				+ 4 * current_squared[3] + 5 * last;
			fx += group1 * group1 + group2 * group2;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 1.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};
//-----------------------  3  -----------------------
//precondition: n >= 4 
class BROYDN7D : public problem
{
public:
	double EPS{};
	long int n{};
	BROYDN7D() {}
	BROYDN7D(double const tolerance) { EPS = tolerance;  n = 5000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "BROYDN7D"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double first(-2.0 * x[1] + 1 + (3. - 2.0 * x[0]) * x[0]);
		double last(-x[n - 2] + 1 + (3. - 2.0 * x[n - 1]) * x[n - 1]);
		double fx = pow(fabs(first), 7 / 3.0) + pow(fabs(last), 7 / 3.0);
		double powFabsFirst4over3{}, powFabsLast4over3{};
		for (int i = 0; i < n; i++)
			g[i] = 0;

		g[0] = (7.0 / 3) * pow(fabs(first), 4 / 3.0) * ((first > 0) ? (1) : (-1)) * (3. - 4. * x[0]);
		g[1] = -(14.0 / 3) * pow(fabs(first), 4 / 3.0) * ((first > 0) ? (1) : (-1));
		g[n - 2] = -(7.0 / 3) * pow(fabs(last), 4 / 3.0) * ((last > 0) ? (1) : (-1));
		g[n - 1] = (7.0 / 3) * pow(fabs(last), 4 / 3.0) * ((last > 0) ? (1) : (-1)) * (3. - 4. * x[n - 1]);

		last = x[0] + x[n / 2];
		fx += pow(fabs(last), 7 / 3.0);
		g[0] += (7.0 / 3) * pow(fabs(last), 4 / 3.0) * ((last > 0) ? (1) : (-1));
		g[n / 2] += (7.0 / 3) * pow(fabs(last), 4 / 3.0) * ((last > 0) ? (1) : (-1));

		for (int i = 1; i < n / 2; i++)
		{
			first = 1 - x[i - 1] - 2.0 * x[i + 1] + (3. - 2.0 * x[i]) * x[i];
			last = x[i] + x[i + n / 2];
			fx += pow(fabs(first), 7 / 3.0) + pow(fabs(last), 7 / 3.0);

			powFabsFirst4over3 = (7.0 / 3) * pow(fabs(first), 4 / 3.0) * ((first > 0) ? (1) : (-1));
			powFabsLast4over3 = (7.0 / 3) * pow(fabs(last), 4 / 3.0) * ((last > 0) ? (1) : (-1));

			g[i - 1] += -powFabsFirst4over3;
			g[i] += powFabsFirst4over3 * (3 - 4 * x[i])
				+ powFabsLast4over3;
			g[i + 1] += -2 * powFabsFirst4over3;
			g[i + n / 2] += powFabsLast4over3;
		}
		for (int i = n / 2; i < n - 1; i++)
		{
			first = 1 - x[i - 1] - 2.0 * x[i + 1] + (3. - 2.0 * x[i]) * x[i];
			fx += pow(fabs(first), 7 / 3.0);
			powFabsFirst4over3 = (7.0 / 3) * pow(fabs(first), 4 / 3.0) * ((first > 0) ? (1) : (-1));
			g[i - 1] += -powFabsFirst4over3;
			g[i] += powFabsFirst4over3 * (3 - 4 * x[i]);
			g[i + 1] += -2 * powFabsFirst4over3;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double first(-2.0 * x[1] + 1 + (3. - 2.0 * x[0]) * x[0]);
		double last(-x[n - 2] + 1 + (3. - 2.0 * x[n - 1]) * x[n - 1]);
		double fx = pow(fabs(first), 7 / 3.0) + pow(fabs(last), 7 / 3.0);
		last = x[0] + x[n / 2];
		fx += pow(fabs(last), 7 / 3.0);

		for (int i = 1; i < n / 2; i++)
		{
			first = 1 - x[i - 1] - 2.0 * x[i + 1] + (3. - 2.0 * x[i]) * x[i];
			last = x[i] + x[i + n / 2];
			fx += pow(fabs(first), 7 / 3.0) + pow(fabs(last), 7 / 3.0);
		}
		for (int i = n / 2; i < n - 1; i++)
		{
			first = 1 - x[i - 1] - 2.0 * x[i + 1] + (3. - 2.0 * x[i]) * x[i];
			fx += pow(fabs(first), 7 / 3.0);
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 1.0;
	}
	bool virtual stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};
/*===== 4 =================== BRYBND Function ================ 10000 ===================*/
//i < n - must be
class BRYBND : public problem
{
public:
	double EPS{};
	long int n{};
	BRYBND() {}
	BRYBND(double const tolerance) { EPS = tolerance;  n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "BRYBND"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		double group(0.0), next(x[1] * (1 + x[1])), secGroup(next), current(x[0] * (1 + x[0]));
		for (int i = 0; i < n; i++)
			g[i] = 0;
		group = x[0] * (2.0 + 5.0 * x[0] * x[0]) + 1 - secGroup;
		fx += group * group;
		g[0] += 2.0 * group * (2.0 + 15.0 * x[0] * x[0]);
		g[1] -= 2.0 * group * (1 + 2.0 * x[1]);
		for (int i = 1; i < 6; i++)
		{
			secGroup -= next;
			secGroup += current;
			current = next;
			next = x[i + 1] * (1 + x[i + 1]);
			secGroup += next;
			group = x[i] * (2.0 + 5.0 * x[i] * x[i]) + 1 - secGroup;
			fx += group * group;
			g[i] += 2.0 * group * (2.0 + 15.0 * x[i] * x[i]);

			int  lo((0 > i - 5) ? (0) : (i - 5));
			for (int j = lo; j < i; j++)
				g[j] -= 2.0 * group * (1 + 2.0 * x[j]);
			g[i + 1] -= 2.0 * group * (1 + 2.0 * x[i + 1]);
		}
		for (int i = 6; i < n - 1; i++)
		{
			secGroup -= next;
			secGroup += current;
			current = next;
			next = x[i + 1] * (1 + x[i + 1]);
			secGroup += next;
			secGroup -= x[i - 6] * (1 + x[i - 6]);
			group = x[i] * (2.0 + 5.0 * x[i] * x[i]) + 1 - secGroup;
			fx += group * group;
			g[i] += 2.0 * group * (2.0 + 15.0 * x[i] * x[i]);

			for (int j = i - 5; j < i; j++)
				g[j] -= 2.0 * group * (1 + 2.0 * x[j]);
			g[i + 1] -= 2.0 * group * (1 + 2.0 * x[i + 1]);
		}
		secGroup -= next;
		secGroup -= x[n - 7] * (1 + x[n - 7]);
		secGroup += current;
		group = x[n - 1] * (2.0 + 5.0 * x[n - 1] * x[n - 1]) + 1 - secGroup;
		fx += group * group;
		g[n - 1] += 2.0 * group * (2.0 + 15.0 * x[n - 1] * x[n - 1]);

		for (int j = n - 6; j < n - 1; j++)
			g[j] -= 2.0 * group * (1 + 2.0 * x[j]);

		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double group(0.0), next(x[1] * (1 + x[1])), secGroup(next), current(x[0] * (1 + x[0]));
		group = x[0] * (2.0 + 5.0 * x[0] * x[0]) + 1 - secGroup;
		fx += group * group;

		for (int i = 1; i < 6; i++)
		{
			secGroup -= next;
			secGroup += current;
			current = next;
			next = x[i + 1] * (1 + x[i + 1]);
			secGroup += next;
			group = x[i] * (2.0 + 5.0 * x[i] * x[i]) + 1 - secGroup;
			fx += group * group;
		}
		for (int i = 6; i < n - 1; i++)
		{
			secGroup -= next;
			secGroup += current;
			current = next;
			next = x[i + 1] * (1 + x[i + 1]);
			secGroup += next;
			secGroup -= x[i - 6] * (1 + x[i - 6]);
			group = x[i] * (2.0 + 5.0 * x[i] * x[i]) + 1 - secGroup;
			fx += group * group;
		}
		secGroup -= next;
		secGroup -= x[n - 7] * (1 + x[n - 7]);
		secGroup += current;
		group = x[n - 1] * (2.0 + 5.0 * x[n - 1] * x[n - 1]) + 1 - secGroup;
		fx += group * group;

		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = -1.0;
	}
	bool virtual stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//------------ CHAINWOO -----------  5  -----------------------
//    Precondition: n > 4
class CHAINWOO : public problem
{
	double item1{}, item2{}, item3{}, item4{}, item5{}, item6{};
public:
	double EPS{};
	long int n{};
	CHAINWOO() {}
	CHAINWOO(double const tolerance) { EPS = tolerance;  n = 5000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "CHAINWOO"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 1.0;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		for (int i = 0; i < n - 3; i += 2)
		{
			item1 = (x[i + 1] - x[i] * x[i]);
			item2 = (1 - x[i]);
			item3 = (x[i + 3] - x[i + 2] * x[i + 2]);
			item4 = (1 - x[i + 2]);
			item5 = (x[i + 1] + x[i + 3] - 2.0);
			item6 = (x[i + 1] - x[i + 3]);
			fx += 100 * item1 * item1 + item2 * item2 + 90 * item3 * item3
				+ item4 * item4 + 10.0 * item5 * item5 + 0.1 * item6 * item6;
			g[i] -= (400 * item1 * x[i] + 2 * item2);
			g[i + 1] += 200 * item1 + 20.0 * item5 + 0.2 * item6;
			g[i + 2] -= (360 * item3 * x[i + 2] + 2.0 * item4);
			g[i + 3] += 180 * item3 + 20.0 * item5 - 0.2 * item6;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 1.0;
		for (int i = 0; i < n - 3; i += 2)
		{
			item1 = (x[i + 1] - x[i] * x[i]);
			item2 = (1 - x[i]);
			item3 = (x[i + 3] - x[i + 2] * x[i + 2]);
			item4 = (1 - x[i + 2]);
			item5 = (x[i + 1] + x[i + 3] - 2.0);
			item6 = (x[i + 1] - x[i + 3]);
			fx += 100 * item1 * item1 + item2 * item2 + 90 * item3 * item3
				+ item4 * item4 + 10.0 * item5 * item5 + 0.1 * item6 * item6;
		}
		return fx;
	}
	void initialize(double* x)
	{
		x[0] = -3.0;
		x[1] = -1.0;
		x[2] = -3.0;
		x[3] = -1.0;
		for (int i = 4; i < n; i++)
			x[i] = -2.0;
	}
	bool virtual stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};
//----------- COSINE ------------  6  -----------------------
class COSINE : public problem
{
	double tmp{};
public:
	double EPS{};
	long int n{};
	COSINE() {}
	COSINE(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "COSINE"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item;
		double fx(0.0);

		for (int i = 0; i < n; i++)
			g[i] = 0.0;

		for (int i = 0; i < n - 1; i++)
		{
			item = -0.5 * x[i + 1] + x[i] * x[i];
			tmp = sin(item);
			g[i] -= 2.0 * tmp * x[i];
			g[i + 1] += 0.5 * tmp;
			fx += cos(item);
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item;
		double fx(0.0);

		for (int i = 0; i < n - 1; i++)
		{
			item = -0.5 * x[i + 1] + x[i] * x[i];
			fx += cos(item);
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 1.0;
	}
	bool virtual stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};
//-----------------------  7  -----------------------
class CRAGGLVY : public problem
{
public:
	double EPS{};
	long int n{};
	CRAGGLVY() {}
	CRAGGLVY(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "CRAGGLVY"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		double item1, item2, element, item3, item4;
		double item1Squared, item2Squared, item3Squared, item4Squared;
		double xipow2, xipow4;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		for (int i = 0; i < n - 3; i += 2)
		{
			item1 = exp(x[i]) - x[i + 1];
			item2 = x[i + 1] - x[i + 2];
			element = x[i + 2] - x[i + 3];
			item3 = tan(element) + element;
			item4 = (x[i + 3] - 1);

			item1Squared = item1 * item1;
			item2Squared = item2 * item2;
			item3Squared = item3 * item3;
			item4Squared = item4 * item4;
			xipow2 = x[i] * x[i];
			xipow4 = xipow2 * xipow2;

			fx += item1Squared * item1Squared + 100 * item2Squared * item2Squared * item2Squared
				+ item3Squared * item3Squared + xipow4 * xipow4 + item4Squared;

			g[i] += 4 * item1Squared * item1 * exp(x[i]) + 8 * x[i] * xipow4 * xipow2;
			g[i + 1] += -4 * item1 * item1Squared + 600 * item2 * item2Squared * item2Squared;
			g[i + 2] += -600 * item2 * item2Squared * item2Squared
				+ 4 * item3 * item3Squared * (1 / pow(cos(element), 2.0) + 1);

			g[i + 3] += -4 * item3 * item3Squared * (1 / pow(cos(element), 2.0) + 1)
				+ 2 * item4;

		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double item1, item2, element, item3, item4;
		double item1Squared, item2Squared, item3Squared, item4Squared;
		double xipow2, xipow4;

		for (int i = 0; i < n - 3; i += 2)
		{
			item1 = exp(x[i]) - x[i + 1];
			item2 = x[i + 1] - x[i + 2];
			element = x[i + 2] - x[i + 3];
			item3 = tan(element) + element;
			item4 = (x[i + 3] - 1);

			item1Squared = item1 * item1;
			item2Squared = item2 * item2;
			item3Squared = item3 * item3;
			item4Squared = item4 * item4;
			xipow2 = x[i] * x[i];
			xipow4 = xipow2 * xipow2;

			fx += item1Squared * item1Squared + 100 * item2Squared * item2Squared * item2Squared
				+ item3Squared * item3Squared + xipow4 * xipow4 + item4Squared;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool virtual stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};
//-----------------------  8  -----------------------
class CURLY10 : public problem
{
	int k{ 10 };
	double fx{};
	double cube{};
	int i{}, j{};
	double q{};
public:
	double EPS{};
	long int n{};
	CURLY10() {}
	CURLY10(double const tolerance) { EPS = tolerance;   n = 500; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "CURLY10"; }
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
		fx += pow(q, 4.0) - 20 * q * q - 0.1 * q;
		for (int j = 0; j <= k; j++)
			g[j] += 4 * pow(q, 3.0) - 40 * q - 0.1;

		for (i = 1; i < n - k; i++)
		{
			q = q - x[i - 1] + x[i + k];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
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
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
			cube = 4 * t0 * t1 - 40 * t0 - 0.1;
			for (int j = i; j <= n - 1; j++)
				g[j] += cube;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		fx = 0.;
		q = 0.0;
		for (int j = 0; j <= k; j++)
			q += x[j];
		fx += pow(q, 4.0) - 20 * q * q - 0.1 * q;

		for (i = 1; i < n - k; i++)
		{
			q = q - x[i - 1] + x[i + k];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
		}
		for (; i < n; i++)
		{
			q -= x[i - 1];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
		}
		return fx;
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
//-----------------------  9  -----------------------
class CURLY20 : public problem
{
	int k{ 20 };
	double fx{};
	double cube{};
	int i{}, j{};
	double q{};
public:
	double EPS{};
	long int n{};
	CURLY20() {}
	CURLY20(double const tolerance) { EPS = tolerance;   n = 500; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "CURLY20"; }
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
		fx += pow(q, 4.0) - 20 * q * q - 0.1 * q;
		for (int j = 0; j <= k; j++)
			g[j] += 4 * pow(q, 3.0) - 40 * q - 0.1;

		for (i = 1; i < n - k; i++)
		{
			q = q - x[i - 1] + x[i + k];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
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
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
			cube = 4 * t0 * t1 - 40 * t0 - 0.1;
			for (int j = i; j <= n - 1; j++)
				g[j] += cube;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		fx = 0.;
		q = 0.0;
		for (int j = 0; j <= k; j++)
			q += x[j];
		fx += pow(q, 4.0) - 20 * q * q - 0.1 * q;

		for (i = 1; i < n - k; i++)
		{
			q = q - x[i - 1] + x[i + k];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
		}
		for (; i < n; i++)
		{
			q -= x[i - 1];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
		}
		return fx;
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
//-----------------------  10  -----------------------
class CURLY30 : public problem
{
	int k{ 30 };
	double fx{};
	double cube{};
	int i{}, j{};
	double q{};
public:
	double EPS{};
	long int n{};
	CURLY30() {}
	CURLY30(double const tolerance) { EPS = tolerance;   n = 500; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "CURLY30"; }
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
		fx += pow(q, 4.0) - 20 * q * q - 0.1 * q;
		for (int j = 0; j <= k; j++)
			g[j] += 4 * pow(q, 3.0) - 40 * q - 0.1;

		for (i = 1; i < n - k; i++)
		{
			q = q - x[i - 1] + x[i + k];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
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
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
			cube = 4 * t0 * t1 - 40 * t0 - 0.1;
			for (int j = i; j <= n - 1; j++)
				g[j] += cube;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		fx = 0.;
		q = 0.0;
		for (int j = 0; j <= k; j++)
			q += x[j];
		fx += pow(q, 4.0) - 20 * q * q - 0.1 * q;

		for (i = 1; i < n - k; i++)
		{
			q = q - x[i - 1] + x[i + k];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
		}
		for (; i < n; i++)
		{
			q -= x[i - 1];
			double t0 = q, t1 = t0 * t0, t2 = t1 * t1;
			fx += t2 - 20 * t1 - 0.1 * t0;;
		}
		return fx;
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
//-----------------------  11  ----------------------- 10000
class DIXMAANA : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANA() {}
	DIXMAANA(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANA"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 1.0;
		double item(0.0);
		int m(n / 3);
		for (int i = 0; i < n; i++)
			g[i] = 0;
		for (int i = 0; i < m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.125 * pow(item, 2) + 0.125 * x[i] * x[i + 2 * m];
			g[i] += 2 * x[i] + 0.25 * item * pow(x[i + m], 2) + 0.125 * x[i + 2 * m];
			g[i + m] += 0.5 * item * x[i] * x[i + m];
			g[i + 2 * m] += 0.125 * x[i];
		}
		for (int i = m; i < 2 * m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.125 * pow(item, 2);
			g[i] += 2 * x[i] + 0.25 * item * pow(x[i + m], 2);
			g[i + m] += 0.5 * item * x[i] * x[i + m];
		}
		for (int i = 2 * m; i < n; i++)
		{
			fx += pow(x[i], 2);
			g[i] += 2 * x[i];
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 1.0;
		double item(0.0);
		int m(n / 3);

		for (int i = 0; i < m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.125 * pow(item, 2) + 0.125 * x[i] * x[i + 2 * m];
		}
		for (int i = m; i < 2 * m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.125 * pow(item, 2);
		}
		for (int i = 2 * m; i < n; i++)
		{
			fx += pow(x[i], 2);
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  12  ----------------------- 10000
class DIXMAANB : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANB() {}
	DIXMAANB(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANB"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = n / 3;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		for (int i = 0; i < m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + item1 * item1 + item2 * item2 + (0.0625 * x[i] * x[i + 2 * m]);
			g[i] += 2 * x[i] + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.5 * item2 * pow(x[i + m], 2) +
				(0.0625 * x[i + 2 * m]);
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += item2 * x[i] * x[i + m];
			g[i + 2 * m] += 0.0625 * x[i];
		}
		for (int i = m; i < 2 * m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + item1 * item1 + item2 * item2;
			g[i] += 2 * x[i] + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.5 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += item2 * x[i] * x[i + m];
		}
		for (int i = 2 * m; i < n - 1; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) + item1 * item1;
			g[i] += 2 * x[i] + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = n / 3;

		for (int i = 0; i < m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + item1 * item1 + item2 * item2 + (0.0625 * x[i] * x[i + 2 * m]);
		}
		for (int i = m; i < 2 * m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + item1 * item1 + item2 * item2;
		}
		for (int i = 2 * m; i < n - 1; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) + item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  13  ----------------------- 10000
class DIXMAANC : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANC() {}
	DIXMAANC(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANC"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = n / 3;
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.125 * item1 * item1 + 0.125 * item2 * item2 + 0.125 * x[i] * x[i + 2 * m];
			g[i] += 2 * x[i] + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.25 * item2 * pow(x[i + m], 2) + 0.125 * x[i + 2 * m];
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 0.5 * item2 * x[i] * x[i + m];
			g[i + 2 * m] += 0.125 * x[i];
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.125 * item1 * item1 + 0.125 * item2 * item2;
			g[i] += 2 * x[i] + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.25 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 0.5 * item2 * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) + 0.125 * item1 * item1;
			g[i] += 2 * x[i] + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = n / 3;

		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.125 * item1 * item1 + 0.125 * item2 * item2 + 0.125 * x[i] * x[i + 2 * m];
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.125 * item1 * item1 + 0.125 * item2 * item2;
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) + 0.125 * item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  14  ----------------------- 10000
class DIXMAAND : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAAND() {}
	DIXMAAND(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAAND"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = n / 3;
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.26 * item1 * item1 + 0.26 * item2 * item2 + 0.26 * x[i] * x[i + 2 * m];
			g[i] += 2 * x[i] + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.52 * item2 * pow(x[i + m], 2) + 0.26 * x[i + 2 * m];
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 1.04 * item2 * x[i] * x[i + m];
			g[i + 2 * m] += 0.26 * x[i];
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.26 * item1 * item1 + 0.26 * item2 * item2;
			g[i] += 2 * x[i] + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.52 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 1.04 * item2 * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) + 0.26 * item1 * item1;
			g[i] += 2 * x[i] + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = n / 3;

		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.26 * item1 * item1 + 0.26 * item2 * item2 + 0.26 * x[i] * x[i + 2 * m];
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) + 0.26 * item1 * item1 + 0.26 * item2 * item2;
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) + 0.26 * item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  15  ----------------------- 5000
class DIXMAANE : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANE() {}
	DIXMAANE(double const tolerance) { EPS = tolerance;   n = 5000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANE"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item = 0.0;
		int m = n / 3;
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * pow(item, 2) + (0.125 * x[i] * x[i + 2 * m] * (i + 1)) / n;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.25 * item * pow(x[i + m], 2) + (0.125 * x[i + 2 * m] * (i + 1)) / n;
			g[i + m] += 0.5 * item * x[i] * x[i + m];
			g[i + 2 * m] += (0.125 * x[i] * (i + 1)) / n;
		}
		for (i = m; i < 2 * m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * pow(item, 2);
			g[i] += (2 * x[i] * (i + 1)) / n + 0.25 * item * pow(x[i + m], 2);
			g[i + m] += 0.5 * item * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n; i++)
		{
			fx += (pow(x[i], 2) * i) / n;
			g[i] += (2 * x[i] * (i + 1)) / n;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item = 0.0;
		int m = n / 3;

		for (i = 0; i < m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * pow(item, 2) + (0.125 * x[i] * x[i + 2 * m] * (i + 1)) / n;
		}
		for (i = m; i < 2 * m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * pow(item, 2);
		}
		for (i = 2 * m; i < n; i++)
		{
			fx += (pow(x[i], 2) * i) / n;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  16  ----------------------- 5000
class DIXMAANF : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANF() {}
	DIXMAANF(double const tolerance) { EPS = tolerance;   n = 5000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANF"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = n / 3;
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + item1 * item1 + item2 * item2 + (0.0625 * x[i] * x[i + 2 * m] * (i + 1)) / n;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.5 * item2 * pow(x[i + m], 2) +
				(0.0625 * x[i + 2 * m] * (i + 1)) / n;
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += item2 * x[i] * x[i + m];
			g[i + 2 * m] += (0.0625 * x[i] * (i + 1)) / n;
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + item1 * item1 + item2 * item2;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.5 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += item2 * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += (pow(x[i], 2) * (i + 1)) / n + item1 * item1;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = n / 3;

		for (i = 0; i < m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + item1 * item1 + item2 * item2 + (0.0625 * x[i] * x[i + 2 * m] * (i + 1)) / n;
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + item1 * item1 + item2 * item2;
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += (pow(x[i], 2) * (i + 1)) / n + item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};


//-----------------------  17  ----------------------- 5000
class DIXMAANG : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANG() {}
	DIXMAANG(double const tolerance) { EPS = tolerance;   n = 5000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANG"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = (n / 3);
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * item1 * item1 + 0.125 * item2 * item2
				+ (0.125 * x[i] * x[i + 2 * m] * (i + 1)) / n;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.25 * item2 * pow(x[i + m], 2)
				+ (0.125 * x[i + 2 * m] * (i + 1)) / n;
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 0.5 * item2 * x[i] * x[i + m];
			g[i + 2 * m] += (0.125 * x[i] * (i + 1)) / n;
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * item1 * item1 + 0.125 * item2 * item2;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.25 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 0.5 * item2 * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * item1 * item1;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = (n / 3);

		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * item1 * item1 + 0.125 * item2 * item2
				+ (0.125 * x[i] * x[i + 2 * m] * (i + 1)) / n;
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * item1 * item1 + 0.125 * item2 * item2;
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.125 * item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};
//-----------------------  18  ----------------------- 5000
class DIXMAANH : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANH() {}
	DIXMAANH(double const tolerance) { EPS = tolerance;   n = 5000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANH"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = (n / 3);
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.26 * item1 * item1 + 0.26 * item2 * item2
				+ (0.26 * x[i] * x[i + 2 * m] * (i + 1)) / n;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.52 * item2 * pow(x[i + m], 2)
				+ (0.26 * x[i + 2 * m] * (i + 1)) / n;
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 1.04 * item2 * x[i] * x[i + m];
			g[i + 2 * m] += (0.26 * x[i] * (i + 1)) / n;
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.26 * item1 * item1 + 0.26 * item2 * item2;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]) + 0.52 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 1.04 * item2 * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.26 * item1 * item1;
			g[i] += (2 * x[i] * (i + 1)) / n + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item1 = 0.0;
		double item2 = 0.0;
		int m = (n / 3);

		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.26 * item1 * item1 + 0.26 * item2 * item2
				+ (0.26 * x[i] * x[i + 2 * m] * (i + 1)) / n;
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.26 * item1 * item1 + 0.26 * item2 * item2;
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += (pow(x[i], 2) * (i + 1)) / n + 0.26 * item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  19  -----------------------3000
class DIXMAANI : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANI() {}
	DIXMAANI(double const tolerance) { EPS = tolerance;   n = 3000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANI"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		double item(0.0);
		int m(n / 3);
		for (int i = 0; i < n; i++)
			g[i] = 0;
		for (int i = 0; i < m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / double(n), 2) + 0.125 * pow(item, 2)
				+ 0.125 * x[i] * x[i + 2 * m] * pow((i + 1) / double(n), 2);
			g[i] += 2 * x[i] * pow((i + 1) / double(n), 2) + 0.25 * item * pow(x[i + m], 2)
				+ 0.125 * x[i + 2 * m] * pow((i + 1) / double(n), 2);
			g[i + m] += 0.5 * item * x[i] * x[i + m];
			g[i + 2 * m] += 0.125 * x[i] * pow((i + 1) / double(n), 2);
		}
		for (int i = m; i < 2 * m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / double(n), 2) + 0.125 * pow(item, 2);
			g[i] += 2 * x[i] * pow((i + 1) / double(n), 2) + 0.25 * item * pow(x[i + m], 2);
			g[i + m] += 0.5 * item * x[i] * x[i + m];
		}
		for (int i = 2 * m; i < n; i++)
		{
			fx += pow(x[i], 2) * pow((i + 1) / double(n), 2);
			g[i] += 2 * x[i] * pow((i + 1) / double(n), 2);
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double item(0.0);
		int m(n / 3);

		for (int i = 0; i < m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / double(n), 2) + 0.125 * pow(item, 2)
				+ 0.125 * x[i] * x[i + 2 * m] * pow((i + 1) / double(n), 2);
		}
		for (int i = m; i < 2 * m; i++)
		{
			item = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / double(n), 2) + 0.125 * pow(item, 2);
		}
		for (int i = 2 * m; i < n; i++)
			fx += pow(x[i], 2) * pow((i + 1) / double(n), 2);

		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  20  ----------------------- 3000
class DIXMAANJ : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANJ() {}
	DIXMAANJ(double const tolerance) { EPS = tolerance;   n = 3000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANJ"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item1 = (0.0);
		double item2 = (0.0);
		int m = (n / 3);
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + item1 * item1 + item2 * item2
				+ 0.0625 * x[i] * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1])
				+ 0.5 * item2 * pow(x[i + m], 2) + 0.0625 * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += item2 * x[i] * x[i + m];
			g[i + 2 * m] += 0.0625 * x[i] * pow((i + 1) / (double)(n), 2);
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + item1 * item1 + item2 * item2;
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1])
				+ 0.5 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += item2 * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + item1 * item1;
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.5 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.5 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item1 = (0.0);
		double item2 = (0.0);
		int m = (n / 3);

		for (i = 0; i < m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + item1 * item1 + item2 * item2
				+ 0.0625 * x[i] * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = 0.25 * x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + item1 * item1 + item2 * item2;
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = 0.25 * x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  21  ----------------------- 3000
class DIXMAANK : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANK() {}
	DIXMAANK(double const tolerance) { EPS = tolerance;   n = 3000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANK"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item1 = (0.0);
		double item2 = (0.0);
		int m = (n / 3);
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.125 * item1 * item1 + 0.125 * item2 * item2
				+ 0.125 * x[i] * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1])
				+ 0.25 * item2 * pow(x[i + m], 2) + 0.125 * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 0.5 * item2 * x[i] * x[i + m];
			g[i + 2 * m] += 0.125 * x[i] * pow((i + 1) / (double)(n), 2);
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.125 * item1 * item1 + 0.125 * item2 * item2;
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1])
				+ 0.25 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 0.5 * item2 * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.125 * item1 * item1;
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.25 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.25 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item1 = (0.0);
		double item2 = (0.0);
		int m = (n / 3);

		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.125 * item1 * item1 + 0.125 * item2 * item2
				+ 0.125 * x[i] * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.125 * item1 * item1 + 0.125 * item2 * item2;
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.125 * item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  22  ----------------------- 3000
class DIXMAANL : public problem
{
public:
	double EPS{};
	long int n{};
	DIXMAANL() {}
	DIXMAANL(double const tolerance) { EPS = tolerance;   n = 3000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXMAANL"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 1.0;
		double item1 = (0.0);
		double item2 = (0.0);
		int m = (n / 3);
		for (i = 0; i < n; i++)
			g[i] = 0;
		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.26 * item1 * item1 + 0.26 * item2 * item2
				+ 0.26 * x[i] * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1])
				+ 0.52 * item2 * pow(x[i + m], 2) + 0.26 * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 1.04 * item2 * x[i] * x[i + m];
			g[i + 2 * m] += 0.26 * x[i] * pow((i + 1) / (double)(n), 2);
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.26 * item1 * item1 + 0.26 * item2 * item2;
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1])
				+ 0.52 * item2 * pow(x[i + m], 2);
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
			g[i + m] += 1.04 * item2 * x[i] * x[i + m];
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.26 * item1 * item1;
			g[i] += 2 * x[i] * pow((i + 1) / (double)(n), 2) + 0.52 * item1 * (x[i + 1] + x[i + 1] * x[i + 1]);
			g[i + 1] += 0.52 * item1 * x[i] * (1 + 2 * x[i + 1]);
		}
		fx += pow(x[n - 1], 2);
		g[n - 1] += 2 * x[n - 1];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 1.0;
		double item1 = (0.0);
		double item2 = (0.0);
		int m = (n / 3);

		for (i = 0; i < m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.26 * item1 * item1 + 0.26 * item2 * item2
				+ 0.26 * x[i] * x[i + 2 * m] * pow((i + 1) / (double)(n), 2);
		}
		for (i = m; i < 2 * m; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			item2 = x[i] * pow(x[i + m], 2);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.26 * item1 * item1 + 0.26 * item2 * item2;
		}
		for (i = 2 * m; i < n - 1; i++)
		{
			item1 = x[i] * (x[i + 1] + x[i + 1] * x[i + 1]);
			fx += pow(x[i], 2) * pow((i + 1) / (double)(n), 2) + 0.26 * item1 * item1;
		}
		fx += pow(x[n - 1], 2);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  23  ----------------------- 10000
class DIXON3DQ : public problem
{
public:
	double EPS{};
	long int n{};
	DIXON3DQ() {}
	DIXON3DQ(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DIXON3DQ"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		double item{};

		item = x[0] - 1;
		fx += item * item;
		g[0] = 2.0 * item;

		for (int i = 1; i < n - 1; i++)
			g[i] = 0.0;

		item = x[n - 1] - 1;
		fx += item * item;
		g[n - 1] = 2.0 * item;


		for (int i = 1; i < n - 1; i++)
		{
			item = (x[i] - x[i + 1]);
			fx += item * item;
			g[i] += 2.0 * item;
			g[i + 1] -= 2.0 * item;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double item{};
		item = x[0] - 1;
		fx += item * item;
		item = x[n - 1] - 1;
		fx += item * item;

		for (int i = 1; i < n - 1; i++)
		{
			item = (x[i] - x[i + 1]);
			fx += item * item;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = -1.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  24  -----------------------15000
class DQDRTIC : public problem
{
public:
	double EPS{};
	long int n{};
	DQDRTIC() {}
	DQDRTIC(double const tolerance) { EPS = tolerance;   n = 15000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DQDRTIC"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		double t0 = x[0] * x[0], t1 = x[1] * x[1], t2 = x[2] * x[2];
		double fx = t0 + 100 * (t1 + t2);
		g[0] += 2 * x[0];
		g[1] += 200 * x[1];
		g[2] += 200 * x[2];
		for (int i = 1; i < n - 2; i++)
		{
			t0 = t1;
			t1 = t2;
			t2 = x[i + 2] * x[i + 2];
			fx += t0 + 100 * (t1 + t2);
			g[i] += 2 * x[i];
			g[i + 1] += 200 * x[i + 1];
			g[i + 2] += 200 * x[i + 2];
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double t0 = x[0] * x[0], t1 = x[1] * x[1], t2 = x[2] * x[2];
		double fx = t0 + 100 * (t1 + t2);

		for (int i = 1; i < n - 2; i++)
		{
			t0 = t1;
			t1 = t2;
			t2 = x[i + 2] * x[i + 2];
			fx += t0 + 100 * (t1 + t2);
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 3.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  25  ----------------------- 15000
class DQRTIC : public problem
{
public:
	double EPS{};
	long int n{};
	DQRTIC() {}
	DQRTIC(double const tolerance) { EPS = tolerance;   n = 15000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "DQRTIC"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		double item, squaredItem;
		for (int i = 0; i < n; i++)
		{
			item = x[i] - i - 1;
			squaredItem = item * item;
			fx += squaredItem * squaredItem;
			g[i] = 4 * squaredItem * item;
		}
		return fx;
	}

	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double item, squaredItem;
		for (int i = 0; i < n; i++)
		{
			item = x[i] - i - 1;
			squaredItem = item * item;
			fx += squaredItem * squaredItem;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  26  -----------------------8000
class EDENSCH : public problem
{
public:
	double EPS{};
	long int n{};
	EDENSCH() {}
	EDENSCH(double const tolerance) { EPS = tolerance;   n = 8000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "EDENSCH"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		double item1, item2, item3;
		double squaredItem1;
		for (int i = 0; i < n - 1; i++)
		{
			item1 = x[i] - 2;
			item2 = x[i] * x[i + 1] - 2 * x[i + 1];
			item3 = x[i + 1] + 1;
			squaredItem1 = item1 * item1;
			fx += 16 + squaredItem1 * squaredItem1 + item2 * item2 + item3 * item3;
			g[i] += 4 * squaredItem1 * item1 + 2 * item2 * x[i + 1];
			g[i + 1] += 2 * item2 * (x[i] - 2.0) + 2 * item3;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double item1, item2, item3;
		double squaredItem1;

		for (int i = 0; i < n - 1; i++)
		{
			item1 = x[i] - 2;
			item2 = x[i] * x[i + 1] - 2 * x[i + 1];
			item3 = x[i + 1] + 1;
			squaredItem1 = item1 * item1;
			fx += 16 + squaredItem1 * squaredItem1 + item2 * item2 + item3 * item3;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = .0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  27  ----------------------- 15000
class EG2 : public problem
{
public:
	double EPS{};
	long int n{};
	EG2() {}
	EG2(double const tolerance) { EPS = tolerance;   n = 15000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "EG2"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		double fx = 0.5 * sin(pow(x[n - 1], 2));
		g[n - 1] = cos(pow(x[n - 1], 2)) * x[n - 1];
		double item;
		for (int i = 0; i < n - 1; i++)
		{
			item = x[0] + x[i] * x[i] - 1;;
			fx += sin(item);
			g[0] += cos(item);
			g[i] += 2 * cos(item) * x[i];
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.5 * sin(pow(x[n - 1], 2));
		double item;
		for (int i = 0; i < n - 1; i++)
		{
			item = x[0] + x[i] * x[i] - 1;;
			fx += sin(item);
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = .0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  28  -----------------------
class ENGVAL1 : public problem
{
public:
	double EPS{};
	long int n{};
	ENGVAL1() {}
	ENGVAL1(double const tolerance) { EPS = tolerance;  n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "ENGVAL1"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		double item;
		for (int i = 0; i < n; i++)
			g[i] = 0.0;

		for (int i = 0; i < n - 1; i++)
		{
			item = x[i] * x[i] + x[i + 1] * x[i + 1];
			fx += item * item + (3 - 4.0 * x[i]);
			g[i] += 4.0 * item * x[i] - 4.0;
			g[i + 1] += 4.0 * item * x[i + 1];
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double item;

		for (int i = 0; i < n - 1; i++)
		{
			item = x[i] * x[i] + x[i + 1] * x[i + 1];
			fx += item * item + (3 - 4.0 * x[i]);
		}
		return fx;
	}

	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 2.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  29  -----------------------
class EXTROSNB : public problem
{
public:
	double EPS{};
	long int n{};
	EXTROSNB() {}
	EXTROSNB(double const tolerance) { EPS = tolerance;   n = 2000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "EXTROSNB"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item(x[0] - 1);
		double fx(pow(item, 2));
		for (int i = 1; i < n; i++)
			g[i] = 0.0;
		g[0] = 2 * item;
		for (int i = 1; i < n; i++) {
			item = 10 * (x[i] - x[i - 1] * x[i - 1]);
			fx += item * item;
			g[i - 1] += -40.0 * item * x[i - 1];
			g[i] += 20.0 * item;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item(x[0] - 1);
		double fx(pow(item, 2));

		for (int i = 1; i < n; i++)
		{
			item = 10 * (x[i] - x[i - 1] * x[i - 1]);
			fx += item * item;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = -1.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  30  -----------------------
class FLETCHR : public problem
{
public:
	double EPS{};
	long int n{};
	FLETCHR() {}
	FLETCHR(double const tolerance) { EPS = tolerance;  n = 2000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "FLETCHR"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item;
		double fx = 0.0;
		for (int i = 0; i < n; i++)
			g[i] = 0;

		for (int i = 0; i < n - 1; i++)
		{
			item = x[i + 1] - x[i] + 1 - x[i] * x[i];
			fx += item * item;
			g[i] += 20.0 * item * (-2.0 * x[i] - 1.0);
			g[i + 1] += 20.0 * item;
		}
		return 100. * fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item;
		double fx = 0.0;

		for (int i = 0; i < n - 1; i++)
		{
			item = x[i + 1] - x[i] + 1 - x[i] * x[i];
			fx += item * item;
		}
		return 100. * fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = .0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  31  -----------------------
class FREUROTH : public problem
{
public:
	double EPS{};
	long int n{};
	FREUROTH() {}
	FREUROTH(double const tolerance) { EPS = tolerance;   n = 15000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "FREUROTH"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		double item1, item2;;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		for (int i = 0; i < n - 1; i++)
		{
			item1 = (-13 + x[i] + ((5 - x[i + 1]) * x[i + 1] - 2.0) * x[i + 1]);
			item2 = (-29 + x[i] + ((1 + x[i + 1]) * x[i + 1] - 14.0) * x[i + 1]);
			fx += item1 * item1 + item2 * item2;
			g[i] += 2.0 * item1 + 2.0 * item2;
			g[i + 1] += 2.0 * item1 * (10 * x[i + 1] - 3.0 * x[i + 1] * x[i + 1] - 2.0) +
				2.0 * item2 * (2 * x[i + 1] + 3.0 * x[i + 1] * x[i + 1] - 14.0);
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double item1, item2;;

		for (int i = 0; i < n - 1; i++)
		{
			item1 = (-13 + x[i] + ((5 - x[i + 1]) * x[i + 1] - 2.0) * x[i + 1]);
			item2 = (-29 + x[i] + ((1 + x[i + 1]) * x[i + 1] - 14.0) * x[i + 1]);
			fx += item1 * item1 + item2 * item2;
		}
		return fx;
	}
	void initialize(double* x)
	{
		x[0] = 0.5;
		x[1] = -2.0;
		for (int i = 2; i < n; i++)
			x[i] = 0.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  32  -----------------------
class GENHUMPS : public problem
{
public:
	double EPS{};
	long int n{};
	GENHUMPS() {}
	GENHUMPS(double const tolerance) { EPS = tolerance;   n = 7500; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "GENHUMPS"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		double item1 = sin(2.0 * x[0]);
		double item2 = sin(2.0 * x[1]);
		double item11 = item1 * item1;
		double item22 = item2 * item2;
		double t0 = x[0] * x[0], t1 = x[1] * x[1];
		double fx = item11 * item22 + 0.05 * (t0 + t1);
		g[0] = 4.0 * item1 * cos(2.0 * x[0]) * item22 + 0.1 * x[0];
		g[1] = 4.0 * item2 * cos(2.0 * x[1]) * item11 + 0.1 * x[1];

		for (int i = 1; i < n - 1; i++)
		{

			item1 = item2;
			item2 = sin(2.0 * x[i + 1]);
			item11 = item22;
			item22 = item2 * item2;
			t0 = t1;
			t1 = x[i + 1] * x[i + 1];
			fx += item11 * item22 + 0.05 * (t0 + t1);
			t0 = 4.0 * item1 * item2;
			g[i] += t0 * cos(2.0 * x[i]) * item2 + 0.1 * x[i];
			g[i + 1] += t0 * cos(2.0 * x[i + 1]) * item1 + 0.1 * x[i + 1];
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item1 = sin(2.0 * x[0]);
		double item2 = sin(2.0 * x[1]);
		double item11 = item1 * item1;
		double item22 = item2 * item2;
		double t0 = x[0] * x[0], t1 = x[1] * x[1];
		double fx = item11 * item22 + 0.05 * (t0 + t1);

		for (int i = 1; i < n - 1; i++)
		{

			item1 = item2;
			item2 = sin(2.0 * x[i + 1]);
			item11 = item22;
			item22 = item2 * item2;
			t0 = t1;
			t1 = x[i + 1] * x[i + 1];
			fx += item11 * item22 + 0.05 * (t0 + t1);
		}
		return fx;
	}
	void initialize(double* x)
	{
		x[0] = -506.0;
		for (int i = 1; i < n; i++)
			x[i] = 506.2;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  33  -----------------------
class GENROSE : public problem
{
public:
	double EPS{};
	long int n{};
	GENROSE() {}
	GENROSE(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "GENROSE"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 1.0;
		double item1, item2;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		for (int i = 1; i < n; i++)
		{
			item1 = 10.0 * (x[i] - x[i - 1] * x[i - 1]);
			item2 = x[i] - 1.0;
			fx += item1 * item1 + item2 * item2;
			g[i - 1] -= 40.0 * item1 * x[i - 1];
			g[i] += 20 * item1 + 2 * item2;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 1.0;
		double item1, item2;

		for (int i = 1; i < n; i++)
		{
			item1 = 10.0 * (x[i] - x[i - 1] * x[i - 1]);
			item2 = x[i] - 1.0;
			fx += item1 * item1 + item2 * item2;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 1.0 / n + 1;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  34  -----------------------
class LIARWDH : public problem
{
public:
	double EPS{};
	long int n{};
	LIARWDH() {}
	LIARWDH(double const tolerance) { EPS = tolerance;   n = 5000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "LIARWDH"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item1, item2;
		double fx(0.0);
		for (int i = 0; i < n; i++)
		{
			item1 = 2.0 * (x[i] * x[i] - x[0]);
			item2 = (x[i] - 1);
			fx += item1 * item1 + item2 * item2;
			g[i] = 8.0 * item1 * x[i] + 2.0 * item2;
			g[0] -= 4.0 * item1;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item1, item2;
		double fx(0.0);
		for (int i = 0; i < n; i++)
		{
			item1 = 2.0 * (x[i] * x[i] - x[0]);
			item2 = (x[i] - 1);
			fx += item1 * item1 + item2 * item2;
		}
		return fx;
	}

	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 4.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  35  -----------------------
class MOREBV : public problem
{
public:
	double EPS{};
	long int n{};
	MOREBV() {}
	MOREBV(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "MOREBV"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double h(1.0 / n);
		double element, item;
		double fx(0.0);
		for (int i = 0; i < n; i++)
			g[i] = 0;
		//first term
		element = x[1] + h + 1;
		item = 2 * x[1] - x[2] + (h * h * pow(element, 3)) / 2;
		fx += item * item;
		g[1] = 2 * item * (2.0 + h * h * 1.5 * pow(element, 2));
		g[2] = -2 * item;
		//last term
		element = x[n - 1] + (n - 1) * h + 1;
		item = 2 * x[n - 1] - x[n - 2] + (h * h * pow(element, 3)) / 2;
		fx += item * item;
		g[n - 1] = 4 * item * (2.0 + h * h * 1.5 * pow(element, 2));
		g[n - 2] = -2 * item;
		for (int i = 2; i < n - 1; i++)
		{
			element = x[i] + i * h + 1;
			item = 2 * x[i] - x[i - 1] - x[i + 1] + (h * h * pow(element, 3)) / 2;
			fx += item * item;
			g[i - 1] -= 2 * item;
			g[i] += 2 * item * (2.0 + h * h * 1.5 * pow(element, 2));
			g[i + 1] -= 2 * item;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double h(1.0 / n);
		double element, item;
		double fx(0.0);

		//first term
		element = x[1] + h + 1;
		item = 2 * x[1] - x[2] + (h * h * pow(element, 3)) / 2;
		fx += item * item;
		//last term
		element = x[n - 1] + (n - 1) * h + 1;
		item = 2 * x[n - 1] - x[n - 2] + (h * h * pow(element, 3)) / 2;
		fx += item * item;

		for (int i = 2; i < n - 1; i++)
		{
			element = x[i] + i * h + 1;
			item = 2 * x[i] - x[i - 1] - x[i + 1] + (h * h * pow(element, 3)) / 2;
			fx += item * item;
		}
		return fx;
	}
	void initialize(double* x)
	{
		double h(1.0 / n);
		x[0] = 0.;
		for (int i = 1; i < n; i++)
			x[i] = i * h * (i * h - 1.0);
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  36  -----------------------
class NONCVXU2 : public problem
{
public:
	double EPS{};
	long int n{};
	NONCVXU2() {}
	NONCVXU2(double const tolerance) { EPS = tolerance;   n = 3000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "NONCVXU2"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx = 0.0;
		for (int i = 0; i < n; i++)
			g[i] = 0;
		double item1, item2;
		for (int j = 0; j < n; j++)
		{
			int i(j + 1);
			item1 = (x[j] + x[(3 * i - 2) % n] + x[(7 * i - 3) % n]);
			item2 = x[j] + x[(3 * i - 2) % n] + x[(7 * i - 3) % n];
			fx += item1 * item1 + 4 * cos(item2);
			g[j] += 2 * item1 - 4 * sin(item2);
			g[(3 * i - 2) % n] += 2 * item1 - 4 * sin(item2);
			g[(7 * i - 3) % n] += 2 * item1 - 4 * sin(item2);
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.0;
		double item1, item2;
		for (int j = 0; j < n; j++)
		{
			int i(j + 1);
			item1 = (x[j] + x[(3 * i - 2) % n] + x[(7 * i - 3) % n]);
			item2 = x[j] + x[(3 * i - 2) % n] + x[(7 * i - 3) % n];
			fx += item1 * item1 + 4 * cos(item2);
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = i + 1;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  37  -----------------------
class NONDIA : public problem
{
public:
	double EPS{};
	long int n{};
	NONDIA() {}
	NONDIA(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "NONDIA"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item(x[0] - 1.0);
		double fx(item * item);
		g[0] = 2.0 * item;

		for (int i = 1; i < n; i++)
		{
			item = 10.0 * (x[0] - x[i] * x[i]);
			fx += item * item;
			g[0] += 20.0 * item;
			g[i] = -40.0 * x[i] * item;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item(x[0] - 1.0);
		double fx(item * item);

		for (int i = 1; i < n; i++)
		{
			item = 10.0 * (x[0] - x[i] * x[i]);
			fx += item * item;
		}
		return fx;
	}

	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = -1.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  38  -----------------------1000
class NONDQUAR : public problem
{
public:
	double EPS{};
	long int n{};
	NONDQUAR() {}
	NONDQUAR(double const tolerance) { EPS = tolerance;   n = 1000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "NONQUAR"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item, tmp;
		double fx(0.0);

		for (int i = 0; i < n; i++)
			g[i] = 0.0;

		item = x[0] - x[1];
		fx += item * item;
		g[0] = 2.0 * item;
		g[1] = -2.0 * item;

		item = x[n - 2] + x[n - 1];
		fx += item * item;
		g[n - 2] = 2.0 * item;
		g[n - 1] = 2.0 * item;

		for (int i = 0; i < n - 2; i++)
		{
			double t0 = x[i] + x[i + 1] + x[n - 1], t1 = t0 * t0, t2 = t1 * t1, t3 = t0 * t1;

			fx += t2;
			tmp = 4.0 * t3;
			g[i] += tmp;
			g[i + 1] += tmp;
			g[n - 1] += tmp;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item;
		double fx(0.0);
		item = x[0] - x[1];
		fx += item * item;
		item = x[n - 2] + x[n - 1];
		fx += item * item;

		for (int i = 0; i < n - 2; i++)
		{
			double t0 = x[i] + x[i + 1] + x[n - 1], t1 = t0 * t0, t2 = t1 * t1;
			fx += t2;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = -1.0;
	}	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  39  -----------------------
class PENALTY1 : public problem
{
public:
	double EPS{};
	long int n{};
	PENALTY1() {}
	PENALTY1(double const tolerance) { EPS = tolerance;   n = 500; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "PENALTY1"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item, tmp;
		double tail(0.0);
		double a(1E-5);

		double fx = 0.0;

		for (int i = 0; i < n; i++)
		{
			item = x[i] - 1;
			fx += item * item;
			g[i] = 2.0 * a * item;
			tail += x[i] * x[i];
		}
		fx *= a;

		fx += pow(tail - 0.25, 2);
		tmp = 4 * (tail - 0.25);
		for (int i = 0; i < n; i++)
			g[i] += tmp * x[i];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item{}, tmp{};
		double tail(0.0);
		double a(1E-5);
		double fx = 0.0;

		for (int i = 0; i < n; i++)
		{
			item = x[i] - 1;
			fx += item * item;
			tail += x[i] * x[i];
		}
		fx *= a;
		fx += pow(tail - 0.25, 2);

		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = i + 1;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  40  -----------------------
class PENALTY2 : public problem
{
public:
	double EPS{};
	long int n{};
	PENALTY2() {}
	PENALTY2(double const tolerance) { EPS = tolerance;   n = 150; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "PENALTY2"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item1, item2;
		double tail(0.0);
		double a(1E-7);
		double a1(0.2 * a);
		const double ExpMinus1by10 = exp(-1 / 10.0);
		for (int i = 0; i < n; i++)
			g[i] = 0.0;
		item1 = x[0] - 0.2;
		double fx(pow(item1, 2));
		g[0] = 2 * item1;
		tail += (n)*pow(x[0], 2);
		double currentTerm = exp(x[0] / 10);
		double prevTer;
		double currentI = exp(1 / 10.0);
		double prevI;
		for (int i = 1; i < n; i++)
		{
			prevTer = currentTerm;
			currentTerm = exp(x[i] / 10);
			prevI = currentI;
			currentI = exp((i + 1) / 10.0);
			item1 = currentTerm + prevTer - currentI - prevI;
			item2 = currentTerm - ExpMinus1by10;
			tail += (n - i) * x[i] * x[i];
			fx += a * (item1 * item1 + item2 * item2);
			g[i - 1] += a1 * item1 * prevTer;
			g[i] += a1 * currentTerm * (item1 + item2);
		}
		fx += (tail - 1) * (tail - 1);
		for (int i = 0; i < n; i++)
			g[i] += 4 * (tail - 1) * (n - i) * x[i];
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item1, item2;
		double tail(0.0);
		double a(1E-5);
		double a1(0.2 * a);
		const double ExpMinus1by10 = exp(-1 / 10.0);
		item1 = x[0] - 0.2;
		double fx(pow(item1, 2));
		tail += (n)*pow(x[0], 2);
		double currentTerm = exp(x[0] / 10);
		double prevTer;
		double currentI = exp(1 / 10.0);
		double prevI;

		for (int i = 1; i < n; i++)
		{
			prevTer = currentTerm;
			currentTerm = exp(x[i] / 10);
			prevI = currentI;
			currentI = exp((i + 1) / 10.0);
			item1 = currentTerm + prevTer - currentI - prevI;
			item2 = currentTerm - ExpMinus1by10;
			tail += (n - i) * x[i] * x[i];
			fx += a * (item1 * item1 + item2 * item2);
		}
		fx += (tail - 1) * (tail - 1);
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 0.5;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  41  -----------------------
class POWER : public problem
{
public:
	double EPS{};
	long int n{};
	POWER() {}
	POWER(double const tolerance) { EPS = tolerance;   n = 5000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "POWER"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item;
		double fx(0.0);

		for (int i = 0; i < n; i++)
		{
			item = (i + 1) * x[i];
			fx += item * item;
			g[i] = 2.0 * item * (i + 1);
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item;
		double fx(0.0);

		for (int i = 0; i < n; i++)
		{
			item = (i + 1) * x[i];
			fx += item * item;
		}
		return fx;
	}

	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 1.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  42  -----------------------
class  SROSENBR : public problem
{
public:
	double EPS{};
	long int n{};
	SROSENBR() {}
	SROSENBR(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "SROSENBR"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		int i;
		double fx = 0.0;

		for (i = 0; i < n; i += 2) {
			double t1 = 1.0 - x[i];
			double t2 = 10.0 * (x[i + 1] - x[i] * x[i]);
			g[i + 1] = 20.0 * t2;
			g[i] = -2.0 * (x[i] * g[i + 1] + t1);
			fx += t1 * t1 + t2 * t2;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		int i;
		double fx = 0.0;

		for (i = 0; i < n; i += 2)
		{
			double t1 = 1.0 - x[i];
			double t2 = 10.0 * (x[i + 1] - x[i] * x[i]);
			fx += t1 * t1 + t2 * t2;
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i += 2)
		{
			x[i] = -1.2;
			x[i + 1] = 1.0;
		}
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  43  -----------------------
class  TRIDIA : public problem
{
public:
	double EPS{};
	long int n{};
	TRIDIA() {}
	TRIDIA(double const tolerance) { EPS = tolerance;   n = 10000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "TRIDIA"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double item(x[0] - 1.0);
		double fx(item * item);
		for (int i = 0; i < n; i++)
			g[i] = 0;
		g[0] = 2.0 * item;

		for (int i = 1; i < n; i++)
		{
			item = (2.0 * x[i] - x[i - 1]);
			fx += (i + 1) * item * item;
			g[i] += 4.0 * item * (i + 1);
			g[i - 1] -= 2.0 * (i + 1) * item;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double item(x[0] - 1.0);
		double fx(item * item);

		for (int i = 1; i < n; i++)
		{
			item = (2.0 * x[i] - x[i - 1]);
			fx += (i + 1) * item * item;
		}
		return fx;
	}

	void initialize(double* x)
	{
		for (int i = 0; i < n; i++)
			x[i] = 1.0;
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};

//-----------------------  44  -----------------------
class Woods : public problem
{
public:
	double EPS{};
	long int n{};
	Woods() {}
	Woods(double const tolerance) { EPS = tolerance;   n = 15000; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "Woods"; }
	double valGrad
	(
		double* x,
		double* g
	)
	{
		++ValGradEvaluations;
		double fx(0.0);
		double item1, item2, item3, item4, item5, item6;

		for (int i = 0; i < n; i += 4)
		{
			item1 = (x[i + 1] - x[i] * x[i]);
			item2 = (1 - x[i]);
			item3 = (x[i + 3] - x[i + 2] * x[i + 2]);
			item4 = (1 - x[i + 2]);
			item5 = (x[i + 1] + x[i + 3] - 2.0);
			item6 = (x[i + 1] - x[i + 3]);
			fx += (100 * item1 * item1 + item2 * item2 + 90 * item3 * item3
				+ item4 * item4 + 10.0 * item5 * item5 + 0.1 * item6 * item6);
			g[i] = -400 * item1 * x[i] - 2 * item2;
			g[i + 1] = 200 * item1 + 20.0 * item5 + 0.2 * item6;
			g[i + 2] = -360 * item3 * x[i + 2] - 2.0 * item4;
			g[i + 3] = 180 * item3 + 20.0 * item5 - 0.2 * item6;
		}
		return fx;
	}
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx(0.0);
		double item1, item2, item3, item4, item5, item6;

		for (int i = 0; i < n; i += 4)
		{
			item1 = (x[i + 1] - x[i] * x[i]);
			item2 = (1 - x[i]);
			item3 = (x[i + 3] - x[i + 2] * x[i + 2]);
			item4 = (1 - x[i + 2]);
			item5 = (x[i + 1] + x[i + 3] - 2.0);
			item6 = (x[i + 1] - x[i + 3]);
			fx += (100 * item1 * item1 + item2 * item2 + 90 * item3 * item3
				+ item4 * item4 + 10.0 * item5 * item5 + 0.1 * item6 * item6);
		}
		return fx;
	}
	void initialize(double* x)
	{
		for (int i = 0; i < n; i += 2)
		{
			x[i] = -3;
			x[i + 1] = -1;
		}
	}
	bool  stoppingCondition(double* g) const
	{
		return (infNorm(g, n) < EPS);
	}
};
//-----------------------  45  -----------------------
#include<numeric>
#include<vector>
using namespace std;
class quadrPenalSMG : public problem
{
	double Dev{};
	int i{}, j{};
	vector<vector<double>> A;
public:
	double EPS{};
	long int n{};
	quadrPenalSMG() {}
	quadrPenalSMG(double const tolerance) { EPS = tolerance; n = 400; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "quadrPenalSMG"; }
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
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.;
		Dev = 0.;
		double prod{};
		double sum{};
		double* p;
		for (i = 0; i < n; i++)
			sum += x[i];

		for (i = 0; i < n; i++)
		{
			p = A[i].data();
			prod = std::inner_product(p, p + n, x, double{});
			if (prod > 0.)
			{
				if (Dev < prod) Dev = prod;
				fx += prod * prod;
			}
		}
		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		fx += (sum - 1) * (sum - 1);

		for (i = 0; i < n; i++)
		{
			if (x[i] < 0.)
			{
				if (-x[i] > Dev)
					Dev = -x[i];
				prod = x[i];
				fx += prod * prod;
			}
		}
		return fx / 2;
	}
	void printVector()
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
		for (i = 0; i < n; ++i)
		{
			x[i] = 1. / n;
		}
		A.resize(n);
		for (i = 0; i < n; ++i)
		{
			for (j = 0; j < i; ++j)
				A[i].push_back((2.0 * rand()) / RAND_MAX - 1.0);
			A[i].push_back(0.);
		}
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				A[i].push_back(-1.0 * A[j][i]);
	}
	bool  stoppingCondition(double* g) const
	{
		return (Dev < EPS);
	}
};
//-----------------------  45  -----------------------
#include<numeric>
#include<vector>
using namespace std;
class CubicPenalSMG : public problem
{
	double Dev{};
	int i{}, j{};
	vector<vector<double>> A;
public:
	double EPS{};
	long int n{};
	CubicPenalSMG() {}
	CubicPenalSMG(double const tolerance) { EPS = tolerance; n = 800; }
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "CubicPenalSMG"; }
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

		for (i = 0; i < n; i++)
		{
			g[i] = 0;
			sum += x[i];
		}
		for (i = 0; i < n; i++)
		{
			p = A[i].data();
			prod = std::inner_product(p, p + n, x, double{});
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
		sq = (sum - 1) * (sum - 1) * ((sum < 1) ? (-1) : 1);
		fx += sq * (sum - 1);
		prod = sq;
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
	double value
	(
		double* x
	)
	{
		++ValueEvaluations;
		double fx = 0.;
		Dev = 0.;
		double prod{};
		double sum{};
		double* p;
		double qu{}, sq{};
		int sigh{};

		for (i = 0; i < n; i++)
			sum += x[i];

		for (i = 0; i < n; i++)
		{
			p = A[i].data();
			prod = std::inner_product(p, p + n, x, double{});
			if (prod > 0.)
			{
				if (Dev < prod) Dev = prod;
				qu = prod * prod;
				fx += qu * prod;
			}
		}
		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		sigh = (sum < 1) ? (-1) : 1;
		sq = (sum - 1) * (sum - 1);
		fx += sq * (sum - 1) * sigh;
		prod = sq * sigh;

		for (i = 0; i < n; i++)
		{
			if (x[i] < 0.)
			{
				if (-x[i] > Dev)
					Dev = -x[i];
				qu = x[i];
				sq = qu * qu;
				fx -= sq * qu;
			}
		}

		return fx / 3;
	}
	void printVector()
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
		for (i = 0; i < n; ++i)
		{
			x[i] = 1. / n;
		}
		A.resize(n);
		for (i = 0; i < n; ++i)
		{
			for (j = 0; j < i; ++j)
				A[i].push_back((2.0 * rand()) / RAND_MAX - 1.0);
			A[i].push_back(0.);
		}
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				A[i].push_back(-1.0 * A[j][i]);
	}
	bool  stoppingCondition(double* g) const
	{
		return (Dev < EPS);
	}
};
//-----------------------  46  -----------------------
#include<numeric>
#include<vector>
using namespace std;
class CubPenSMG_Eigen : public problem
{
	double Dev{};
	int i{}, j{};
//	vector<vector<double>> A;
	Matrix <double, Dynamic, Dynamic> A;
	Vector<double, Dynamic> x_vec;		//to store a copy of "x"
	Vector<double, Dynamic> g_vec;		//to store a copy of "g"
	vector<double> product;
public:
	double EPS{};
	long int n{};
	CubPenSMG_Eigen() {}
	CubPenSMG_Eigen(double const tolerance) 
	{ 
		EPS = tolerance; n = 800;
		A.resize(n, n);
		x_vec.resize(n);
		g_vec.resize(n);
	}
	long int virtual getSize()  const { return n; }
	std::string getName() const { return "CubicPenalSMG with Eigen"; }
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

		std::memmove(x_vec.data(), x, n * sizeof(double));
		sum = x_vec.sum();

		g_vec = A * x_vec;		//temporary use of "g_vec"
		double tmp = fabs(g_vec.maxCoeff());
		if (tmp > Dev)
			Dev = tmp;

		x_vec = g_vec.array().cwiseMax(0).pow(3);
		fx = x_vec.sum();
		g_vec = (g_vec.array().cwiseMax(0).pow(2));
		g_vec = g_vec.transpose()* A;
		
		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		sq = (sum - 1) * (sum - 1) * ((sum < 1) ? (-1) : 1);
		fx += sq * (sum - 1);
		prod = sq;
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
	
	double value
	(
		double* x
	)
	{ 
		++ValueEvaluations;
		std::memmove(x_vec.data(), x, n * sizeof(double));
		double fx = 0.;
		Dev = 0.;
		double prod{};
		double sum{};
		double qu{}, sq{};
		int sigh{};

		sum = x_vec.sum();
		x_vec = (A * x_vec).array().cwiseMax(0).pow(3);
		fx = x_vec.sum();

		if (Dev < fabs(sum - 1)) Dev = fabs(sum - 1);
		sigh = (sum < 1) ? (-1) : 1;
		sq = (sum - 1) * (sum - 1);
		fx += sq * (sum - 1) * sigh;
		prod = sq * sigh;

		for (i = 0; i < n; i++)
		{
			if (x[i] < 0.)
			{
				if (-x[i] > Dev)
					Dev = -x[i];
				qu = x[i];
				sq = qu * qu;
				fx -= sq * qu;
			}
		}
		return fx / 3;
	}

	void initialize(double* x)
	{
		for (i = 0; i < n; ++i)
		{
			x[i] = 1. / n;
		}

		for (i = 0; i < n; ++i)
		{
			for (j = 0; j < i; ++j)
				A(i, j) = (2.0 * rand()) / RAND_MAX - 1.0;
			A(i, i) = 0.;
		}

		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				A(i, j) = -1.0 * A(j,i);
	}
	bool  stoppingCondition(double* g) const
	{
		return (Dev < EPS);
	}
};