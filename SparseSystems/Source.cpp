#include"Solvers.h"

//Uncomment one of them
int main()
{
    //From Eigen
    //	conjugate_gradient_Eigen();

    //Experimental, it does the same as cg, but without the formula
    //	SteepestDescent_2d(1e-6);

    //Usual conjugate gradient for quadratic systems, with Eigen
    cg_usual(1e-6);
}
