/*
 * Sample program for CDA5125
 *
 * A generic 3 level feedforward neural network from scratch
 * 
 * Change N0 (input size), N1 (size of the hidden layer),
 *        and N2 (size of the output layer) to change the neural network
 * 
 * driver program train 
 */

#include <iostream>
#include <math.h> 
#include <stdlib.h>

using namespace std;

#define N0  2
#define N1  4
#define N2  2

double IN[N0];
double W0[N0][N1];
double B1[N1];
double HS[N1];
double HO[N1];
double W1[N1][N2];
double B2[N2];
double OS[N2];
double OO[N2];

double err;
double rate = 0.1;

double sigmoid(double x)
{
	return 1/(1+exp(-x));
}


// forward progagation with input: input[N0]
void forward(double *input)
{

        for (int i = 0; i<N0; i++) 
		IN[i] = input[i];

        // compute the weighted sum HS in the hidden layer
        for (int i=0; i<N1; i++) {
		HS[i] = B1[i];
	}
        for (int i=0; i<N1; i++) {
		for (int j=0; j<N0; j++)
			HS[i] += IN[j]*W0[j][i];
	}

        // Comput the output of the hidden layer, HO[N1];

        for (int i=0; i<N1; i++) {
		HO[i] = sigmoid(HS[i]);
	}

        // compute the weighted sum OS in the output layer
        for (int i=0; i<N2; i++) {
		OS[i] = B2[i];
	}
        for (int i=0; i<N2; i++) {
		for (int j=0; j<N1; j++)
			OS[i] += HO[j]*W1[j][i];
	}

        // Comput the output of the output layer, OO[N2];

        for (int i=0; i<N2; i++) {
		OO[i] = sigmoid(OS[i]);
	}
}

double dE_OO[N2];
double dOO_OS[N2];
double dE_OS[N2];
double dE_B2[N2];
double dE_W1[N1][N2];
double dE_HO[N1];
double dHO_HS[N1];
double dE_HS[N1];
double dE_B1[N1];
double dE_W0[N0][N1];

void print_1d(double *a, int size, const char* aa)
{
	for (int i=0; i<size; i++)
		cout << aa << "[" << i << "]=" << a[i] << "\n";
}

void print_01(double a[N0][N1], const char* aa)
{
	for (int i=0; i<N0; i++)
		for (int j=0; j<N1; j++)
			cout << aa << "[" << i << "][" << j 
			     << "]=" << a[i][j]<< "\n";
}

void print_12(double a[N1][N2], const char* aa)
{
	for (int i=0; i<N1; i++)
		for (int j=0; j<N2; j++)
			cout << aa << "[" << i << "][" << j 
			     << "]=" << a[i][j]<< "\n";
}

// 
double backward(double *O, double *Y)
{
        // compute error
	err = 0.0;
        for (int i=0; i<N2; i++) 
		err += (O[i] - Y[i])*(O[i]-Y[i]);
	err = err / N2;

        // compute dE_OO
        for (int i=0; i<N2; i++) 
		dE_OO[i] = (O[i] - Y[i])*2.0/N2;

        // compute dOO_OS = OO dot (1-OO)
        for (int i=0; i<N2; i++)
		dOO_OS[i] = OO[i] * (1.0-OO[i]);

        // compute dE_OS = dE_OO dot dOO_OS
        for (int i=0; i<N2; i++)
		dE_OS[i] = dE_OO[i] * dOO_OS[i];

        // compute dE_B2 = dE_OS
        for (int i=0; i<N2; i++)
		dE_B2[i] = dE_OS[i];

        // compute dE_W1
        for (int i=0; i<N1; i++)
		for (int j = 0; j<N2; j++) 
			dE_W1[i][j] = dE_OS[j]*HO[i];

	// compute dE_HO
	for (int i=0; i<N1; i++) {
		dE_HO[i] = 0;
		for (int j = 0; j<N2; j++)
			dE_HO[i] += dE_OS[j]*W1[i][j];
	}

        // compute dHO_HS = HO dot (1-HO)
        for (int i=0; i<N1; i++)
		dHO_HS[i] = HO[i] * (1-HO[i]);

        // compute dE_HS = dE_HO dot dHO_HS
        for (int i=0; i<N1; i++)
		dE_HS[i] = dE_HO[i] * dHO_HS[i];

        // compute dE_B1 = dE_HS
        for (int i=0; i<N1; i++)
		dE_B1[i] = dE_HS[i];

        // compute dE_W0
        for (int i=0; i<N0; i++)
		for (int j = 0; j<N1; j++) 
			dE_W0[i][j] = dE_HS[j]*IN[i];
	/*
	cout << "err = " << err << "\n";
	print_1d(IN, N0, "IN");
	print_1d(dE_OO, N2, "dE_OO");
	print_1d(dOO_OS, N2, "dOO_OS");
	print_1d(OO, N2, "OO");
	print_1d(dE_OS, N2, "dE_OS");
        print_1d(dE_B2, N2, "dE_B2");
        print_12(dE_W1, "dE_W1");
        print_1d(dE_B1, N1, "dE_B1");
        print_01(dE_W0, "dE_W0");
	*/

        // update W0, W1, B1, B2;

	for (int i=0; i<N0; i++)
		for (int j=0; j<N1; j++)
			W0[i][j] = W0[i][j] - rate * dE_W0[i][j];

	for (int i=0; i<N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];

	for (int i=0; i<N1; i++)
		for (int j=0; j<N2; j++)
			W1[i][j] = W1[i][j] - rate * dE_W1[i][j];

	for (int i=0; i<N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];

}  

double X[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
double Y[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};
//double Y[4][2] = {{0.0, 0.0}, {1.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
	
void train(int iter)
{
	for (int i = 0; i< iter; i++) {
		//int ii = random () % 4;
		int ii = i % 4;
                //int ii= 3;
		forward(&(X[ii][0]));
		backward(OO, &(Y[ii][0]));

		if (i % 10000 == 0) 
			cout << "Iter " << i << ": err =" << err << "\n";
		// break;
	}
}

int main(int argc, char *argv[]) 
{
	// randomize weights
        for (int i = 0; i<N1; i++)
		B1[i] = random()*1.0/RAND_MAX;
        for (int i = 0; i<N0; i++)
		for (int j = 0; j<N1; j++)
			W0[i][j] = random()*1.0/RAND_MAX;
        for (int i = 0; i<N2; i++)
		B2[i] = random()*1.0/RAND_MAX;
        for (int i = 0; i<N1; i++)
		for (int j = 0; j<N2; j++)
			W1[i][j] = random()*1.0/RAND_MAX;
			

	if (argc == 2) train(atoi(argv[1]));
        else train(100000);

	//        cout << "w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << "\n";

        forward(X[0]);
	cout << "(0, 0) -> " << "(" << OO[0] << ", " << OO[1]  << ")\n";
        forward(X[1]);
	cout << "(0, 1) -> " << "(" << OO[0] << ", " << OO[1]  << ")\n";
        forward(X[2]);
	cout << "(1, 0) -> " << "(" << OO[0] << ", " << OO[1]  << ")\n";
        forward(X[3]);
	cout << "(1, 1) -> " << "(" << OO[0] << ", " << OO[1]  << ")\n";
}