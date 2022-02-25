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
#include <random>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <chrono>

using namespace std;
#define N0  784
#define N1  1000
#define N2  500
#define N3  10

#define DEBUG 1
#define HEIGHT 28
#define WIDTH 28

double IN[N0]; // Input Layer
double W0[N0][N1]; //Input to hidden layer1
double B1[N1]; 
double HS_1[N1];
double HO_1[N1];

double W1[N1][N2];
double B2[N2];
double HS_2[N2]; //2nd hidden layer sum
double HO_2[N2]; //2nd hidden layer output

double W2[N2][N3];
double B3[N3];
double OS[N3];
double OO[N3];


typedef unsigned char uchar;
vector<vector <double> >  data_X;
vector<vector <double> >  data_Y;





double err;
double rate = 0.001; //Learning Rate


double scaled_tanh(double x)
{
	double A = 1.7159;
	double B = 0.6666;

	double result;
	result = A * tanh(B * x);

	return result;
}
/*
void flatten_convert_2D(void * input){
	for (int i=0; i < HEIGHT; i++){ 
		for (int j =0; j< WIDTH; j++){
 			scale_input[i * WIDTH + j] = (input[i][j] / 127.5) - 10; //(Scale PI acc to formula PI/127.5 - 10)
		}
 	} 
}
*/
////////////////////////////////Read Image//////////////////////////////////////

void read_mnist_images(string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };


    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
	/**
	i=1 [786]
        i=2 [786]
        i=3 [786]
        ....
        i=60000 [786]

	[[786] [786] [786] .... [786]]
	[0 1 2 .....60000]

	**/
	vector<double> temp;
        for(int ii=0; ii < number_of_images; ii++){
		for(int jj=0; jj < image_size; jj++){
			temp.push_back( (double(_dataset[ii][jj]) / 127.5) - 10 ); //Scale (input[i][j] / 127.5) - 10
		}
		data_X.push_back(temp);
		temp.clear();
	}
	/* For Display
        for(int ii=0; ii < number_of_images; ii++){
		for(int jj=0; jj < image_size; jj++){
			if (jj % 28 == 0){
				cout << endl;
			}
			if (_dataset[ii][jj] == 0){
				cout << ".";
			}
			else{
				cout << "@";
			}
		}
		cout << endl;
	}
	*/
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}



//////////////////////////Read Label//////////////////////////////

void read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
 	vector<double> temp(10, 0.0);
	for (int ii = 0; ii < number_of_labels; ii++){
		temp[(int)_dataset[ii]] = 1.0;
		data_Y.push_back(temp);
		temp[(int)_dataset[ii]] = 0.0;
	} 
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}


// forward progagation with input: input[N0]
void forward(vector<double> input)
{

        for (int i = 0; i<N0; i++) 
		IN[i] = input[i];

        // compute the weighted sum HS in the hidden layer
        for (int i=0; i<N1; i++) {
		HS_1[i] = B1[i];
	}

	int bk = 8; //blocking
	for (int j=0; j<N0; j+=bk){ //loop interchange
		for (int i=0; i<N1; i+=bk){
			for (int jj=j;jj<j+bk;jj++){
				HS_1[i+0] += IN[jj]*W0[jj][i+0];
				HS_1[i+1] += IN[jj]*W0[jj][i+1];
				HS_1[i+2] += IN[jj]*W0[jj][i+2];
				HS_1[i+3] += IN[jj]*W0[jj][i+3];
				HS_1[i+4] += IN[jj]*W0[jj][i+4];
				HS_1[i+5] += IN[jj]*W0[jj][i+5];
				HS_1[i+6] += IN[jj]*W0[jj][i+6];
				HS_1[i+7] += IN[jj]*W0[jj][i+7];
			}	
		}
	}

        // Comput the output of the hidden layer, HO[N1];
        for (int i=0; i<N1; i++) {
		HO_1[i] = scaled_tanh(HS_1[i]);
	}

        // compute the weighted sum HS in the hidden layer
        for (int i=0; i<N2; i++) {
		HS_2[i] = B2[i];
	}

	bk = 20;
	for (int j=0; j<N1; j+=bk) {
		for (int i=0; i<N2; i+=bk){
			for (int jj=j;jj<j+bk;jj++){
				HS_2[i+0] += HO_1[jj]* W1[jj][i+0];
				HS_2[i+1] += HO_1[jj]* W1[jj][i+1];
				HS_2[i+2] += HO_1[jj]* W1[jj][i+2];
				HS_2[i+3] += HO_1[jj]* W1[jj][i+3];
				HS_2[i+4] += HO_1[jj]* W1[jj][i+4];
				HS_2[i+5] += HO_1[jj]* W1[jj][i+5];
				HS_2[i+6] += HO_1[jj]* W1[jj][i+6];
				HS_2[i+7] += HO_1[jj]* W1[jj][i+7];
				HS_2[i+8] += HO_1[jj]* W1[jj][i+8];
				HS_2[i+9] += HO_1[jj]* W1[jj][i+9];
				HS_2[i+10] += HO_1[jj]* W1[jj][i+10];
				HS_2[i+11] += HO_1[jj]* W1[jj][i+11];
				HS_2[i+12] += HO_1[jj]* W1[jj][i+12];
				HS_2[i+13] += HO_1[jj]* W1[jj][i+13];
				HS_2[i+14] += HO_1[jj]* W1[jj][i+14];
				HS_2[i+15] += HO_1[jj]* W1[jj][i+15];
				HS_2[i+16] += HO_1[jj]* W1[jj][i+16];
				HS_2[i+17] += HO_1[jj]* W1[jj][i+17];
				HS_2[i+18] += HO_1[jj]* W1[jj][i+18];
				HS_2[i+19] += HO_1[jj]* W1[jj][i+19];
			}
				
		}
	}
 
        // Comput the output of the hidden layer, HO[N1];
        for (int i=0; i<N2; i++) {
		HO_2[i] = scaled_tanh(HS_2[i]);
	}

        // compute the weighted sum OS in the output layer
        for (int i=0; i<N3; i++) {
		OS[i] = B3[i];
	}

	bk = 10;
        for (int j=0; j<N2; j+=bk) {
		for (int i=0; i<N3; i+=bk)
			for(int jj=j; jj< j+bk; jj++){
				OS[i+0] += HO_2[jj]*W2[jj][i+0];
				OS[i+1] += HO_2[jj]*W2[jj][i+1];
				OS[i+2] += HO_2[jj]*W2[jj][i+2];
				OS[i+3] += HO_2[jj]*W2[jj][i+3];
				OS[i+4] += HO_2[jj]*W2[jj][i+4];
				OS[i+5] += HO_2[jj]*W2[jj][i+5];
				OS[i+6] += HO_2[jj]*W2[jj][i+6];
				OS[i+7] += HO_2[jj]*W2[jj][i+7];
				OS[i+8] += HO_2[jj]*W2[jj][i+8];
				OS[i+9] += HO_2[jj]*W2[jj][i+9];
			}
	}

        // Comput the output of the output layer, OO[N2];
        for (int i=0; i<N3; i++) {
		OO[i] = scaled_tanh(OS[i]);
	}
}

double dE_OO[N3];
double dOO_OS[N3];
double dE_OS[N3];
double dE_B3[N3];
double dE_W2[N2][N3];


double dE_HO_2[N2];
double dHO_HS_2[N2];
double dE_HS_2[N2];
double dE_B2[N2];
double dE_W1[N1][N2];

double dE_HO_1[N1];
double dHO_HS_1[N1];
double dE_HS_1[N1];
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


double backward(double *O, vector<double> Y)
{
        // compute error
	double A = 1.7159;
	double B = 0.6666;
	err = 0.0;

        for (int i=0; i<N3; i++) 
		err += (O[i] - Y[i])*(O[i]-Y[i]);
	err = err / N3;

	double temp_OOi;
        for (int i=0; i<N3; i++){
		/*OO[i] = AtanH(Bx)
 * 		A * B (1 - (tanh(Bx) * tanh(Bx)))
 * 		B * (A - (A * tanhx(Bx) * tanhx(Bx)))
 * 		B * (A - OO[i] * OO[i])
 * 		A * B (1 - (tanh(Bx) * tanh(Bx)))*/

		dE_OO[i] = (O[i] - Y[i])*2.0/N3; //loop fusion
		temp_OOi =  OO[i];
		dOO_OS[i] = B * (A - (temp_OOi * temp_OOi / A)); //A * B (1 - (tanh(Bx) * tanh(Bx)))
		dE_OS[i] = dE_OO[i] * dOO_OS[i];
		dE_B3[i] = dE_OS[i];
	}

        for (int i=0; i<N2; i++)
		for (int j = 0; j<N3; j++) 
			dE_W2[i][j] = dE_OS[j]*HO_2[i];

	for (int i=0; i<N2; i++) {
		dE_HO_2[i] = 0;
		for (int j = 0; j<N3; j++)
			dE_HO_2[i] += dE_OS[j]*W2[i][j];
	}


        // compute dHO_HS_2 = HO_2 dot (1-HO_2)
        for (int i=0; i<N2; i++)
		dHO_HS_2[i] = B * (A - (HO_2[i] * HO_2[i] / A));

        // compute dE_HS_2 = dE_HO_2 dot dHO_HS_2
        for (int i=0; i<N2; i++)
		dE_HS_2[i] = dE_HO_2[i] * dHO_HS_2[i];

        // compute dE_B2 = dE_HS_2
        for (int i=0; i<N2; i++)
		dE_B2[i] = dE_HS_2[i];


	int bk = 20;
	for (int i=0; i<N1; i+=bk){
		for (int j = 0; j<N2; j+=bk){
			for(int ii=i; ii< i+bk; ii++){
				dE_W1[ii][j+0] = dE_HS_2[j+0]*HO_1[ii];
				dE_W1[ii][j+1] = dE_HS_2[j+1]*HO_1[ii];
				dE_W1[ii][j+2] = dE_HS_2[j+2]*HO_1[ii];
				dE_W1[ii][j+3] = dE_HS_2[j+3]*HO_1[ii];
				dE_W1[ii][j+4] = dE_HS_2[j+4]*HO_1[ii];	
				
				dE_W1[ii][j+5] = dE_HS_2[j+5]*HO_1[ii];
                                dE_W1[ii][j+6] = dE_HS_2[j+6]*HO_1[ii];
                                dE_W1[ii][j+7] = dE_HS_2[j+7]*HO_1[ii];
                                dE_W1[ii][j+8] = dE_HS_2[j+8]*HO_1[ii];
                                dE_W1[ii][j+9] = dE_HS_2[j+9]*HO_1[ii];

				dE_W1[ii][j+10] = dE_HS_2[j+10]*HO_1[ii];
                                dE_W1[ii][j+11] = dE_HS_2[j+11]*HO_1[ii];
                                dE_W1[ii][j+12] = dE_HS_2[j+12]*HO_1[ii];
                                dE_W1[ii][j+13] = dE_HS_2[j+13]*HO_1[ii];
                                dE_W1[ii][j+14] = dE_HS_2[j+14]*HO_1[ii];

				dE_W1[ii][j+15] = dE_HS_2[j+15]*HO_1[ii];
                                dE_W1[ii][j+16] = dE_HS_2[j+16]*HO_1[ii];
                                dE_W1[ii][j+17] = dE_HS_2[j+17]*HO_1[ii];
                                dE_W1[ii][j+18] = dE_HS_2[j+18]*HO_1[ii];
                                dE_W1[ii][j+19] = dE_HS_2[j+19]*HO_1[ii];

			}
		}
	}

	// compute dE_HO_1
	double temp_dE_HO_1;
        bk=20;
	for (int i=0; i<N1; i++) {
		temp_dE_HO_1=0.0;
		for (int j = 0; j<N2; j+=bk){
				
				temp_dE_HO_1 += dE_HS_2[j+0]*W1[i][j+0];
				temp_dE_HO_1 += dE_HS_2[j+1]*W1[i][j+1];
				temp_dE_HO_1 += dE_HS_2[j+2]*W1[i][j+2];
				temp_dE_HO_1 += dE_HS_2[j+3]*W1[i][j+3];
				temp_dE_HO_1 += dE_HS_2[j+4]*W1[i][j+4];
				temp_dE_HO_1 += dE_HS_2[j+5]*W1[i][j+5];
				temp_dE_HO_1 += dE_HS_2[j+6]*W1[i][j+6];
				temp_dE_HO_1 += dE_HS_2[j+7]*W1[i][j+7];
				temp_dE_HO_1 += dE_HS_2[j+8]*W1[i][j+8];
				temp_dE_HO_1 += dE_HS_2[j+9]*W1[i][j+9];
				temp_dE_HO_1 += dE_HS_2[j+10]*W1[i][j+10];
				temp_dE_HO_1 += dE_HS_2[j+11]*W1[i][j+11];
				temp_dE_HO_1 += dE_HS_2[j+12]*W1[i][j+12];
				temp_dE_HO_1 += dE_HS_2[j+13]*W1[i][j+13];
				temp_dE_HO_1 += dE_HS_2[j+14]*W1[i][j+14];
				temp_dE_HO_1 += dE_HS_2[j+15]*W1[i][j+15];
				temp_dE_HO_1 += dE_HS_2[j+16]*W1[i][j+16];
				temp_dE_HO_1 += dE_HS_2[j+17]*W1[i][j+17];
				temp_dE_HO_1 += dE_HS_2[j+18]*W1[i][j+18];
				temp_dE_HO_1 += dE_HS_2[j+19]*W1[i][j+19];
				
		}
		dE_HO_1[i] = temp_dE_HO_1;
	}

        // compute dHO_HS_1 = HO_1 dot (1-HO_1)
        for (int i=0; i<N1; i++)
		dHO_HS_1[i] = B * (A - (HO_1[i] * HO_1[i] / A));

        // compute dE_HS_1 = dE_HO_1 dot dHO_HS_1
        for (int i=0; i<N1; i++)
		dE_HS_1[i] = dE_HO_1[i] * dHO_HS_1[i];

        // compute dE_B1 = dE_HS_1
        for (int i=0; i<N1; i++)
		dE_B1[i] = dE_HS_1[i];
        
	bk = 16;
	for (int i=0; i<N0; i+=bk){
			for (int j = 0; j<N1; j+=bk){
				for(int ii=i; ii<i+bk; ii++){
					dE_W0[ii][j+0] = dE_HS_1[j+0]*IN[ii];
					dE_W0[ii][j+1] = dE_HS_1[j+1]*IN[ii];
					dE_W0[ii][j+2] = dE_HS_1[j+2]*IN[ii];
					dE_W0[ii][j+3] = dE_HS_1[j+3]*IN[ii];

					dE_W0[ii][j+4] = dE_HS_1[j+4]*IN[ii];
					dE_W0[ii][j+5] = dE_HS_1[j+5]*IN[ii];
					dE_W0[ii][j+6] = dE_HS_1[j+6]*IN[ii];
					dE_W0[ii][j+7] = dE_HS_1[j+7]*IN[ii];

					dE_W0[ii][j+8] = dE_HS_1[j+8]*IN[ii];
					dE_W0[ii][j+9] = dE_HS_1[j+9]*IN[ii];
					dE_W0[ii][j+10] = dE_HS_1[j+10]*IN[ii];
					dE_W0[ii][j+11] = dE_HS_1[j+11]*IN[ii];

					dE_W0[ii][j+12] = dE_HS_1[j+12]*IN[ii];
					dE_W0[ii][j+13] = dE_HS_1[j+13]*IN[ii];
					dE_W0[ii][j+14] = dE_HS_1[j+14]*IN[ii];
					dE_W0[ii][j+15] = dE_HS_1[j+15]*IN[ii];

				}
			}
		}
 
        // update W0, W1, W2, B1, B2, B3;

	bk=8;
	for (int i=0; i<N0; i+=bk)
                for (int j=0; j<N1; j+=bk)
			for(int ii=i; ii< i+bk; ii++){
                        	W0[ii][j+0] = W0[ii][j+0] - rate * dE_W0[ii][j+0];
				 W0[ii][j+1] = W0[ii][j+1] - rate * dE_W0[ii][j+1];
				 W0[ii][j+2] = W0[ii][j+2] - rate * dE_W0[ii][j+2];
				 W0[ii][j+3] = W0[ii][j+3] - rate * dE_W0[ii][j+3];

				W0[ii][j+4] = W0[ii][j+4] - rate * dE_W0[ii][j+4];
                                 W0[ii][j+5] = W0[ii][j+5] - rate * dE_W0[ii][j+5];
                                 W0[ii][j+6] = W0[ii][j+6] - rate * dE_W0[ii][j+6];
                                 W0[ii][j+7] = W0[ii][j+7] - rate * dE_W0[ii][j+7];
			}

	for (int i=0; i<N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];


	bk=20;
	double temp_w1;
	for (int i=0; i<N1; i++){
		for (int j=0; j<N2; j+=bk){
			
				W1[i][j+0] = W1[i][j+0] - rate * dE_W1[i][j+0];
				W1[i][j+1] = W1[i][j+1] - rate * dE_W1[i][j+1];
				W1[i][j+2] = W1[i][j+2] - rate * dE_W1[i][j+2];
				W1[i][j+3] = W1[i][j+3] - rate * dE_W1[i][j+3];
				W1[i][j+4] = W1[i][j+4] - rate * dE_W1[i][j+4];

				W1[i][j+5] = W1[i][j+5] - rate * dE_W1[i][j+5];
				W1[i][j+6] = W1[i][j+6] - rate * dE_W1[i][j+6];
				W1[i][j+7] = W1[i][j+7] - rate * dE_W1[i][j+7];
				W1[i][j+8] = W1[i][j+8] - rate * dE_W1[i][j+8];
				W1[i][j+9] = W1[i][j+9] - rate * dE_W1[i][j+9];

				W1[i][j+10] = W1[i][j+10] - rate * dE_W1[i][j+10];
				W1[i][j+11] = W1[i][j+11] - rate * dE_W1[i][j+11];
				W1[i][j+12] = W1[i][j+12] - rate * dE_W1[i][j+12];
				W1[i][j+13] = W1[i][j+13] - rate * dE_W1[i][j+13];
				W1[i][j+14] = W1[i][j+14] - rate * dE_W1[i][j+14];

				W1[i][j+15] = W1[i][j+15] - rate * dE_W1[i][j+15];
				W1[i][j+16] = W1[i][j+16] - rate * dE_W1[i][j+16];
				W1[i][j+17] = W1[i][j+17] - rate * dE_W1[i][j+17];
				W1[i][j+18] = W1[i][j+18] - rate * dE_W1[i][j+18];
				W1[i][j+19] = W1[i][j+19] - rate * dE_W1[i][j+19];
			
		}
	}

	for (int i=0; i<N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];
	
	for (int i=0; i<N2; i++)
		for (int j=0; j<N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i=0; i<N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];

}  

	

void train(int iter)
{
	
	time_t t; // t passed as argument in function time()
   	struct tm * tt; // decalring variable for localtime()
   	time (&t); //passing argument to time()
   	tt = localtime(&t);
	ofstream myfile;
	myfile.open ("output_3_6_400.txt");
	int ii,jj;
	int i = 0;
        double constant = 1.5;
	for (int kk = 0; kk < iter; kk++) {
		
		ii = kk % data_X.size();
		forward(data_X[ii]);
		backward(OO, data_Y[ii]);
	        

		if (kk % 10000 == 0) {

			int winner = 0;
			double max = OO[0];

			for (int i = 0; i < N3; i++){
				if (data_Y[ii][i] ) {
					jj = i;
				}
			}
			for (int i = 0; i < N3; i++){
				if (OO[i] > max) {
					max = OO[i];
					winner = i;
				}
			} 

			myfile << "[Train] Iter " << kk << ": err =" << err << ", Y = " << jj <<"\n";			
			for (int i = 0; i < N3; i++)
				myfile << "OO[" <<i<< "] = " << OO[i] << "\n";
			
			myfile << "Max OO["<<winner<<"] VS " << "Y = " << jj<<"\n";
			time (&t);
			myfile << asctime(localtime(&t)) << "\n";
			myfile.flush();
		}
		if ( kk % 50000 == 0) {
			if (rate > 0.000001){
				myfile << "Learning rate changed from " << rate << " to " << rate / (double) constant << "\n";
				rate = rate / (double) constant;
				myfile.flush();
			}
		}
	}
	myfile.close();
}

void test(vector<double> data){
        forward(data);
	for (int j = 0; j < data.size(); j++){
			if (j % 28 == 0)
				cout << endl;
			double mm =  (data[j]+10)*127.5;
			if (mm == 0.0){
				cout << ".";
			}
			else{
				cout << "@";
			}
		}
	cout << endl;
        for (int i = 0; i < 10; i++){
		cout << OO[i] << "\t";
	}
	cout << endl;
} 

int main(int argc, char *argv[]) 
{


	int number_of_images = 0, size = 0;
	read_mnist_images("./data/train_data/input/train-images-idx3-ubyte", number_of_images, size);
	cout << "IMAGE READ COMPLETE" << endl;	
	

	int number_of_labels = 0;
	read_mnist_labels("./data/train_data/output/train-labels-idx1-ubyte", number_of_labels);
	cout << "LABEL READ COMPLETE" << endl;
	
	// randomize weights
	int seed = 30;
	default_random_engine generator(seed); // rd() provides a random seed
	uniform_real_distribution<double> distribution(-0.05, 0.05);
        
	for (int i = 0; i<N1; i++)
		B1[i] = distribution(generator);
        for (int i = 0; i<N0; i++)
		for (int j = 0; j<N1; j++)
			W0[i][j] = distribution(generator);
        
	for (int i = 0; i<N2; i++)
		B2[i] = distribution(generator);
        for (int i = 0; i<N1; i++)
		for (int j = 0; j<N2; j++)
			W1[i][j] = distribution(generator);		
        
	for (int i = 0; i<N3; i++)
		B3[i] = distribution(generator);
        for (int i = 0; i<N2; i++)
		for (int j = 0; j<N3; j++)
			W2[i][j] = distribution(generator);		
	cout << "WEIGHT DISTRIBUTION COMPLETE" << endl;	

	if (argc == 2) train(atoi(argv[1]));
        else train(2500000);


	/* For Testing */
	int m = 4;
	test(data_X[m]);


}
