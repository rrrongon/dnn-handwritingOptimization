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

using namespace std;
#define N0  784
#define N1  1
#define N2  2
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
double rate = 0.1; //Learning Rate


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
//////
//Read Image

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


///////

/////

//Read Label

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
        for (int i=0; i<N1; i++) {
		for (int j=0; j<N0; j++)
			HS_1[i] += IN[j]*W0[j][i];
	}
        
        // Comput the output of the hidden layer, HO[N1];

        for (int i=0; i<N1; i++) {
		HO_1[i] = scaled_tanh(HS_1[i]);
	}

        // compute the weighted sum HS in the hidden layer
        for (int i=0; i<N2; i++) {
		HS_2[i] = B2[i];
	}
        for (int i=0; i<N2; i++) {
		for (int j=0; j<N1; j++)
			HS_2[i] += HO_1[j]*W1[j][i];
	}
        
        // Comput the output of the hidden layer, HO[N1];

        for (int i=0; i<N2; i++) {
		HO_2[i] = scaled_tanh(HS_2[i]);
	}



        // compute the weighted sum OS in the output layer
        for (int i=0; i<N3; i++) {
		OS[i] = B3[i];
	}
        for (int i=0; i<N3; i++) {
		for (int j=0; j<N2; j++)
			OS[i] += HO_2[j]*W2[j][i];
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

// 

double backward(double *O, vector<double> Y)
{
        // compute error
	double A = 1.7159;
	double B = 0.6666;
	err = 0.0;

        for (int i=0; i<N3; i++) 
		err += (O[i] - Y[i])*(O[i]-Y[i]);
	err = err / N3;

        // compute dE_OO
        for (int i=0; i<N3; i++) 
		dE_OO[i] = (O[i] - Y[i])*2.0/N3;

        // compute dOO_OS = OO dot (1-OO)
        for (int i=0; i<N3; i++)
		//OO[i] = AtanH(Bx)
		//
		// A * B (1 - (tanh(Bx) * tanh(Bx)))
		//B * (A - (A * tanhx(Bx) * tanhx(Bx)))
		//B * (A - OO[i] * OO[i])
		dOO_OS[i] = B * (A - (A * OO[i] * OO[i])); //A * B (1 - (tanh(Bx) * tanh(Bx)))

        // compute dE_OS = dE_OO dot dOO_OS
        for (int i=0; i<N3; i++)
		dE_OS[i] = dE_OO[i] * dOO_OS[i];

        // compute dE_B3 = dE_OS
        for (int i=0; i<N3; i++)
		dE_B3[i] = dE_OS[i];

        // compute dE_W2
        for (int i=0; i<N2; i++)
		for (int j = 0; j<N3; j++) 
			dE_W2[i][j] = dE_OS[j]*HO_2[i];

	// compute dE_HO_2
	for (int i=0; i<N2; i++) {
		dE_HO_2[i] = 0;
		for (int j = 0; j<N3; j++)
			dE_HO_2[i] += dE_OS[j]*W2[i][j];
	}

        // compute dHO_HS_2 = HO_2 dot (1-HO_2)
        for (int i=0; i<N2; i++)
		dHO_HS_2[i] = B * (A - (A * HO_2[i] *HO_2[i]));

        // compute dE_HS_2 = dE_HO_2 dot dHO_HS_2
        for (int i=0; i<N2; i++)
		dE_HS_2[i] = dE_HO_2[i] * dHO_HS_2[i];

        // compute dE_B2 = dE_HS_2
        for (int i=0; i<N2; i++)
		dE_B2[i] = dE_HS_2[i];

////////////////////////////////

        // compute dE_W1
        for (int i=0; i<N1; i++)
		for (int j = 0; j<N2; j++) 
			dE_W1[i][j] = dE_HS_2[j]*HO_1[i];

	// compute dE_HO_1
	for (int i=0; i<N1; i++) {
		dE_HO_1[i] = 0;
		for (int j = 0; j<N2; j++)
			dE_HO_1[i] += dE_HS_2[j]*W1[i][j];
	}

        // compute dHO_HS_1 = HO_1 dot (1-HO_1)
        for (int i=0; i<N1; i++)
		dHO_HS_1[i] = B * (A - (A * HO_1[i] *HO_1[i]));

        // compute dE_HS_1 = dE_HO_1 dot dHO_HS_1
        for (int i=0; i<N1; i++)
		dE_HS_1[i] = dE_HO_1[i] * dHO_HS_1[i];

        // compute dE_B1 = dE_HS_1
        for (int i=0; i<N1; i++)
		dE_B1[i] = dE_HS_1[i];
        
	// compute dE_W0
        for (int i=0; i<N0; i++)
		for (int j = 0; j<N1; j++) 
			dE_W0[i][j] = dE_HS_1[j]*IN[i];
	
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
	

        // update W0, W1, W2, B1, B2, B3;

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
	

	for (int i=0; i<N2; i++)
		for (int j=0; j<N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i=0; i<N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];

}  


//double X[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
//double Y[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};
//double Y[4][2] = {{0.0, 0.0}, {1.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
	
void train(int iter)
{
	for (int i = 0; i< iter; i++) {
		//int ii = random () % data_X.size();
		int ii = i % data_X.size();
                //int ii= 3;
		forward(data_X[ii]);
	        cout <<"Iter : " << i << "FORWARD PROPAGATION" << endl;	
		backward(OO, data_Y[ii]);
	        cout <<"Iter : " << i << "BACKWARD PROPAGATION" << endl;	

		if (i % 10 == 0) 
			cout << "Iter " << i << ": err =" << err << "\n";
		// break;
	}
}

int main(int argc, char *argv[]) 
{


	int number_of_images = 0, size = 0;
	read_mnist_images("/Users/saptarshibhowmik/Documents/CDA_5125/project_1/data/train_data/input/train-images-idx3-ubyte", number_of_images, size);
	cout << "IMAGE READ COMPLETE" << endl;	
	

	int number_of_labels = 0;
	read_mnist_labels("/users/saptarshibhowmik/documents/cda_5125/project_1/data/train_data/output/train-labels-idx1-ubyte", number_of_labels);
		


	cout << "LABEL READ COMPLETE" << endl;
        /*
	for (int i = 0; i < data_X.size(); i++){
		for (int j = 0; j < data_X[i].size(); j++){
			if (j % 28 == 0)
				cout << endl;
			double mm =  (data_X[i][j]+10)*127.5;
			if (mm == 0.0){
				cout << ".";
			}
			else{
				cout << "@";
			}
		}
		cout << endl;
		cout << "Label : ";
		for (int k = 0; k < data_Y[i].size(); k++){
			cout << data_Y[i][k] << "\t";
		}
		cout << endl;
	}*/
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
			
	cout << "WEIGHT DISTRIBUTION COMPLETE" << endl;	

	if (argc == 2) train(atoi(argv[1]));
        else train(10);

	//        cout << "w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << "\n";
        int m = 4;
        forward(data_X[m]);
	for (int j = 0; j < data_X[m].size(); j++){
			if (j % 28 == 0)
				cout << endl;
			double mm =  (data_X[m][j]+10)*127.5;
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
/*
	cout << "(0, 0) -> " << "(" << OO[0] << ", " << OO[1]  << ")\n";
        forward(X[1]);
	cout << "(0, 1) -> " << "(" << OO[0] << ", " << OO[1]  << ")\n";
        forward(X[2]);
	cout << "(1, 0) -> " << "(" << OO[0] << ", " << OO[1]  << ")\n";
        forward(X[3]);
	cout << "(1, 1) -> " << "(" << OO[0] << ", " << OO[1]  << ")\n";
*/
}
