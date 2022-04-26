/**
 * @file project4.cpp
 * @author  Rubayet Rongon, Saptarshi Bhowmik (bhowmik@cs.fsu.edu, rongon@cs.fsu.edu)
 * @brief
 * @version 0.1
 * @date 2022-04-12
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <emmintrin.h>
#include <time.h>
#include <stdio.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
using namespace std;

long long int print_duration(struct timespec *b, struct timespec *c)
{
	long long r = c->tv_nsec - b->tv_nsec;
	r += ((long long)(c->tv_sec - b->tv_sec)) * 1000000000;
	return r;
}

int nearest_lowest_divisor(int layer_size, int world_size, int rank)
{
	if (rank > layer_size)
	{
		return 0;
	}
	else
	{
		if (layer_size % world_size == 0)
		{
			return 1;
		}
		else
		{
			int nl;
			for (int i = world_size; i > 0; i--)
			{
				if ((layer_size % i == 0))
				{
					nl = i;
					break;
				}
			}
			if (0 <= rank && rank <= (nl - 1))
				return 1;
			else
				return 0;
		}
	}
}
// mpirun -n 502
#define N0 784
#define N1 1000
#define N2 500
#define N3 10

#define DEBUG 1
#define HEIGHT 28
#define WIDTH 28

double IN[N0];	   // Input Layer
double W0[N0][N1]; // Input to hidden layer1
double B1[N1];
double HS_1[N1];
double HO_1[N1];

double W1[N1][N2];
double B2[N2];
double HS_2[N2]; // 2nd hidden layer sum
double HO_2[N2]; // 2nd hidden layer output

double W2[N2][N3];
double B3[N3];
double OS[N3];
double OO[N3];

char sub_N0;
char sub_N2;

MPI_Comm N0_comm;
MPI_Comm N2_comm;

int rank_N2;
int rank_N0;
int size_N2;
int size_N0;

int size_N0_A;
int size_N2_A;

int comm_size_A_N2 = 0;

int my_rank;
int size;

/**
 * @brief The LOOP Preprocessor is used for determining which mode the code will optimize
 * 			1 = General LOOP based optimization
 * 			2 = NO optimmization
 * 			3 = OpenMP parallelized
 * 			4 = MPI parallelized
 *
 */
#define LOOP 5

double loop_HS_1 = 0;
double loop_HS_2 = 0;

double loop_OS = 0;

double loop_W0 = 0;
double loop_W1 = 0;
double loop_W2 = 0;

double loop_dE_HO_2 = 0;
double loop_dE_HO_1 = 0;

double loop_dE_W2 = 0;
double loop_dE_W1 = 0;
double loop_dE_W0 = 0;

double loop_dE_OS_dE_B3 = 0;
double loop_trainTime = 0;

double start_time, end_time, start_time_HS_1, end_time_HS_1, start_time_HS_2, end_time_HS_2;

typedef unsigned char uchar;
vector<vector<double> > data_X;
vector<vector<double> > data_Y;
vector<int> data_pos;
struct timespec bb, ee, pp, qq, itbb, itee;

double err = 0.0;
double rate = 0.0001; // Learning Rate

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double scaled_tanh(double x)
{
	double A = 1.7159;
	double B = 0.6666;

	double result;
	result = A * tanh(B * x);

	return result;
}

/**
 * @brief The function is used to read image data
 *
 * @param full_path
 * @param number_of_images
 * @param image_size
 */

void read_mnist_images(string full_path, int &number_of_images, int &image_size)
{
	auto reverseInt = [](int i)
	{
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	ifstream file(full_path, ios::binary);

	if (file.is_open())
	{
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051)
			throw runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		uchar **_dataset = new uchar *[number_of_images];
		for (int i = 0; i < number_of_images; i++)
		{
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
		for (int ii = 0; ii < number_of_images; ii++)
		{
			for (int jj = 0; jj < image_size; jj++)
			{
				temp.push_back((double(_dataset[ii][jj]) / 127.5) - 1); // Scale (input[i][j] / 127.5) - 10
			}
			data_X.push_back(temp);
			temp.clear();
		}
	}
	else
	{
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}

/**
 * @brief The function is used to read image label
 *
 * @param full_path
 * @param number_of_labels
 */

void read_mnist_labels(string full_path, int &number_of_labels)
{
	auto reverseInt = [](int i)
	{
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open())
	{
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049)
			throw runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		uchar *_dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++)
		{
			file.read((char *)&_dataset[i], 1);
		}

		for (int ii = 0; ii < number_of_labels; ii++)
		{
			int digit = (int)_dataset[ii];
			if (digit < N3)
			{
				vector<double> temp(N3, -1.7159);
				temp[(int)_dataset[ii]] = 1.7159;
				data_Y.push_back(temp);
				data_pos.push_back(ii);
			}
			else
			{
				vector<double> temp(N3, -1.7159);
				data_Y.push_back(temp);
			}
		}
	}
	else
	{
		throw runtime_error("Unable to open file `" + full_path + "`!");
	}
}

/**
 * @brief The function is used to process the forward propagation of a neural network
 *
 */
#if LOOP == 1
void forward(vector<double> input)
{
	for (int i = 0; i < N0; i++)
		IN[i] = input[i];

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N1; i++)
	{
		HS_1[i] = B1[i];
	}

	int bk = 8;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int j = 0; j < N0; j += bk)
	{ // loop interchange
		for (int i = 0; i < N1; i += bk)
		{
			for (int jj = j; jj < j + bk; jj++)
			{
				HS_1[i + 0] += IN[jj] * W0[jj][i + 0];
				HS_1[i + 1] += IN[jj] * W0[jj][i + 1];
				HS_1[i + 2] += IN[jj] * W0[jj][i + 2];
				HS_1[i + 3] += IN[jj] * W0[jj][i + 3];
				HS_1[i + 4] += IN[jj] * W0[jj][i + 4];
				HS_1[i + 5] += IN[jj] * W0[jj][i + 5];
				HS_1[i + 6] += IN[jj] * W0[jj][i + 6];
				HS_1[i + 7] += IN[jj] * W0[jj][i + 7];
			}
		}
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_HS_1 += print_duration(&bb, &ee);

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N1; i++)
	{
		HO_1[i] = scaled_tanh(HS_1[i]);
		// HO_1[i] = sigmoid(HS_1[i]);
	}

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N2; i++)
	{
		HS_2[i] = B2[i];
	}

	bk = 20;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int j = 0; j < N1; j += bk)
	{
		for (int i = 0; i < N2; i += bk)
		{
			for (int jj = j; jj < j + bk; jj++)
			{
				HS_2[i + 0] += HO_1[jj] * W1[jj][i + 0];
				HS_2[i + 1] += HO_1[jj] * W1[jj][i + 1];
				HS_2[i + 2] += HO_1[jj] * W1[jj][i + 2];
				HS_2[i + 3] += HO_1[jj] * W1[jj][i + 3];
				HS_2[i + 4] += HO_1[jj] * W1[jj][i + 4];
				HS_2[i + 5] += HO_1[jj] * W1[jj][i + 5];
				HS_2[i + 6] += HO_1[jj] * W1[jj][i + 6];
				HS_2[i + 7] += HO_1[jj] * W1[jj][i + 7];
				HS_2[i + 8] += HO_1[jj] * W1[jj][i + 8];
				HS_2[i + 9] += HO_1[jj] * W1[jj][i + 9];
				HS_2[i + 10] += HO_1[jj] * W1[jj][i + 10];
				HS_2[i + 11] += HO_1[jj] * W1[jj][i + 11];
				HS_2[i + 12] += HO_1[jj] * W1[jj][i + 12];
				HS_2[i + 13] += HO_1[jj] * W1[jj][i + 13];
				HS_2[i + 14] += HO_1[jj] * W1[jj][i + 14];
				HS_2[i + 15] += HO_1[jj] * W1[jj][i + 15];
				HS_2[i + 16] += HO_1[jj] * W1[jj][i + 16];
				HS_2[i + 17] += HO_1[jj] * W1[jj][i + 17];
				HS_2[i + 18] += HO_1[jj] * W1[jj][i + 18];
				HS_2[i + 19] += HO_1[jj] * W1[jj][i + 19];
			}
		}
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_HS_2 += print_duration(&bb, &ee);

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N2; i++)
	{
		HO_2[i] = scaled_tanh(HS_2[i]);
		// HO_2[i] = sigmoid(HS_2[i]);
	}

	// compute the weighted sum OS in the output layer
	for (int i = 0; i < N3; i++)
	{
		OS[i] = B3[i];
	}

	bk = 10;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int j = 0; j < N2; j += bk)
	{
		for (int i = 0; i < N3; i += bk)
		{
			for (int jj = j; jj < j + bk; jj++)
			{
				OS[i + 0] += HO_2[jj] * W2[jj][i + 0];
				OS[i + 1] += HO_2[jj] * W2[jj][i + 1];
				OS[i + 2] += HO_2[jj] * W2[jj][i + 2];
				OS[i + 3] += HO_2[jj] * W2[jj][i + 3];
				OS[i + 4] += HO_2[jj] * W2[jj][i + 4];
				OS[i + 5] += HO_2[jj] * W2[jj][i + 5];
				OS[i + 6] += HO_2[jj] * W2[jj][i + 6];
				OS[i + 7] += HO_2[jj] * W2[jj][i + 7];
				OS[i + 8] += HO_2[jj] * W2[jj][i + 8];
				OS[i + 9] += HO_2[jj] * W2[jj][i + 9];
			}
		}
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_OS += print_duration(&bb, &ee);

	// Compute the output of the output layer, OO[N2];
	for (int i = 0; i < N3; i++)
	{
		OO[i] = scaled_tanh(OS[i]);
		// OO[i] = sigmoid(OS[i]);
	}
	cout << "In LOOP" << endl;
}
#elif LOOP == 2
void forward(vector<double> input)
{

	for (int i = 0; i < N0; i++)
		IN[i] = input[i];

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N1; i++)
	{
		HS_1[i] = B1[i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N1; i++)
	{
		for (int j = 0; j < N0; j++)
			HS_1[i] += IN[j] * W0[j][i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_HS_1 += print_duration(&bb, &ee);

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N1; i++)
	{
		HO_1[i] = scaled_tanh(HS_1[i]);
		// HO_1[i] = sigmoid(HS_1[i]);
	}

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N2; i++)
	{
		HS_2[i] = B2[i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N2; i++)
	{
		for (int j = 0; j < N1; j++)
			HS_2[i] += HO_1[j] * W1[j][i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_HS_2 += print_duration(&bb, &ee);

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N2; i++)
	{
		HO_2[i] = scaled_tanh(HS_2[i]);
		// HO_2[i] = sigmoid(HS_2[i]);
	}

	// compute the weighted sum OS in the output layer
	for (int i = 0; i < N3; i++)
	{
		OS[i] = B3[i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N3; i++)
	{
		for (int j = 0; j < N2; j++)
			OS[i] += HO_2[j] * W2[j][i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_OS += print_duration(&bb, &ee);

	// Comput the output of the output layer, OO[N2];
	for (int i = 0; i < N3; i++)
	{
		OO[i] = scaled_tanh(OS[i]);
		// OO[i] = sigmoid(OS[i]);
	}
	// cout << "In normal" << endl;
}

#elif LOOP == 3
void forward(vector<double> input)
{

	for (int i = 0; i < N0; i++)
		IN[i] = input[i];

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N1; i++)
	{
		HS_1[i] = B1[i];
	}

#pragma omp parallel for num_threads(12)
	for (int i = 0; i < N1; i++)
	{
		for (int j = 0; j < N0; j++)
			HS_1[i] += IN[j] * W0[j][i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_HS_1 += print_duration(&bb, &ee);
	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N1; i++)
	{
		HO_1[i] = scaled_tanh(HS_1[i]);
	}

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N2; i++)
	{
		HS_2[i] = B2[i];
	}

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
#pragma omp parallel for num_threads(12)
	for (int i = 0; i < N2; i++)
	{
		for (int j = 0; j < N1; j++)
			HS_2[i] += HO_1[j] * W1[j][i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_HS_2 += print_duration(&bb, &ee);

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N2; i++)
	{
		HO_2[i] = scaled_tanh(HS_2[i]);
	}

	// compute the weighted sum OS in the output layer
	for (int i = 0; i < N3; i++)
	{
		OS[i] = B3[i];
	}

	int bk = 10;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &pp);
	for (int i = 0; i < N3; i++)
	{
		for (int j = 0; j < N2; j++)
			OS[i] += HO_2[j] * W2[j][i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &qq);
	loop_OS += print_duration(&pp, &qq);

	// Comput the output of the output layer, OO[N2];
	for (int i = 0; i < N3; i++)
	{
		OO[i] = scaled_tanh(OS[i]);
	}
	cout << "In OMP" << endl;
}
#elif LOOP == 4
void forward(vector<double> input)
{
	int partition_N1 = N1 / size;
	int start_N1 = rank * partition_N1;
	int end_N1 = start_N1 + partition_N1;

	int partition_N2 = N2 / size;
	int start_N2 = rank * partition_N2;
	int end_N2 = start_N2 + partition_N2;

	int partition_N3 = N3 / size;
	int start_N3 = rank * partition_N3;
	int end_N3 = start_N3 + partition_N3;

	for (int i = 0; i < N0; i++)
		IN[i] = input[i];

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N1; i++)
	{
		HS_1[i] = B1[i];
	}

	double hs_1[partition_N1];
	for (int i = start_N1; i < end_N1; i++)
	{
		for (int j = 0; j < N0; j++)
			hs_1[i - start_N1] += IN[j] * W0[j][i];
	}
	MPI_Allgather(hs_1, partition_N1, MPI_DOUBLE, HS_1, partition_N1, MPI_DOUBLE, MPI_COMM_WORLD);

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N1; i++)
	{
		HO_1[i] = scaled_tanh(HS_1[i]);
		// HO_1[i] = sigmoid(HS_1[i]);
	}

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N2; i++)
	{
		HS_2[i] = B2[i];
	}

	double hs_2[partition_N2];
	for (int i = start_N2; i < end_N2; i++)
	{
		for (int j = 0; j < N1; j++)
			hs_2[i - start_N2] += HO_1[j] * W1[j][i];
	}
	MPI_Allgather(hs_2, partition_N2, MPI_DOUBLE, HS_2, partition_N2, MPI_DOUBLE, MPI_COMM_WORLD);

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N2; i++)
	{
		HO_2[i] = scaled_tanh(HS_2[i]);
		// HO_2[i] = sigmoid(HS_2[i]);
	}

	// compute the weighted sum OS in the output layer
	for (int i = 0; i < N3; i++)
	{
		OS[i] = B3[i];
	}

	double os[partition_N3];
	for (int i = start_N3; i < end_N3; i++)
	{
		for (int j = 0; j < N2; j++)
			os[i - start_N3] += HO_2[j] * W2[j][i];
	}
	MPI_Allgather(os, partition_N3, MPI_DOUBLE, OS, partition_N3, MPI_DOUBLE, MPI_COMM_WORLD);

	// Comput the output of the output layer, OO[N2];
	for (int i = 0; i < N3; i++)
	{
		OO[i] = scaled_tanh(OS[i]);
		// OO[i] = sigmoid(OS[i]);
	}
}
#else
void forward(vector<double> input)
{

	int partition_N1, start_N1, end_N1, partition_N2, start_N2, end_N2;
	if (sub_N2 == 'A')
	{
		partition_N1 = N1 / size_N2;
		start_N1 = rank_N2 * partition_N1;
		end_N1 = start_N1 + partition_N1;

		partition_N2 = N2 / size_N2;
		start_N2 = rank_N2 * partition_N2;
		end_N2 = start_N2 + partition_N2;
	}
	// cout << "partition" << partition_N1 << endl;

	for (int i = 0; i < N0; i++)
		IN[i] = input[i];

	// compute the weighted sum HS in the hidden layer

	for (int i = 0; i < N1; i++)
	{
		HS_1[i] = B1[i];
	}
	if (sub_N2 == 'A')
	{
		double hs_1[partition_N1];
		for (int i = start_N1; i < end_N1; i++)
		{
			hs_1[i - start_N1] = B1[i];
		}
		if (my_rank == 0)
		{
			start_time_HS_1 = MPI_Wtime();
		}
		for (int i = start_N1; i < end_N1; i++)
		{
			for (int j = 0; j < N0; j++)
				hs_1[i - start_N1] += IN[j] * W0[j][i];
		}
		MPI_Allgather(hs_1, partition_N1, MPI_DOUBLE, HS_1, partition_N1, MPI_DOUBLE, N2_comm);
		if (my_rank == 0)
		{
			end_time_HS_1 = MPI_Wtime();
			loop_HS_1 += (end_time_HS_1 - start_time_HS_1);
			start_time_HS_1 = 0;
			end_time_HS_1 = 0;
			// cout << loop_HS_1 << endl;
		}
	}

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N1; i++)
	{
		HO_1[i] = scaled_tanh(HS_1[i]);
		// HO_1[i] = sigmoid(HS_1[i]);
	}

	// compute the weighted sum HS in the hidden layer
	for (int i = 0; i < N2; i++)
	{
		HS_2[i] = B2[i];
	}
	if (sub_N2 == 'A')
	{
		double hs_2[partition_N2];
		for (int i = start_N2; i < end_N2; i++)
		{
			hs_2[i - start_N2] = B2[i];
		}
		if (my_rank == 0)
		{
			start_time_HS_2 = MPI_Wtime();
		}
		for (int i = start_N2; i < end_N2; i++)
		{
			for (int j = 0; j < N1; j++)
				hs_2[i - start_N2] += HO_1[j] * W1[j][i];
		}
		MPI_Allgather(hs_2, partition_N2, MPI_DOUBLE, HS_2, partition_N2, MPI_DOUBLE, N2_comm);
		if (my_rank == 0)
		{
			end_time_HS_2 = MPI_Wtime();
			loop_HS_2 += (end_time_HS_2 - start_time_HS_2);
			start_time_HS_2 = 0;
			end_time_HS_2 = 0;
		}
	}

	// Comput the output of the hidden layer, HO[N1];
	for (int i = 0; i < N2; i++)
	{
		HO_2[i] = scaled_tanh(HS_2[i]);
		// HO_2[i] = sigmoid(HS_2[i]);
	}

	// compute the weighted sum OS in the output layer
	for (int i = 0; i < N3; i++)
	{
		OS[i] = B3[i];
	}

	for (int i = 0; i < N3; i++)
	{
		for (int j = 0; j < N2; j++)
			OS[i] += HO_2[j] * W2[j][i];
	}

	// Comput the output of the output layer, OO[N2];
	for (int i = 0; i < N3; i++)
	{
		OO[i] = scaled_tanh(OS[i]);
		// OO[i] = sigmoid(OS[i]);
	}
	// cout << "In normal" << endl;
}
#endif

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

void print_1d(double *a, int size, const char *aa)
{
	for (int i = 0; i < size; i++)
		cout << aa << "[" << i << "]=" << a[i] << "\n";
}

void print_01(double a[N0][N1], const char *aa)
{
	for (int i = 0; i < N0; i++)
		for (int j = 0; j < N1; j++)
			cout << aa << "[" << i << "][" << j
				 << "]=" << a[i][j] << "\n";
}

void print_12(double a[N1][N2], const char *aa)
{
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
			cout << aa << "[" << i << "][" << j
				 << "]=" << a[i][j] << "\n";
}

/**
 * @brief The funtion is used to process backward propagation of a Neural network
 *
 */
#if LOOP == 1
double backward(double *O, vector<double> Y)
{
	// compute error
	double A = 1.7159;
	double B = 0.6666;
	err = 0.0;

	for (int i = 0; i < N3; i++)
		err += (O[i] - Y[i]) * (O[i] - Y[i]);
	err = err / N3;

	double temp_OOi;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N3; i++)
	{
		/*OO[i] = AtanH(Bx)
		 * 		A * B (1 - (tanh(Bx) * tanh(Bx)))
		 * 		B * (A - (A * tanhx(Bx) * tanhx(Bx)))
		 * 		B * (A - OO[i] * OO[i])
		 * 		A * B (1 - (tanh(Bx) * tanh(Bx)))*/

		dE_OO[i] = (O[i] - Y[i]) * 2.0 / N3; // loop fusion
		temp_OOi = OO[i];
		dOO_OS[i] = B * (A - (temp_OOi * temp_OOi / A)); // A * B (1 - (tanh(Bx) * tanh(Bx)))
		dE_OS[i] = dE_OO[i] * dOO_OS[i];
		dE_B3[i] = dE_OS[i];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_OS_dE_B3 += print_duration(&bb, &ee);

	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			dE_W2[i][j] = dE_OS[j] * HO_2[i];

	for (int i = 0; i < N2; i++)
	{
		dE_HO_2[i] = 0;
		for (int j = 0; j < N3; j++)
			dE_HO_2[i] += dE_OS[j] * W2[i][j];
	}

	// compute dHO_HS_2 = HO_2 dot (1-HO_2)
	for (int i = 0; i < N2; i++)
		dHO_HS_2[i] = B * (A - (HO_2[i] * HO_2[i] / A));

	// compute dE_HS_2 = dE_HO_2 dot dHO_HS_2
	for (int i = 0; i < N2; i++)
		dE_HS_2[i] = dE_HO_2[i] * dHO_HS_2[i];

	// compute dE_B2 = dE_HS_2
	for (int i = 0; i < N2; i++)
		dE_B2[i] = dE_HS_2[i];

	int bk = 20;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N1; i += bk)
	{
		for (int j = 0; j < N2; j += bk)
		{
			for (int ii = i; ii < i + bk; ii++)
			{
				dE_W1[ii][j + 0] = dE_HS_2[j + 0] * HO_1[ii];
				dE_W1[ii][j + 1] = dE_HS_2[j + 1] * HO_1[ii];
				dE_W1[ii][j + 2] = dE_HS_2[j + 2] * HO_1[ii];
				dE_W1[ii][j + 3] = dE_HS_2[j + 3] * HO_1[ii];
				dE_W1[ii][j + 4] = dE_HS_2[j + 4] * HO_1[ii];

				dE_W1[ii][j + 5] = dE_HS_2[j + 5] * HO_1[ii];
				dE_W1[ii][j + 6] = dE_HS_2[j + 6] * HO_1[ii];
				dE_W1[ii][j + 7] = dE_HS_2[j + 7] * HO_1[ii];
				dE_W1[ii][j + 8] = dE_HS_2[j + 8] * HO_1[ii];
				dE_W1[ii][j + 9] = dE_HS_2[j + 9] * HO_1[ii];

				dE_W1[ii][j + 10] = dE_HS_2[j + 10] * HO_1[ii];
				dE_W1[ii][j + 11] = dE_HS_2[j + 11] * HO_1[ii];
				dE_W1[ii][j + 12] = dE_HS_2[j + 12] * HO_1[ii];
				dE_W1[ii][j + 13] = dE_HS_2[j + 13] * HO_1[ii];
				dE_W1[ii][j + 14] = dE_HS_2[j + 14] * HO_1[ii];

				dE_W1[ii][j + 15] = dE_HS_2[j + 15] * HO_1[ii];
				dE_W1[ii][j + 16] = dE_HS_2[j + 16] * HO_1[ii];
				dE_W1[ii][j + 17] = dE_HS_2[j + 17] * HO_1[ii];
				dE_W1[ii][j + 18] = dE_HS_2[j + 18] * HO_1[ii];
				dE_W1[ii][j + 19] = dE_HS_2[j + 19] * HO_1[ii];
			}
		}
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_W1 += print_duration(&bb, &ee);

	// compute dE_HO_1
	double temp_dE_HO_1;
	bk = 20;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N1; i++)
	{
		temp_dE_HO_1 = 0.0;
		for (int j = 0; j < N2; j += bk)
		{

			temp_dE_HO_1 += dE_HS_2[j + 0] * W1[i][j + 0];
			temp_dE_HO_1 += dE_HS_2[j + 1] * W1[i][j + 1];
			temp_dE_HO_1 += dE_HS_2[j + 2] * W1[i][j + 2];
			temp_dE_HO_1 += dE_HS_2[j + 3] * W1[i][j + 3];
			temp_dE_HO_1 += dE_HS_2[j + 4] * W1[i][j + 4];
			temp_dE_HO_1 += dE_HS_2[j + 5] * W1[i][j + 5];
			temp_dE_HO_1 += dE_HS_2[j + 6] * W1[i][j + 6];
			temp_dE_HO_1 += dE_HS_2[j + 7] * W1[i][j + 7];
			temp_dE_HO_1 += dE_HS_2[j + 8] * W1[i][j + 8];
			temp_dE_HO_1 += dE_HS_2[j + 9] * W1[i][j + 9];
			temp_dE_HO_1 += dE_HS_2[j + 10] * W1[i][j + 10];
			temp_dE_HO_1 += dE_HS_2[j + 11] * W1[i][j + 11];
			temp_dE_HO_1 += dE_HS_2[j + 12] * W1[i][j + 12];
			temp_dE_HO_1 += dE_HS_2[j + 13] * W1[i][j + 13];
			temp_dE_HO_1 += dE_HS_2[j + 14] * W1[i][j + 14];
			temp_dE_HO_1 += dE_HS_2[j + 15] * W1[i][j + 15];
			temp_dE_HO_1 += dE_HS_2[j + 16] * W1[i][j + 16];
			temp_dE_HO_1 += dE_HS_2[j + 17] * W1[i][j + 17];
			temp_dE_HO_1 += dE_HS_2[j + 18] * W1[i][j + 18];
			temp_dE_HO_1 += dE_HS_2[j + 19] * W1[i][j + 19];
		}
		dE_HO_1[i] = temp_dE_HO_1;
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_HO_1 += print_duration(&bb, &ee);

	// compute dHO_HS_1 = HO_1 dot (1-HO_1)
	for (int i = 0; i < N1; i++)
		dHO_HS_1[i] = B * (A - (HO_1[i] * HO_1[i] / A));

	// compute dE_HS_1 = dE_HO_1 dot dHO_HS_1
	for (int i = 0; i < N1; i++)
		dE_HS_1[i] = dE_HO_1[i] * dHO_HS_1[i];

	// compute dE_B1 = dE_HS_1
	for (int i = 0; i < N1; i++)
		dE_B1[i] = dE_HS_1[i];

	bk = 16;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N0; i += bk)
	{
		for (int j = 0; j < N1; j += bk)
		{
			for (int ii = i; ii < i + bk; ii++)
			{
				dE_W0[ii][j + 0] = dE_HS_1[j + 0] * IN[ii];
				dE_W0[ii][j + 1] = dE_HS_1[j + 1] * IN[ii];
				dE_W0[ii][j + 2] = dE_HS_1[j + 2] * IN[ii];
				dE_W0[ii][j + 3] = dE_HS_1[j + 3] * IN[ii];

				dE_W0[ii][j + 4] = dE_HS_1[j + 4] * IN[ii];
				dE_W0[ii][j + 5] = dE_HS_1[j + 5] * IN[ii];
				dE_W0[ii][j + 6] = dE_HS_1[j + 6] * IN[ii];
				dE_W0[ii][j + 7] = dE_HS_1[j + 7] * IN[ii];

				dE_W0[ii][j + 8] = dE_HS_1[j + 8] * IN[ii];
				dE_W0[ii][j + 9] = dE_HS_1[j + 9] * IN[ii];
				dE_W0[ii][j + 10] = dE_HS_1[j + 10] * IN[ii];
				dE_W0[ii][j + 11] = dE_HS_1[j + 11] * IN[ii];

				dE_W0[ii][j + 12] = dE_HS_1[j + 12] * IN[ii];
				dE_W0[ii][j + 13] = dE_HS_1[j + 13] * IN[ii];
				dE_W0[ii][j + 14] = dE_HS_1[j + 14] * IN[ii];
				dE_W0[ii][j + 15] = dE_HS_1[j + 15] * IN[ii];
			}
		}
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_W0 += print_duration(&bb, &ee);

	// update W0, W1, W2, B1, B2, B3;

	bk = 8;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N0; i += bk)
		for (int j = 0; j < N1; j += bk)
			for (int ii = i; ii < i + bk; ii++)
			{
				W0[ii][j + 0] = W0[ii][j + 0] - rate * dE_W0[ii][j + 0];
				W0[ii][j + 1] = W0[ii][j + 1] - rate * dE_W0[ii][j + 1];
				W0[ii][j + 2] = W0[ii][j + 2] - rate * dE_W0[ii][j + 2];
				W0[ii][j + 3] = W0[ii][j + 3] - rate * dE_W0[ii][j + 3];
				W0[ii][j + 4] = W0[ii][j + 4] - rate * dE_W0[ii][j + 4];
				W0[ii][j + 5] = W0[ii][j + 5] - rate * dE_W0[ii][j + 5];
				W0[ii][j + 6] = W0[ii][j + 6] - rate * dE_W0[ii][j + 6];
				W0[ii][j + 7] = W0[ii][j + 7] - rate * dE_W0[ii][j + 7];
			}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_W0 += print_duration(&bb, &ee);

	for (int i = 0; i < N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];

	bk = 20;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	double temp_w1;
	for (int i = 0; i < N1; i++)
	{
		for (int j = 0; j < N2; j += bk)
		{

			W1[i][j + 0] = W1[i][j + 0] - rate * dE_W1[i][j + 0];
			W1[i][j + 1] = W1[i][j + 1] - rate * dE_W1[i][j + 1];
			W1[i][j + 2] = W1[i][j + 2] - rate * dE_W1[i][j + 2];
			W1[i][j + 3] = W1[i][j + 3] - rate * dE_W1[i][j + 3];
			W1[i][j + 4] = W1[i][j + 4] - rate * dE_W1[i][j + 4];

			W1[i][j + 5] = W1[i][j + 5] - rate * dE_W1[i][j + 5];
			W1[i][j + 6] = W1[i][j + 6] - rate * dE_W1[i][j + 6];
			W1[i][j + 7] = W1[i][j + 7] - rate * dE_W1[i][j + 7];
			W1[i][j + 8] = W1[i][j + 8] - rate * dE_W1[i][j + 8];
			W1[i][j + 9] = W1[i][j + 9] - rate * dE_W1[i][j + 9];

			W1[i][j + 10] = W1[i][j + 10] - rate * dE_W1[i][j + 10];
			W1[i][j + 11] = W1[i][j + 11] - rate * dE_W1[i][j + 11];
			W1[i][j + 12] = W1[i][j + 12] - rate * dE_W1[i][j + 12];
			W1[i][j + 13] = W1[i][j + 13] - rate * dE_W1[i][j + 13];
			W1[i][j + 14] = W1[i][j + 14] - rate * dE_W1[i][j + 14];

			W1[i][j + 15] = W1[i][j + 15] - rate * dE_W1[i][j + 15];
			W1[i][j + 16] = W1[i][j + 16] - rate * dE_W1[i][j + 16];
			W1[i][j + 17] = W1[i][j + 17] - rate * dE_W1[i][j + 17];
			W1[i][j + 18] = W1[i][j + 18] - rate * dE_W1[i][j + 18];
			W1[i][j + 19] = W1[i][j + 19] - rate * dE_W1[i][j + 19];
		}
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_W1 += print_duration(&bb, &ee);

	for (int i = 0; i < N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];

	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i = 0; i < N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];
	cout << "In LOOP" << endl;
}
#elif LOOP == 2
double backward(double *O, vector<double> Y)
{
	// compute error
	double A = 1.7159;
	double B = 0.6666;
	err = 0.0;
	for (int i = 0; i < N3; i++)
		err += (O[i] - Y[i]) * (O[i] - Y[i]);
	err = err / N3;

	// compute dE_OO
	for (int i = 0; i < N3; i++)
		dE_OO[i] = (O[i] - Y[i]) * 2.0 / N3;

	// compute dOO_OS = OO dot (1-OO)
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N3; i++)
		// OO[i] = AtanH(Bx)
		//  A * B (1 - (tanh(Bx) * tanh(Bx)))
		//  B * (A - (A * tanhx(Bx) * tanhx(Bx)))
		//  A * B (1 - (tanh(Bx) * tanh(Bx)))
		dOO_OS[i] = B * (A - (OO[i] * OO[i] / A)); // A * B (1 - (tanh(Bx) * tanh(Bx)))
	// dOO_OS[i] = OO[i] * (1.0-OO[i]);
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_OS_dE_B3 += print_duration(&bb, &ee);

	// compute dE_OS = dE_OO dot dOO_OS
	for (int i = 0; i < N3; i++)
		dE_OS[i] = dE_OO[i] * dOO_OS[i];

	// compute dE_B3 = dE_OS
	for (int i = 0; i < N3; i++)
		dE_B3[i] = dE_OS[i];

	// compute dE_W2
	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			dE_W2[i][j] = dE_OS[j] * HO_2[i];

	// compute dE_HO_2
	for (int i = 0; i < N2; i++)
	{
		dE_HO_2[i] = 0;
		for (int j = 0; j < N3; j++)
			dE_HO_2[i] += dE_OS[j] * W2[i][j];
	}

	// compute dHO_HS_2 = HO_2 dot (1-HO_2)
	for (int i = 0; i < N2; i++)
		// dHO_HS_2[i] = HO_2[i] * (1-HO_2[i]);
		dHO_HS_2[i] = B * (A - (HO_2[i] * HO_2[i] / A));

	// compute dE_HS_2 = dE_HO_2 dot dHO_HS_2
	for (int i = 0; i < N2; i++)
		dE_HS_2[i] = dE_HO_2[i] * dHO_HS_2[i];

	// compute dE_B2 = dE_HS_2
	for (int i = 0; i < N2; i++)
		dE_B2[i] = dE_HS_2[i];

	// compute dE_W1
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
			dE_W1[i][j] = dE_HS_2[j] * HO_1[i];
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_W1 += print_duration(&bb, &ee);

	// compute dE_HO_1
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N1; i++)
	{
		dE_HO_1[i] = 0;
		for (int j = 0; j < N2; j++)
			dE_HO_1[i] += dE_HS_2[j] * W1[i][j];
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_HO_1 += print_duration(&bb, &ee);

	// compute dHO_HS_1 = HO_1 dot (1-HO_1)
	for (int i = 0; i < N1; i++)
		// dHO_HS_1[i] = HO_1[i] * (1-HO_1[i]);
		dHO_HS_1[i] = B * (A - (HO_1[i] * HO_1[i] / A));

	// compute dE_HS_1 = dE_HO_1 dot dHO_HS_1
	for (int i = 0; i < N1; i++)
		dE_HS_1[i] = dE_HO_1[i] * dHO_HS_1[i];

	// compute dE_B1 = dE_HS_1
	for (int i = 0; i < N1; i++)
		dE_B1[i] = dE_HS_1[i];

	// compute dE_W0
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N0; i++)
		for (int j = 0; j < N1; j++)
			dE_W0[i][j] = dE_HS_1[j] * IN[i];
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_W0 += print_duration(&bb, &ee);

	// update W0, W1, W2, B1, B2, B3;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N0; i++)
		for (int j = 0; j < N1; j++)
			W0[i][j] = W0[i][j] - rate * dE_W0[i][j];
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_W0 += print_duration(&bb, &ee);

	for (int i = 0; i < N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
			W1[i][j] = W1[i][j] - rate * dE_W1[i][j];
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_W1 += print_duration(&bb, &ee);

	for (int i = 0; i < N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];

	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i = 0; i < N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];
	// cout << "In Normal" << endl;
}
#elif LOOP == 3
double backward(double *O, vector<double> Y)
{
	// compute error
	double A = 1.7159;
	double B = 0.6666;
	err = 0.0;

	for (int i = 0; i < N3; i++)
		err += (O[i] - Y[i]) * (O[i] - Y[i]);
	err = err / N3;

	double temp_OOi;
	for (int i = 0; i < N3; i++)
	{
		/*
		 *        OO[i] = AtanH(Bx)
		 * 		A * B (1 - (tanh(Bx) * tanh(Bx)))
		 * 		B * (A - (A * tanhx(Bx) * tanhx(Bx)))
		 * 		B * (A - OO[i] * OO[i])
		 * 		A * B (1 - (tanh(Bx) * tanh(Bx)))
		 */

		dE_OO[i] = (O[i] - Y[i]) * 2.0 / N3; // loop fusion
		temp_OOi = OO[i];
		dOO_OS[i] = B * (A - (temp_OOi * temp_OOi / A)); // A * B (1 - (tanh(Bx) * tanh(Bx)))
		dE_OS[i] = dE_OO[i] * dOO_OS[i];
		dE_B3[i] = dE_OS[i];
	}

	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			dE_W2[i][j] = dE_OS[j] * HO_2[i];

	for (int i = 0; i < N2; i++)
	{
		dE_HO_2[i] = 0;
		for (int j = 0; j < N3; j++)
			dE_HO_2[i] += dE_OS[j] * W2[i][j];
	}

	// compute dHO_HS_2 = HO_2 dot (1-HO_2)
	for (int i = 0; i < N2; i++)
		dHO_HS_2[i] = B * (A - (HO_2[i] * HO_2[i] / A));

	// compute dE_HS_2 = dE_HO_2 dot dHO_HS_2
	for (int i = 0; i < N2; i++)
		dE_HS_2[i] = dE_HO_2[i] * dHO_HS_2[i];

	// compute dE_B2 = dE_HS_2
	for (int i = 0; i < N2; i++)
		dE_B2[i] = dE_HS_2[i];

	int bk = 20;

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);

	int i, j, ii;
#pragma omp parallel
#pragma omp for private(j, ii)
	for (i = 0; i < N1; i += bk)
	{
		for (j = 0; j < N2; j += bk)
		{
			for (ii = i; ii < i + bk; ii++)
			{

				dE_W1[ii][j + 0] = dE_HS_2[j + 0] * HO_1[ii];
				dE_W1[ii][j + 1] = dE_HS_2[j + 1] * HO_1[ii];
				dE_W1[ii][j + 2] = dE_HS_2[j + 2] * HO_1[ii];
				dE_W1[ii][j + 3] = dE_HS_2[j + 3] * HO_1[ii];
				dE_W1[ii][j + 4] = dE_HS_2[j + 4] * HO_1[ii];

				dE_W1[ii][j + 5] = dE_HS_2[j + 5] * HO_1[ii];
				dE_W1[ii][j + 6] = dE_HS_2[j + 6] * HO_1[ii];
				dE_W1[ii][j + 7] = dE_HS_2[j + 7] * HO_1[ii];
				dE_W1[ii][j + 8] = dE_HS_2[j + 8] * HO_1[ii];
				dE_W1[ii][j + 9] = dE_HS_2[j + 9] * HO_1[ii];

				dE_W1[ii][j + 10] = dE_HS_2[j + 10] * HO_1[ii];
				dE_W1[ii][j + 11] = dE_HS_2[j + 11] * HO_1[ii];
				dE_W1[ii][j + 12] = dE_HS_2[j + 12] * HO_1[ii];
				dE_W1[ii][j + 13] = dE_HS_2[j + 13] * HO_1[ii];
				dE_W1[ii][j + 14] = dE_HS_2[j + 14] * HO_1[ii];

				dE_W1[ii][j + 15] = dE_HS_2[j + 15] * HO_1[ii];
				dE_W1[ii][j + 16] = dE_HS_2[j + 16] * HO_1[ii];
				dE_W1[ii][j + 17] = dE_HS_2[j + 17] * HO_1[ii];
				dE_W1[ii][j + 18] = dE_HS_2[j + 18] * HO_1[ii];
				dE_W1[ii][j + 19] = dE_HS_2[j + 19] * HO_1[ii];
			}
		}
	}

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_W1 += print_duration(&bb, &ee);

	// compute dE_HO_1
	double temp_dE_HO_1;
	bk = 20;
	for (int i = 0; i < N1; i++)
	{
		temp_dE_HO_1 = 0.0;
		for (int j = 0; j < N2; j += bk)
		{

			temp_dE_HO_1 += dE_HS_2[j + 0] * W1[i][j + 0];
			temp_dE_HO_1 += dE_HS_2[j + 1] * W1[i][j + 1];
			temp_dE_HO_1 += dE_HS_2[j + 2] * W1[i][j + 2];
			temp_dE_HO_1 += dE_HS_2[j + 3] * W1[i][j + 3];
			temp_dE_HO_1 += dE_HS_2[j + 4] * W1[i][j + 4];
			temp_dE_HO_1 += dE_HS_2[j + 5] * W1[i][j + 5];
			temp_dE_HO_1 += dE_HS_2[j + 6] * W1[i][j + 6];
			temp_dE_HO_1 += dE_HS_2[j + 7] * W1[i][j + 7];
			temp_dE_HO_1 += dE_HS_2[j + 8] * W1[i][j + 8];
			temp_dE_HO_1 += dE_HS_2[j + 9] * W1[i][j + 9];
			temp_dE_HO_1 += dE_HS_2[j + 10] * W1[i][j + 10];
			temp_dE_HO_1 += dE_HS_2[j + 11] * W1[i][j + 11];
			temp_dE_HO_1 += dE_HS_2[j + 12] * W1[i][j + 12];
			temp_dE_HO_1 += dE_HS_2[j + 13] * W1[i][j + 13];
			temp_dE_HO_1 += dE_HS_2[j + 14] * W1[i][j + 14];
			temp_dE_HO_1 += dE_HS_2[j + 15] * W1[i][j + 15];
			temp_dE_HO_1 += dE_HS_2[j + 16] * W1[i][j + 16];
			temp_dE_HO_1 += dE_HS_2[j + 17] * W1[i][j + 17];
			temp_dE_HO_1 += dE_HS_2[j + 18] * W1[i][j + 18];
			temp_dE_HO_1 += dE_HS_2[j + 19] * W1[i][j + 19];
		}
		dE_HO_1[i] = temp_dE_HO_1;
	}

	// compute dHO_HS_1 = HO_1 dot (1-HO_1)
	for (int i = 0; i < N1; i++)
		dHO_HS_1[i] = B * (A - (HO_1[i] * HO_1[i] / A));

	// compute dE_HS_1 = dE_HO_1 dot dHO_HS_1
	for (int i = 0; i < N1; i++)
		dE_HS_1[i] = dE_HO_1[i] * dHO_HS_1[i];

	// compute dE_B1 = dE_HS_1
	for (int i = 0; i < N1; i++)
		dE_B1[i] = dE_HS_1[i];

	bk = 16;

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
#pragma omp parallel
#pragma omp for private(j, ii)
	for (i = 0; i < N0; i += bk)
	{
		for (j = 0; j < N1; j += bk)
		{
			for (ii = i; ii < i + bk; ii++)
			{

				dE_W0[ii][j + 0] = dE_HS_1[j + 0] * IN[ii];
				dE_W0[ii][j + 1] = dE_HS_1[j + 1] * IN[ii];
				dE_W0[ii][j + 2] = dE_HS_1[j + 2] * IN[ii];
				dE_W0[ii][j + 3] = dE_HS_1[j + 3] * IN[ii];

				dE_W0[ii][j + 4] = dE_HS_1[j + 4] * IN[ii];
				dE_W0[ii][j + 5] = dE_HS_1[j + 5] * IN[ii];
				dE_W0[ii][j + 6] = dE_HS_1[j + 6] * IN[ii];
				dE_W0[ii][j + 7] = dE_HS_1[j + 7] * IN[ii];

				dE_W0[ii][j + 8] = dE_HS_1[j + 8] * IN[ii];
				dE_W0[ii][j + 9] = dE_HS_1[j + 9] * IN[ii];
				dE_W0[ii][j + 10] = dE_HS_1[j + 10] * IN[ii];
				dE_W0[ii][j + 11] = dE_HS_1[j + 11] * IN[ii];

				dE_W0[ii][j + 12] = dE_HS_1[j + 12] * IN[ii];
				dE_W0[ii][j + 13] = dE_HS_1[j + 13] * IN[ii];
				dE_W0[ii][j + 14] = dE_HS_1[j + 14] * IN[ii];
				dE_W0[ii][j + 15] = dE_HS_1[j + 15] * IN[ii];
			}
		}
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
	loop_dE_W0 += print_duration(&bb, &ee);

	// update W0, W1, W2, B1, B2, B3;
	bk = 8;
	{

		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
		int i, j, ii;
#pragma omp parallel
#pragma omp for private(ii)
		for (int i = 0; i < N0; i += bk)
			for (int j = 0; j < N1; j += bk)
			{
				for (int ii = i; ii < i + bk; ii++)
				{

					W0[ii][j + 0] = W0[ii][j + 0] - rate * dE_W0[ii][j + 0];
					W0[ii][j + 1] = W0[ii][j + 1] - rate * dE_W0[ii][j + 1];
					W0[ii][j + 2] = W0[ii][j + 2] - rate * dE_W0[ii][j + 2];
					W0[ii][j + 3] = W0[ii][j + 3] - rate * dE_W0[ii][j + 3];
					W0[ii][j + 4] = W0[ii][j + 4] - rate * dE_W0[ii][j + 4];
					W0[ii][j + 5] = W0[ii][j + 5] - rate * dE_W0[ii][j + 5];
					W0[ii][j + 6] = W0[ii][j + 6] - rate * dE_W0[ii][j + 6];
					W0[ii][j + 7] = W0[ii][j + 7] - rate * dE_W0[ii][j + 7];
				}
			}
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
		loop_W0 += print_duration(&bb, &ee);
	}

	for (int i = 0; i < N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];

	bk = 20;
	{

		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
		double temp_w1;

		int i, j;
#pragma omp parallel
#pragma omp for private(i)
		for (i = 0; i < N1; i++)
		{
			for (j = 0; j < N2; j += bk)
			{

				W1[i][j + 0] = W1[i][j + 0] - rate * dE_W1[i][j + 0];
				W1[i][j + 1] = W1[i][j + 1] - rate * dE_W1[i][j + 1];
				W1[i][j + 2] = W1[i][j + 2] - rate * dE_W1[i][j + 2];
				W1[i][j + 3] = W1[i][j + 3] - rate * dE_W1[i][j + 3];
				W1[i][j + 4] = W1[i][j + 4] - rate * dE_W1[i][j + 4];
				W1[i][j + 5] = W1[i][j + 5] - rate * dE_W1[i][j + 5];
				W1[i][j + 6] = W1[i][j + 6] - rate * dE_W1[i][j + 6];
				W1[i][j + 7] = W1[i][j + 7] - rate * dE_W1[i][j + 7];
				W1[i][j + 8] = W1[i][j + 8] - rate * dE_W1[i][j + 8];
				W1[i][j + 9] = W1[i][j + 9] - rate * dE_W1[i][j + 9];
				W1[i][j + 10] = W1[i][j + 10] - rate * dE_W1[i][j + 10];
				W1[i][j + 11] = W1[i][j + 11] - rate * dE_W1[i][j + 11];
				W1[i][j + 12] = W1[i][j + 12] - rate * dE_W1[i][j + 12];
				W1[i][j + 13] = W1[i][j + 13] - rate * dE_W1[i][j + 13];
				W1[i][j + 14] = W1[i][j + 14] - rate * dE_W1[i][j + 14];
				W1[i][j + 15] = W1[i][j + 15] - rate * dE_W1[i][j + 15];
				W1[i][j + 16] = W1[i][j + 16] - rate * dE_W1[i][j + 16];
				W1[i][j + 17] = W1[i][j + 17] - rate * dE_W1[i][j + 17];
				W1[i][j + 18] = W1[i][j + 18] - rate * dE_W1[i][j + 18];
				W1[i][j + 19] = W1[i][j + 19] - rate * dE_W1[i][j + 19];
			}
		}

#pragma omp barrier

		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ee);
		loop_W1 += print_duration(&bb, &ee);
	}

#pragma omp parallel
#pragma omp for private(i)
	for (i = 0; i < N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];

#pragma omp parallel
#pragma omp for private(i)
	for (i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i = 0; i < N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];
	cout << "In OMP" << endl;
}
#elif LOOP == 4
double backward(double *O, vector<double> Y) int partition_N1 = N1 / size;
int start_N1 = rank * partition_N1;
int end_N1 = start_N1 + partition_N1;

int partition_N2 = N2 / size;
int start_N2 = rank * partition_N2;
int end_N2 = start_N2 + partition_N2;

int partition_N3 = N3 / size;
int start_N3 = rank * partition_N3;
int end_N3 = start_N3 + partition_N3;

// compute error
double A = 1.7159;
double B = 0.6666;
err = 0.0;
for (int i = 0; i < N3; i++)
	err += (O[i] - Y[i]) * (O[i] - Y[i]);
err = err / N3;

// compute dE_OO
for (int i = 0; i < N3; i++)
	dE_OO[i] = (O[i] - Y[i]) * 2.0 / N3;

// compute dOO_OS = OO dot (1-OO)
for (int i = 0; i < N3; i++)
	// OO[i] = AtanH(Bx)
	//  A * B (1 - (tanh(Bx) * tanh(Bx)))
	//  B * (A - (A * tanhx(Bx) * tanhx(Bx)))
	//  A * B (1 - (tanh(Bx) * tanh(Bx)))
	dOO_OS[i] = B * (A - (OO[i] * OO[i] / A)); // A * B (1 - (tanh(Bx) * tanh(Bx)))
											   // dOO_OS[i] = OO[i] * (1.0-OO[i]);

// compute dE_OS = dE_OO dot dOO_OS
for (int i = 0; i < N3; i++)
	dE_OS[i] = dE_OO[i] * dOO_OS[i];

// compute dE_B3 = dE_OS
for (int i = 0; i < N3; i++)
	dE_B3[i] = dE_OS[i];

// compute dE_W2
double *dE_w2 = (double *)malloc(partition_N2 * N3 * sizeof(double));
for (int i = start_N2; i < end_N2; i++)
	for (int j = 0; j < N3; j++)
	{
		int temp_pos = (i - start_N2) * N3 + j;
		*(dE_w2 + temp_pos) = dE_OS[j] * HO_2[i];
		// dE_w2[i - start_N2][j] = dE_OS[j] * HO_2[i];
	}
MPI_Allgather(dE_w2, partition_N2 *N3, MPI_DOUBLE, dE_W2, partition_N2 *N3, MPI_DOUBLE, MPI_COMM_WORLD);

// compute dE_HO_2
double dE_ho_2[partition_N2];
for (int i = start_N2; i < end_N2; i++)
{
	dE_ho_2[i - start_N2] = 0;
	for (int j = 0; j < N3; j++)
		dE_ho_2[i - start_N2] += dE_OS[j] * W2[i][j];
}
MPI_Allgather(dE_ho_2, partition_N2, MPI_DOUBLE, dE_HO_2, partition_N2, MPI_DOUBLE, MPI_COMM_WORLD);

// compute dHO_HS_2 = HO_2 dot (1-HO_2)
for (int i = 0; i < N2; i++)
	// dHO_HS_2[i] = HO_2[i] * (1-HO_2[i]);
	dHO_HS_2[i] = B * (A - (HO_2[i] * HO_2[i] / A));

// compute dE_HS_2 = dE_HO_2 dot dHO_HS_2
for (int i = 0; i < N2; i++)
	dE_HS_2[i] = dE_HO_2[i] * dHO_HS_2[i];

// compute dE_B2 = dE_HS_2
for (int i = 0; i < N2; i++)
	dE_B2[i] = dE_HS_2[i];

// compute dE_W1
double dE_w1[partition_N1][N2];
for (int i = start_N1; i < end_N1; i++)
	for (int j = 0; j < N2; j++)
		dE_w1[i - start_N1][j] = dE_HS_2[j] * HO_1[i];
MPI_Allgather(dE_w1, partition_N1 *N2, MPI_DOUBLE, dE_W1, partition_N1 *N2, MPI_DOUBLE, MPI_COMM_WORLD);

// compute dE_HO_1
double dE_ho_1[partition_N1];
for (int i = start_N1; i < end_N1; i++)
{
	dE_ho_1[i - start_N1] = 0;
	for (int j = 0; j < N2; j++)
		dE_ho_1[i - start_N1] += dE_HS_2[j] * W1[i][j];
}
MPI_Allgather(dE_ho_1, partition_N2, MPI_DOUBLE, dE_HO_1, partition_N2, MPI_DOUBLE, MPI_COMM_WORLD);

// compute dHO_HS_1 = HO_1 dot (1-HO_1)
for (int i = 0; i < N1; i++)
	// dHO_HS_1[i] = HO_1[i] * (1-HO_1[i]);
	dHO_HS_1[i] = B * (A - (HO_1[i] * HO_1[i] / A));

// compute dE_HS_1 = dE_HO_1 dot dHO_HS_1
for (int i = 0; i < N1; i++)
	dE_HS_1[i] = dE_HO_1[i] * dHO_HS_1[i];

// compute dE_B1 = dE_HS_1
for (int i = 0; i < N1; i++)
	dE_B1[i] = dE_HS_1[i];

// compute dE_W0
for (int i = 0; i < N0; i++)
	for (int j = 0; j < N1; j++)
		dE_W0[i][j] = dE_HS_1[j] * IN[i];

// update W0, W1, W2, B1, B2, B3;

for (int i = 0; i < N0; i++)
	for (int j = 0; j < N1; j++)
		W0[i][j] = W0[i][j] - rate * dE_W0[i][j];

for (int i = 0; i < N1; i++)
	B1[i] = B1[i] - rate * dE_B1[i];

double w1[partition_N1][N2];
for (int i = start_N1; i < end_N1; i++)
	for (int j = 0; j < N2; j++)
		w1[i - start_N1][j] = W1[i][j] - rate * dE_W1[i][j];
MPI_Allgather(w1, partition_N1 *N2, MPI_DOUBLE, W1, partition_N1 *N2, MPI_DOUBLE, MPI_COMM_WORLD);

for (int i = 0; i < N2; i++)
	B2[i] = B2[i] - rate * dE_B2[i];

double w2[partition_N2][N3];
for (int i = start_N2; i < end_N2; i++)
	for (int j = 0; j < N3; j++)
		w2[i - start_N2][j] = W2[i][j] - rate * dE_W2[i][j];
MPI_Allgather(w2, partition_N2 *N3, MPI_DOUBLE, W2, partition_N2 *N3, MPI_DOUBLE, MPI_COMM_WORLD);

for (int i = 0; i < N3; i++)
	B3[i] = B3[i] - rate * dE_B3[i];
}
#else
double backward(double *O, vector<double> Y)
{
	int partition_N1, start_N1, end_N1, partition_N2, start_N2, end_N2, partition_N0, start_N0, end_N0;
	if (sub_N2 == 'A')
	{

		partition_N1 = N1 / size_N2;
		start_N1 = rank_N2 * partition_N1;
		end_N1 = start_N1 + partition_N1;

		partition_N2 = N2 / size_N2;
		start_N2 = rank_N2 * partition_N2;
		end_N2 = start_N2 + partition_N2;
	}

	if (sub_N0 == 'A')
	{

		partition_N0 = N0 / size_N0;
		start_N0 = rank_N0 * partition_N0;
		end_N0 = start_N0 + partition_N0;
	}

	// compute error
	double A = 1.7159;
	double B = 0.6666;
	err = 0;
	for (int i = 0; i < N3; i++)
		err += (O[i] - Y[i]) * (O[i] - Y[i]);
	err = err / N3;

	// compute dE_OO
	for (int i = 0; i < N3; i++)
		dE_OO[i] = (O[i] - Y[i]) * 2.0 / N3;

	// compute dOO_OS = OO dot (1-OO)
	for (int i = 0; i < N3; i++)
		dOO_OS[i] = B * (A - (OO[i] * OO[i] / A)); // A * B (1 - (tanh(Bx) * tanh(Bx)))

	// compute dE_OS = dE_OO dot dOO_OS
	for (int i = 0; i < N3; i++)
		dE_OS[i] = dE_OO[i] * dOO_OS[i];

	// compute dE_B3 = dE_OS
	for (int i = 0; i < N3; i++)
		dE_B3[i] = dE_OS[i];
	/**
	 * @brief Removing MPI parallelization here because N3 is small, Taking more time
	 *
	 */
	// compute dE_W2
	// if (sub_N2 == 'A')
	// {
	// 	double *dE_w2 = (double *)malloc(partition_N2 * N3 * sizeof(double));
	// 	if (my_rank == 0)
	// 	{
	// 		start_time = MPI_Wtime();
	// 	}
	// 	for (int i = start_N2; i < end_N2; i++)
	// 		for (int j = 0; j < N3; j++)
	// 		{
	// 			int temp_pos = (i - start_N2) * N3 + j;
	// 			*(dE_w2 + temp_pos) = dE_OS[j] * HO_2[i];
	// 		}
	// 	MPI_Allgather(dE_w2, partition_N2 * N3, MPI_DOUBLE, dE_W2, partition_N2 * N3, MPI_DOUBLE, N2_comm);
	// 	if (my_rank == 0)
	// 	{
	// 		end_time = MPI_Wtime();
	// 		loop_dE_W2 += (end_time - start_time);
	// 		start_time = 0;
	// 		end_time = 0;
	// 	}
	// 	free(dE_w2);

	// 	// compute dE_HO_2
	// 	if (my_rank == 0)
	// 	{
	// 		start_time = MPI_Wtime();
	// 	}
	// 	for (int i = start_N2; i < end_N2; i++)
	// 	{
	// 		dE_HO_2[i] = 0;
	// 		for (int j = 0; j < N3; j++)
	// 			dE_HO_2[i] += dE_OS[j] * W2[i][j];
	// 	}
	// 	MPI_Allgather(MPI_IN_PLACE, partition_N2, MPI_DOUBLE, dE_HO_2, partition_N2, MPI_DOUBLE, N2_comm);
	// 	if (my_rank == 0)
	// 	{
	// 		end_time = MPI_Wtime();
	// 		loop_dE_HO_2 += (end_time - start_time);
	// 		start_time = 0;
	// 		end_time = 0;
	// 	}
	// }

	// compute dE_W2
	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			dE_W2[i][j] = dE_OS[j] * HO_2[i];

	// compute dE_HO_2
	for (int i = 0; i < N2; i++)
	{
		dE_HO_2[i] = 0;
		for (int j = 0; j < N3; j++)
			dE_HO_2[i] += dE_OS[j] * W2[i][j];
	}
	// compute dHO_HS_2 = HO_2 dot (1-HO_2)
	for (int i = 0; i < N2; i++)
		// dHO_HS_2[i] = HO_2[i] * (1-HO_2[i]);
		dHO_HS_2[i] = B * (A - (HO_2[i] * HO_2[i] / A));

	// compute dE_HS_2 = dE_HO_2 dot dHO_HS_2
	for (int i = 0; i < N2; i++)
		dE_HS_2[i] = dE_HO_2[i] * dHO_HS_2[i];

	// compute dE_B2 = dE_HS_2
	for (int i = 0; i < N2; i++)
		dE_B2[i] = dE_HS_2[i];

	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
			dE_W1[i][j] = dE_HS_2[j] * HO_1[i];
	if (sub_N2 == 'A')
	{
		// compute dE_W1
		// if (my_rank == 0)
		// {
		// 	start_time = MPI_Wtime();
		// }
		// for (int i = start_N1; i < end_N1; i++)
		// 	for (int j = 0; j < N2; j++)
		// 	{
		// 		dE_W1[i][j] = dE_HS_2[j] * HO_1[i];
		// 	}
		// MPI_Allgather(MPI_IN_PLACE, partition_N1 * N2, MPI_DOUBLE, dE_W1, partition_N1 * N2, MPI_DOUBLE, N2_comm);
		// if (my_rank == 0)
		// {
		// 	end_time = MPI_Wtime();
		// 	loop_dE_W1 += (end_time - start_time);
		// 	start_time = 0;
		// 	end_time = 0;
		// }

		// compute dE_HO_1
		if (my_rank == 0)
		{
			start_time = MPI_Wtime();
		}
		for (int i = start_N1; i < end_N1; i++)
		{
			dE_HO_1[i] = 0;
			for (int j = 0; j < N2; j++)
				dE_HO_1[i] += dE_HS_2[j] * W1[i][j];
		}
		MPI_Allgather(MPI_IN_PLACE, partition_N1, MPI_DOUBLE, dE_HO_1, partition_N1, MPI_DOUBLE, N2_comm);
		if (my_rank == 0)
		{
			end_time = MPI_Wtime();
			loop_dE_HO_1 += (end_time - start_time);
			start_time = 0;
			end_time = 0;
		}
	}

	if (size_N0 > size_N2)
	{
		MPI_Bcast(dE_HO_1, N1, MPI_DOUBLE, 0, N0_comm);
	}

	// compute dHO_HS_1 = HO_1 dot (1-HO_1)
	for (int i = 0; i < N1; i++)
		// dHO_HS_1[i] = HO_1[i] * (1-HO_1[i]);
		dHO_HS_1[i] = B * (A - (HO_1[i] * HO_1[i] / A));

	// compute dE_HS_1 = dE_HO_1 dot dHO_HS_1
	for (int i = 0; i < N1; i++)
		dE_HS_1[i] = dE_HO_1[i] * dHO_HS_1[i];

	// compute dE_B1 = dE_HS_1
	for (int i = 0; i < N1; i++)
		dE_B1[i] = dE_HS_1[i];
	if (sub_N0 == 'A')
	{
		// compute dE_W0

		if (my_rank == 0)
		{
			start_time = MPI_Wtime();
		}
		for (int i = start_N0; i < end_N0; i++)
			for (int j = 0; j < N1; j++)
			{
				dE_W0[i][j] = dE_HS_1[j] * IN[i];
			}
		MPI_Allgather(MPI_IN_PLACE, partition_N0 * N1, MPI_DOUBLE, dE_W0, partition_N0 * N1, MPI_DOUBLE, N0_comm);
		if (my_rank == 0)
		{
			end_time = MPI_Wtime();
			loop_dE_W0 += (end_time - start_time);
			start_time = 0;
			end_time = 0;
		}

		// update W0, W1, W2, B1, B2, B3;

		// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &bb);
		// if (my_rank == 0)
		// {
		// 	start_time = MPI_Wtime();
		// }
		// for (int i = start_N0; i < end_N0; i++)
		// 	for (int j = 0; j < N1; j++)
		// 	{
		// 		W0[i][j] = W0[i][j] - rate * dE_W0[i][j];
		// 	}

		// MPI_Allgather(MPI_IN_PLACE, partition_N0 * N1, MPI_DOUBLE, W0, partition_N0 * N1, MPI_DOUBLE, N0_comm);
		// if (my_rank == 0)
		// {
		// 	end_time = MPI_Wtime();
		// 	loop_W0 += (end_time - start_time);
		// 	start_time = 0;
		// 	end_time = 0;
		// }
	}
	for (int i = 0; i < N0; i++)
		for (int j = 0; j < N1; j++)
			W0[i][j] = W0[i][j] - rate * dE_W0[i][j];

	if (size_N2_A > size_N0_A)
	{
		if (my_rank + size_N0_A < size_N2_A)
		{
			MPI_Request request;
			MPI_Isend(&dE_W0, N0 * N1, MPI_DOUBLE, my_rank + size_N0_A, 0, MPI_COMM_WORLD, &request);
		}
	}
	if (size_N2_A > size_N0_A)
	{
		if (rank_N2 > size_N0_A)
		{
			MPI_Recv(&dE_W0, N0 * N1, MPI_DOUBLE, rank_N2 - size_N0_A, 0, MPI_COMM_WORLD,
					 MPI_STATUS_IGNORE);
		}
	}
	/**
	 * @brief Two large data takes more time to broadcast
	 *
	 */
	// if (size_N2 > size_N0)
	// {
	// 	MPI_Bcast(dE_W0, N0 * N1, MPI_DOUBLE, 0, N2_comm);
	// 	MPI_Bcast(W0, N0 * N1, MPI_DOUBLE, 0, N2_comm);
	// }
	for (int i = 0; i < N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];

	// if (sub_N2 == 'A')
	// {
	// 	if (my_rank == 0)
	// 	{
	// 		start_time = MPI_Wtime();
	// 	}
	// 	for (int i = start_N1; i < end_N1; i++)
	// 		for (int j = 0; j < N2; j++)
	// 		{
	// 			W1[i][j] = W1[i][j] - rate * dE_W1[i][j];
	// 		}
	// 	MPI_Allgather(MPI_IN_PLACE, partition_N1 * N2, MPI_DOUBLE, W1, partition_N1 * N2, MPI_DOUBLE, N2_comm);
	// 	if (my_rank == 0)
	// 	{
	// 		end_time = MPI_Wtime();
	// 		loop_W1 += (end_time - start_time);
	// 		start_time = 0;
	// 		end_time = 0;
	// 	}
	// }
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
			W1[i][j] = W1[i][j] - rate * dE_W1[i][j];

	for (int i = 0; i < N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];
	/**
	 * @brief Removing MPI parallelization here because N3 is small, Taking more time
	 *
	 */
	// if (sub_N2 == 'A')
	// {
	// 	double *w2 = (double *)malloc(partition_N2 * N3 * sizeof(double));
	// 	if (my_rank == 0)
	// 	{
	// 		start_time = MPI_Wtime();
	// 	}
	// 	for (int i = start_N2; i < end_N2; i++)
	// 		for (int j = 0; j < N3; j++)
	// 		{
	// 			int temp_pos = (i - start_N2) * N3 + j;
	// 			*(w2 + temp_pos) = W2[i][j] - rate * dE_W2[i][j];
	// 		}
	// 	MPI_Allgather(w2, partition_N2 * N3, MPI_DOUBLE, W2, partition_N2 * N3, MPI_DOUBLE, N2_comm);
	// 	if (my_rank == 0)
	// 	{
	// 		end_time = MPI_Wtime();
	// 		loop_W2 += (end_time - start_time);
	// 		start_time = 0;
	// 		end_time = 0;
	// 	}
	// 	free(w2);
	// }
	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i = 0; i < N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];
	// cout << "In Normal" << endl;
}
#endif
/**
 * @brief The function is used to train neural network
 *
 * @param iter
 * @param filename
 */
void train(int iter, string filename)
{

	time_t t;	   // t passed as argument in function time()
	struct tm *tt; // decalring variable for localtime()
	time(&t);	   // passing argument to time()
	tt = localtime(&t);
	ofstream myfile;
	myfile.open(filename);
	int jj;
	int index;
	for (int kk = 0; kk < iter; kk++)
	{
		index = kk % data_pos.size();
		forward(data_X[data_pos[index]]);

		backward(OO, data_Y[data_pos[index]]);

		if (my_rank == 0)
		{
			if (kk % 10000 == 0)
			{

				int winner = 0;
				double max = OO[0];

				for (int i = 0; i < N3; i++)
				{
					if (data_Y[data_pos[index]][i] == 1.7159)
					{
						jj = i;
					}
				}
				for (int i = 0; i < N3; i++)
				{
					if (OO[i] > 0)
					{
						winner = i;
					}
				}

				myfile << "[Train] Iter " << kk << ": err =" << err << ", Y = " << jj << "\n";
				for (int i = 0; i < N3; i++)
					myfile << "OO[" << i << "] = " << OO[i] << "\n";

				time(&t);
				myfile << asctime(localtime(&t)) << "\n";
				myfile.flush();
			}
		}
	}
	myfile.close();
}
/**
 * @brief The function is used to test neural network
 *
 * @param filename
 */
void test(string filename)
{
	ofstream myfile;
	myfile.open(filename);
	vector<int> correct(N3, 0);
	vector<int> incorrect(N3, 0);
	int iter = 10000;
	int count = 0;
	int jj = 0, winner = 0;
	for (int kk = 0; kk < iter; kk++)
	{
		int index = kk % data_pos.size();
		forward(data_X[data_pos[index]]);
		for (int i = 0; i < N3; i++)
		{
			if (data_Y[data_pos[index]][i] == 1.7159)
			{
				jj = i;
			}
		}
		for (int i = 0; i < N3; i++)
		{
			if (OO[i] > 0)
			{
				winner = i;
				count++;
			}
		}
		if (winner == jj && count == 1)
		{
			correct[jj]++;
		}
		else
		{
			incorrect[jj]++;
		}
		count = 0;
	}

	for (int i = 0; i < N3; i++)
	{
		double total = 0;
		double accuracy = 0.0;
		total = correct[i] + incorrect[i];
		if (total != 0)
		{
			accuracy = (double)correct[i] / total;
		}
		if (my_rank == 0)
		{
			myfile << "Total : " << total << " Correct : " << correct[i] << " Accuracy of " << i << " is : " << accuracy * 100 << "%" << endl;
		}
	}
	myfile.close();
}

/**
 * @brief Main function call for MPI
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char *argv[])
{

	if (argc < 4)
	{
		cout << " The command should follow the syntax below " << endl;
		cout << " ./project1.x <iter> <train_file_output.txt> <test_file_output.txt>" << endl;
		exit(0);
	}
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int number_of_images = 0, img_size = 0;
	read_mnist_images("./data/train_data/input/train-images-idx3-ubyte", number_of_images, img_size);

	int number_of_labels = 0;
	read_mnist_labels("./data/train_data/output/train-labels-idx1-ubyte", number_of_labels);

	if (my_rank == 0)
	{
		// randomize weights
		int seed = 30;
		default_random_engine generator(seed);
		uniform_real_distribution<double> distribution(-0.05, 0.05);

		for (int i = 0; i < N1; i++)
			B1[i] = distribution(generator);
		for (int i = 0; i < N0; i++)
			for (int j = 0; j < N1; j++)
				W0[i][j] = distribution(generator);

		for (int i = 0; i < N2; i++)
			B2[i] = distribution(generator);
		for (int i = 0; i < N1; i++)
			for (int j = 0; j < N2; j++)
				W1[i][j] = distribution(generator);

		for (int i = 0; i < N3; i++)
			B3[i] = distribution(generator);
		for (int i = 0; i < N2; i++)
			for (int j = 0; j < N3; j++)
				W2[i][j] = distribution(generator);
	}
	MPI_Bcast(B1, N1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B2, N2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B3, N3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(W0, N0 * N1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(W1, N1 * N2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(W2, N2 * N3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	/**
	 * @brief Color for every Layer in NN
	 *
	 */
	int color_N0 = -1;
	int key_N0;

	int color_N2 = -1;
	int key_N2;

	if (nearest_lowest_divisor(N2, size, my_rank))
	{
		color_N2 = 1;
		sub_N2 = 'A';
	}
	else
	{
		color_N2 = 0;
		sub_N2 = 'B';
	}

	if (nearest_lowest_divisor(N0, size, my_rank))
	{
		color_N0 = 1;
		sub_N0 = 'A';
	}
	else
	{
		color_N0 = 0;
		sub_N0 = 'B';
	}

	MPI_Comm_split(MPI_COMM_WORLD, color_N0, my_rank, &N0_comm);
	MPI_Comm_split(MPI_COMM_WORLD, color_N2, my_rank, &N2_comm);

	MPI_Comm_size(N2_comm, &size_N2);
	MPI_Comm_rank(N2_comm, &rank_N2);
	MPI_Comm_size(N0_comm, &size_N0);
	MPI_Comm_rank(N0_comm, &rank_N0);

	MPI_Comm_size(N2_comm, &size_N2_A);
	MPI_Comm_size(N0_comm, &size_N0_A);
	MPI_Bcast(&size_N2_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&size_N0_A, 1, MPI_INT, 0, MPI_COMM_WORLD);

	printf("[MPI process %d] I am now MPI process %d. subcommunicator N0 %c and MPI process %d. in subcommunicator N2 %c  size %d %d.\n", my_rank, rank_N0, sub_N0, rank_N2, sub_N2, size_N0, size_N2);

	if (argc == 4)
	{
		if (sub_N2 == 'A' || sub_N0 == 'A')
		{
			start_time = MPI_Wtime();
			train(atoi(argv[1]), argv[2]);
			end_time = MPI_Wtime();
			loop_trainTime += (end_time - start_time);
		}
	}
	if (my_rank == 0)
	{
		cout << "loop_HS_1:   My Rank : " << my_rank << " MPI time : " << loop_HS_1 << endl;
		cout << "loop_HS_2:  My Rank : " << my_rank << " MPI time : " << loop_HS_2 << endl;
		cout << "loop_dE_HO_1:  My Rank : " << my_rank << " MPI time : " << loop_dE_HO_1 << endl;
		cout << "loop_dE_W0: My Rank : " << my_rank << " MPI time : " << loop_dE_W0 << endl;
		cout << "Loop whole training time: My Rank : " << my_rank << " MPI time : " << end_time - start_time << endl;
	}
	// /* For Testing */
	if (sub_N2 == 'A')
	{
		test(argv[3]);
	}
	MPI_Finalize();
}
