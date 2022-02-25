/*
/Users/saptarshibhowmik/Documents/CDA_5125/project_1/data/train_data/input/train-images-idx3-ubyte
*/
#include <iostream>
#include <fstream>
using namespace std;
typedef unsigned char uchar;
uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
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
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}


int main(){
	int number_of_images = 0, size = 0;
	read_mnist_images("/Users/saptarshibhowmik/Documents/CDA_5125/project_1/data/train_data/input/train-images-idx3-ubyte", number_of_images, size);
}

