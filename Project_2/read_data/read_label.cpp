/*
/Users/saptarshibhowmik/Documents/CDA_5125/project_1/data/train_data/output/train-labels-idx1-ubyte
*/
#include <iostream>
#include <fstream>
using namespace std;
typedef unsigned char uchar;
uchar* read_mnist_labels(string full_path, int& number_of_labels) {
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
	for (int ii = 0; ii < number_of_labels; ii++){
		cout << (int)_dataset[ii] << endl;
	} 
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

int main(){
	int number_of_labels = 0;
	read_mnist_labels("/Users/saptarshibhowmik/Documents/CDA_5125/project_1/data/train_data/output/train-labels-idx1-ubyte", number_of_labels);
}

