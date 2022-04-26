CC=mpicc
CFLAGS=-I.
CXX=mpicxx
CXXFLAGS = -msse2 -std=c++11 -Wall


project4.x:
	$(CXX) $< $(CXXFLAGS) project4.cpp -o project4.x

clean:
	rm -rf *.o *.x

cleantxt:
	rm -rf *.txt

test_mpi:
	mpirun -n 4 project4.x 10 train_mpi.txt test_mpi.txt > prob_16.txt
test_non_mpi:
	mpirun -n 1 project4.x 10 train.txt test.txt > prob_1.txt

