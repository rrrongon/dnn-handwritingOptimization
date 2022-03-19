CC=gcc
CFLAGS=-I.
CXX =g++
CXXFLAGS = -fopenmp -O3 -msse2 -std=c++11 -Wall

run:
	$(CXX) $< $(CXXFLAGS) project2.cpp -o project2.x

clean:
	rm -rf *.o *.x
