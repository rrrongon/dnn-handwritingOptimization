CC=gcc
CFLAGS=-I.
CXX = g++
CXXFLAGS = -O3 -msse2 --std=c++11 -Wall

%.o: %.cpp 
	$(CXX) -c -o $@ $< $(CXXFLAGS)

project2.x: project2.o
	$(CXX) -o project2.x project2.o

clean:
	rm -rf *.o *.x
