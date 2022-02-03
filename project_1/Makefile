CC=gcc
CFLAGS=-I.
CXX = g++
CXXFLAGS = --std=c++11 -Wall

%.o: %.cpp 
	$(CXX) -c -o $@ $< $(CXXFLAGS)

project1.x: project1.o
	$(CXX) -o project1.x project1.o

clean:
	rm -rf *.o *.x
