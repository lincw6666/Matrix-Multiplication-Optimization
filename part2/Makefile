CXXFLAGS := -std=c++17 -g3 -O3 -Wall
CXXFLAGS += -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
CXXFLAGS += -mavx2

all: matmul

matmul: main.o matmul.o
	OMPI_CXX=clang++-11 mpicxx ${CXXFLAGS} main.o matmul.o -o matmul

main.o: main.cc
	OMPI_CXX=clang++-11 mpicxx ${CXXFLAGS} -c main.cc -o main.o

matmul.o: matmul.c
	OMPI_CXX=clang++-11 mpicxx ${CXXFLAGS} -c matmul.c -o matmul.o

clean:
	rm *.o matmul

