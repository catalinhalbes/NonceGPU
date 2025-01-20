CXXFLAGS = --std=c++20 -O3
NVCCFLAGS = $(CXXFLAGS) --Werror cross-execution-space-call
LDFLAGS = 
TESTFILE = testfile.txt

.PHONY: test clean

nonce: nonce.o sha1.o
	nvcc $(NVCCFLAGS) -o nonce nonce.o sha1.o $(LDFLAGS)

sha1.o: sha1.c
	g++ $(CXXFLAGS) -c sha1.c

nonce.o: nonce.cu
	nvcc $(NVCCFLAGS) -c nonce.cu

clean:
	rm -f *.o

test: 
	./nonce $(TESTFILE) nonce.bin
	cat $(TESTFILE) nonce.bin | sha1sum
