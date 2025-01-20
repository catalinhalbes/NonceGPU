.PHONY: all clean run

all: compute

compute-hash-unoptimized.spv: compute-hash-only-uint32.comp
	glslc -O compute-hash-only-uint32.comp -o compute-hash-unoptimized.spv

compute-hash.spv: compute-hash-unoptimized.spv
	spirv-opt -O compute-hash-unoptimized.spv -o compute-hash.spv

compute: compute.cpp sha1-fast.c compute-hash.spv
	g++ --std=c++20 -Wall -O3 compute.cpp sha1-fast.c -o compute `pkg-config --cflags --libs vulkan`

run: compute
	./compute
	printf "Hash:  %s %s\n" `cat testfile.txt nonce.bin | sha1sum`

clean:
	rm -f compute compute-hash-unoptimized.spv compute-hash.spv
