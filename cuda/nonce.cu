/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <chrono>

#include <cuda_runtime.h>

#include "sha1.h"

#define PREFIX_LEN 3
#define MAX_NONCE_LEN 5
uint64_t GPU_THREADS = 256;
uint64_t MAX_N_BATCH = 0x100000;

__device__ void cuda_sha1_compress(uint32_t state[5], uint32_t block[16]) {
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	
	uint32_t schedule[16];
	uint32_t temp;
	ROUND0a(a, b, c, d, e,  0)
	ROUND0a(e, a, b, c, d,  1)
	ROUND0a(d, e, a, b, c,  2)
	ROUND0a(c, d, e, a, b,  3)
	ROUND0a(b, c, d, e, a,  4)
	ROUND0a(a, b, c, d, e,  5)
	ROUND0a(e, a, b, c, d,  6)
	ROUND0a(d, e, a, b, c,  7)
	ROUND0a(c, d, e, a, b,  8)
	ROUND0a(b, c, d, e, a,  9)
	ROUND0a(a, b, c, d, e, 10)
	ROUND0a(e, a, b, c, d, 11)
	ROUND0a(d, e, a, b, c, 12)
	ROUND0a(c, d, e, a, b, 13)
	ROUND0a(b, c, d, e, a, 14)
	ROUND0a(a, b, c, d, e, 15)
	ROUND0b(e, a, b, c, d, 16)
	ROUND0b(d, e, a, b, c, 17)
	ROUND0b(c, d, e, a, b, 18)
	ROUND0b(b, c, d, e, a, 19)
	ROUND1(a, b, c, d, e, 20)
	ROUND1(e, a, b, c, d, 21)
	ROUND1(d, e, a, b, c, 22)
	ROUND1(c, d, e, a, b, 23)
	ROUND1(b, c, d, e, a, 24)
	ROUND1(a, b, c, d, e, 25)
	ROUND1(e, a, b, c, d, 26)
	ROUND1(d, e, a, b, c, 27)
	ROUND1(c, d, e, a, b, 28)
	ROUND1(b, c, d, e, a, 29)
	ROUND1(a, b, c, d, e, 30)
	ROUND1(e, a, b, c, d, 31)
	ROUND1(d, e, a, b, c, 32)
	ROUND1(c, d, e, a, b, 33)
	ROUND1(b, c, d, e, a, 34)
	ROUND1(a, b, c, d, e, 35)
	ROUND1(e, a, b, c, d, 36)
	ROUND1(d, e, a, b, c, 37)
	ROUND1(c, d, e, a, b, 38)
	ROUND1(b, c, d, e, a, 39)
	ROUND2(a, b, c, d, e, 40)
	ROUND2(e, a, b, c, d, 41)
	ROUND2(d, e, a, b, c, 42)
	ROUND2(c, d, e, a, b, 43)
	ROUND2(b, c, d, e, a, 44)
	ROUND2(a, b, c, d, e, 45)
	ROUND2(e, a, b, c, d, 46)
	ROUND2(d, e, a, b, c, 47)
	ROUND2(c, d, e, a, b, 48)
	ROUND2(b, c, d, e, a, 49)
	ROUND2(a, b, c, d, e, 50)
	ROUND2(e, a, b, c, d, 51)
	ROUND2(d, e, a, b, c, 52)
	ROUND2(c, d, e, a, b, 53)
	ROUND2(b, c, d, e, a, 54)
	ROUND2(a, b, c, d, e, 55)
	ROUND2(e, a, b, c, d, 56)
	ROUND2(d, e, a, b, c, 57)
	ROUND2(c, d, e, a, b, 58)
	ROUND2(b, c, d, e, a, 59)
	ROUND3(a, b, c, d, e, 60)
	ROUND3(e, a, b, c, d, 61)
	ROUND3(d, e, a, b, c, 62)
	ROUND3(c, d, e, a, b, 63)
	ROUND3(b, c, d, e, a, 64)
	ROUND3(a, b, c, d, e, 65)
	ROUND3(e, a, b, c, d, 66)
	ROUND3(d, e, a, b, c, 67)
	ROUND3(c, d, e, a, b, 68)
	ROUND3(b, c, d, e, a, 69)
	ROUND3(a, b, c, d, e, 70)
	ROUND3(e, a, b, c, d, 71)
	ROUND3(d, e, a, b, c, 72)
	ROUND3(c, d, e, a, b, 73)
	ROUND3(b, c, d, e, a, 74)
	ROUND3(a, b, c, d, e, 75)
	ROUND3(e, a, b, c, d, 76)
	ROUND3(d, e, a, b, c, 77)
	ROUND3(c, d, e, a, b, 78)
	ROUND3(b, c, d, e, a, 79)
	
	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
}

// __global__ void kernel_find_nonce_sha1(
// 	const uint32_t state[5], 
// 	const uint32_t message[16], 
// 	uint32_t len, 
// 	uint32_t prefix_len, 
// 	uint64_t* found_nonce, 
// 	uint32_t nonce_len, 
// 	uint64_t offset)
// {
// 	offset += blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t sink[5], block[16];
// 	uint8_t* sink_ptr  = (uint8_t*) sink;
// 	uint8_t* block_ptr = (uint8_t*) block;
// 	memcpy(sink_ptr, state, 20);
// 	uint32_t rem = len % 64;
// 	memcpy(block_ptr, message, rem);
// 	if (rem + nonce_len >= 64) {
// 		uint32_t comp = 64 - rem;
// 		memcpy(block_ptr + rem, &offset, comp);
// 		cuda_sha1_compress(sink, block);
// 		rem = nonce_len - comp;
// 		// memset(block_ptr, 0, 64);
// 		memcpy(block_ptr, ((uint8_t*)(&offset)) + comp, rem);
// 	}
// 	else {
// 		memcpy(block_ptr + rem, &offset, nonce_len);
// 		rem += nonce_len;
// 	}
// 	len += nonce_len;
// 	block_ptr[rem] = 0x80;
// 	rem += 1;
// 	if (rem <= 56) {
// 		memset(block_ptr + rem, 0, 56 - rem);
// 	}
// 	else {
// 		memset(block_ptr + rem, 0, 64 - rem);
// 		cuda_sha1_compress(sink, block);
// 		memset(block, 0, 56);
// 	}
// 	uint64_t longLen = ((uint64_t)len) << 3;
// 	block[14] = REVERSE_32((uint32_t) (longLen >> 32));
// 	block[15] = REVERSE_32((uint32_t) longLen);
// 	cuda_sha1_compress(sink, block);
// 	sink[0] = REVERSE_32(sink[0]);
// 	sink[1] = REVERSE_32(sink[1]);
// 	sink[2] = REVERSE_32(sink[2]);
// 	sink[3] = REVERSE_32(sink[3]);
// 	sink[4] = REVERSE_32(sink[4]);
// 	uint32_t i;
//     for (i = 0; i < prefix_len; i++) {
//         if (sink_ptr[i] != 0) break;
//     }
// 	if (i == prefix_len)
// 		*found_nonce = offset;
// }

__constant__ uint32_t device_state[5];
__constant__ uint32_t device_message[16];
__constant__ uint32_t device_len;
__constant__ uint32_t device_prefix_len;

__global__ void kernel_find_nonce_sha1(
	uint64_t* device_found_nonce, 
	const uint32_t device_nonce_len, 
	uint64_t device_offset)
{
	device_offset += blockIdx.x * blockDim.x + threadIdx.x;
	uint tid = device_offset;
	uint extraNonce = device_offset >> 32;

    uint hash[5];
	hash[0] = device_state[0];
    hash[1] = device_state[1];
    hash[2] = device_state[2];
    hash[3] = device_state[3];
    hash[4] = device_state[4];
    
	// just to avoid some conditional execution make the buffer larger 
	// all threads will hash a message of equal length, meaning that tey will follow the same conditional path
	// but the code was becoming way too complex to handle :D
	uint block[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    uint rem                 = (device_len % 64) >> 2;
	uint remBytes            = device_len % 4;
    uint remBits             = remBytes << 3;
	uint remBits32Complement = 32 - remBits;
    
	// copy the bytes from the message (should pe padded with 0s so it's safe to copy more)
    for (uint j = 0; j < rem; j++) {
        block[j] = device_message[j];
    }

	if (remBytes > 0) {
		block[rem    ] = (device_message[rem] & (0xffffffffu >> remBits32Complement))                         | ((tid        & (0xffffffffu >> remBits)) << remBits);
		block[rem + 1] = ((tid                & (0xffffffffu << remBits32Complement)) >> remBits32Complement) | ((extraNonce & (0xffffffffu >> remBits)) << remBits);
		block[rem + 2] = ((extraNonce         & (0xffffffffu << remBits32Complement)) >> remBits32Complement);
	} 
	else {
		block[rem    ] = tid;
		block[rem + 1] = extraNonce;
	}
	
	if (rem >= 14 && (remBytes + device_nonce_len) >= ((16 - rem) * 4)) {
		// the buffer is completely full
		cuda_sha1_compress(hash, block);

		for (uint j = 0; j < 16; j++) {
			block[j] = 0;
		}
		
		block[0] = block[16];
		block[1] = block[17];
		rem = (remBytes + device_nonce_len) / 4 - 16 + rem;
	}
	else {
		// the buffer is not full continue normally
		rem += (remBytes + device_nonce_len) / 4;
	}

	remBytes            = (remBytes + device_nonce_len) % 4;
	remBits             = remBytes << 3;
	remBits32Complement = 32 - remBits;

	// add the last 0x80
	if (remBytes > 0) {
		block[rem] = (block[rem] & (0xffffffffu >> remBits32Complement)) | (0x80 << remBits);
	}
	else {
		block[rem] = 0x80;
	}

	block[rem + 1] = 0;
	block[rem + 2] = 0;

	if (rem >= 14) {
		cuda_sha1_compress(hash, block);
        for (uint j = 0; j < 14; j++) {
            block[j] = 0;
        }
	}

	uint bitLen = (device_len + device_nonce_len) << 3; // let's hope the file isn't greater than 512MB
	bitLen = REVERSE_32(bitLen);
	block[14] = 0;
    block[15] = bitLen;
    
    cuda_sha1_compress(hash, block);

	// remember that the bytes are in big endian order, you need to check from the most significant to the least significant
	uint i, ok = 1;
	rem = device_prefix_len / 4;
	for (i = 0; i < rem; i++) {
		if (hash[i] != 0) {
			ok = 0;
			break;
		}
	}

	if ((hash[i] & (0xffffffffu << ((4 - device_prefix_len % 4) << 3))) != 0) {
		ok = 0;
	}

	if (ok > 0) {
		*device_found_nonce = device_offset;
	}
}

uint32_t find_sha1_nonce(
	const uint32_t state[5], 
	const uint32_t message[16], 
	uint32_t len, 
	uint32_t prefix_len, 
	uint64_t* nonce, 
	uint32_t max_nonce_len) 
{
	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_find_nonce_sha1);

	printf("minGridSize: %d\nblockSize: %d\n", minGridSize, blockSize);
    
	uint64_t *cuda_nonce;
    *nonce = 0l;

	cudaMallocManaged(&cuda_nonce, sizeof(uint64_t));
	memcpy(cuda_nonce, nonce, sizeof(uint64_t)); // unified memory is managed by the driver, we can simply copy data using memcpy

	cudaMemcpyToSymbol(device_state, state, 5 * sizeof(uint32_t));
	cudaMemcpyToSymbol(device_message, message, 16 * sizeof(uint32_t));
	cudaMemcpyToSymbol(device_len, &len, 1 * sizeof(uint32_t));
	cudaMemcpyToSymbol(device_prefix_len, &prefix_len, 1 * sizeof(uint32_t));

	uint64_t max_val = 1;
	uint32_t current_nonce_len;

	for (current_nonce_len = 1; current_nonce_len <= max_nonce_len; current_nonce_len++) {
		uint64_t prev_max_val = max_val;
		max_val *= 256; // one uint8_t more
		uint64_t end = max_val - 1;

		if (prev_max_val > max_val) {
			end = max_val = UINT64_MAX;
		}

		uint32_t n_batch = (max_val > MAX_N_BATCH)? MAX_N_BATCH: max_val;
		uint32_t thread = GPU_THREADS;
		uint32_t block = (n_batch + thread - 1) / thread;
		uint64_t offset = 0l;

		bool run = true;
		uint64_t aux = 0;
		while (run) {
			if (offset >= end) run = false;

			if (offset >> 32 != aux) {
				printf("Start: %08lx\n", offset >> 32);
				aux = offset >> 32;
			}

			kernel_find_nonce_sha1 <<< block, thread >>> (
				cuda_nonce, 
				current_nonce_len, 
				offset);
			cudaMemcpy(nonce, cuda_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess) {
				printf("Error cuda sha1 hash: %s \n", cudaGetErrorString(error));
				break;
			}

			if (*nonce != 0) break;

			uint32_t prev = offset;
			offset += n_batch;
			if (prev > offset) // int overflow
				offset = max_val;
		}

		if (*nonce != 0) break;
	}

	cudaFree(cuda_nonce);

	if (*nonce == 0) return 0;
	return current_nonce_len;
}

void start_hashing_file(char* filename, uint32_t state[5], uint8_t buf[64], uint32_t *len) {
    FILE* f = fopen(filename, "rb");

    state[0] = 0x67452301;
    state[1] = 0xEFCDAB89;
    state[2] = 0x98BADCFE;
    state[3] = 0x10325476;
    state[4] = 0xC3D2E1F0;
    
    while (true) {
        uint64_t read_size = fread(buf, 1, 64, f); // not too efficient to read this little, but anyway
        *len += read_size;

        if (read_size != 64) {
            //printf("read: %d\nlen: %d\n", read_size, len);
            memset(buf + read_size, 0x00, 64 - read_size);
            break; 
        }

        sha1_compress(state, (uint32_t*) buf);
    }
}

void save_nonce(char* filename, uint64_t nonce, uint32_t nonce_len) {
	FILE* f = fopen(filename, "wb");
	fwrite(&nonce, sizeof(uint8_t), nonce_len, f);
	fclose(f);
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		printf("Usage: %s <filename> <nonce.out>\n", argv[0]);
		return 1;
	}

	uint64_t nonce;

    printf ("Prefix length: %d\n", PREFIX_LEN);

	uint32_t state[5], message[64], len;
	start_hashing_file(argv[1], state, (uint8_t*) message, &len);

	auto start = std::chrono::high_resolution_clock::now();
    uint32_t nonce_len = find_sha1_nonce(state, message, len, PREFIX_LEN, &nonce, MAX_NONCE_LEN);
	auto end = std::chrono::high_resolution_clock::now();

	double elapsed = std::chrono::duration<double>(end - start).count();

	if (nonce_len == 0) {
		printf("\nDidn't find a valid nonce!\n\n");
	}
	else {
		printf("\nFound nonce!\n\n");
		save_nonce(argv[2], nonce, nonce_len);
	}
	printf("Elapsed: %lfs\n", elapsed);
}
