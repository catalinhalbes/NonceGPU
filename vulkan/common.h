#define REVERSE_16(x) ((x << 8) | (x >> 8))
#define REVERSE_32(x) (((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) | (x >> 24))
//#define REVERSE_64(x) (((x & 0xFF) << 56) | ((x & 0xFF00) << 40) | ((x & 0xFF0000) << 24) | ((x & 0xFF000000) << 8) | ((x & 0xFF00000000) >> 8) | ((x & 0xFF0000000000) >> 24) | (x << 56))
#define GPU_THREADS_PER_BLOCK 256
#define MAX_GPU_BLOCKS 0x400000
#define MAX_NONCE_LEN 5
