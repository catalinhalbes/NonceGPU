#include <stdint.h>

#define SCHEDULE(i)  \
	temp = schedule[(i-3)&0xF] ^ schedule[(i-8)&0xF] ^ schedule[(i-14)&0xF] ^ schedule[(i-16)&0xF];  \
	schedule[i & 0xF] = temp << 1 | temp >> 31;

#define ROUND0a(a,b,c,d,e,i)  \
	schedule[i] = (block[i] << 24) | ((block[i] & 0xFF00) << 8) | ((block[i] >> 8) & 0xFF00) | (block[i] >> 24);  \
	ROUNDTAIL(a, b, e, ((b & c) | (~b & d)), i, 0x5A827999)

#define ROUND0b(a,b,c,d,e,i)  \
	SCHEDULE(i)  \
	ROUNDTAIL(a, b, e, ((b & c) | (~b & d)), i, 0x5A827999)

#define ROUND1(a,b,c,d,e,i)  \
	SCHEDULE(i)  \
	ROUNDTAIL(a, b, e, (b ^ c ^ d), i, 0x6ED9EBA1)

#define ROUND2(a,b,c,d,e,i)  \
	SCHEDULE(i)  \
	ROUNDTAIL(a, b, e, ((b & c) ^ (b & d) ^ (c & d)), i, 0x8F1BBCDC)

#define ROUND3(a,b,c,d,e,i)  \
	SCHEDULE(i)  \
	ROUNDTAIL(a, b, e, (b ^ c ^ d), i, 0xCA62C1D6)

#define ROUNDTAIL(a,b,e,f,i,k)  \
	e += (a << 5 | a >> 27) + f + k + schedule[i & 0xF];  \
	b = b << 30 | b >> 2;

#define REVERSE_16(x) ((x << 8) | (x >> 8))
#define REVERSE_32(x) (((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) | (x >> 24))
#define REVERSE_64(x) (((x & 0xFF) << 56) | ((x & 0xFF00) << 40) | ((x & 0xFF0000) << 24) | ((x & 0xFF000000) << 8) | ((x & 0xFF00000000) >> 8) | ((x & 0xFF0000000000) >> 24) | (x << 56))

void sha1_compress(uint32_t state[5], uint32_t block[16]);