#include <stdlib.h>
typedef int u32;
#define FOR(var,count) for(int var = 0; var < (count); var++)
float fade(float t){
    return t*t*t*(t*(t*6-15)+10);
}
int fastfloor(float x){
    return (int)x<x ? (int)x : (int)x-1;
}
float lerp(float t, float a, float b){
    return a+t*(b-a);
}
unsigned char perm[512];
void pnoiseInit(u32 seed){
    srand(seed);
    FOR(i,sizeof(perm)) perm[i]=i;
    FOR(i,sizeof(perm)){
        int x = perm[i];
        int j = rand()%sizeof(perm);
        perm[i] = perm[j];
        perm[j] = x;
    }
}
float grad1(int hash, float x){
    int h = hash & 15;
    float grad = 1.0 + (h & 7);  // Gradient value 1.0, 2.0, ..., 8.0
    if (h&8) grad = -grad;         // and a random sign for the gradient
    return grad * x;           // Multiply the gradient with the distance
}
float grad2(int hash, float x, float y){
    int h = hash & 7;      // Convert low 3 bits of hash code
    float u = h<4 ? x : y;  // into 8 simple gradient directions,
    float v = h<4 ? y : x;  // and compute the dot product with (x,y).
    return ((h&1)? -u : u) + ((h&2)? -2.0*v : 2.0*v);
}
float grad3(int hash, float x, float y, float z){
    int h = hash & 15;     // Convert low 4 bits of hash code into 12 simple
    float u = h<8 ? x : y; // gradient directions, and compute dot product.
    float v = h<4 ? y : h==12||h==14 ? x : z; // Fix repeats at h = 12 to 15
    return ((h&1) ? -u : u) + ((h&2) ? -v : v);
}
float pnoise1(float x){
    int ix0, ix1;
    float fx0, fx1;
    float s, n0, n1;
    ix0 = fastfloor(x); // Integer part of x
    fx0 = x - ix0;       // Fractional part of x
    fx1 = fx0 - 1.0f;
    ix1 = (ix0+1) & 0xff;
    ix0 = ix0 & 0xff;    // Wrap to 0..255
    s = fade(fx0);
    n0 = grad1(perm[ix0], fx0);
    n1 = grad1(perm[ix1], fx1);
    return 0.188f * lerp(s, n0, n1);
}
float pnoise2(float x, float y){
    int ix0, iy0, ix1, iy1;
    float fx0, fy0, fx1, fy1;
    float s, t, nx0, nx1, n0, n1;
    ix0 = fastfloor(x); // Integer part of x
    iy0 = fastfloor(y); // Integer part of y
    fx0 = x - ix0;        // Fractional part of x
    fy0 = y - iy0;        // Fractional part of y
    fx1 = fx0 - 1.0f;
    fy1 = fy0 - 1.0f;
    ix1 = (ix0 + 1) & 0xff;  // Wrap to 0..255
    iy1 = (iy0 + 1) & 0xff;
    ix0 = ix0 & 0xff;
    iy0 = iy0 & 0xff;
    t = fade(fy0);
    s = fade(fx0);
    nx0 = grad2(perm[ix0 + perm[iy0]], fx0, fy0);
    nx1 = grad2(perm[ix0 + perm[iy1]], fx0, fy1);
    n0 = lerp(t, nx0, nx1);
    nx0 = grad2(perm[ix1 + perm[iy0]], fx1, fy0);
    nx1 = grad2(perm[ix1 + perm[iy1]], fx1, fy1);
    n1 = lerp(t, nx0, nx1);
    return 0.507f * lerp(s, n0, n1);
}
float pnoise3(float x, float y, float z){
    int ix0, iy0, ix1, iy1, iz0, iz1;
    float fx0, fy0, fz0, fx1, fy1, fz1;
    float s, t, r;
    float nxy0, nxy1, nx0, nx1, n0, n1;
    ix0 = fastfloor(x); // Integer part of x
    iy0 = fastfloor(y); // Integer part of y
    iz0 = fastfloor(z); // Integer part of z
    fx0 = x - ix0;        // Fractional part of x
    fy0 = y - iy0;        // Fractional part of y
    fz0 = z - iz0;        // Fractional part of z
    fx1 = fx0 - 1.0f;
    fy1 = fy0 - 1.0f;
    fz1 = fz0 - 1.0f;
    ix1 = (ix0 + 1) & 0xff; // Wrap to 0..255
    iy1 = (iy0 + 1) & 0xff;
    iz1 = (iz0 + 1) & 0xff;
    ix0 = ix0 & 0xff;
    iy0 = iy0 & 0xff;
    iz0 = iz0 & 0xff;
    r = fade(fz0);
    t = fade(fy0);
    s = fade(fx0);
    nxy0 = grad3(perm[ix0 + perm[iy0 + perm[iz0]]], fx0, fy0, fz0);
    nxy1 = grad3(perm[ix0 + perm[iy0 + perm[iz1]]], fx0, fy0, fz1);
    nx0 = lerp(r, nxy0, nxy1);
    nxy0 = grad3(perm[ix0 + perm[iy1 + perm[iz0]]], fx0, fy1, fz0);
    nxy1 = grad3(perm[ix0 + perm[iy1 + perm[iz1]]], fx0, fy1, fz1);
    nx1 = lerp(r, nxy0, nxy1);
    n0 = lerp(t, nx0, nx1);
    nxy0 = grad3(perm[ix1 + perm[iy0 + perm[iz0]]], fx1, fy0, fz0);
    nxy1 = grad3(perm[ix1 + perm[iy0 + perm[iz1]]], fx1, fy0, fz1);
    nx0 = lerp(r, nxy0, nxy1);
    nxy0 = grad3(perm[ix1 + perm[iy1 + perm[iz0]]], fx1, fy1, fz0);
    nxy1 = grad3(perm[ix1 + perm[iy1 + perm[iz1]]], fx1, fy1, fz1);
    nx1 = lerp(r, nxy0, nxy1);
    n1 = lerp(t, nx0, nx1);
    return 0.936f * lerp(s, n0, n1);
}