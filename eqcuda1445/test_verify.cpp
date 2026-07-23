#include "eqcuda1445/eqcuda1445.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
static unsigned char g_hdr[180];
static int g_ok = 0, g_bad = 0;
static int cb(void *, void *sol) {
    int rc = equihash_verify_c((const char *)g_hdr, sizeof g_hdr, (const unsigned char *)sol);
    rc == 0 ? g_ok++ : g_bad++;
    if (rc != 0) printf("VERIFY FAILED rc=%d\n", rc);
    return 0;
}
int main() {
    EqSolver *s = eq_create(0);
    if (!s) return 1;
    int total = 0;
    for (uint32_t n = 0; n < 20; n++) {
        memset(g_hdr, 0x42, sizeof g_hdr);
        memcpy(&g_hdr[140], &n, 4); // patch nonce like the solver does
        int r = eq_solve(s, g_hdr, sizeof g_hdr, n, cb, nullptr);
        if (r < 0) { printf("solve error %d\n", r); return 1; }
        total += r;
    }
    printf("20 nonces: %d solutions, %d verified OK, %d FAILED\n", total, g_ok, g_bad);
    eq_destroy(s);
    return g_bad || !g_ok;
}
