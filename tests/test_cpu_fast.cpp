// Fast CPU kernel vs the double-accum reference, across shapes chosen to
// stress the query-tile blocking (seq below / at / just past QBLK=8 and
// KBLK=64), causal masks that kill whole and partial tiles, and both
// head dims. Same float-vs-double tolerance as the scalar CPU test.

#include <cstdio>
#include <vector>
#include "attention.h"

static int check(Shape s, bool causal)
{
    std::vector<float> Q(s.elems()), K(s.elems()), V(s.elems());
    std::vector<float> Oref(s.elems()), Ofast(s.elems());
    size_t rows = (size_t)s.batch * s.heads * s.seq;
    std::vector<float> Lref(rows), Lfast(rows);

    fill_random(Q, 0xA11CE + s.seq);
    fill_random(K, 0xB0B + s.seq * 7);
    fill_random(V, 0xC0FFEE + s.seq * 13);

    attention_reference(s, Q, K, V, causal, Oref, &Lref);
    attention_flash_fast(s, Q, K, V, causal, Ofast, &Lfast);

    Error eo = max_error(Ofast, Oref);
    Error el = max_error(Lfast, Lref);
    int ok = eo.abs < 2e-5 && el.abs < 2e-5;
    printf("%-30s B=%d H=%d N=%-5d d=%-3d %s  O err %.2e  lse err %.2e\n",
           causal ? "flash_fast vs ref (causal)" : "flash_fast vs ref",
           s.batch, s.heads, s.seq, s.d, ok ? "ok  " : "FAIL",
           eo.abs, el.abs);
    return ok ? 0 : 1;
}

int main()
{
    int fails = 0;
    const Shape shapes[] = {
        { 1, 1, 1, 32 },    { 1, 1, 7, 64 },    { 1, 1, 8, 64 },
        { 1, 1, 9, 32 },    { 1, 1, 64, 32 },   { 2, 3, 128, 64 },
        { 1, 2, 257, 64 },  { 1, 1, 33, 64 },   { 2, 2, 300, 32 },
        { 1, 2, 65, 64 },   { 1, 1, 129, 32 },  { 3, 4, 512, 64 },
        { 1, 4, 1024, 64 }, { 1, 1, 1000, 32 },
    };
    for (const Shape &s : shapes) {
        fails += check(s, false);
        fails += check(s, true);
    }
    printf(fails ? "FAILED (%d)\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
