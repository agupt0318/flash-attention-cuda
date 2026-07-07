// Tiled algorithm vs. naive reference, across shapes chosen to hurt:
// seq lengths that don't divide the tile size, causal masks that kill
// whole tiles and partial tiles, head dims the kernel will support.

#include <cstdio>
#include <vector>
#include "attention.h"

static int check(Shape s, bool causal)
{
    std::vector<float> Q(s.elems()), K(s.elems()), V(s.elems());
    std::vector<float> Oref(s.elems()), Oflash(s.elems());
    size_t rows = (size_t)s.batch * s.heads * s.seq;
    std::vector<float> Lref(rows), Lflash(rows);

    fill_random(Q, 0xA11CE);
    fill_random(K, 0xB0B);
    fill_random(V, 0xC0FFEE);

    attention_reference(s, Q, K, V, causal, Oref, &Lref);
    attention_flash_cpu(s, Q, K, V, causal, Oflash, &Lflash);

    Error eo = max_error(Oflash, Oref);
    Error el = max_error(Lflash, Lref);
    // float vs double-accum reference: rounding, not algorithm error
    int ok = eo.abs < 2e-5 && el.abs < 2e-5;
    printf("%-30s B=%d H=%d N=%-4d d=%-3d %s  O err %.2e  lse err %.2e\n",
           causal ? "flash_cpu vs ref (causal)" : "flash_cpu vs ref",
           s.batch, s.heads, s.seq, s.d, ok ? "ok  " : "FAIL",
           eo.abs, el.abs);
    return ok ? 0 : 1;
}

int main()
{
    int fails = 0;
    const Shape shapes[] = {
        { 1, 1, 64, 32 },       // one tile exactly
        { 2, 3, 128, 64 },      // multiple tiles, multiple heads
        { 1, 2, 257, 64 },      // ragged: 257 = 4*64 + 1
        { 1, 1, 33, 64 },       // shorter than one tile
        { 2, 2, 300, 32 },      // ragged again, small head dim
    };
    for (const Shape &s : shapes) {
        fails += check(s, false);
        fails += check(s, true);
    }
    printf(fails ? "FAILED (%d)\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
