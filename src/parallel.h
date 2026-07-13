// A tiny persistent thread pool. Spawning std::thread per call costs tens
// of microseconds each, which is fine for a handful of calls (prefill) but
// ruinous in a decode loop that issues dozens of parallel ops per token.
// The pool spawns its workers once and hands them work through parallel_for.
//
// FLASH_CPU_THREADS (read once, at first use) caps the worker count.
#pragma once
#include <cstdint>
#include <functional>

// Split [0, n) into contiguous ranges and run fn(begin, end) on each across
// the pool, returning when all ranges are done. Runs inline (no dispatch)
// when n <= min_chunk or only one thread is available, so it is safe to call
// on tiny work. The calling thread participates, so no core sits idle.
void parallel_for(int64_t n, int64_t min_chunk,
                  const std::function<void(int64_t, int64_t)> &fn);

// Effective worker count (respects FLASH_CPU_THREADS).
int parallel_num_threads();
