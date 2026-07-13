#include "parallel.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct Pool {
    std::vector<std::thread> workers;
    std::mutex mtx;
    std::condition_variable cv_work, cv_done;

    // Current job, published under mtx before workers are woken.
    const std::function<void(int64_t, int64_t)> *job = nullptr;
    int64_t job_n = 0, job_chunk = 0;
    int job_K = 0;
    std::atomic<int> next{0};    // next chunk index, pulled by all participants
    int active = 0;              // workers still running the current job
    uint64_t epoch = 0;          // bumped once per job so workers wake exactly once
    bool stop = false;
    int nthreads = 1;

    Pool()
    {
        unsigned hw = std::thread::hardware_concurrency();
        if (const char *e = std::getenv("FLASH_CPU_THREADS")) {
            int r = std::atoi(e);
            if (r > 0) hw = (unsigned)r;
        }
        nthreads = hw ? (int)hw : 1;
        for (int i = 1; i < nthreads; i++)     // caller acts as the nth worker
            workers.emplace_back([this] { worker_loop(); });
    }

    ~Pool()
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv_work.notify_all();
        for (auto &t : workers)
            t.join();
    }

    // Pull and run chunks until the job is exhausted.
    void drain()
    {
        for (;;) {
            int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= job_K) break;
            int64_t b = (int64_t)i * job_chunk;
            if (b >= job_n) break;             // guard any empty trailing chunk
            int64_t e = b + job_chunk < job_n ? b + job_chunk : job_n;
            (*job)(b, e);
        }
    }

    void worker_loop()
    {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_work.wait(lk, [&] { return stop || epoch != seen; });
                if (stop) return;
                seen = epoch;
            }
            drain();
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (--active == 0)
                    cv_done.notify_one();
            }
        }
    }

    void run(const std::function<void(int64_t, int64_t)> &fn, int64_t n,
             int K, int64_t chunk)
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            job = &fn;
            job_n = n;
            job_K = K;
            job_chunk = chunk;
            next.store(0, std::memory_order_relaxed);
            active = (int)workers.size();
            epoch++;
        }
        cv_work.notify_all();
        drain();                               // caller participates
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [&] { return active == 0; });
    }
};

Pool &pool()
{
    static Pool p;
    return p;
}

}   // namespace

int parallel_num_threads()
{
    return pool().nthreads;
}

void parallel_for(int64_t n, int64_t min_chunk,
                  const std::function<void(int64_t, int64_t)> &fn)
{
    if (n <= 0) return;
    Pool &p = pool();
    if (p.nthreads <= 1 || n <= min_chunk) {
        fn(0, n);
        return;
    }
    if (min_chunk < 1) min_chunk = 1;
    int64_t max_chunks = (n + min_chunk - 1) / min_chunk;
    int K = (int)(max_chunks < p.nthreads ? max_chunks : p.nthreads);
    int64_t chunk = (n + K - 1) / K;
    K = (int)((n + chunk - 1) / chunk);        // exact chunk count, no empties
    p.run(fn, n, K, chunk);
}
