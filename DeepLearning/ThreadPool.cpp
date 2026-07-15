//Copyright (c) 2026 Denys Dragunov, dragunovdenis@gmail.com
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
//copies of the Software, and to permit persons to whom the Software is furnished
//to do so, subject to the following conditions :

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
//PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ThreadPool.h"

namespace DeepLearning
{
    void ThreadPool::start(const std::size_t& threads_cnt) {
        _threads.resize(threads_cnt);
        for (auto local_thread_id = 0ull; local_thread_id < threads_cnt; local_thread_id++)
            _threads[local_thread_id] = std::thread([&]() { this->ThreadLoop(); });
    }

    ThreadPool::ThreadPool(const std::size_t& threads_cnt)
    {
        start(threads_cnt);
    }

    void ThreadPool::ThreadLoop() {
        while (true) {
            job_func_t job;
            {
                std::unique_lock lock(_queue_mutex);
                _mutex_condition.wait(lock, [this] {
                    return !_jobs.empty() || _should_terminate;
                    });
                if (_should_terminate) {
                    return;
                }
                job = _jobs.front();
                _jobs.pop();
            }

            job();

            {
                std::unique_lock lock(_queue_mutex);
                ++_done_jobs_count;
            }
            _done_condition.notify_one();
        }
    }

    ThreadPool::~ThreadPool()
    {
        stop();
    }

    void ThreadPool::queue_job(const job_func_t& job) {
        {
            std::unique_lock lock(_queue_mutex);
            ++_queued_jobs_count;
            _jobs.push(job);
        }
        _mutex_condition.notify_one();
    }

    void ThreadPool::wait_all_jobs_done()
    {
        std::unique_lock lock(_queue_mutex);
        _done_condition.wait(lock, [&]() { return _done_jobs_count == _queued_jobs_count; });

        _done_jobs_count = 0;
        _queued_jobs_count = 0;
    }

    void ThreadPool::stop()
    {
        {
            std::unique_lock lock(_queue_mutex);
            _should_terminate = true;
        }
        _mutex_condition.notify_all();
        for (std::thread& active_thread : _threads) {
            active_thread.join();
        }
        _threads.clear();
    }
}