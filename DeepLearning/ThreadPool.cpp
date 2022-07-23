//Copyright (c) 2022 Denys Dragunov, dragunovdenis@gmail.com
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
    void ThreadPool::start() {
        const uint32_t num_threads = std::thread::hardware_concurrency(); // Max # of threads the system supports
        threads.resize(num_threads);
        for (uint32_t i = 0; i < num_threads; i++) {
            threads.at(i) = std::thread([&]() { this->ThreadLoop(); });
        }
    }

    ThreadPool::ThreadPool()
    {
        start();
    }

    void ThreadPool::ThreadLoop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                mutex_condition.wait(lock, [this] {
                    return !jobs.empty() || should_terminate;
                    });
                if (should_terminate) {
                    return;
                }
                job = jobs.front();
                jobs.pop();
            }
            job();

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                done_jobs_count++;
            }
            done_condition.notify_all();
        }
    }

    ThreadPool::~ThreadPool()
    {
        stop();
    }

    void ThreadPool::queue_job(const std::function<void()>& job) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            jobs.push(job);
        }
        mutex_condition.notify_one();
    }

    void ThreadPool::wait_until_jobs_done(const int expected_number_of_done_jobs)
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        done_condition.wait(lock, [&]() { return done_jobs_count == expected_number_of_done_jobs; });
    }

    void ThreadPool::reset_done_jobs_counter()
    {
        done_jobs_count = 0;
    }

    void ThreadPool::stop()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            should_terminate = true;
        }
        mutex_condition.notify_all();
        for (std::thread& active_thread : threads) {
            active_thread.join();
        }
        threads.clear();
    }
}