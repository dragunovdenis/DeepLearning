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

#pragma once
#include <functional>
#include <thread>
#include <mutex>
#include <queue>

namespace DeepLearning
{
    /// <summary>
    /// Inspired by the answer of PhD AP EcE 
    /// (https://stackoverflow.com/questions/15752659/thread-pooling-in-c11)
    /// </summary>
    class ThreadPool {
        /// <summary>
        /// Starts the pool with the given number of threads
        /// </summary>
        void start(const std::size_t& threads_cnt);
        /// <summary>
        /// Stops the all the threads in the pool
        /// </summary>
        void stop();
    public:
        /// <summary>
        /// Type of a job function.
        /// </summary>
        using job_func_t = std::function<void()>;

        /// <summary>
        /// Constructs thread pool with the given number of threads
        /// </summary>
        ThreadPool(const std::size_t& threads_cnt);
        /// <summary>
        /// Ads new job to the queue
        /// </summary>
        void queue_job(const job_func_t& job);
        /// <summary>
        /// Blocks the calling thread until all currently queued jobs have completed, then resets the internal
        /// job counters. All jobs must be queued before calling this method; queuing additional jobs
        /// concurrently with an active wait leads to undefined behaviour.
        /// </summary>
        void wait_all_jobs_done();

        /// <summary>
        /// Destructor
        /// </summary>
        ~ThreadPool();
    private:
        void ThreadLoop();
        std::size_t _done_jobs_count{};
        std::size_t _queued_jobs_count{};
        bool _should_terminate = false;           // Tells threads to stop looking for jobs
        std::mutex _queue_mutex;                  // Prevents data races to the job queue
        std::condition_variable _mutex_condition; // Allows threads to wait on new jobs or termination 
        std::condition_variable _done_condition;
        std::vector<std::thread> _threads;
        std::queue<job_func_t> _jobs;
    };
}
