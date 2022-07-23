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
        /// Starts the pool
        /// </summary>
        void start();
        /// <summary>
        /// Stops the all the threads in the pool
        /// </summary>
        void stop();
    public:
        /// <summary>
        /// Default constructor (starts the pool immediately)
        /// </summary>
        ThreadPool();
        /// <summary>
        /// Ads new job to the queue
        /// </summary>
        void queue_job(const std::function<void()>& job);
        /// <summary>
        /// Blocks the thread in which it is called and waits until the counter of "done jobs" reaches the "expected value"
        /// </summary>
        void wait_until_jobs_done(const int expected_numbero_of_done_jobs);
        /// <summary>
        /// Resets the "done jobs" counter to "0"
        /// </summary>
        void reset_done_jobs_counter();
        /// <summary>
        /// Destructor
        /// </summary>
        ~ThreadPool();
    private:
        void ThreadLoop();
        int done_jobs_count{};
        bool should_terminate = false;           // Tells threads to stop looking for jobs
        std::mutex queue_mutex;                  // Prevents data races to the job queue
        std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination 
        std::condition_variable done_condition;
        std::vector<std::thread> threads;
        std::queue<std::function<void()>> jobs;
    };
}
