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
#include <unordered_map>

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
        /// Type of a job function. It will be called by the corresponding thread
        /// with the argument equal to the local thread.
        /// </summary>
        using job_func_t = std::function<void(const std::size_t&)>;

        /// <summary>
        /// Constructs thread pool with the given number of threads
        /// </summary>
        ThreadPool(const std::size_t& threads_cnt);
        /// <summary>
        /// Ads new job to the queue
        /// </summary>
        void queue_job(const job_func_t& job);
        /// <summary>
        /// Blocks the thread in which it is called and waits until the counter of "done jobs" reaches the "expected value"
        /// </summary>
        void wait_until_jobs_done(const int expected_number_of_done_jobs, const bool reset_job_counter = true);
        /// <summary>
        /// Resets the "done jobs" counter to "0"
        /// </summary>
        void reset_done_jobs_counter();

        /// <summary>
        /// Returns local thread Id that corresponds to the given global thread id.
        /// In the current implementation all the possible local thread Ids are supposed to form
        /// as range of integer values from 0 to `number of threads in pool` - 1 (without gaps).
        /// The corresponding local thread Id is supposed to be passed to the job function so that 
        /// based on it the corresponding distributed (per-thread) resources can be accessed.
        /// If the corresponding mapping does not contain a key equal to the given global thread Id,
        /// an exception will be thrown
        /// </summary>
        std::size_t retrieve_local_thread_id(const std::thread::id& global_id);

        /// <summary>
        /// Destructor
        /// </summary>
        ~ThreadPool();
    private:
        void ThreadLoop();
        int _done_jobs_count{};
        bool _should_terminate = false;           // Tells threads to stop looking for jobs
        std::mutex _queue_mutex;                  // Prevents data races to the job queue
        std::condition_variable _mutex_condition; // Allows threads to wait on new jobs or termination 
        std::condition_variable _done_condition;
        std::vector<std::thread> _threads;
        std::queue<job_func_t> _jobs;
        /// <summary>
        /// Mapping from global to local [0,1,2...N_threads - 1] IDs
        /// </summary>
        std::unordered_map<std::thread::id, std::size_t> _global_to_local_thread_id;
    };
}
