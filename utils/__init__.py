import sys
import os
import git
import threading
import Queue

root_path  = git.Repo(__file__, search_parent_directories=True).working_dir
caffe_path = os.path.join(root_path, 'caffe/python')


def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


class WorkerThread(threading.Thread):
    def __init__(self, kernel_func, task_q, lock, count):
        self.kernel_func = kernel_func
        self.task_q = task_q
        self.lock   = lock
        self.count  = count
        self.stop_request = threading.Event()
        super(WorkerThread, self).__init__()
        return

    def run(self):
        while not self.stop_request.is_set():
            try:
                task = self.task_q.get(True, 0.05)
                self.kernel_func(**task)
                have_lock = self.lock.acquire(True)
                if have_lock:
                    self.count[0] += 1
                    self.lock.release()
            except Queue.Empty:
                continue
        return

    def join(self, timeout=None):
        self.stop_request.set()
        super(WorkerThread, self).join(timeout)
        return


def run_with_multithreads(kernel_func, work_list, stop, num_threads):
    task_q = Queue.Queue()
    lock   = threading.Lock()
    count  = [0]   ## Use list for calling by reference
    thread_pool = [WorkerThread(kernel_func, task_q, lock, count)
                   for _ in xrange(num_threads)]

    for thread in thread_pool:
        thread.start()

    ## Put jobs into queue
    for work in work_list:
        task_q.put(work)

    ## Waiting for jobs to finish
    while count[0] < stop:
        sys.stdout.write('prog: {}/{} .....    \r'.format(count[0], stop))
        sys.stdout.flush()

    for thread in thread_pool:
        thread.join()
    return
