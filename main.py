from math import floor
from multiprocessing import Process
import time
import psutil
import test_group


CONFIG_TO_USE = 'collector'
PARALLEL = True


test_processes = []


def is_cpu_available():
    child_processes = psutil.Process().children(recursive=True)
    our_cpu_usage = sum([process.cpu_percent(interval=0.1) for process in child_processes]) / 100
    total_cpu_usage = psutil.cpu_percent(interval=0.2) / 100
    other_cpu_usage = total_cpu_usage - our_cpu_usage
    our_max_cpu_usage = 0.3 * (1-other_cpu_usage)
    cpu_bound = floor(psutil.cpu_count() * our_max_cpu_usage)
    print('Our CPU usage:', our_cpu_usage, 'total usage:', total_cpu_usage, 'other usage:', other_cpu_usage, 'our max usage:', our_max_cpu_usage, 'our bound:', cpu_bound)
    return cpu_bound > len(test_processes)


def is_memory_available():
    total_memory_used = psutil.virtual_memory().percent / 100
    child_processes = psutil.Process().children(recursive=True)
    our_usage_percentage = sum([process.memory_percent() for process in child_processes]) / 100
    other_processes_usage = total_memory_used - our_usage_percentage
    our_usable = 0.3 * (1-other_processes_usage)
    return our_usage_percentage < our_usable


def resources_available():
    return is_memory_available() and is_cpu_available()


def spawn_test_run():
    new_process = Process(target=test_group.run_test_group, args=(CONFIG_TO_USE,))
    new_process.start()
    print('Spawned new process:', new_process.pid)
    test_processes.append(new_process)


def clean_finished_processes():
    for process in test_processes:
        if not process.is_alive():
            print('Process', process.pid, 'finished')
            process.join()
            test_processes.remove(process)


if __name__ == '__main__':
    if not PARALLEL:
        test_group.run_test_group(CONFIG_TO_USE)
    else:
        sleep_time = 30
        while True:
            clean_finished_processes()
            # If there are resources, reduce the sleep time a bit, and vice versa.
            if resources_available():
                sleep_time *= 0.9
                spawn_test_run()
            else:
                sleep_time /= 0.9
            # It can take a while for the memory consumption to settle, so let's wait a bit.
            time.sleep(sleep_time)
