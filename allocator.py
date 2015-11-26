"""
Asynchronous numpy array allocator with a memory limit
"""
import asyncio
import logging
import numpy as np
from collections import deque

logging.basicConfig(format="[%(thread)-5d]%(asctime)s: %(message)s")
logger = logging.getLogger('allocator')
logger.setLevel(logging.INFO)

default_allocator_size = 100000

class AsyncAllocator(object):
    def __init__(self, max_size):
        self._max_size = max_size
        self._allocated_size = 0
        self._waiting_allocations = deque()

    async def allocate_data(self, size):
        if not self.can_allocate_now(size):
            logger.info("Can't allocate {} bytes right now, waiting for memory to be freed".format(size))
            future = asyncio.Future()
            self._waiting_allocations.append((future, size))
            await future
        self._allocated_size += size

    def deallocate_data(self, array):
        self._allocated_size -= array.nbytes
        new_deque = deque()
        for ind, (future, size) in enumerate(self._waiting_allocations):
            if self.can_allocate_now(size):
                future.set_result(0)
            else:
                new_deque.append((future, size))
        self._waiting_allocations = new_deque

    def can_allocate_now(self, size):
        return self._allocated_size + size <= self._max_size

default_allocator = AsyncAllocator(default_allocator_size)


class AsyncArrayTracker(object):
    def __init__(self, array, allocator):
        self._allocator = allocator
        self._array = array
        self._array_copy = None

    async def __aenter__(self):
        await self._allocator.allocate_data(self._array.nbytes)
        self._array_copy = self._array.copy()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._allocator.deallocate_data(self._array)
        self._array_copy = None

    @property
    def array(self):
        assert self._array_copy is not None, "Array has not been allocated (use in 'async with' block)"
        return self._array_copy


def allocator_test():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_allocations())

async def run_allocations():
    allocator = AsyncAllocator(100)
    base_array = np.ones(10, dtype=np.float32)
    await asyncio.gather(
        do_work_on_array_copy("Operation 1", base_array, allocator),
        do_work_on_array_copy("Operation 2", base_array, allocator),
        do_work_on_array_copy("Operation 3", base_array, allocator))

async def do_work_on_array_copy(opname, array, allocator):
    logger.info("Attempting to allocate {} bytes for {}".format(array.nbytes, opname))
    async with AsyncArrayTracker(array, allocator):
        logger.info("Allocated {} bytes for {}".format(array.nbytes, opname))
        await asyncio.sleep(3)
        logger.info("Finished working on {}, deallocating {} bytes".format(opname, array.nbytes))
    logger.info("Deallocation on {} finished".format(opname))
    return 2

if __name__ == '__main__':
    allocator_test()