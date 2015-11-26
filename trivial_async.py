"""
Trivial async example
"""
import asyncio
import logging

logging.basicConfig(format="[%(thread)-5d]%(asctime)s: %(message)s")
logger = logging.getLogger('async')
logger.setLevel(logging.INFO)

async def print_after(msg, wait_secs):
    await asyncio.sleep(wait_secs)
    logger.info(msg)

async def test_async():
    await asyncio.gather(
        print_after("One second", 1),
        print_after("Two seconds", 2),
        print_after("Three seconds", 3))
    logger.info("finished")

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_async())

if __name__ == '__main__':
    main()
