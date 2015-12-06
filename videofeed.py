"""
Mock up a video feed pipeline
"""
import asyncio
import logging
import sys

import cv2

logging.basicConfig(format="[%(thread)-5d]%(asctime)s: %(message)s")
logger = logging.getLogger('async')
logger.setLevel(logging.INFO)


async def process_video(filename):
    cap = cv2.VideoCapture(filename)
    tasks = list()
    frame_ind = 0
    while cap.isOpened():
        ret, frame = cap.read()
        tasks.append(asyncio.ensure_future(process_frame(frame, frame_ind)))
        frame_ind += 1
        await asyncio.sleep(0)
    await asyncio.gather(tasks)


async def process_frame(frame, frame_ind):
    logger.info("Processing frame {}".format(frame_ind))
    await asyncio.sleep(20.0)
    logger.info("Finished processing frame {}".format(frame_ind))


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_video(sys.argv[1]))
    logger.info("Completed")


if __name__ == '__main__':
    main()
