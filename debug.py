import logging
import sys

# create logger
logger = logging.getLogger("Zillow")
logger.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create and setup console handler and file handler
if sys.platform == "win32" or sys.platform == "win64":
    filename = r"C:\Users\cyber_000\Desktop\Zillow\Version5\log.txt"
else:
    filename = "/home/david/Zillow/log.txt"

for handler in [logging.StreamHandler(), logging.FileHandler(filename)]:
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.debug('------------------')
logger.debug('logger activated')
