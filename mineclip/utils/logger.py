import logging


def get_logger(filename=None):
    """
    Setup logger for logging
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    # datefmt='%m/%d/%Y %H:%M:%S'
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        
    return logger
