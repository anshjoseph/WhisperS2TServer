import logging

VALID_LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# TODO : also make it so it can stor logs in file

def get_configure_logger(file_name, enabled=True, logging_level='INFO'):
    if logging_level not in VALID_LOGGING_LEVELS:
        logging_level = "INFO"

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} "
               "[%(funcName)s] [PID:%(process)d | %(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(file_name)

    if not enabled:
        logger.disabled = True
    return logger
