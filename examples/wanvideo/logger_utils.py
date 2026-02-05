import logging
import torch.distributed as dist


def setup_logger(output_path):
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s', "%Y-%m-%d %H:%M:%S")
    formatter.default_msec_format = '%s.%03d'

    file_handler = logging.FileHandler(f"{output_path}/rank{rank}.log")
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    file_handler.addFilter(lambda record: setattr(record, 'rank', rank) or True)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(lambda record: setattr(record, 'rank', rank) or True)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger