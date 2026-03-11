import sys
from loguru import logger
from wfomc.problems import WFOMCProblem, MLN_to_WFOMC
from .wfomcs_parser import parse as wfomcs_parse
from .mln_parser import parse as mln_parse

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def parse_input(input_file: str, debug: bool = False) -> WFOMCProblem:
    level = "DEBUG" if debug else "INFO"
    _handler_id = logger.add(
        sys.stderr, level=level, filter="wfomc", colorize=True, format=_LOG_FORMAT,
    )
    logger.enable("wfomc")
    try:
        if input_file.endswith('.mln'):
            with open(input_file, 'r') as f:
                input_content = f.read()
            mln_problem = mln_parse(input_content)
            return MLN_to_WFOMC(mln_problem)
        elif input_file.endswith('.wfomcs'):
            with open(input_file, 'r') as f:
                input_content = f.read()
            return wfomcs_parse(input_content)
        else:
            raise RuntimeError(f'Unknown input file type: {input_file}')
    finally:
        logger.remove(_handler_id)
        logger.disable("wfomc")
