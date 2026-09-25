import logging as _pele_logging
import sys as _pele_sys

logger = _pele_logging.getLogger("pele")
global_handler = _pele_logging.StreamHandler(_pele_sys.stdout)
logger.addHandler(global_handler)
logger.setLevel(_pele_logging.DEBUG)


def get_include():
    """Directory with pele's C++ sources and headers (`pele/*.hpp`), for
    building extensions against pele (e.g. mcpele). Works for installed
    packages and for in-place builds of a source checkout."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    installed = os.path.join(here, "source")
    return installed if os.path.isdir(installed) else os.path.join(os.path.dirname(here), "source")
