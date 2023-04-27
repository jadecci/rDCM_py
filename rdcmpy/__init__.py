import logging

from rdcmpy.rdcm_model import RegressionDCM


logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('rdcmpy')
log.setLevel(level=logging.DEBUG)

__all__ = [
    'log',
    'RegressionDCM',
]
