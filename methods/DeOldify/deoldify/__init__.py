import logging
import sys

sys.path.append('methods/Deoldify')

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

from ..deoldify._device import _Device

device = _Device()
