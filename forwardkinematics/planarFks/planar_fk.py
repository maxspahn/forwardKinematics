from typing import Union
import re
import numpy as np

from forwardkinematics.fksCommon.fk import ForwardKinematics

class NoLinkIndexFoundInLinkNameError(Exception):
    pass

class ForwardKinematicsPlanar(ForwardKinematics):

    def __init__(self):
        super().__init__()
        self._mount_transformation = np.identity(3)

    def ensure_int_link(self, link: Union[str, int]):
        if isinstance(link, int):
            return link
        regex_match = re.match(r'\D*(\d*)', link)
        try:
            return int(regex_match.group(1))
        except Exception as _:
            raise NoLinkIndexFoundInLinkNameError(f"Link name {link} could not be resolved")

