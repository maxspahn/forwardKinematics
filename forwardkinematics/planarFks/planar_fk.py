import re
import numpy as np

from forwardkinematics.fksCommon.fk import ForwardKinematics

class NoLinkIndexFoundInLinkNameError(Exception):
    pass

class ForwardKinematicsPlanar(ForwardKinematics):

    def __init__(self):
        super().__init__()
        self._mount_transformation = np.identity(3)


    def fk(self, q, link: str, position_only: bool=False):
        if isinstance(link, str):
            return super().fk(q, self.get_link_index(link), position_only=position_only)
        else:
            return super().fk(q, link, position_only=position_only)

    def get_link_index(self, link: str):
        regex_match = re.match(r'\D*(\d*)', link)
        try:
            return int(regex_match.group(1))
        except Exception as _:
            raise NoLinkIndexFoundInLinkNameError(f"Link name {link} could not be resolved")

