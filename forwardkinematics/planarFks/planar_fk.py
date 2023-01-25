import re

from forwardkinematics.fksCommon.fk import ForwardKinematics

class NoLinkIndexFoundInLinkNameError(Exception):
    pass

class ForwardKinematicsPlanar(ForwardKinematics):

    def fk(self, q, link: str, positionOnly: bool=False):
        if isinstance(link, str):
            return super().fk(q, self.get_link_index(link), positionOnly=positionOnly)
        else:
            return super().fk(q, link, positionOnly=positionOnly)

    def get_link_index(self, link: str):
        regex_match = re.match(r'\D*(\d*)', link)
        try:
            return int(regex_match.group(1))
        except Exception as _:
            raise NoLinkIndexFoundInLinkNameError(f"Link name {link} could not be resolved")

