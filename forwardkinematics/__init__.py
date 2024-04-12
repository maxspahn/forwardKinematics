from importlib.metadata import version
__version__ = version(__name__)

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from forwardkinematics.xmlFks.generic_xml_fk import GenericXMLFk
