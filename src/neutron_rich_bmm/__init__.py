from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neutron-rich-bmm")
except PackageNotFoundError:
    # Fallback for local builds or GitHub Pages environments
    __version__ = "0.0.0"

__author__ = "Alexandra C. Semposki, C. Drischler, R. J. Furnstahl, D. R. Phillips"
__credits__ = "Ohio University, Facility for Rare Isotope Beams, The Ohio State University"
