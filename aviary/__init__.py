import importlib.metadata
from os.path import abspath, dirname

try:
    __version__ = importlib.metadata.version("aviary")
except importlib.metadata.PackageNotFoundError:
    __version__ = "vendored"
PKG_DIR = dirname(abspath(__file__))
ROOT = dirname(PKG_DIR)
