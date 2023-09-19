#!/usr/bin/python3

import pathlib
import sys

def application(environ, start_response):
    sys.path.append(str(pathlib.Path(__file__).parent.resolve().parent))
    import daily_viewer
    return daily_viewer.application(environ, start_response)
