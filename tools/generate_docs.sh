#!/bin/bash
set -e
rm -rf doc/
pdoc src/ -o doc/ --no-show-source
