#!/usr/bin/env bash
# No-sudo Python.h for TorchInductor's C++ backend (only needed for fusion_report).
set -euo pipefail
mkdir -p ~/pydev/debs && cd ~/pydev/debs
apt-get download libpython3.12-dev python3.12-dev
for f in *.deb; do dpkg-deb -x "$f" ~/pydev/root; done
find ~/pydev/root -name Python.h -o -name pyconfig.h
