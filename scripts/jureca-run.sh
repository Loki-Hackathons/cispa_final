#!/bin/bash
set -e
# Run a remote shell command string on JURECA via multiplexed SSH.
ssh -o ControlMaster=no jureca bash -lc "$1"
