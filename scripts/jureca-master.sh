#!/bin/bash
set -e
export SSH_ASKPASS=~/.ssh/askpass_totp_once.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
ssh -o NumberOfPasswordPrompts=1 jureca true
ssh -O check jureca >/dev/null 2>&1
