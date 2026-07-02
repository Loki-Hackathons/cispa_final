#!/bin/sh
FILE="$HOME/.ssh/totp_once.txt"
if [ ! -f "$FILE" ]; then
  exit 1
fi
code=$(cat "$FILE")
rm -f "$FILE"
echo "$code"
