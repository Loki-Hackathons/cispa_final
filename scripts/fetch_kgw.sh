#!/bin/sh
set -e
export SSH_ASKPASS=/c/Users/super/.ssh/askpass_totp.sh
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0

REMOTE_DIR=/p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre/output
LOCAL_OUT=/c/Users/super/OneDrive/Desktop/CISPA_Hackathon/cispa_final/task_1_text_watermark/alexandre/kgw_bundle.tar

echo "Connecting..." 1>&2
/usr/bin/ssh -F /c/Users/super/.ssh/config -o NumberOfPasswordPrompts=1 jureca \
  "cd $REMOTE_DIR && tar cf - kgw_train.npz kgw_validation.npz kgw_test.npz" \
  > "$LOCAL_OUT"

echo "SSH exit code: $?" 1>&2
ls -la "$LOCAL_OUT" 1>&2
echo "DONE" 1>&2
