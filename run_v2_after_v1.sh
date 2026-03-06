#!/bin/bash
# Wait for V1 to finish, then run V2
echo "Waiting for V1 (PID 2915846) to finish..."
while kill -0 2915846 2>/dev/null; do sleep 10; done
echo "V1 finished. Starting V2 twin experiment..."
python3 -u run_mlswe_lsmcmc_nldata_V2_twin.py example_input_mlswe_nldata_V2_twin.yml 2>&1 | tee nl_loc_twin_da_run.log
