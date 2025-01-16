#!/bin/bash
make -C build/

for i in {1..30}
do
   (time build/SEP_Pipeline > /dev/null) &>> pipeline_timings
done