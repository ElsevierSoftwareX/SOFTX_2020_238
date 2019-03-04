#!/bin/sh

sum_dir=${HOME}/public_html/run_summary_ER14
mkdir -p ${sum_dir}

while true
do

check_ER14_latency.sh>& ${sum_dir}/summary_ER14_latency.txt

check_ER14_trigger.sh>& ${sum_dir}/summary_ER14_trigger.txt

sleep 300
done
