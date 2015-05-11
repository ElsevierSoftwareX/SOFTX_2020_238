#!/bin/sh

fifo1="cachesrc_test_01.input1"
fifo2="cachesrc_test_01.input2"
cache="cachesrc_test_01.cache"

ln -s Makefile H-CACHESRC_TEST_01-874018527-128.txt
ln -s Makefile H-CACHESRC_TEST_01-874018655-128.txt
ln -s Makefile H-CACHESRC_TEST_01-874018783-128.txt
"ls" H-CACHESRC_TEST_01-*.txt | lalapps_path2cache >$cache
mkfifo $fifo1 $fifo2

cat Makefile Makefile Makefile >$fifo1 &
gst-launch lal_cachesrc location=$cache ! filesink location=$fifo2 &

cmp $fifo1 $fifo2

rm -f $fifo1 $fifo2 $cache H-CACHESRC_TEST_01-*.txt
