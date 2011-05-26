#!/bin/bash
for k in *.sqlite
do
ligolw_sqlite --verbose --tmp-space /dev/shm --extract ${k}.xml --database ${k} && ligolw_sicluster --verbose --cluster-window 1 ${k}.xml 
done

ligolw_add --output all_added.xml --verbose svd_*.xml ../segments/injections.xml && ligolw_sicluster --cluster-window 1 all_added.xml && ligolw_inspinjfind all_added.xml && ligolw_sqlite --verbose --database all.sqlite --tmp-space /dev/shm --replace all_added.xml
