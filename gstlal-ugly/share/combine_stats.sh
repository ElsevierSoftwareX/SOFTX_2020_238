#!/bin/bash
dir=$1
flist=`ls ${dir}/ |grep bank`
start_banktime=0
last_banktime=2000000000
total_duration=0
combine_time=400
fin=""
bankid=""
for fn in $flist
do
  this_duration=`ls ${dir}/$fn|cut -d _ -f 4|cut -d . -f 1` 
  this_banktime=`ls ${dir}/$fn|cut -d _ -f 3`
#  this_bankid=`ls ${dir}/$fn|cut -d _ -f 1`
  this_bankid=`ls ${dir}/$fn|cut -d _ -f 1|cut -d / -f 2`
  if [ $this_banktime -lt $last_banktime ]
  then
    if [ $total_duration -gt $combine_time ]
    then
      fout=${dir}/${bankid}_stats_${start_banktime}_${total_duration}.xml.gz
      echo "gstlal_cohfar_calc_fap --input $fin --ifos L1H1 --output $fout --input-format stats --duration $total_duration"
      gstlal_cohfar_calc_fap --input $fin --ifos L1H1 --output $fout --input-format stats --duration $total_duration
      for del_fn in $(echo $fin | tr "," "\n")
      do
	      echo "remove $del_fn"
	      rm $del_fn
      done
    fi
    total_duration=$this_duration
    last_banktime=$this_banktime
    start_banktime=$this_banktime
    bankid=$this_bankid
    fin="${dir}/${fn}"
    continue
  fi

  if [ $total_duration -gt $combine_time ]
  then
    fout=${dir}/${bankid}_stats_${start_banktime}_${total_duration}.xml.gz
    echo "gstlal_cohfar_calc_fap --input $fin --ifos L1H1 --output $fout --input-format stats --duration $total_duration"
    gstlal_cohfar_calc_fap --input $fin --ifos L1H1 --output $fout --input-format stats --duration $total_duration

      for del_fn in $(echo $fin | tr "," "\n")
      do
	      echo "remove $del_fn"
	      rm $del_fn
      done
    total_duration=0
    last_banktime=$this_banktime
    start_banktime=$this_banktime
    fin="${dir}/${fn}"
  else
    fin="$fin,${dir}/${fn}"
    last_banktime=$this_banktime
  fi
  total_duration=$(($total_duration + $this_duration))
done
