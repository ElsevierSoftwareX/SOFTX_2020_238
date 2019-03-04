date
key=mass

run_dir=/home/spiir/ER14
check_dir=${HOME}/Trigger_Latency_Check/ER14
current_dir=`pwd`

mkdir -p ${check_dir}

condor_q -wide

cd ${check_dir}
ls -ltra  ${run_dir}/*${key}*/trigger_control.txt | awk -F ' ' '{print $9}' >trigger_list

for file in `more trigger_list`
do 
   dirf=`echo $file | awk -F '/trigger'  '{print $1}'`

   echo "#############################################################################"
   echo $dirf
   echo "#############################################################################"


   echo  " (HLV)         time  on  off gap "

   for sfile in `ls ${dirf}/000/*[HLV]1/state*`
   do 

       more   ${sfile}
       echo " "
   done


   echo "#############################################################################"
   echo " Latency History " 
   echo "#############################################################################"

   ls -ltra ${dirf}/0*/latency_history.txt | tail --lines 1
   ls -lta ${dirf}/0*/latency_history.txt | tail --lines 1

   echo "
The most recent latency time from job 000
--------------------------------------"
  tail --lines 1  ${dirf}/000/latency_history.txt

   echo "#############################################################################"
   echo $file
   echo "#############################################################################"


  echo " "
  echo "Last 10 triggers " 
  echo "-----------------------------------"

   tail  --lines=10 $file  | sort 
   
   echo " "
   echo "Last 10 with FAR < 3e-6"
   echo "-----------------------------------"


  awk -F ',' '($2<3e-6){print}' $file | tail --lines 10
  
  echo " "
  echo "Time of trigger and dag file "
  echo "-----------------------------------"

  ls -ltra $file 
  ls -ltra ${dirf}/*.dag | tail --lines 2  


  echo " "
done

cd ${current_dir}

