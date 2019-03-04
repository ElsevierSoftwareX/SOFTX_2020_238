date
run_dir=/home/spiir/ER14
key=mass
check_dir=${HOME}/Trigger_Latency_Check/ER14
current_dir=`pwd`
mkdir -p ${check_dir}

cd ${check_dir}

ls -ltra ${run_dir}/*${key}/000/latency_history.txt | awk -F ' ' '{print $9}' >latency_list
cat latency_list >>latency_list_${date}

###############################" 
#       injection latency files
###############################" 

for file in `more latency_list`
do 

   echo "#################################################################"
   echo $file
   echo "#################################################################"
   tail  --lines=3 $file  | sort 
   echo "------------------------------------------------------------------
          Last 10 with Latency > 10 
--------------------------------------------------------------------"
  awk -F ' ' '($2 > 10 ){print}' $file | tail --lines 10
done

cd ${current_dir}

