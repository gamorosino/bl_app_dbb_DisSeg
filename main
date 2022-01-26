#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:30:00


#parse config.json for input parameters 

t1=$(jq -r .t1 config.json)
mask=$(jq -r .mask config.json)
computed=$(jq -r .computed config.json)
groud_truth=$(jq -r .groud_truth config.json)

if ! ( [ -z "${t1}" ]  || [  "${t1}" == "null" ] ); then

	echo "t1: "${t1}
	echo "mask: "${mask}


	outputdir=${PWD}"/segmentation"
	mkdir -p ${outputdir}
	output=${outputdir}'/segmentation.nii.gz'
	proc_dir=${PWD}"/proc/"
	mkdir -p ${proc_dir}
	reference='./data/IMAGE_0426.nii.gz'

	t1_hm=${proc_dir}'/t1_hm.nii.gz'
	singularity exec -e docker://brainlife/ants:2.2.0-1bc ImageMath 3 ${t1_hm}  HistogramMatch ${t1} ${reference}  

	chkcp_dir=${PWD}
	singularity exec -e  --nv docker://gamorosino/bl_app_dbb_disseg python  predict.py  ${t1_hm} ${output} ${chkcp_dir} --mask ${mask}

	cp './data/label.json' ${outputdir}

elif  ! ( [ -z "${computed}" ]  || [  "${computed}" == "null" ] || [ -z "${groud_truth}" ]  || [  "${groud_truth}" == "null" ]   ); then

	echo "computed: "${computed}
	echo "groud_truth: "${groud_truth}


	outputdir=${PWD}"/dice_score"
	mkdir -p ${outputdir}
	output=${outputdir}'/dice_score.txt'
	
	singularity exec -e  --nv docker://gamorosino/bl_app_dbb_disseg python   dice_score.py  ${computed} ${groud_truth} ${output}	

	


fi
