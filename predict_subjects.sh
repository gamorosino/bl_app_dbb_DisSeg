#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"	
testset_dir=$1
output_dir=$2
for i in $( ls ${testset_dir}/* -d ); do
	b_name_i=$( basename ${i} )
	[ "${b_name_i}" == "bids" ] && { continue; }
	[ -d ${i} ] || { continue; }
	echo ${i}	
	t1_i=$( ls ${i}'/dt-neuro-anat-t1w.id-'*/'t1.nii.gz' )
	mask_i=$( ls ${i}'/dt-neuro-mask.id-'*/'mask.nii.gz' )
	parc_i=$( ls ${i}'/dt-neuro-parcellation-volume.id-'*/'parc.nii.gz' )
	echo ${t1_i}
	echo ${mask_i}
	echo ${parc_i}
	input_dir=$( dirname ${t1_i} )
	proc_dir=${input_dir}"/proc/"
	mkdir -p ${proc_dir}
	t1_hm=${proc_dir}'/t1_hm.nii.gz'	
	chkcp_dir=${SCRIPT_DIR}
	output_dir_i=${output_dir}'/'$( basename ${i}  )'/'
	echo ${output_dir_i}
	mkdir -p ${output_dir_i}
	output_seg=${output_dir_i}'/segmentation.nii.gz'	
	dice_score=${output_dir_i}'/dice_score.txt'
	python  ${SCRIPT_DIR}/predict.py  ${t1_hm} ${output_seg} ${chkcp_dir} --mask ${mask_i} 
	python ${SCRIPT_DIR}/dice_score.py  ${output_seg} ${parc_i} ${dice_score}

done
