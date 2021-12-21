#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"	
testset_dir=$1
output_dir=$2
########################################################################
## Functions
########################################################################
exists () {
                                      			
		if [ $# -lt 1 ]; then
		    echo $0: "usage: exists <filename> "
		    echo "    echo 1 if the file (or folder) exists, 0 otherwise"
		    return 1;		    
		fi 
		
		if [ -d "${1}" ]; then 

			echo 1;
		else
			([ -e "${1}" ] && [ -f "${1}" ]) && { echo 1; } || { echo 0; }	
		fi		
		};

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
	[ $( exists $output_seg ) -eq 0 ] && { python  ${SCRIPT_DIR}/predict.py  ${t1_hm} ${output_seg} ${chkcp_dir} --mask ${mask_i} ; }	\
									 || { echo "Brain tissue segmentation already done for "${b_name_i} ;}
	
	python ${SCRIPT_DIR}/dice_score.py  ${output_seg} ${parc_i} ${dice_score}

done
