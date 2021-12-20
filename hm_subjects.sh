#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"	
reference=${SCRIPT_DIR}'/data/IMAGE_0426.nii.gz'
testset_dir=$1
for i in $( ls ${testset_dir}/* -d ); do
	b_name_i=$( basename ${i} )
	[ "${b_name_i}" == "bids" ] && { continue; }
	[ -d ${i} ] || { continue; }
	echo ${i}
	t1_i=$( ls ${i}'/dt-neuro-anat-t1w.id-'*/'t1.nii.gz' )
	echo 't1': ${t1_i}
	#[ $( exists $output_seg ) -eq 0 ] && { bash ${predict_script} ${t1_i} ${mask_i} ${output_dir_i}'/' ; }	
	input_dir=$( dirname ${t1_i} )
	proc_dir=${input_dir}"/proc/"
	t1_hm=${proc_dir}'/t1_hm.nii.gz'
	ImageMath 3 ${t1_hm}  HistogramMatch ${t1_i} ${reference}  

done
