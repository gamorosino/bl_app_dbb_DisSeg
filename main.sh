#! /bin/bash
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"	
	t1=$1
	mask=$2
	no_hm=0
	if [ $# -lt 2 ]; then												
		echo $0: "usage: "$( basename $0 )" <t1.ext> <mask.ext> [<outputdir>] [--no-histmatch]"
		exit -1;		    
	fi 
	# As long as there is at least one more argument, keep looping
	input_3="$3"
	while [[ $# -gt 0 ]]; do
		key="$3"
		case "$key" in
			--no-histmatch)
			no_hm=1
			;;
			 *)

			[ -z ${input_3} ] || {	outputdir=${input_3} ; }
			;;
		esac
		# Shift after checking all the cass to get the next option
		shift
	done
	
	echo "t1: "${t1}
	echo "mask: "${mask}
	input_dir=$( dirname ${t1} )
	[ -z ${outputdir} ] && { outputdir=${input_dir}"/segmentation" ; }
	[ -z ${no_hm} ] && { no_hm=0 ; }
	echo "outputdir: "${outputdir}
	
	mkdir -p ${outputdir}
	output=${outputdir}'/segmentation.nii.gz'
	proc_dir=${input_dir}"/proc/"
	mkdir -p ${proc_dir}
	reference=${SCRIPT_DIR}'/data/IMAGE_0426.nii.gz'

	t1_hm=${proc_dir}'/t1_hm.nii.gz'
	if [ ${no_hm} -eq 0 ]; then
		singularity exec -e docker://brainlife/ants:2.2.0-1bc ImageMath 3 ${t1_hm}  HistogramMatch ${t1} ${reference}  
	else
		singularity exec -e docker://brainlife/ants:2.2.0-1bc  ImageMath 3  ${t1_hm} Normalize  ${t1} ${mask}  && ImageMath 3 ${t1_hm} m ${t1_hm} 100  
	fi
	chkcp_dir=${SCRIPT_DIR}
	
	singularity exec -e  --nv docker://gamorosino/bl_app_dbb_disseg python  ${SCRIPT_DIR}/predict.py  ${t1_hm} ${output} ${chkcp_dir} --mask ${mask}

	cp ${SCRIPT_DIR}'/data/label.json' ${outputdir}
