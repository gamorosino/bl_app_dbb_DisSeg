#! /bin/bash
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"	
	t1=$1
	mask=$2
	outputdir=$3
	if [ $# -lt 2 ]; then												
		echo $0: "usage: "$( basename $0 )" <t1.ext> <mask.ext> [<outputdir>]"
		return 1;		    
	fi 

	echo "t1: "${t1}
	echo "mask: "${mask}
	input_dir=$( dirname ${t1} )
	[ -z ${outputdir} ] && { outputdir=${input_dir}"/segmentation" ; }
	echo "outputdir: "${outputdir}
	
	mkdir -p ${outputdir}
	output=${outputdir}'/segmentation.nii.gz'
	proc_dir=${input_dir}"/proc/"
	mkdir -p ${proc_dir}
	reference=${SCRIPT_DIR}'/data/IMAGE_0426.nii.gz'

	t1_hm=${proc_dir}'/t1_hm.nii.gz'
	singularity exec -e docker://brainlife/ants:2.2.0-1bc ImageMath 3 ${t1_hm}  HistogramMatch ${t1} ${reference}  

	chkcp_dir=${SCRIPT_DIR}
	
	singularity exec -e  --nv docker://gamorosino/bl_app_dbb_disseg python  ${SCRIPT_DIR}/predict.py  ${t1_hm} ${output} ${chkcp_dir} --mask ${mask}

	cp ${SCRIPT_DIR}'/data/label.json' ${outputdir}
