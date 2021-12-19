#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"	
project_id='60a14ca503bcad0ad27cada9'
outputdir=$1
download_dir=./'DBB_test'
bash ${SCRIPT_DIR}/download_testset.sh
mkdir -p ${outputdir}
tag_list=( ACC PFM MCDs HD )
csv_all=${outputdir}'/dice_score_all.csv'
for tag in ${tag_list[@]}; do
	bash ${SCRIPT_DIR}/predict_testset_local.sh ${download_dir}'/'${tag}'/'proj-${project_id} ${outputdir}'/'${tag}
	echo ${tag},$( cat ${outputdir}'/'${tag}'/dice_score_average.csv' ) > ${csv_all}
done

