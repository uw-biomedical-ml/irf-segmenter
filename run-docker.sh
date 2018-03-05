IN=$1
OUT=$2

INDIR=`dirname $IN`
OUTDIR=`dirname $OUT`

IN=`basename $IN`
OUT=`basename $OUT`

#docker run --runtime=nvidia --rm  -it -v ${PWD}/data:/data -e IN=$IN -e OUT=$OUT irf-segmenter

#NV_GPU=0,1 nvidia-docker run -it -v ${INDIR}:/data_in -v ${OUTDIR}:/data_out -e IN=/data_in/$IN -e OUT=/data_out/$OUT irf-segmenter

nvidia-docker run -it -v ${INDIR}:/data_in -v ${OUTDIR}:/data_out -e IN=/data_in/$IN -e OUT=/data_out/$OUT irf-segmenter

