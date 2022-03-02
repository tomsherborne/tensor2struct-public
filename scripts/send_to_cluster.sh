#!/usr/bin/env bash

USER=$(whoami)

while getopts ':c:' flag; do
  case "${flag}" in
    c ) cluster="${OPTARG}" ;;
    * ) echo "Usage: send_to_cluster.sh -c CLUSTER[albert;mlp;ilcc-cluster]"
       exit ;;
  esac
done

if [[ -z ${cluster} ]]
then
    echo "No cluster arg given, setting default to albert.inf.ed.ac.uk"
    cluster="albert"
fi

CLUSTER_HOSTNAME="${cluster}.inf.ed.ac.uk"
echo "Sending data to ${CLUSTER_HOSTNAME}"

ROOTDIR=$(git rev-parse --show-toplevel)
EXCLUDE_FILE=${ROOTDIR}/scripts/send_exclude.txt
rm -rf ./**/.ipynb_checkpoints  # Delete unwanted stuff
rm -rf ./**/__pycache__
rsync -auzh --progress --exclude-from ${EXCLUDE_FILE} ${ROOTDIR} "${USER}@${CLUSTER_HOSTNAME}:~/sp/"
echo "============"
echo "send finished successfully"
