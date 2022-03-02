#!/usr/bin/env bash

USER="s1833057"

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
echo "Getting data from ${CLUSTER_HOSTNAME}"
ROOTDIR=$(git rev-parse --show-toplevel)
EXCLUDE_FILE=${ROOTDIR}/scripts/get_exclude.txt
rsync -auzh --progress --exclude-from ${EXCLUDE_FILE} "${USER}@${CLUSTER_HOSTNAME}:${ROOTDIR}/log" "${ROOTDIR}/"
rsync -auzh --progress --exclude-from ${EXCLUDE_FILE} "${USER}@${CLUSTER_HOSTNAME}:${ROOTDIR}/wandb" "${ROOTDIR}/"
rsync -auzh --progress --exclude-from ${EXCLUDE_FILE} "${USER}@${CLUSTER_HOSTNAME}:${ROOTDIR}/ie_dir" "${ROOTDIR}/"
echo "============"
echo "receive finished successfully"
