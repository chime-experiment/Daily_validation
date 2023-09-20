#!/bin/bash

# TODO: make this configurable or something
REV=07

SOURCEUSER=dvw
SOURCEID=/root/.ssh/id_cedar_shared
SOURCEHOST=cedar.computecanada.ca
SOURCEDIR=/project/rpp-chime/chime/validation/rev_$REV
SOURCEGLOB="rev${REV}_*.html"

DESTBASE=/mnt/md1/daily
DESTDIR=${DESTBASE}/rendered/

# Log to file
exec >>${DESTBASE}/log.txt 2>&1

echo
echo
echo $date -u
echo

# Rsync
rsync --progress --compress --times --protect-args \
  --chmod=0644 --chown=www-data:www-data \
  --update --stats --rsh="ssh -i $SOURCEID" \
  $SOURCEUSER@$SOURCEHOST:$SOURCEDIR/$SOURCEGLOB \
  $DESTDIR
