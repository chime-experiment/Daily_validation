#!/bin/bash

# TODO: make this configurable or something
REV=07

SOURCEUSER=dvw
SOURCEID=/root/.ssh/id_cedar_shared
SOURCEHOST=cedar.computecanada.ca
SOURCEDIR=/project/rpp-chime/chime/validation/rev_$REV
SOURCEGLOB="rev${REV}_*.html"

SLACK_WEBHOOK=/usr/local/share/slack/daily_pipeline_webhook
VIEWER_LINK="https://bao.chimenet.ca/daily/view?csd="

DESTBASE=/opt/venvs/daily_viewer/viewer
DESTDIR=${DESTBASE}/rendered/
CHECKFILE=${DESTDIR}/.newest_day.txt

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

num_pattern='([[:digit:]]{4})'

# Figure out what the last notified newest day was
if test -f "$CHECKFILE"; then
    # Read from the file
    current_day=$(<${CHECKFILE})
    # Extract the CSD number
    [[ $current_day =~ $num_pattern ]]
    current_day=${BASH_REMATCH[1]}
else
    current_day=0
fi

# Figure out the most recent day after this sync
newest_day=$(ls ${DESTDIR} | sort | tail -1)
[[ $newest_day =~ $num_pattern ]]
newest_day=${BASH_REMATCH[1]}

if (( newest_day > current_day )); then
    # Send a notice in slack and log the day
    payload='{"text":"'"A new day is available for revision $REV: $VIEWER_LINK$newest_day"'"}'
    webhook=$(<${SLACK_WEBHOOK})
    curl -X POST -H 'Content-type: application/json' --data "${payload}" ${webhook}
    echo ${newest_day} > ${CHECKFILE}
fi
