#!/bin/bash


scriptdir=$(dirname -- "${BASH_SOURCE[0]}")
scriptdir="$( realpath -e -- "$scriptdir"; )"
daily_notebook="${scriptdir}/daily_validation.ipynb"
weekly_notebook="${scriptdir}/weekly_validation.ipynb"

# Set the pythonpath so that helper_funcs can be found
export PYTHONPATH="${scriptdir}:${PYTHONPATH}"

# Set this to stop papermill complaining
export PYDEVD_DISABLE_FILE_VALIDATION=1

workingdir=$(mktemp -d)

usage() {
    echo "Usage:"
    echo "  run.sh [-r REV] [-b BASEDIR]"
    echo
    echo "  Generate the validation plots for the given REV at BASEDIR. They will "
    echo "  be written into the current directory. Only missing or updated will "
    echo "  be processed."
    echo
    echo "  BASEDIR should be the path to the revisions."
    echo "  REV can be a integer revision id, or the string latest (which is the default)."
}

revdir () {
    printf "${base}/rev_${1}"
}

optdir () {
    printf "${outputdir}/rev_${1}"
}

htmlfile () {
    echo "rev${rev}_${1}.html"
}

dnbfile () {
    echo "${workingdir}/rev${rev}_${1}.ipynb"
}

wnbfile () {
    echo "${workingdir}/rev${rev}_14days.ipynb"
}

rev="latest"
base="/project/rpp-chime/chime/chime_processed/daily"
outputdir="/project/rpp-chime/chime/validation"

# Process command line options
while getopts ":hr:b:" opt; do
    case ${opt} in
        h )
            usage
            exit 0
            ;;
        b )
            base="${OPTARG}"
            ;;
        r )
            rev="${OPTARG}"
            ;;
        \? )
            echo "Invalid option -$OPTARG"
            usage
            exit 1
            ;;
    esac
done

# Check if base dir exists
if [[ ! -d "${base}" ]]
then
    echo "Base directory ${base} does not exist."
    exit 2
fi

# Check if output dir exists
if [[ ! -d "${outputdir}" ]]
then
    echo "Output directory ${outputdir} does not exist."
    exit 2
fi

# Resolve the latest revision if needed
if [[ "${rev}" == "latest" ]]
then
    echo $base
    rev=$(ls $base | grep rev | sort | tail -n 1 | cut -c 5-)
    printf "Using latest revision rev_${rev}\n"
fi

echo "Scanning ${base} for rev_${rev} days to process..."

days_to_process=()

for f in $(revdir $rev)/*/delayspectrum_hpf*.h5
do
    csd=$(basename $(dirname $f))
    outfile=$(optdir $rev)/$(htmlfile $csd)

    if [[ ! -f "${outfile}" ]] || [[ "${outfile}" -ot "${f}" ]]
    then
        days_to_process+=("$csd")
    fi
done

printf "Found %i days to process.\n" ${#days_to_process[@]}

for i in ${!days_to_process[@]}
do
    csd=${days_to_process[i]}
    printf "[%4i of %i] processing CSD=%i \n" $i ${#days_to_process[@]} $csd

    printf "  Executing notebook.\n"
    papermill ${daily_notebook} $(dnbfile $csd) -p LSD ${csd} -p rev_id ${rev} -k python3 \
        --report-mode --no-log-output --no-progress-bar
    printf "  Converting notebook to html.\n"
    jupyter nbconvert --to html --no-input --output-dir=$(optdir $rev) $(dnbfile $csd)
    rm $(dnbfile $csd)
done

# Update the 14-day notebook as well
printf "Processing grid of recent days.\n"
papermill ${weekly_notebook} $(wnbfile) -p rev_id ${rev} -k python3 --report-mode \
    --no-log-output --no-progress-bar
printf " Converting notebook to html.\n"
jupyter nbconvert --to html --no-input --output-dir=$(optdir $rev) $(wnbfile)
rm $(wnbfile)
