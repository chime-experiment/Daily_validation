#!/bin/bash


scriptdir=$(dirname -- "${BASH_SOURCE[0]}")
scriptdir="$( realpath -e -- "$scriptdir"; )"
notebook="${scriptdir}/daily_validation.ipynb"

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
    printf "${base}/rev_%02i" $1
}

htmlfile () {
    echo "rev${rev}_${1}.html"
}

nbfile () {
    echo "${workingdir}/rev${rev}_${1}.ipynb"
}


rev="latest"
base="/project/rpp-chime/chime/chime_processed/daily"

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

# Resolve the latest revision if needed
if [[ "$rev" == "latest" ]]
then
    echo $base
    rev=$(ls $base | grep rev | sort | tail -n 1 | cut -c 5-)
    printf "Using latest revision rev_%02i\n" $rev
fi

echo "Scanning ${base} for rev_${rev} days to process..."

days_to_process=()

for f in $(revdir $rev)/*/delayspectrum_hpf*.h5
do
    csd=$(basename $(dirname $f))
    outfile=$(htmlfile $csd)

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
    papermill ${notebook} $(nbfile $csd) -p LSD ${csd} -p rev_id 7 -k mlchime \
        --report-mode --no-log-output --no-progress-bar
    printf "  Converting notebook to html.\n"
    jupyter nbconvert --to html --no-input --output-dir=. $(nbfile $csd)
done
