#!/bin/bash -e

image_name=kingston/systems/python-model

declare -a volumes=(
  "$(pwd):/mnt/work:Z"
)

# These are the default env vars in the container:
declare -a default_env_vars=(
)

# These env vars, if set, will be carried into the container:
declare -a env_vars=(
)

# These env vars are always set in the container:
declare -a force_env_vars=(
)

build_opts=""
run_opts="--rm -u $(id -u) -i --cap-add SYS_PTRACE"

### Shouldn't need to touch anything below this line ###

build="docker build ${build_opts} --tag=${image_name} ."
echo ${build}
eval ${build}

if [[ "$(tty -s; echo $?)" == "0" ]]; then
  run_opts="${run_opts} -t"
fi

for vol in ${volumes[@]}; do
  run_opts="${run_opts} -v ${vol}"
done

for var_val in ${default_env_vars[@]}; do
  run_opts="${run_opts} -e ${var_val}"
done

for var in ${env_vars[@]}; do
  if [[ "${!var}" != "" ]]; then
    if [[ "${!var}" =~ (\ |\t) ]]; then
      run_opts="${run_opts} -e ${var}=\"${!var}\""
    else
      run_opts="${run_opts} -e ${var}=${!var}"
    fi
  fi
done

for var_val in ${force_env_vars[@]}; do
  run_opts="${run_opts} -e ${var_val}"
done

run="docker run ${run_opts} ${image_name} $@"
echo ${run}
eval ${run}

