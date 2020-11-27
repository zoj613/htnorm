#!/bin/bash
set -e -u -x
# adapted from pypa's python-manylinux-demo and
# https://github.com/sdispater/pendulum/blob/master/build-wheels.sh

# navigate to the root of the mounted project
cd $(dirname $0)

bin_arr=(
    /opt/python/cp36-cp36m/bin
    /opt/python/cp37-cp37m/bin
    /opt/python/cp38-cp38/bin
)
# add  python to image's path
export PATH=/opt/python/cp38-cp38/bin/:$PATH
# download && install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# install openblas
yum install -y openblas-devel zip

function build_poetry_wheels
{
    # build wheels for 3.6-3.8 with poetry 
    for BIN in "${bin_arr[@]}"; do
        rm -Rf build/*
        BUILD_WHEELS=1 "${BIN}/python" ${HOME}/.poetry/bin/poetry build -f wheel
    done

    # add C libraries to wheels
    for whl in dist/*.whl; do
        auditwheel repair "$whl" --plat $1
        rm "$whl"
    done
}

build_poetry_wheels "$PLAT"
