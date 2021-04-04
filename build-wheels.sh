#!/bin/bash
set -e -u -x
# adapted from pypa's python-manylinux-demo and
# https://github.com/pypa/python-manylinux-demo/blob/7e24ad2c202b6f58c55654f89c141bba749ca5d7/travis/build-wheels.sh

# navigate to the root of the mounted project
cd $(dirname $0)

bin_arr=(
    #/opt/python/cp37-cp37m/bin
    /opt/python/cp38-cp38/bin
    #/opt/python/cp39-cp39/bin
)

# add  python to image's path
export PATH=/opt/python/cp37-cp37m/bin/:$PATH
# download install script
curl -#sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py > get-poetry.py
# install using local archive
python get-poetry.py -y --file poetry-1.1.5-linux.tar.gz
# install openblas
yum install -y openblas-devel

function build_poetry_wheels
{
    # build wheels for 3.7-3.9 with poetry 
    for BIN in "${bin_arr[@]}"; do
        rm -Rf build/*
        # install build deps
        "${BIN}/python" ${HOME}/.poetry/bin/poetry run pip install numpy==1.18.1
        BUILD_WHEELS=1 "${BIN}/python" ${HOME}/.poetry/bin/poetry build -f wheel
        mkdir -p ./tmp
        for whl in dist/*.whl; do
            auditwheel repair "$whl" --plat $1 -w "./tmp" 
            whlname="$(basename "$(echo ./tmp/*.whl)")"
            "${BIN}/python" -m pip install ./tmp/"$whlname"
            # test if installed wheel imports correctly
            "${BIN}/python" -c "from pyhtnorm import *"
            mv ./tmp/*.whl wheelhouse/
            rm "$whl"
        done
    done
}

build_poetry_wheels "$PLAT"
