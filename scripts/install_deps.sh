#!/usr/bin/env bash
set -euo pipefail
export http_proxy="${http_proxy:-http://host.docker.internal:1081}"
export https_proxy="${https_proxy:-http://host.docker.internal:1081}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
if declare -F nros-proxy-on >/dev/null 2>&1; then nros-proxy-on || true; fi

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate tro
cd "${HOME}/TRO-Grasp"

REQ="${HOME}/TRO-Grasp/requirements.server.txt"
sed 's|git+ssh://git@github.com/chungmin99/pyroki.git|git+https://github.com/chungmin99/pyroki.git|' \
  requirements.txt > "$REQ"

# torch/pytorch3d already installed; jaxls vs pyroki conflict if resolved together;
# theseus needs torch visible (no build isolation).
grep -vE '^(torch==|pytorch3d==|torch_cluster==|numpy==|jaxls |.*pyroki|.*theseus_ai|nvidia-|scikit-sparse==)' "$REQ" \
  > /tmp/tro_req_core.txt

pip install "numpy==1.26.4" hatchling hatch-vcs setuptools wheel packaging Cython
conda install -y -c conda-forge suitesparse scikit-sparse

pip install --no-build-isolation -r /tmp/tro_req_core.txt

pip install --no-build-isolation \
  "jaxls @ git+https://github.com/brentyi/jaxls.git@e19f32539a1416f65257c3e9b0bbd843f469d924"
pip install --no-deps --no-build-isolation \
  "git+https://github.com/chungmin99/pyroki.git@70b30a56b1e1ea83fb4c2cac8fe2c63a0624b9ce"

if [[ -d "${HOME}/TRO-Grasp/src/theseus-ai" ]]; then
  pip install --no-build-isolation -e "${HOME}/TRO-Grasp/src/theseus-ai"
else
  pip install --no-build-isolation \
    "theseus_ai @ git+https://github.com/facebookresearch/theseus.git@c8583de41824613fb135ee9bad0e930ded6be404"
fi

echo "deps OK"
