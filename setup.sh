repo_url="https://github.com/fmohr/naiveautoml.git"
branch="55-create-support-for-custom-stoppers-for-early-discarding"
git clone "$repo_url" --branch "$branch"
git clone https://github.com/automl/lcpfn.git
mv pyproject.toml lcpfn/
python3 -m venv ./naiveautoml/.env
python3 -m venv ./lcpfn/.env
source ./naiveautoml/.env/bin/activate
pip install -r ./naiveautoml/python/requirements.txt
pip install -r ./naiveautoml/experiments/requirements.txt
pip install -e ./naiveautoml/python/
pip install -e ./lcpfn/./