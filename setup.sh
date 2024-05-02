repo_url="https://github.com/fmohr/naiveautoml.git"
branch="55-create-support-for-custom-stoppers-for-early-discarding"
git clone "$repo_url" --branch "$branch"
git clone https://github.com/automl/lcpfn.git
python3 -m venv .env
source .env/bin/activate
pip install -r ./naiveautoml/python/requirements.txt
pip install -r ./experiment/requirements.txt
pip install -e ./naiveautoml/python/
mv pyproject.toml ./lcpfn/
pip install -e ./lcpfn/
python experiment/config_experiment.py