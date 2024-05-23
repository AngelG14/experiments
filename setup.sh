git clone https://github.com/automl/lcpfn.git
python3 -m venv .env
source .env/bin/activate
pip install -e 'git+https://github.com/fmohr/naiveautoml.git@55-create-support-for-custom-stoppers-for-early-discarding#egg=naiveautoml&subdirectory=python'
pip install lccv
pip install -r ./experiment/requirements.txt
mv pyproject.toml ./lcpfn/
pip install -e ./lcpfn/
python experiment/config_experiment.py