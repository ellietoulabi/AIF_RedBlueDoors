#!/bin/bash
#SBATCH --account=def-mbowling
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-2:59
#SBATCH --salloc cpu-freq=Performance

# setup and tear down takes .5-2 minutes.
module purge
module load python/3.11.4  scipy-stack

if [ "$SLURM_TMPDIR" == "" ]; then
    exit 1
fi

cd $SLURM_TMPDIR



mkdir project
mkdir virtualenvs

cd project
echo "Cloning repo..."
git clone --quiet https://github.com/ellietoulabi/AIF_RedBlueDoors.git 


cd ../virtualenvs
python3.11 -m venv .venv
source .venv/bin/activate

cd ../project/AIF_RedBlueDoors/
pip install -r requirements.txt


cd runs
echo "Running experiment..."

python run_redbluedoors_ql_random.py

echo "done"




# alias  getcpu='salloc --account=def-jrwright   --cpus-per-task=4  --mem=16G --time=0-2:59'