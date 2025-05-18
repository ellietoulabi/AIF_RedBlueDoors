#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-2:59
#SBATCH --salloc cpu-freq=Performance

module purge
module load python/3.11.4  scipy-stack

if [ "$SLURM_TMPDIR" == "" ]; then
    exit 1
fi

cd $SLURM_TMPDIR
echo "Moved to SLURM_TMPDIR"

mkdir project
mkdir virtualenvs
echo "Created directories"


echo "Cloning repo..."
cd project
git clone --quiet https://github.com/ellietoulabi/AIF_RedBlueDoors.git 
echo "Clone done"


echo "Creating virtual environment..."
cd ../virtualenvs
python3.11 -m venv .venv
source .venv/bin/activate
echo "Virtual environment created"


echo "Installing dependencies..."
cd ../project/AIF_RedBlueDoors/
pip install -r requirements.txt
echo "Dependencies installed"


echo "Running experiment..."
cd runs
python run_redbluedoors_ql_random.py
echo "Experiment done"


echo "Copying logs to home directory..."
cd ..
cp -r logs /home/toulabin/projects/def-jrwright/toulabin/AIF_RedBlueDoors/
echo "Copy done"




# alias  getcpu='salloc --account=def-jrwright   --cpus-per-task=4  --mem=16G --time=0-2:59'