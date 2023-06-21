salloc --time=3:0:0 --ntasks=1 --cpus-per-task=2 --mem=16G --gres=gpu:1 --account=def-emilios srun $VIRTUAL_ENV/bin/notebook.sh
