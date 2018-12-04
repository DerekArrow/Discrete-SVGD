# Binary Neural network
## train full precision
python main.py --fullpre 1 --ensemble 0 --particles 1 --epochs 100  --dataset C
## train BNNs with bagging 
python main.py --fullpre 0 --ensemble 1 --particles 4 --epochs 100  --dataset C
## train BNNs with naive surrogate (sigmoid)
python mainGF.py --naive 1 --particles 4 --epochs 100  --dataset C
## train BNNs with GF-SVGD
python mainGF.py --naive 0 --particles 4 --epochs 100  --dataset C