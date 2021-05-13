# ENZYMES
python -m train --datadir=data --bmname=ENZYMES --cuda=3 --max-nodes=100 --num-classes=6

# ENZYMES - Diffpool
python -m train --bmname=ENZYMES --dim=128 --mlp-dim=64 --num-trans-layers=6  --num-classes=6 --method=GSTransformer --lr=0.01 --num-heads=1 --max-nodes=125 --cuda=3