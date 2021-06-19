conda activate pytorch_gpu
cd /nfs/users/minhuiyu/GSTransformer

# ENZYMES Transformer
python -m train --bmname=ENZYMES --dim=64 --mlp-dim=64 --num-trans-layers=6  --num-classes=6 --method=GSTransformer --lr=0.01 --num-heads=1 --max-nodes=125 --cuda=3

# DD GSTransformer
python -m train --bmname=DD --dim=64 --mlp-dim=64 --num-trans-layers=6  --num-classes=2 --method=GSTransformer --lr=0.001 --num-heads=8 --max-nodes=903 --cuda=3 --pool=mean --sort-type=dfs --dropout=0.1 --name-suffix=527meandfs --mask=adj --batch-size=16