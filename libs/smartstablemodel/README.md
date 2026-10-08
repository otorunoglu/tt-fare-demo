# Smart Stable Model



## Getting started
Run setup scrip, so PANNs data will be copied, where it is expected.
```
python3 ./setup.py install
```

Test on example file:
```
python horse_kick_monitor.py --input-file ./data/stable03_not_trimmed_12_events.wav --output-file results.json --verbose
```

Test live monitoring:
```
python horse_kick_monitor.py --duration 60 --verbose
```

## Model Setup

Train CAE on normal labels only:
```
python smartstablemodel/horse_kick_monitor.py \
    --train-cae-normal-labels /path/to/export.json \
    --audio-base-path /data/local-files \
    --epochs 50
```
Train kick detector on kicks vs normals:
```
python smartstablemodel/horse_kick_monitor.py \
    --train-from-ls-export /path/to/export.json \
    --audio-base-path /data/local-files \
    --epochs 50
```