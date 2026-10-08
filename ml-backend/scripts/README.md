# How to add a fsd50k dataset:

```bash 
```
cd ~/dev/equine-ai-monitor
# CC0 dog sounds → apps/ml-backend/data/datasets/FSD50K/dog_barking/<uploader>/...
python3 apps/ml-backend/scripts/filter_fsd50k.py

# point the ingestion at the result, then rebuild the dataset version
export DOG_BARK_DIR=apps/ml-backend/data/datasets/FSD50K/dog_barking
```
```


