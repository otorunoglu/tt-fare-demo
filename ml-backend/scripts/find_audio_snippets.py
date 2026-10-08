import json
from collections import Counter, defaultdict
PATH = "/datasets/manifests/v2.1.jsonl"
manifest = [json.loads(l) for l in open(PATH)]  # your version
snore_test = [r for r in manifest if r["label"] == "snoring" and r["split"] == "test"]
snore_train = [r for r in manifest if r["label"] == "snoring" and r["split"] == "train"]

print("test snore groups:", Counter(r["group_id"] for r in snore_test))
print("train snore groups:", Counter(r["group_id"] for r in snore_train))
# the key question: do test and train snoring share ANY group?
test_g = {r["group_id"] for r in snore_test}
train_g = {r["group_id"] for r in snore_train}
print("groups only in test:", test_g - train_g)
print("overlap:", test_g & train_g)


# Listen to sounds:
import numpy as np, soundfile as sf
for r in snore_test:
    blob = np.load("/datasets/blobs/" + r["hash"][:2] + "/" + r["hash"] + ".npy")
    sf.write("/datasets/tmp/snore_test_" + r["hash"] + ".wav", blob, 16000)  # listen to it
    print(r["group_id"], r["split"], r["label"])