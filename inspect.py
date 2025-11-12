import pickle

with open("train.pkl", "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
print("Keys:", data.keys())

for k, v in data.items():
    try:
        print(k, "=> type:", type(v), "shape:", v.shape)
    except:
        print(k, "=> type:", type(v))
