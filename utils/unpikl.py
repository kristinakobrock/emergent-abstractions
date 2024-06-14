import pickle
import argparse
from pathlib import Path

"""
This is set up to open the params pickle files, if opening another pickle file
then find out the type used first, if argparse.namespace then you can use this
as is, if not alterations need to be made for the specified type. 
"""
# dont forget, when you call use pickle_path= r"relative_path"
pickle_path = Path(r"results\(3,4)_game_size_10_vsf_3\standard\0\params.pkl")
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Check if the data is an argparse.Namespace instance
if isinstance(data, argparse.Namespace):
    print("Contents of the argparse.Namespace object:")
    # Convert the Namespace object to a dictionary
    data_dict = vars(data)
    # Iterate over the dictionary and print each attribute and its value
    for key, value in data_dict.items():
        print(f"{key}: {value}")
else:
    # If the data is not an argparse.Namespace, handle other types (if needed)
    print("Data type:", type(data))
    print("Content:", data)
