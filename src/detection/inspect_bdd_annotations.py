import json


FILE_PATH = "data/bdd100k/samples.json"


print("Loading annotation file...")

with open(FILE_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)


print("\nTop-level type:")
print(type(data))


if isinstance(data, dict):

    print("\nTop-level keys:")
    print(data.keys())


    # FiftyOne datasets commonly store samples here
    if "samples" in data:

        samples = data["samples"]

        print("\nNumber of samples:")
        print(len(samples))

        first_sample = samples[0]

        print("\nFirst sample keys:")
        print(first_sample.keys())

        print("\nFirst sample:")

        for key, value in first_sample.items():

            if key == "detections":

                print("\nDETECTIONS:")
                print(value)

            else:

                print(
                    f"{key}: {value}"
                )


elif isinstance(data, list):

    print("\nNumber of samples:")
    print(len(data))

    print("\nFirst sample:")
    print(data[0])