import os


def find_files(directory, target_files):
    found_files = {file: False for file in target_files}

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file in target_files:
                found_files[file] = True
                # print(f"Found '{file}' in '{root}'")
                yield f"{root}/{file}"


# Folder to start the search from
search_directory = "runs/train"

# List of files to search for
target_files = ["res_hcb_2.txt"]

with open("res_hcb.csv", "w") as res:
    res.write("defense, selection, seed, attack, map0.5")
    for file in find_files(search_directory, target_files):
        with open(file, "r") as f:
            content = f.read()
            attack = "hcb"
            file_s = file.split("/")[2]
            file_s = file_s.split("_")

            map_0 = float(content.split("\n")[1].split(" ")[1].replace(",", ""))
            map_1 = float(content.split("\n")[5].split(" ")[1].replace(",", ""))
            map_2 = float(content.split("\n")[9].split(" ")[1].replace(",", ""))

            map = min([map_0, map_1, map_2])

            if "det" in file:
                t = "_".join(file_s[:2])
                aug = "_".join(file_s[2:-1])
                seed = int(file_s[-1])
                print(t, aug, seed, attack, map)
            else:
                t = file_s[0]
                aug = " "
                seed = int(file_s[-1])
                print(t, aug, seed, attack, map)
            res.write(f'\n{t},{aug},{seed},{attack},{map}')