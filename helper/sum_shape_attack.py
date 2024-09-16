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
search_directory = "infrared_patch_attack/res"

# List of files to search for
target_files = ["res_clean.txt", "res_patch.txt"]

with open("res_shape.csv", "w") as res:
    res.write("defense, selection, seed, attack, map0.5")
    for file in find_files(search_directory, target_files):
        with open(file, "r") as f:
            content = f.read().replace("[", "").replace("]","")
            content = [i for i in content.split(" ") if i != ""]
            attack = file.split("/")[-1].split("_")[-1].replace(".txt","")
            file_s = file.split("/")[2]
            file_s = file_s.split("_")
            if "det" in file:
                t = "_".join(file_s[:2])
                aug = "_".join(file_s[2:-1])
                seed = int(file_s[-1])
                print(t, aug, seed, attack, float(content[1]))
            else:
                t = file_s[0]
                aug = " "
                seed = int(file_s[-1])
                print(t, aug, seed, attack, float(content[1]))
            map = float(content[1])
            res.write(f'\n{t},{aug},{seed},{attack},{map}')