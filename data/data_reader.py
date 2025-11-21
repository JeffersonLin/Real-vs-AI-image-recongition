import os

dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DeepGuardDB_v1", "DeepGuardDB_v1"))

FOLDERS = {
    "dalle": os.path.join(dataset, "DALLE_dataset"),
    "glide": os.path.join(dataset, "GLIDE_dataset"),
    "imagen": os.path.join(dataset,"IMAGEN_dataset"),
    "sd": os.path.join(dataset,"SD_dataset"),
    "json": os.path.join(dataset,"json_files"),
} #access a specific dataset


def load_all_data():
    data = {}
    #'dalle': {'real':[], 'fake':[]}

    for name, root in FOLDERS.items(): #name = dalle
        real_dir = os.path.join(root, "real") #access real dataset within dalle
        fake_dir = os.path.join(root, "fake") #access fake dataset within dalle

        real_files = []
        if os.path.isdir(real_dir):
            for dirpath, dirnames, filenames in os.walk(real_dir):
                for f in filenames:
                    real_files.append(os.path.join(dirpath, f))

        fake_files = []
        if os.path.isdir(fake_dir):
            for dirpath, dirnames, filenames in os.walk(fake_dir):
                for f in filenames:
                    fake_files.append(os.path.join(dirpath, f)) #adds all the files within the fake folder into fake_files
                    # appends "...\\DeepGuardDB_v1\\DeepGuardDB_v1\\DALLE_dataset\\fake"
                    # and "000000000042.jpg.png"

        data[name] = {"real": real_files, "fake": fake_files}

    return data


def load_dataset(name):
    root = FOLDERS[name]

    real_dir = os.path.join(root, "real")
    fake_dir = os.path.join(root, "fake")

    real_files = []
    fake_files = []

    if os.path.isdir(real_dir):
        for dirpath, dirnames, filenames in os.walk(real_dir):
            for f in filenames:
                real_files.append(os.path.join(dirpath, f))

    if os.path.isdir(fake_dir):
        for dirpath, dirnames, filenames in os.walk(fake_dir):
            for f in filenames:
                fake_files.append(os.path.join(dirpath, f))

    if real_files or fake_files:
        return {"real": real_files, "fake": fake_files}

    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            all_files.append(os.path.join(dirpath, f))

    return {"all": all_files}