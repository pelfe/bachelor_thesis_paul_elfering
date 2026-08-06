import zipfile
import tempfile
import os
import shutil
import h5py


src = "data/models/external models/model_humam/baseline_20260528_133156_fold3.keras"
dst = "data/models/external models/model_humam/baseline_20260528_133156_fold3_fixed.keras"


def copy_item(src_obj, dst_parent, name):

    new_name = name.replace("\\", "/")

    if isinstance(src_obj[name], h5py.Group):

        # Split path components and recreate hierarchy
        current = dst_parent

        for part in new_name.split("/"):
            if part not in current:
                current = current.create_group(part)
            else:
                current = current[part]

        copy_group(src_obj[name], current)

    else:
        dst_parent.create_dataset(
            new_name,
            data=src_obj[name][:]
        )


def copy_group(src_group, dst_group):

    for name in src_group.keys():
        copy_item(src_group, dst_group, name)



with tempfile.TemporaryDirectory() as tmp:

    # unpack keras archive
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)

    old_weights = os.path.join(tmp, "model.weights.h5")
    new_weights = os.path.join(tmp, "model.weights_fixed.h5")


    with h5py.File(old_weights, "r") as fin:
        with h5py.File(new_weights, "w") as fout:
            copy_group(fin, fout)


    os.remove(old_weights)
    os.rename(new_weights, old_weights)


    # repack
    with zipfile.ZipFile(dst, "w") as z:
        for root, dirs, files in os.walk(tmp):
            for f in files:
                path = os.path.join(root, f)
                z.write(
                    path,
                    os.path.relpath(path, tmp)
                )


print("Created:")
print(dst)