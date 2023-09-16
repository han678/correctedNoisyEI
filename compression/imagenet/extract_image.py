import os
import os.path
import errno

from PIL import Image

# root = "/users/visics/hzhou/data/"
# root = "D:/codes/data/"
root = "/data/leuven/326/vsc32665/"
fn_train = root + "val.txt"
image_names = []
image_labels = []
with open(fn_train) as f:
    for line in f:
        name, label = line.strip().split()
        image_names.append(name)
        image_labels.append(label)
f.close()

img_root = root + "ILSVRC2012"
n = 50000
fn_random_n = root + "random_{}.txt".format(n)
image_names_n = image_names[0:n]
with open(fn_random_n, "w") as outfile:
    for i in range(n):
        outfile.write(str(image_names[i]) + "\n")

image_labels_n = image_labels[0:n]
fn_res_n = root + "random_{}_with_label.txt".format(n)
with open(fn_res_n, 'w') as outfile:
    for i in range(n):
        outfile.write(str(image_names[i]) + " " + str(image_labels[i]) + "\n")
outfile.close()

content = []
fn_random_n = root + "random_{}.txt".format(n)
for i in range(len(image_names_n)):
    # content.append("{} {}\n".format(os.path.join(img_root, image_names_n[i]), image_labels_n[i]))
    content.append("{} {}\n".format(img_root + "/" + image_names_n[i], image_labels_n[i]))


def _make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


scaled_ime_dir = root + "scaled_ime_dir"
_make_dir(scaled_ime_dir)

new_content = []
content_with_label = []
size = (256, 256)
i = 0
total = n
for line in content:
    path, label = line.strip().split()
    try:
        im = Image.open(path).resize(size)
        # print(im.mode)
    except IOError:
        print(path)
        continue

    if im.mode != 'RGB':
        # cause axes don't match array error
        continue

    i += 1
    if i > total:
        break
    # new_path = os.path.join(scaled_ime_dir, os.path.basename(path))
    im.save(path)
    new_content.append("{}\n".format(path))
    content_with_label.append("{} {}\n".format(path, label))

fn_res_n = root + "random_{}".format(n)
with open(fn_res_n, 'w') as f:
    f.writelines(new_content)

fn_label_n = root + "random_{}_with_label".format(n)
with open(fn_label_n, 'w') as f:
    f.writelines(content_with_label)
