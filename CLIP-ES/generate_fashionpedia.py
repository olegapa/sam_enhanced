import json

DATA_PATH = "../../datasets/Fashionpedia"


def prepare_file(attr_file, out_file):
    with open(f'{DATA_PATH}/{attr_file}') as f:
        data = json.load(f)
    with open(f'{DATA_PATH}/annotation/info_test2020.json') as f:
        info = json.load(f)

    attr = info['attributes']
    interesting_attributes = list()
    for a in attr:
        if a['supercategory'] == "nickname":
            interesting_attributes.append(a['id'])

    images_info = list(data['images'])
    images_an = list(data['annotations'])

    with open(f'{out_file}', 'w') as f:
        while len(images_info) != 0:
            image_an = None
            image = images_info.pop(0)
            for j in range(len(images_an)):
                if images_an[j]['image_id'] == image['id']:
                    image_an = images_an.pop(j)
                    break

            line = image['file_name'][0:-4]
            if image_an:
                add = False
                for a in image_an['attribute_ids']:
                    if a in interesting_attributes:
                        line += f' {a}'
                        add = True
                if add:
                    f.write(f'{line}\n')


def print_attributes():
    with open(f'{DATA_PATH}/annotation/info_test2020.json') as f:
        info = json.load(f)

    attr = info['attributes']
    attributes = list()
    for a in attr:
        attributes.append(a['name'])
    print(attributes)


# prepare_file('annotation/attributes_train2020.json', './custom_dataset/train.txt')
# prepare_file('annotation/attributes_val2020.json', './custom_dataset/val.txt')
print_attributes()
