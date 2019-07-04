import numpy
import os
from torch.utils.data import Dataset, DataLoader


class ICDataset(Dataset):
    def __init__(self, images_names, visual_feats, captions):
        self.visual_feats = dict(zip(images_names, visual_feats))
        self.captions = captions

    def __getitem__(self, item):
        img_name, caption = self.captions[item]
        visual_feat = self.visual_feats[img_name]

        return img_name, visual_feat, caption

    def __len__(self):
        return len(self.captions)


def get_loader(image_names, viusal_feats, captions, batch_size, shuffle, num_workers, pin_memory):
    dataset = ICDataset(image_names, viusal_feats, captions)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory)


def load_coco_files(file_names, file_vectors, file_captions, vector_dimensions):
    assert os.path.isfile(file_names), "no existe archivo " + file_names
    assert os.path.isfile(file_vectors), "no existe archivo " + file_vectors
    assert os.path.isfile(file_captions), "no existe archivo " + file_captions

    print("leyendo " + file_names)
    names = [line.strip() for line in open(file_names)]

    print("leyendo " + file_vectors)
    mat = numpy.fromfile(file_vectors, dtype=numpy.float32)
    vectors = numpy.reshape(mat, (len(names), vector_dimensions))
    print(str(len(names)) + " vectores de largo " + str(vector_dimensions))

    print("leyendo " + file_captions)
    captions = [line.strip().split("\t") for line in open(file_captions, encoding='utf-8')]

    return names, vectors, captions
