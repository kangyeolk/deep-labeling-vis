import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class SiameseTrainDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    """
    def __init__(self, image_folder):
        
        self.dataset = image_folder
        self.samples = self.dataset.samples
        self.transform = self.dataset.transform
        
        self.train_data = [ss[0] for ss in self.samples]
        self.train_labels = [ss[1] for ss in self.samples]
        self.train_labels = np.asarray(self.train_labels)
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                    for label in self.labels_set}

    def __getitem__(self, index):
        
        target = np.random.randint(0, 2)
        img1, label1 = self.train_data[index], self.train_labels[index].item()
        # Positive sample
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        # Negative sample
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2 = self.train_data[siamese_index]
        
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.samples)

class SiameseTestDataset(Dataset):
    """
    Test: Creates fixed pairs for testing
    """
    def __init__(self, image_folder):
        
        self.dataset = image_folder
        self.samples = self.dataset.samples
        self.transform = self.dataset.transform
       
        # generate fixed pairs for testing
        self.test_data = [ss[0] for ss in self.samples]
        self.test_labels = [ss[1] for ss in self.samples]
        self.test_labels = np.asarray(self.test_labels)
        self.labels_set = set(self.test_labels)
        self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                    for label in self.labels_set}

        random_state = np.random.RandomState(29)

        positive_pairs = [[i, 
                            random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                            1]
                            for i in range(0, len(self.test_data), 2)]

        negative_pairs = [[i,
                            random_state.choice(self.label_to_indices[
                                                    np.random.choice(
                                                        list(self.labels_set - set([self.test_labels[i].item()]))
                                                    )]),0]
                            for i in range(1, len(self.test_data), 2)]
        
        self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        img1 = self.test_data[self.test_pairs[index][0]]
        img2 = self.test_data[self.test_pairs[index][1]]
        target = self.test_pairs[index][2]
        
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.dataset)



""" Currently not be used """

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
