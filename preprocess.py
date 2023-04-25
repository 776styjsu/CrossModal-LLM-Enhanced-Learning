import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.data as data
import clip.embed as clip
# from torch.utils.data import Subset


def get_data(device='cuda' if torch.cuda.is_available() else 'cpu',
             mode='standard', dataset='mnist', val_split=0.2, batch_size=64,
             save_embedding=True, class_wise_embedding=True):
    train_dataset, test_dataset = None, None
    if dataset == 'mnist':
        # Download MNIST dataset
        train_dataset = datasets.MNIST(
            root='dataset/',
            train=True,
            transform=None,
            download=True
        )

        # Create a subset of the training set that contains only the first 100 data points
        # This is for debugging purposes
        '''
        indices_range = 1000
        subset_indices = list(range(indices_range))
        train_dataset = Subset(train_dataset, subset_indices)
        '''

        test_dataset = datasets.MNIST(
            root='dataset/',
            train=False,
            transform=None,
            download=True
        )

    elif dataset == 'cifar10':
        # Download CIFAR10 dataset
        train_dataset = datasets.CIFAR10(
            root='dataset/',
            train=True,
            transform=None,
            download=True
        )

        test_dataset = datasets.CIFAR10(
            root='dataset/',
            train=False,
            transform=None,
            download=True
        )

    if mode == 'standard':
        to_tensor = transforms.ToTensor()
        train_dataset = [(to_tensor(img), label) for img, label in train_dataset]
        test_dataset = [(to_tensor(img), label) for img, label in test_dataset]

    elif mode == 'clip':

        class PreprocessedImagesDataset(data.Dataset):
            def __init__(self, images, labels, preprocess):
                self.images = images
                self.labels = labels
                self.preprocess = preprocess

            def __getitem__(self, index):
                preprocessed_image = self.preprocess(self.images[index])
                label = self.labels[index]
                return preprocessed_image, label

            def __len__(self):
                return len(self.images)

        with torch.no_grad():
            # Embed the images using CLIP with pretrained weights
            model, preprocess = clip.create_model_and_transforms()

            # Get the images and labels from the train and test datasets
            train_images, train_labels = zip(*train_dataset)
            test_images, test_labels = zip(*test_dataset)

            # Turn the images and labels into datasets
            train_dataset = PreprocessedImagesDataset(train_images, train_labels, preprocess)
            test_dataset = PreprocessedImagesDataset(test_images, test_labels, preprocess)

            train_image_embeds = []
            train_text_embeds = []
            train_labels = []

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            print("Embedding training images...")

            # TODO: Need to implement a separate loader for different modalities
            #  e.g. Obtain the text through LLM/template mining and added to a text dataset
            #  then load the text and image datasets separately
            for batch_images, batch_labels in train_dataloader:
                batch_images = batch_images.float().to(device)
                batch_image_embeds = model.encode_image(batch_images).float().to(device)
                batch_image_embeds /= batch_image_embeds.norm(dim=-1, keepdim=True)

                texts = batch_labels.numpy().astype(str)
                # Fixed template for text embedding
                text_tokens = clip.tokenizer.tokenize(["This is a photo of " + text for text in texts]).to(device)
                batch_text_embeds = model.encode_text(text_tokens).float().to(device)
                batch_text_embeds /= batch_text_embeds.norm(dim=-1, keepdim=True)

                train_image_embeds.append(batch_image_embeds)
                train_text_embeds.append(batch_text_embeds)
                train_labels.append(batch_labels)

            train_image_embeds = torch.cat(train_image_embeds, dim=0).to(device)
            train_text_embeds = torch.cat(train_text_embeds, dim=0).to(device)
            train_labels = torch.cat(train_labels, dim=0).to(device)

            # TODO: I suspect that the concatenation of the image and text embeddings
            #  causes accuracy to be reported in an unintended and weird way
            # Concatenate the image and text embeddings
            train_vectors = torch.cat((train_image_embeds, train_text_embeds), dim=0).to(device)
            train_labels_vector = torch.cat((train_labels, train_labels), dim=0).to(device)
            train_dataset = data.TensorDataset(train_vectors, train_labels_vector)

            test_image_embeds = []
            test_labels = []

            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print("Embedding test images...")

            for batch_images, batch_labels in test_dataloader:
                batch_images = batch_images.float().to(device)
                batch_image_embeds = model.encode_image(batch_images).float().to(device)
                batch_image_embeds /= batch_image_embeds.norm(dim=-1, keepdim=True)

                test_image_embeds.append(batch_image_embeds)
                test_labels.append(batch_labels)

            test_image_embeds = torch.cat(test_image_embeds, dim=0).to(device)
            test_labels = torch.cat(test_labels, dim=0).to(device)

            test_dataset = data.TensorDataset(test_image_embeds, test_labels)

            # Save the embedded datasets to disk
            save_dir = 'embeddings/'
            if save_dir is not None and save_embedding:
                torch.save(train_dataset, f"{save_dir}/{dataset}_train_dataset_embedded.pt")
                torch.save(test_dataset, f"{save_dir}/{dataset}_test_dataset_embedded.pt")

    else:
        raise ValueError(f"Invalid mode: {mode}")

    train_loader, val_loader = split_train_val(train_dataset, val_split)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, val_loader, test_loader


def load_data(dataset='mnist', val_split=0.2, batch_size=64):
    print(f"Files found, loading {dataset} dataset...")
    train_dataset = torch.load(f'embeddings/{dataset}_train_dataset_embedded.pt')
    test_dataset = torch.load(f'embeddings/{dataset}_test_dataset_embedded.pt')
    train_loader, val_loader = split_train_val(train_dataset, val_split)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, val_loader, test_loader


def split_train_val(dataset, val_split):
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader
