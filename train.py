import torch
import torch.nn as nn
import torch.optim as optim

import clip.embed
import models.softmax_regression as softmax_regression

import preprocess
from preprocess import get_data


class Trainer:
    def __init__(self, mode='standard', dataset='mnist', load_embedding=True, model=None, learning_rate=0.001,
                 batch_size=64, num_epochs=5, val_split=0.1):
        print("Setting up trainer...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.dataset = dataset
        self.load_embedding = load_embedding
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.val_split = val_split

        self.train_dataset, self.test_dataset, self.train_loader, self.val_loader, self.test_loader,\
            self.input_size, self.num_classes = self._get_data()

        self.model = None
        if model == 'softmax_regression':
            self.model = softmax_regression.SoftmaxRegression(
                self.input_size, self.num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # TODO: Add scheduler

        print("Trainer set up complete")
        print("Trainer specifications:")
        print(f"Mode: {self.mode}")
        print(f"Dataset: {self.dataset}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Validation split: {self.val_split}")

    # TODO: Further separate the data preparation from the model training to make the code more modular
    def _get_data(self):
        if self.load_embedding and self.mode == 'clip':
            try:
                train_dataset, test_dataset, train_loader, \
                    val_loader, test_loader = preprocess.load_data(self.dataset,
                                                                   self.val_split, self.batch_size)
            except FileNotFoundError:
                print("No embedding found. Generating new embedding...")
                train_dataset, test_dataset, train_loader,\
                    val_loader, test_loader = get_data(self.device,
                                                       self.mode, self.dataset,
                                                       self.val_split, self.batch_size)
        else:
            train_dataset, test_dataset, train_loader, \
                val_loader, test_loader = get_data(self.device,
                                                   self.mode, self.dataset, self.val_split,
                                                   self.batch_size)

        # Inspect the shape of the first example in the train_loader to get the input size
        example_input, _ = next(iter(train_loader))
        input_size = example_input.shape[1:]  # ignore the batch dimension

        # Inspect the labels in the train_loader to get the number of classes
        _, example_label = next(iter(train_loader))
        num_classes = len(torch.unique(example_label))

        return train_dataset, test_dataset, train_loader, val_loader, test_loader, input_size, num_classes

    def train(self):
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            self.model.train()
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)
                data = data.reshape(data.shape[0], -1)

                # forward
                scores = self.model(data)
                loss = self.criterion(scores, targets)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()

            # Evaluate the model on the validation set
            self.check_accuracy(self.val_loader)

    def check_accuracy(self, loader, is_test=False):
        num_correct = 0
        num_samples = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                x = x.reshape(x.shape[0], -1)

                scores = self.model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(
                f'Got {num_correct} / {num_samples} '
                f'with accuracy {float(num_correct)/float(num_samples)*100:.2f}%'
            )

    # TODO: Save checkpoint and save accuracy for early stopping

    def check_similarity_and_predict(self):
        # Check similarity between test image and each label in the dataset
        # and predict the label with the highest similarity
        print("Checking similarity and predicting...")
        similarity_test_loader = preprocess.load_data(self.dataset, self.val_split, self.batch_size, False)[4]
        self.model.eval()
        predictions = []
        # Get the class labels
        labels = preprocess.get_labels(self.dataset)
        # TODO: Code here has a pretty poor style. embed.py should be refactored
        #  to be more modular and reusable
        encode_model, _ = clip.embed.create_model_and_transforms()
        label_tokens = clip.embed.tokenizer.tokenize(["This is a photo of " + label for label in labels]) \
            .to(self.device)
        label_embeddings = clip.embed.get_text_embed(label_tokens, encode_model)
        for batch_images, _ in similarity_test_loader:
            # Compute the similarity between the test image and each label
            batch_images = batch_images.to(self.device)
            batch_images = batch_images.reshape(batch_images.shape[0], -1)
            similarity = torch.matmul(batch_images, label_embeddings.T)
            # Get the label with the highest similarity
            _, prediction = similarity.max(1)
            predictions.append(prediction)
        predictions = torch.cat(predictions)
        # Get the ground truth labels
        ground_truth = self.test_dataset[:][1].clone().detach()
        # Compute the accuracy
        num_correct = (predictions == ground_truth).sum()
        num_samples = predictions.size(0)
        print(
            f'Got {num_correct} / {num_samples} '
            f'with accuracy {float(num_correct) / float(num_samples) * 100:.2f}%'
        )

    def save_model(self):
        torch.save(self.model.state_dict(), 'simple_linear_model.pth')
