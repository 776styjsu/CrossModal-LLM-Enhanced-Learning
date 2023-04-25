from train import Trainer


def main():
    model = 'softmax_regression'
    # dataset = 'mnist'
    dataset = 'cifar10'
    mode = 'clip'
    # mode = 'standard'
    check_similarity = False

    trainer = Trainer(mode=mode, dataset=dataset, load_embedding=True,
                      model=model, learning_rate=1e-2, batch_size=64, num_epochs=1)
    print("Trainer set up complete")
    print("Starting training loop...")
    trainer.train()

    print("Checking accuracy on training data:")
    trainer.check_accuracy(trainer.train_loader, False)

    print("Checking accuracy on validation data:")
    trainer.check_accuracy(trainer.val_loader, False)

    print("Checking accuracy on test data:")
    trainer.check_accuracy(trainer.test_loader, True)

    if mode == 'clip' and check_similarity:
        print("Checking similarity between image and text:")
        trainer.check_similarity_and_predict()


if __name__ == '__main__':
    main()