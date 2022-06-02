from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained("efficientnet-b0")

# TODO: Model input: fixed and moving image. Model output: optimal varreg parameters. Loss: MSE between fixed and warped moving (using predicted params). Which loss when differnt moddalities? Training: Pairs of images, different modalities. Augmentation: Fixed + moving==fixed image or fixed (CT) + moving==fixed (MRI) image (i.e. no registration necessary).
# TODO: Network input: Stage wise histogram of fixed and moving -> reg hyperparameter, loss = MSE
