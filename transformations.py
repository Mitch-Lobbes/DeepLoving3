import torchvision.transforms as transforms


def transformation(type):

    # Transformations for question 9
    if type == 9.1:
        transformation = transforms.CenterCrop(18)
    elif type == 9.2:
        transformation = transforms.CenterCrop(128)
    elif type == 9.3:
        transformation = transforms.ColorJitter(brightness=0.50, contrast=0.50, saturation=0.05, hue=0.05)
    elif type == 9.4:
        transformation = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    elif type == 9.5:
        transformation = transforms.RandomRotation(degrees=45)

    # Transformation for question 10
    elif type == 10:
        transformation = transforms.Resize(size=(28,28))
    else:
        raise Exception("Only 6 different transformations are implemented.")
    
    # Setup relevant transformation pipeline
    transform = transforms.Compose([
                #transforms.ToPILImage(),
                transformation,
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    return transform
