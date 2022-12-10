import torchvision.transforms as transforms


def transformation(split):
    if split == 1:
        transformation = transforms.CenterCrop(18)
    elif split == 2:
        transformation = transforms.CenterCrop(128)
    elif split == 3:
        transformation = transforms.ColorJitter(brightness=0.50, contrast=0.50, saturation=0.05, hue=0.05)
    elif split == 4:
        transformation = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    elif split == 5:
        transformation = transforms.RandomRotation(degrees=45)
    else:
        raise Exception("Only 5 different transformations are implemented.")
    
    transform = transforms.Compose([
                #transforms.ToPILImage(),
                transformation,
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    return transform
