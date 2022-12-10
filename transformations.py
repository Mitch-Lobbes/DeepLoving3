import torchvision.transforms as transforms

class Transformations:
    CenterCropSmall = transforms.Compose([transforms.CenterCrop(18), transforms.ToTensor()])
    CenterCropLarge = transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor()])  

    ColorJitter = transforms.Compose([
        transforms.ColorJitter(brightness=0.50, contrast=0.50, saturation=0.05, hue=0.05), 
        transforms.ToTensor()])    

    GaussianBlur = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(5, 9), 
        sigma=(0.1, 5)), transforms.ToTensor()])

    RandomRotation = transforms.Compose([transforms.RandomRotation(degrees=45), transforms.ToTensor()])

    Resize = transforms.Compose([
        transforms.Resize(size=(28,28)), 
        transforms.Grayscale(), 
        transforms.ToTensor()])
        
    Grayscale = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
