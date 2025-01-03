from torch.utils.data import DataLoader

# Load the pre-trained ResNet18 model and modify it for feature extraction
resnet18 = models.resnet18(pretrained=True)
resnet18 = torch.nn.Sequential(
    *list(resnet18.children())[:-1]
)  # Remove the final classification layer
resnet18.eval()  # Set to evaluation mode to avoid any training behavior

# Define image transformation pipeline (matching the training setup of ResNet)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize as per ResNet
    ]
)

# DataLoader for the training images
train_loader = DataLoader(image_datasets["train"], batch_size=32, shuffle=False)


def extract_resnet18_features(data_loader):

    features_list = []
    with torch.no_grad():  # Disable gradient calculation for faster inference
        for inputs, _ in data_loader:
            inputs = inputs.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            resnet18.to(inputs.device)
            outputs = resnet18(inputs)
            features_list.append(
                outputs.view(outputs.size(0), -1)
            )  # Flatten the outputs

    features = torch.cat(features_list, dim=0)
    return features.cpu().numpy()


train_features = extract_resnet18_features(train_loader)

test_loader = DataLoader(image_datasets["val"], batch_size=32, shuffle=False)

test_features = extract_resnet18_features(test_loader)
print("Shape of extracted test features:", test_features.shape)  # Should be Mx512
print("Shape of extracted features:", train_features.shape)  # Should be Nx512
