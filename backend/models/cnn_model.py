import torch
import torch.nn as nn
import torchvision.models as models

class BananaCNNModel(nn.Module):
    """
    CNN-based banana ripeness classifier using MobileNetV2.
    Matches the architecture used during training.
    """
    def __init__(self, num_classes=4):
        super(BananaCNNModel, self).__init__()
        # Initialize MobileNetV2 with default weights as base
        self.model = models.mobilenet_v2(weights=None)
        
        # Modify the classifier to match the 4-class output
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
        self.class_mapping = {
            0: 'unripe',
            1: 'ripe',
            2: 'overripe',
            3: 'dispose'
        }

    def forward(self, x):
        """
        Input: (B, 3, 224, 224) or (B, 3, 300, 300) Tensor
        Output: Logits (B, 4)
        """
        return self.model(x)

    def predict(self, x_tensor):
        """
        Predict high-level wrapper
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            status = self.class_mapping.get(predicted.item(), 'unknown')
            return status, confidence.item()
