import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
TEMPERATURE = 4.0
ALPHA = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


teacher = models.resnet18(pretrained=False)
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
teacher.to(DEVICE)

student = StudentNet().to(DEVICE)


for param in teacher.parameters():
    param.requires_grad = False


ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student.parameters(), lr=LR)

def distillation_loss(student_logits, teacher_logits, labels):
    soft_targets = F.softmax(teacher_logits / TEMPERATURE, dim=1)
    soft_preds = F.log_softmax(student_logits / TEMPERATURE, dim=1)

    kd_loss = F.kl_div(
        soft_preds, soft_targets, reduction="batchmean"
    ) * (TEMPERATURE ** 2)

    hard_loss = ce_loss(student_logits, labels)

    return ALPHA * kd_loss + (1 - ALPHA) * hard_loss


def train():
    student.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_outputs = teacher(images.repeat(1, 3, 1, 1))

            student_outputs = student(images)

            loss = distillation_loss(
                student_outputs, teacher_outputs, labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")


def evaluate():
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = student(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
    evaluate()
