import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import optim
from tqdm import tqdm

from models import TeacherResNet18, StudentSmallCNN
from utils import evaluate, count_parameters

def get_loaders(batch_size):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def kd_loss(student_logits, teacher_logits, targets, alpha, T):
    """
    L = (1-alpha)*CE(y, s) + alpha*T^2*KL( softmax(t/T) || softmax(s/T) )
    """
    ce = F.cross_entropy(student_logits, targets)

    # KL divergence expects log-probs for input and probs for target
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)

    kl = F.kl_div(p_s, p_t, reduction="batchmean")
    return (1 - alpha) * ce + alpha * (T * T) * kl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=str, default="teacher_resnet18_cifar10.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--out", type=str, default="student_kd.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainloader, testloader = get_loaders(args.batch_size)

    teacher = TeacherResNet18(num_classes=10).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = StudentSmallCNN(num_classes=10).to(device)

    print("Teacher params:", count_parameters(teacher))
    print("Student params:", count_parameters(student))

    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in tqdm(trainloader, desc=f"KD train ep {epoch}", leave=False):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                t_logits = teacher(x)

            s_logits = student(x)
            loss = kd_loss(s_logits, t_logits, y, alpha=args.alpha, T=args.temp)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        test_loss, test_acc = evaluate(student, testloader, device)

        print(f"Epoch {epoch:02d} | kd-train loss {train_loss:.4f} "
              f"| test loss {test_loss:.4f} acc {test_acc:.4f} "
              f"(alpha={args.alpha}, T={args.temp})")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": student.state_dict(),
                        "alpha": args.alpha,
                        "temp": args.temp}, args.out)

    print("Best KD student test acc:", best_acc)
    print("Saved to:", args.out)

if __name__ == "__main__":
    main()
