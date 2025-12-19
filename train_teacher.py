import argparse
import torch



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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="teacher_resnet18_cifar10.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainloader, testloader = get_loaders(args.batch_size)

    model = TeacherResNet18(num_classes=10).to(device)
    print("Teacher params:", count_parameters(model))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

        train_loss, train_acc = train_one_epoch_supervised(model, trainloader, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| test loss {test_loss:.4f} acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict()}, args.out)

    print("Best teacher test acc:", best_acc)
    print("Saved to:", args.out)

if __name__ == "__main__":
    main()
