import torch
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from visual_weight import plot_attention_heatmap



def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_outputs = []
    all_attens = []  

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, atten = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
            all_attens.append(atten.detach().cpu())

    avg_atten = torch.mean(torch.stack(all_attens), dim=0)

    avg_loss = total_loss / len(loader.dataset)
    auc_score = roc_auc_score(all_labels, all_outputs)
    return avg_loss, auc_score, avg_atten


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)  
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_outputs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())


    avg_loss = running_loss / len(loader.dataset)
    auc_score = roc_auc_score(all_labels, all_outputs)
    return avg_loss, auc_score


def train_cnn(train_loader, val_loader, test_loader, model, optimizer, criterion, num_epochs=20, device="cuda", log_dir="runs/cnn_experiment", save_path="checkpoint/best_model.pth", enable_atten=True):
    model.to(device)
    writer = SummaryWriter(log_dir)

    best_val_auc = 0.0  
    best_epoch = -1

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
        val_loss, val_auc, atten = evaluate_model(model, val_loader, criterion, device)
        if enable_atten:
            plot_attention_heatmap(atten, epoch, title="Attention Heatmap", save_path="./attention_pic/heatmap")
            
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("AUC/Train", train_auc, epoch)
        writer.add_scalar("AUC/Validation", val_auc, epoch)

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] : "
            f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} // "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path) 
            print(f"Best model saved at epoch {epoch + 1} with Validation AUC: {val_auc:.4f}")

    print(f"Training complete. Best Validation AUC: {best_val_auc:.4f} at epoch {best_epoch + 1}")

    model.load_state_dict(torch.load(save_path))  
    test_loss, test_auc, _ = evaluate_model(model, test_loader, criterion, device)
    writer.add_scalar("AUC/Test", test_auc, epoch)
    writer.add_scalar("Loss/Test", test_loss, epoch)
    print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
    writer.close()

def test_cnn(test_loader, model, criterion, device="cuda"):
    model.to(device)
    test_loss, test_auc, _ = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
    return test_loss, test_auc

