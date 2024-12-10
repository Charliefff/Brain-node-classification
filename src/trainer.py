import torch
from torch.utils.tensorboard import SummaryWriter

def train_model(train_loader, val_loader, model, optimizer, criterion , num_epochs=20, device="cpu", log_dir="./runs"):

    writer = SummaryWriter(log_dir)  # 初始化 TensorBoard 寫入器
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for x_batch, y_batch, adj_batch in train_loader:
            x_batch, y_batch, adj_batch = x_batch.to(device), y_batch.to(device), adj_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch, adj_batch)
            outputs = outputs.mean(dim=1)

            loss = criterion(outputs, y_batch)
            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

            loss.backward()
            optimizer.step()

        train_acc = train_correct / train_total

        # 記錄到 TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)

        # 驗證階段
        if val_loader:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0

            with torch.no_grad():
                for x_batch, y_batch, adj_batch in val_loader:
                    x_batch, y_batch, adj_batch = x_batch.to(device), y_batch.to(device), adj_batch.to(device)

                    outputs = model(x_batch, adj_batch)
                    outputs = outputs.mean(dim=1)

                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)

            val_acc = val_correct / val_total

            # 記錄到 TensorBoard
            writer.add_scalar("Loss/Val", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/Val", val_acc, epoch + 1)
            
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} // Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        

    writer.close()  # 關閉 TensorBoard 寫入器

def test_model(test_loader, model, criterion, device="cpu", log_dir="./logs"):

    writer = SummaryWriter(log_dir)
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x_batch, y_batch, adj_batch in test_loader:
            x_batch, y_batch, adj_batch = x_batch.to(device), y_batch.to(device), adj_batch.to(device)

            outputs = model(x_batch, adj_batch)
            outputs = outputs.mean(dim=1)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # 記錄到 TensorBoard
    writer.add_scalar("Loss/Test", avg_loss, 0)
    writer.add_scalar("Accuracy/Test", accuracy, 0)
    writer.close()
    return avg_loss, accuracy
