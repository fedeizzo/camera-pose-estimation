def train(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for index, (x, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            predictions = model(x)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}: loss={loss}")
