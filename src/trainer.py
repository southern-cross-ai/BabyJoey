# train.py
import torch
from torchtnt.engine import Engine
from torchtnt.state import State

def train_step(state: State, batch):
    model = state.model
    optimizer = state.optimizer
    loss_fn = state.loss_fn

    model.train()
    data, target = batch
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item()}

def validate(state: State):
    model = state.model
    loss_fn = state.loss_fn
    dataloader = state.val_dataloader

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            data, target = batch
            output = model(data)
            loss = loss_fn(output, target)
            val_loss += loss.item()

    val_loss /= len(dataloader)
    return {"val_loss": val_loss}

def fit(model, optimizer, loss_fn, train_loader, val_loader, num_epochs):
    # Initialize state
    state = State(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        max_epochs=num_epochs
    )

    # Define the training loop
    def train_loop(state: State):
        for epoch in range(state.max_epochs):
            # Training
            for batch in state.dataloader:
                train_output = train_step(state, batch)
                print(f"Epoch {epoch}, Train Loss: {train_output['loss']}")

            # Validation
            val_output = validate(state)
            print(f"Epoch {epoch}, Validation Loss: {val_output['val_loss']}")

    # Create the Engine
    engine = Engine(train_loop)

    # Run the training and validation process
    engine.run(state)
