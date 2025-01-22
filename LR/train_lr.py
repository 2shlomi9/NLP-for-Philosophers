import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Fully connected layer for multi-class classification

    def forward(self, x):
        return self.fc(x)  # CrossEntropyLoss includes it internally


# Helper function for calculating accuracy
def calculate_accuracy(y_pred, y_true):
    """Calculate accuracy by comparing predicted and true labels."""
    y_pred_classes = torch.argmax(y_pred, dim=1)
    accuracy = (y_pred_classes == y_true).sum().item() / len(y_true)
    return accuracy


# --------------------------------- Logistic Regression without cross validation ---------------------------------
def train_logistic_regression(
    x_data,
    y_data,
    model_path,
    epochs=500,
    lr=0.01,
    validation_split=0.2,
    regularization=None,
    reg_lambda=0.01,
    optimize = 'sgd'
):
    """
    Train a Logistic Regression model with optional regularization and a learning rate scheduler.

    Parameters:
        x_data (numpy.ndarray): Input features.
        y_data (numpy.ndarray): Labels (as integers for class indices).
        model_path (str): Path where the trained model will be saved.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        validation_split (float): Proportion of data to be used as validation set.
        regularization (str): Type of regularization ('l1' or 'l2'). Default is None.
        reg_lambda (float): Regularization strength. Default is 0.01.

    Returns:
        model (LogisticRegressionModel): The trained Logistic Regression model.
    """

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=validation_split, random_state=42
    )

    # Convert data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)  # Convert One-Hot to indices
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)

    # Initialize the model
    input_dim = x_train.shape[1]
    num_classes = len(torch.unique(y_train))
    model = LogisticRegressionModel(input_dim, num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimize == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)  # Use SGD optimizer 
    if optimize == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Use Adam optimizer 
    if optimize == 'lbfgs':
        optimizer = optim.LBFGS(model.parameters(), lr=lr)  # Use LBFGS optimizer 

    # Learning rate scheduler to reduce learning rate on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    train_final_accuary = 0

    validation_final_accuary = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()

        if optimize == 'lbfgs':
            y_pred = None  # Initialize y_pred outside the closure
            def closure():
                nonlocal y_pred  # Access y_pred in the outer scope
                optimizer.zero_grad()
                y_pred = model(x_train)  # Update y_pred
                loss = criterion(y_pred, y_train)

                # Add regularization if specified
                if regularization == "l1":
                    l1_norm = sum(param.abs().sum() for param in model.parameters())
                    loss += reg_lambda * l1_norm
                elif regularization == "l2":
                    l2_norm = sum(param.pow(2).sum() for param in model.parameters())
                    loss += reg_lambda * l2_norm

                loss.backward()
                return loss

            optimizer.step(closure)  # Use the closure for LBFGS
            loss = closure()  # Calculate the final loss
        else:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            # Add regularization if specified
            if regularization == "l1":
                l1_norm = sum(param.abs().sum() for param in model.parameters())
                loss += reg_lambda * l1_norm
            elif regularization == "l2":
                l2_norm = sum(param.pow(2).sum() for param in model.parameters())
                loss += reg_lambda * l2_norm

            loss.backward()
            optimizer.step()

        # Calculate train accuracy
        train_accuracy = calculate_accuracy(y_pred, y_train)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val).item()
            val_accuracy = calculate_accuracy(val_pred, y_val)

        # Adjust learning rate if validation loss plateaus
        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print training progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"
            )
        train_final_accuary = train_accuracy * 100
        validation_final_accuary = val_accuracy * 100
    print(
                f"Train Accuracy: {train_final_accuary:.2f}%, "
                f"Val Accuracy: {validation_final_accuary:.2f}%"
            )
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies, epochs 


# --------------------------------- Logistic Regression with cross validation ---------------------------------

def train_logistic_regression_with_cv(
    x_data,
    y_data,
    model_path,
    epochs=500,
    lr=0.01,
    k=5,
    regularization=None,
    reg_lambda=0.01,
    optimize='sgd',
):
    """
    Train a Logistic Regression model using K-Fold Cross Validation.

    Parameters:
        x_data (numpy.ndarray): Input features.
        y_data (numpy.ndarray): Labels (as integers for class indices or one-hot encoded).
        model_path (str): Path where the trained model will be saved.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        k (int): Number of folds for cross-validation.
        regularization (str): Type of regularization ('l1' or 'l2'). Default is None.
        reg_lambda (float): Regularization strength. Default is 0.01.
        optimize (str): Optimizer to use ('sgd', 'adam', 'lbfgs'). Default is 'sgd'.

    Returns:
        float: Average validation accuracy across all folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_accuracies = []

    fold = 1
    for train_index, val_index in kf.split(x_data):
        print(f"[INFO] Fold {fold}/{k}")
        
        # Split data into training and validation sets for this fold
        x_train, x_val = x_data[train_index], x_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        # If y_data is one-hot encoded, convert to indices
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
            y_val = np.argmax(y_val, axis=1)

        # Check that the labels are within the correct range
        num_classes = len(np.unique(y_train))  # Determine the number of classes based on training data
        assert np.all(y_train < num_classes), f"Some labels in y_train are out of bounds. Max label is {np.max(y_train)}"
        assert np.all(y_val < num_classes), f"Some labels in y_val are out of bounds. Max label is {np.max(y_val)}"

        # Convert data to PyTorch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        # Initialize the model
        input_dim = x_train.shape[1]
        model = LogisticRegressionModel(input_dim, num_classes)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        if optimize == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimize == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimize == 'lbfgs':
            optimizer = optim.LBFGS(model.parameters(), lr=lr)

        # Scheduler to adjust learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

        for epoch in range(epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            if optimize == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    y_pred = model(x_train)
                    loss = criterion(y_pred, y_train)
                    if regularization == 'l1':
                        l1_norm = sum(param.abs().sum() for param in model.parameters())
                        loss += reg_lambda * l1_norm
                    elif regularization == 'l2':
                        l2_norm = sum(param.pow(2).sum() for param in model.parameters())
                        loss += reg_lambda * l2_norm
                    loss.backward()
                    return loss
                optimizer.step(closure)
                # Calculate loss after LBFGS step
                with torch.no_grad():
                    y_pred = model(x_train)
                    loss = criterion(y_pred, y_train)
                    if regularization == 'l1':
                        l1_norm = sum(param.abs().sum() for param in model.parameters())
                        loss += reg_lambda * l1_norm
                    elif regularization == 'l2':
                        l2_norm = sum(param.pow(2).sum() for param in model.parameters())
                        loss += reg_lambda * l2_norm
            else:
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                if regularization == 'l1':
                    l1_norm = sum(param.abs().sum() for param in model.parameters())
                    loss += reg_lambda * l1_norm
                elif regularization == 'l2':
                    l2_norm = sum(param.pow(2).sum() for param in model.parameters())
                    loss += reg_lambda * l2_norm
                loss.backward()
                optimizer.step()

            train_accuracy = calculate_accuracy(y_pred, y_train)
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = criterion(val_pred, y_val).item()
                val_accuracy = calculate_accuracy(val_pred, y_val)

            # Adjust learning rate
            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"
                )
        train_final_accuary = train_accuracy * 100
        val_accuracies.append(val_accuracy)
        fold += 1

    # Calculate average validation accuracy
    avg_val_accuracy = np.mean(val_accuracies)*100
    print(
            f"Train Accuracy: {train_final_accuary:.2f}%, "
            f"Average Validation Accuracy: {avg_val_accuracy:.2f}%"
        )
    
    # Save the final model
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}")

    return model

