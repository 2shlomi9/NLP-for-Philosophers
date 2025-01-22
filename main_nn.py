import torch
from NN.text_processor import TextProcessor
from NN.model import FullyConnectedNN
from NN.train_model import train_fnn
from NN.evaluate_model import evaluate_model
from sklearn.preprocessing import StandardScaler

# Paths and configurations
data_path = "Dataset/philosophy_data.csv"  # Path to your dataset
model_save_path = "Models/fnn_model.pth"         # Path to save the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_dim = 100         # Adjust based on vectorization output
hidden_dim = 512        # Number of neurons in the hidden layer
num_classes = 4         # Adjust based on your dataset
dropout_rate = 0.3      # Dropout rate for regularization
epochs = 200            # Number of training epochs
learning_rate = 0.0001  # Learning rate for optimizer
batch_size = 32         # Batch size for training
class_weights = []

# Step 1: Data Preprocessing
print("[INFO] Processing data...")
processor = TextProcessor()

# Choose vectorization method (1 - BoW, 2 - TF-IDF, 3 - Word2Vec)
vectorizer_choice = 3  # Adjust based on your preference

if vectorizer_choice == 1:
    train_data, test_data, y_train, y_test, classes_names = processor.read_data(data_path)

    train_vectors = processor.convert_to_vector(train_data, train_data, y_train, vectorizer='bow')
    test_vectors = processor.convert_to_vector(train_data, test_data, y_test, vectorizer='bow')
    # Normalize the input data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_vectors)
    x_test = scaler.fit_transform(test_vectors)

elif vectorizer_choice == 2:
    train_data, test_data, y_train, y_test, classes_names = processor.read_data(data_path)
    train_vectors = processor.convert_to_vector(train_data, train_data, y_train, vectorizer='tfidf')
    test_vectors = processor.convert_to_vector(train_data, test_data, y_test, vectorizer='tfidf')
    # Normalize the input data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_vectors)
    x_test = scaler.fit_transform(test_vectors)
    
elif vectorizer_choice == 3:
    print("[INFO] Read Data .....")
    train_data, test_data, y_train, y_test, classes_names, classes_count = processor.read_data(data_path)
    class_count = [count for count in classes_count.values]
    class_weights = [sum(class_count) / c for c in class_count]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("[INFO] Training Word2Vec model .....")
    word2vec_model = processor.train_word2vec(
        train_data['sentence_str'].tolist(),
        vector_size=input_dim,
        window=10,
        min_count=2,
    )
    print("[INFO] Convert train data to Word2Vec embeddings .....")
    train_embeddings = processor.convert_to_word2vec(word2vec_model, train_data['sentence_str'].tolist())
    x_train = train_embeddings

    # Normalize the input data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    print(f"Shape of train embeddings: {x_train.shape}")


    print("[INFO] Convert test data to Word2Vec embeddings .....")
    test_embeddings = processor.convert_to_word2vec(word2vec_model, test_data['sentence_str'].tolist())
    x_test = test_embeddings

    x_test = scaler.fit_transform(x_test)

    print(f"Shape of test embeddings: {x_test.shape}")
else:
    raise ValueError("Invalid vectorizer choice. Please select 1, 2, or 3.")

# Prepare data for training
y_train = y_train.argmax(axis=1)  # Convert one-hot labels to indices
y_test = y_test.argmax(axis=1)    # Convert one-hot labels to indices

# Step 2: Define the Model
print("[INFO] Defining the neural network model...")
model = FullyConnectedNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout_rate=dropout_rate)
model.to(device)

# Step 3: Train the Model
print("[INFO] Training the model...")
model = train_fnn(x_train, y_train, model, class_weights, epochs=epochs, lr=learning_rate, batch_size=batch_size, device=device)

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"[INFO] Model saved to {model_save_path}")

# Step 4: Evaluate the Model
print("[INFO] Evaluating the model...")
accuracy = evaluate_model(model, x_test, y_test, classes_names, device=device)

print(f"[INFO] Final Test Accuracy: {accuracy * 100:.2f}%")
