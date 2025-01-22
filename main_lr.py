import torch
import pandas as pd
import numpy as np
from LR.evaluate_model import evaluate_model
from LR.text_processor import TextProcessor
from LR.train_lr import train_logistic_regression, train_logistic_regression_with_cv
from plot_graph import plot_graph
from sklearn.preprocessing import StandardScaler

data_path = "Dataset/philosophy_data.csv"
edit_data_path = "Dataset/philosophy_data_edit.csv"
model_path = "Models/lr_model.pth"

processor = TextProcessor()


choosen_convert_data = 2 # Choose a conversion method : 1 - Bag-Of-Words, 2 - TF-IDF, 3 - Word2Vec
choosen_training_model = 1 # Choose with or without cross-validation : 0 - without, 1 - with
choosen_regularization = 2 # Choose regularization type or without regularization : 0 - without, 1 - Lasso, 2 - Ridged
choosen_optimization = 3 # Choose optimization : 1 - SGD, 2 - ADAM, 3 - LBFGS


# --------------------------- Convert Data to vectors with Bag-Of-Words ---------------------------
if choosen_convert_data == 1:
    print("[INFO] Read Data .....")
    train_data, test_data, y_train, y_test, classes_names = processor.read_data(edit_data_path)

    print("[INFO] Convert train Data to vectors using Bag-Of-Words.....")
    train_matrix = processor.convert_to_vector(train_data, train_data, y_train, 'bow') # bow - Bag Of Words, tfidf - TFIDF

    num_label_columns = y_train.shape[1]
    x_train = train_matrix[:, num_label_columns:]  # Remaining columns are the bag of bow features
    y_train = train_matrix[:, :num_label_columns]  # First column is the label

    print("[INFO] Convert test Data to vectors using Bag-Of-Words.....")
    test_matrix = processor.convert_to_vector(train_data, test_data, y_test, 'bow')

    num_label_columns = y_test.shape[1]
    x_test = test_matrix[:, num_label_columns:]  # Remaining columns are the bag of bow features
    y_test = test_matrix[:, :num_label_columns]  # First column is the label

# --------------------------- Convert Data to vectors with TF-IDF ---------------------------
if choosen_convert_data == 2:
    print("[INFO] Read Data .....")
    train_data, test_data, y_train, y_test, classes_names = processor.read_data(edit_data_path)

    print("[INFO] Convert train Data to vectors using Bag-Of-Words.....")
    train_matrix = processor.convert_to_vector(train_data, train_data, y_train, 'tfidf') # bow - Bag Of Words, tfidf - TFIDF

    num_label_columns = y_train.shape[1]
    x_train = train_matrix[:, num_label_columns:]  # Remaining columns are the bag of bow features
    y_train = train_matrix[:, :num_label_columns]  # First column is the label

    print("[INFO] Convert test Data to vectors using Bag-Of-Words.....")
    test_matrix = processor.convert_to_vector(train_data, test_data, y_test, 'tfidf')

    num_label_columns = y_test.shape[1]
    x_test = test_matrix[:, num_label_columns:]  # Remaining columns are the bag of bow features
    y_test = test_matrix[:, :num_label_columns]  # First column is the label


# --------------------------- Convert Data to vectors with Word2Vec ---------------------------
if choosen_convert_data == 3:
    print("[INFO] Read Data .....")
    train_data, test_data, y_train, y_test, classes_names = processor.read_data(data_path)

    print("[INFO] Training Word2Vec model .....")
    word2vec_model = processor.train_word2vec(
        train_data['sentence_str'].tolist(),
        vector_size=200,  
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
 
# --------------------------------- Train model without cross validation ---------------------------------
if choosen_training_model == 0:

    #    ----------------- regularization - None, optimizer - SGD ----------------- 
    if choosen_regularization == 0 and choosen_optimization == 1:
        print("[INFO] Training the model : regularization - None, optimizer - SGD...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=150, lr=0.01, regularization=None, reg_lambda=0.001, optimize='sgd'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Lasso, optimizer - SGD ----------------- 
    if choosen_regularization == 1 and choosen_optimization == 1:
        print("[INFO] Training the model : regularization - Lasso, optimizer - SGD...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=150, lr=0.01, regularization="l1", reg_lambda=0.001, optimize='sgd'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Ridged, optimizer - SGD ----------------- 
    if choosen_regularization == 2 and choosen_optimization == 1:
        print("[INFO] Training the model : regularization - Ridged, optimizer - SGD...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=150, lr=0.01, regularization="l2", reg_lambda=0.001, optimize='sgd'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - None, optimizer - ADAM ----------------- 
    if choosen_regularization == 0 and choosen_optimization == 2:
        print("[INFO] Training the model : regularization - None, optimizer - ADAM...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=150, lr=0.01, regularization=None, reg_lambda=0.001, optimize='adam'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Lasso, optimizer - ADAM ----------------- 
    if choosen_regularization == 1 and choosen_optimization == 2:
        print("[INFO] Training the model : regularization - Lasso, optimizer - ADAM...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=60, lr=0.01, regularization="l1", reg_lambda=0.001, optimize='adam'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Ridged, optimizer - ADAM ----------------- 
    if choosen_regularization == 2 and choosen_optimization == 2:
        print("[INFO] Training the model : regularization - Ridged, optimizer - ADAM...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=60, lr=0.01, regularization="l2", reg_lambda=0.001, optimize='adam'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - None, optimizer - LBFGS ----------------- 
    if choosen_regularization == 0 and choosen_optimization == 3:
        print("[INFO] Training the model : regularization - None, optimizer - LBFGS...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=30, lr=0.01, regularization=None, reg_lambda=0.001, optimize='lbfgs'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Lasso, optimizer - LBFGS ----------------- 
    if choosen_regularization == 1 and choosen_optimization == 3:
        print("[INFO] Training the model : regularization - Lasso, optimizer - LBFGS...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=30, lr=0.01, regularization="l1", reg_lambda=0.001, optimize='lbfgs'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Ridged, optimizer - LBFGS ----------------- 
    if choosen_regularization == 2 and choosen_optimization == 3:
        print("[INFO] Training the model : regularization - Ridged, optimizer - LBFGS...")
        model, train_losses, val_losses, train_accuracies, val_accuracies, epochs = train_logistic_regression(
            x_train, y_train, model_path, epochs=30, lr=0.01, regularization="l2", reg_lambda=0.001, optimize='lbfgs'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)

    plot_graph(train_losses, val_losses, train_accuracies, val_accuracies, epochs)    

# --------------------------------- Train model with cross validation ---------------------------------
if choosen_training_model == 1:
    
    #    ----------------- regularization - None, optimizer - SGD ----------------- 
    if choosen_regularization == 0 and choosen_optimization == 1:
        print("[INFO] Training the model : regularization - None, optimizer - SGD...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=150, lr=0.01, regularization=None, reg_lambda=0.001, optimize='sgd'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Lasso, optimizer - SGD ----------------- 
    if choosen_regularization == 1 and choosen_optimization == 1:
        print("[INFO] Training the model : regularization - Lasso, optimizer - SGD...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=150, lr=0.01, regularization="l1", reg_lambda=0.001, optimize='sgd'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Ridged, optimizer - SGD ----------------- 
    if choosen_regularization == 2 and choosen_optimization == 1:
        print("[INFO] Training the model : regularization - Ridged, optimizer - SGD...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=150, lr=0.01, regularization="l2", reg_lambda=0.001, optimize='sgd'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - None, optimizer - ADAM ----------------- 
    if choosen_regularization == 0 and choosen_optimization == 2:
        print("[INFO] Training the model : regularization - None, optimizer - ADAM...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=150, lr=0.01, regularization=None, reg_lambda=0.001, optimize='adam'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Lasso, optimizer - ADAM ----------------- 
    if choosen_regularization == 1 and choosen_optimization == 2:
        print("[INFO] Training the model : regularization - Lasso, optimizer - ADAM...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=60, lr=0.01, regularization="l1", reg_lambda=0.001, optimize='adam'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Ridged, optimizer - ADAM ----------------- 
    if choosen_regularization == 2 and choosen_optimization == 2:
        print("[INFO] Training the model : regularization - Ridged, optimizer - ADAM...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=60, lr=0.01, regularization="l2", reg_lambda=0.001, optimize='adam'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - None, optimizer - LBFGS ----------------- 
    if choosen_regularization == 0 and choosen_optimization == 3:
        print("[INFO] Training the model : regularization - None, optimizer - LBFGS...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=30, lr=0.01, regularization=None, reg_lambda=0.001, optimize='lbfgs'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Lasso, optimizer - LBFGS ----------------- 
    if choosen_regularization == 1 and choosen_optimization == 3:
        print("[INFO] Training the model : regularization - Lasso, optimizer - LBFGS...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=30, lr=0.1, regularization="l1", reg_lambda=0.001, optimize='lbfgs'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)


    #    ----------------- regularization - Ridged, optimizer - LBFGS ----------------- 
    if choosen_regularization == 2 and choosen_optimization == 3:
        print("[INFO] Training the model : regularization - Ridged, optimizer - LBFGS...")
        model = train_logistic_regression_with_cv(
            x_train, y_train, model_path, epochs=30, lr=0.01, regularization="l2", reg_lambda=0.001, optimize='lbfgs'
        )
        print("[INFO] Evaluate model...")
        precision, recall, f1 = evaluate_model(model, x_test, y_test, classes_names)



# ---------------------------------- Manual Testing ----------------------------------

print("[INFO] Test model classification with sample (text)...")

model.eval()

text_class1 = "When things have only a name in common and the definition of being which corresponds to the name is different, they are called homonymous."
text_class2 = "Why this Critique is titled a critique not of pure practical reason but simply of practical reason as such, although its parallelism with the critique of speculative reas seems to require the former on this the treatise provides sufficient information."
text_class3 = "What's new, Socrates, to make you leave your usual haunts in the Lyceum and spend your time here by the king archon's court?"
text_class4 = "For us, the human body defines, by natural right, the space of origin and of distribution of disease: a space whose lines, volumes, surfaces, and routes are laid down, in accordance with a now familiar geometry, by the anatomical atlas."

if choosen_convert_data == 1:

    class_1 = pd.DataFrame({
    'sentence_str': [text_class1]
    })
    class_2 = pd.DataFrame({
    'sentence_str': [text_class2]
    })
    class_3 = pd.DataFrame({
    'sentence_str': [text_class3]
    })
    class_4 = pd.DataFrame({
    'sentence_str': [text_class4]
    })

    label_1 = np.array([[1, 0, 0, 0]])
    label_2 = np.array([[0, 1, 0, 0]])
    label_3 = np.array([[0, 0, 1, 0]])
    label_4 = np.array([[0, 0, 0, 1]])


    label_1 = label_1.reshape(1, 4)
    label_2 = label_2.reshape(1, 4)
    label_3 = label_3.reshape(1, 4)
    label_4 = label_4.reshape(1, 4)


    text1_embedding = processor.convert_to_vector(train_data, class_1, label_1, vectorizer='bow')
    num_label_columns = label_1.shape[1]
    text1_embedding = text1_embedding[:, num_label_columns:]
    text1_embedding = torch.tensor(text1_embedding, dtype=torch.float32)

    text2_embedding = processor.convert_to_vector(train_data, class_2, label_2, vectorizer='bow')
    num_label_columns = label_2.shape[1]
    text2_embedding = text2_embedding[:, num_label_columns:]
    text2_embedding = torch.tensor(text2_embedding, dtype=torch.float32)

    text3_embedding = processor.convert_to_vector(train_data, class_3, label_3, vectorizer='bow')
    num_label_columns = label_3.shape[1]
    text3_embedding = text3_embedding[:, num_label_columns:]
    text3_embedding = torch.tensor(text3_embedding, dtype=torch.float32)    

    text4_embedding = processor.convert_to_vector(train_data, class_4, label_4, vectorizer='bow')
    num_label_columns = label_4.shape[1]
    text4_embedding = text4_embedding[:, num_label_columns:]
    text4_embedding = torch.tensor(text4_embedding, dtype=torch.float32)        


if choosen_convert_data == 2:

    class_1 = pd.DataFrame({
    'sentence_str': [text_class1]
    })
    class_2 = pd.DataFrame({
    'sentence_str': [text_class2]
    })
    class_3 = pd.DataFrame({
    'sentence_str': [text_class3]
    })
    class_4 = pd.DataFrame({
    'sentence_str': [text_class4]
    })


    label_1 = np.array([[1, 0, 0, 0]])
    label_2 = np.array([[0, 1, 0, 0]])
    label_3 = np.array([[0, 0, 1, 0]])
    label_4 = np.array([[0, 0, 0, 1]])

    label_1 = label_1.reshape(1, 4)
    label_2 = label_2.reshape(1, 4)
    label_3 = label_3.reshape(1, 4)
    label_4 = label_4.reshape(1, 4)

    text1_embedding = processor.convert_to_vector(train_data, class_1, label_1, vectorizer='tfidf')
    num_label_columns = label_1.shape[1]
    text1_embedding = text1_embedding[:, num_label_columns:]
    text1_embedding = torch.tensor(text1_embedding, dtype=torch.float32)

    text2_embedding = processor.convert_to_vector(train_data, class_2, label_2, vectorizer='tfidf')
    num_label_columns = label_2.shape[1]
    text2_embedding = text2_embedding[:, num_label_columns:]
    text2_embedding = torch.tensor(text2_embedding, dtype=torch.float32)

    text3_embedding = processor.convert_to_vector(train_data, class_3, label_3, vectorizer='tfidf')
    num_label_columns = label_3.shape[1]
    text3_embedding = text3_embedding[:, num_label_columns:]
    text3_embedding = torch.tensor(text3_embedding, dtype=torch.float32)    

    text4_embedding = processor.convert_to_vector(train_data, class_4, label_4, vectorizer='tfidf')
    num_label_columns = label_4.shape[1]
    text4_embedding = text4_embedding[:, num_label_columns:]
    text4_embedding = torch.tensor(text4_embedding, dtype=torch.float32)        


if choosen_convert_data == 3:
    text1_embedding = processor.convert_to_word2vec(word2vec_model, [text_class1])
    text1_embedding = torch.tensor(text1_embedding, dtype=torch.float32)

    text2_embedding = processor.convert_to_word2vec(word2vec_model, [text_class2])
    text2_embedding = torch.tensor(text2_embedding, dtype=torch.float32)

    text3_embedding = processor.convert_to_word2vec(word2vec_model, [text_class3])
    text3_embedding = torch.tensor(text3_embedding, dtype=torch.float32)

    text4_embedding = processor.convert_to_word2vec(word2vec_model, [text_class4])
    text4_embedding = torch.tensor(text4_embedding, dtype=torch.float32)

with torch.no_grad():

    y_pred1 = model(text1_embedding)
    y_pred2 = model(text2_embedding)
    y_pred3 = model(text3_embedding)  
    y_pred4 = model(text4_embedding)

    predicted_class1 = torch.argmax(y_pred1, dim=1).item()  
    predicted_class2 = torch.argmax(y_pred2, dim=1).item()  
    predicted_class3 = torch.argmax(y_pred3, dim=1).item()  
    predicted_class4 = torch.argmax(y_pred4, dim=1).item()  


print(f'Text_1: {text_class1}\n'
      f'True answer : {classes_names[0]}\n'
      f'predicted answer : {classes_names[predicted_class1]}\n')

print(f'\nText_2: {text_class2}\n'
      f'True answer : {classes_names[1]}\n'
      f'predicted answer : {classes_names[predicted_class2]}\n')

print(f'\nText_3: {text_class3}\n'
      f'True answer : {classes_names[2]}\n'
      f'predicted answer : {classes_names[predicted_class3]}')

print(f'\nText_4: {text_class4}\n'
      f'True answer : {classes_names[3]}\n'
      f'predicted answer : {classes_names[predicted_class4]}')

