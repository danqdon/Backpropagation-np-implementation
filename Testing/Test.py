

def load_mnist_data():
    train_data = pd.read_csv('mnist_train.csv')
    test_data = pd.read_csv('mnist_test.csv')
    return train_data, test_data

def prepare_data(data):
    data = data.dropna()
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values / 255.0
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1
    return X, y_one_hot

def main():
    train_data, test_data = load_mnist_data()
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    network = NeuralNetwork(
        DenseLayer(784, 128),
        Sigmoid(),
        DenseLayer(128, 64),
        Sigmoid(),
        DenseLayer(64, 10),
        Softmax()
    )

    loss_history = network.train(X_train, y_train, epochs=100, learning_rate=0.001, loss='crossentropy')

    # Graficar la curva de aprendizaje
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

    # Hacer predicciones y evaluar
    predictions = network.predict(X_test)
    predictions_rounded = np.argmax(predictions, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    conf_matrix = confusion_matrix(y_test_labels, predictions_rounded)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_test_labels, predictions_rounded))

if __name__ == "__main__":
    main()
