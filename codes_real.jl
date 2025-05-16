

function get_mnist()
    url_train = "https://www.openml.org/data/get_csv/52667/mnist_784.arff"
    file_train = "data/mnist_train.csv"

    Downloads.download(url_train, file_train)

    data = readdlm(file_train, ',', header=true)

    X = Float32.(data[1][:, 1:end-1])
    y = Int.(data[1][:, end]) .+ 1
    X_train = X[1:60000,:];
    y_train = y[1:60000,:];
    X_test = X[60001:end,:];
    y_test = y[60001:end,:];
    return X, y, X_train, y_train, X_test, y_test
end

