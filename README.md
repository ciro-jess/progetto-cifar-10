def build_mlp():
    model =keras.Sequential(
        [
            layers.Input(shape=(32,32,3),name="immagine_input"), //specifica che ogni sigolo esempio ha :altezza0 32,laghezza=32,canali= 3

            layers.Flatten(name="Flatten"),  // trasforma gli input che abbiamo impostato (32,32,3) in 3072
            layers.Dense(512,activation="relu",name="dance_512"), // riceve l'input e produce 512 attivatori , poi applica ReLU (ogni neurone possiede :3072pesi,1 bias)
            layers.Dropout(0.40,name="dropout"), // duerante il training annulla casualmente il 40% delle attivazioni ricevute in ingresso 
            layers.Dense(256,activation="relu"),
            layers.Dropout(0.30,name="dropout")
            layers.Dense(10,activation="softmax",name="output")
        ],
        name="mlp_cifar10"
    )

    return model

mlp= build_mlp()