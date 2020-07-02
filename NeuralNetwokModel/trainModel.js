const tf = require("@tensorflow/tfjs");
const w = require("@tensorflow/tfjs-node");
const data = require("../../trainingData.json");

const getGameData = function () {
    return data.game;
};

function getOutputData() {
    const outputData = tf.tensor2d(
        data.direction.map((item) => {
            var arr = [0, 0, 0, 0];
            // 0 Up, 1 Down, 2 Left, 3 Right
            arr[item] = 1;
            return arr;
        }),
        [data.direction.length, 4]
    );
    return outputData;
}

var gameDataFormat = getGameData();
var outputData = getOutputData();

function applyML() {
    var start = Date.now();
    const trainingData = tf.tensor2d(gameDataFormat, [
        gameDataFormat.length,
        gameDataFormat[0].length,
    ]);

    // build neural network
    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [gameDataFormat[0].length],
            activation: "relu",
            units: 20,
        })
    );
    model.add(
        tf.layers.dense({
            inputShape: [20],
            activation: "sigmoid",
            units: 4,
        })
    );

    console.log("Compiling");
    model.compile({
        loss: "meanSquaredError",
        optimizer: tf.train.sgd(0.8),
    });

    console.log("Fitting");
    const trainedModel = model.fit(trainingData, outputData, { epochs: 100 });
    trainedModel.then(async (history) => {
        console.log("Fitted");
        console.log(history);
        var prediction = model.predict(trainingData);
        prediction.print();
        console.log("Evaluating");
        var result = model.evaluate(trainingData, outputData);
        result.print();

        // Save model
        await model
            .save("file://./savedModel")
            .then((res) => console.log("After Saving", res))
            .catch((err) => console.log(err));

        var end = Date.now();
        var timee = Math.floor(Math.round((end - start) / 1000));

        console.log("Time Taken: ", timee / 60, " min ", timee % 60, " sec");
    });
}

console.log("Starting");
applyML();
