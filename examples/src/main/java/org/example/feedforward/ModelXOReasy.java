package org.example.feedforward;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static java.lang.System.exit;

public class ModelXOReasy
{
    public static void main(String[] args) throws InterruptedException {

        INDArray input = Nd4j.create(new float[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
        });

        INDArray expectedOutput = Nd4j.create(new float[][]{
                {0},
                {1},
                {1},
                {0}
        });


        DataSet dataSet = new DataSet(input, expectedOutput);
        System.out.println(dataSet);

        // network definition
        var cfg = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.UNIFORM)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .nIn(2)
                        .nOut(3)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation((Activation.SIGMOID))
                        .nIn(3)
                        .nOut(1)
                        .build())
                .build();
        var network = new MultiLayerNetwork(cfg);
        network.init();
        network.setLearningRate(0.7);
//        var uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//        network.setListeners(new StatsListener(statsStorage));
////        network.setListeners(new ScoreIterationListener(30));


        // train a model
        for (int i = 0; i < 3000; i++) {
            network.fit(dataSet);
        }

        // test the model
        INDArray output = network.output(input);
        var evaluation = new Evaluation();
        evaluation.eval(expectedOutput, output);
        System.out.println(evaluation.stats());
    }
}
