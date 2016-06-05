package cs517;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by allen on 6/3/2016.
 */
public class RNN {

    MultiLayerNetwork net;
    int batchSize = 32;
    int vectorSize = 300;
    int nEpochs = 5;
    int maxLength = 50;

    public RNN() {
        int vectorSize = 100;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .learningRate(0.0018)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(vectorSize).nOut(vectorSize)
                        .activation("softsign")
                        .build())
                .layer(1, new GravesLSTM.Builder().nIn(vectorSize).nOut(vectorSize)
                        .activation("softsign")
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .nIn(vectorSize)
                        .nOut(8)
                        .build())
                .pretrain(false).backprop(true).build();


        net = new MultiLayerNetwork(conf);

    }



}

