package cs517;

import cs517.data.DataSetManager;
import cs517.data.MultiClassIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

/**
 * Created by allen on 6/3/2016.
 */
public class RNN {

    private MultiLayerNetwork model;
    private int batchSize = 32;
    private int vectorSize = 100;
    private int nEpochs = 5;
    private int maxLength = 100;

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
                        .nIn(vectorSize).nOut(200)
                        .activation("softsign")
                        .build())
                .layer(1, new GravesLSTM.Builder().nIn(200).nOut(200)
                        .activation("softsign")
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .nIn(200)
                        .nOut(8)
                        .build())
                .pretrain(false).backprop(true).build();


        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new HistogramIterationListener(1));


        try {
            WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File("sentimentWordVectors.txt"));
        } catch (FileNotFoundException e) {
            System.err.println("Can't find word vector file.");
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            System.err.println("shouldn't occur");
            e.printStackTrace();
        }
        DataSetManager dm = new DataSetManager();
        try {
//            DataSetIterator train = new MultiClassIterator(dm, 1, true);
            DataSetIterator train = dm.makeTrainingIterator();
        } catch (IOException e) {
            System.err.println("MultiClassIterator called from RNN failed.");
            e.printStackTrace();
        }
    }

}
