package cs517;

import cs517.data.DataSetManager;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by allen on 6/3/2016.
 */
public class RNN {



    public RNN() {
    }

    public static void main(String[] args) {

        int batchSize = 1;
        int vectorSize=100;
        int nEpochs = 5;
        int maxLength = 50;

        RNN myNN = new RNN();

        DataSetManager dm = new DataSetManager();
        File ldf = new File("src/main/resources/movieData/toyMaasDataset/toyLabeledTrainData.tsv");
        File tdf = new File("src/main/resources/movieData/toyMaasDataset/toyTestData.tsv");
        try {
            dm.importData(ldf);
            dm.importData(tdf);
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<DataSetIterator> iterators = dm.makeIterators(batchSize);
        DataSetIterator train = iterators.get(0);
        DataSetIterator cv = iterators.get(1);
        DataSetIterator test = iterators.get(2);


        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(0.0018)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(200)
                        .activation("tanh").build())
                .layer(1, new RnnOutputLayer.Builder().activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(8).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();




//            WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File("sentimentWordVectors.txt"));
//            vectorSize = wordVectors.lookupTable().layerSize();
//        myNN.net.init();
//            myNN.net.setListeners(new ScoreIterationListener(1));
//            myNN.net.setListeners(new HistogramIterationListener(1));

        System.out.println("Starting training");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = new Evaluation();

            while (test.hasNext()) {

                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray labels = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features, false, inMask, outMask);

                evaluation.evalTimeSeries(labels, predicted, outMask);
                i++;
            }
            test.reset();

            System.out.println(evaluation.stats());
        }
        System.out.println("----- Example complete -----");
    }



}

