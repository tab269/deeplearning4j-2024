package org.example.feedforward;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTClassifyOwnDataSetWithReader {

    public static final String MODEL_PATH = "/home/tom/tmp/MATHEMA-Campus-2024/DeepLearning4j/models/MNISTSingleLayer.2024-04-09_12-31-14.model.dl4j";
    public static final String IMAGE_PATH = "/home/tom/tmp/MATHEMA-Campus-2024/DeepLearning4j/datasets/mnistOwnWithReader/";
    public static final int HEIGHT = 28;
    public static final int WIDTH = 28;
    public static final int CHANNELS = 1;
    public static final int BATCH_SIZE = 10;

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model = MultiLayerNetwork.load(new File(MODEL_PATH), true);
        System.out.println(model.summary());

//        INDArray intVector5D = Nd4j.zeros(DataType.INT, 1, 5);
//        INDArray doubleMatrix2x2 = Nd4j.create(new int[][]{{1, 2}, {3, 4}});
//        INDArray doubleTensor2x3x4 = Nd4j.rand(new int[]{2, 3, 4});
//        System.out.println("intVector5D = " + intVector5D);
//        System.out.println("doubleMatrix2x2 = " + doubleMatrix2x2);
//        System.out.println("intTensor2x3x4 = " + doubleTensor2x3x4);
//        System.out.println("intVector5D transposed = " + intVector5D.transpose());
//        INDArray v0 = Nd4j.create(new double[]{1, 2});
//        System.out.println("v0 = " + v0);
//        System.out.println("M2002 times v0 = " + Nd4j.create(new int[][]{{2, 0}, {0, 2}}).mul(v0));
//        exit(-1);

        File parentDirOfTestImages = new File(IMAGE_PATH);
        ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, new ParentPathLabelGenerator());
        FileSplit fileSplit = new FileSplit(parentDirOfTestImages, NativeImageLoader.ALLOWED_FORMATS);
        recordReader.initialize(fileSplit);
        int classesCount = (int)fileSplit.length();
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, classesCount);
        testIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        Evaluation eval = new Evaluation();
        while (testIter.hasNext()) {
            DataSet ds = testIter.next();
            INDArray output = model.output(ds.getFeatures());
            eval.eval(ds.getLabels(), output);
        }

        System.out.println(eval.stats());

        recordReader.close();
    }

    public static INDArray imageToINDArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Skalieren Sie das Bild auf 28x28, falls erforderlich
        BufferedImage scaledImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = scaledImage.createGraphics();
        g.drawImage(image, 0, 0, 28, 28, null);
        g.dispose();

        // Erstellen eines INDArray mit den Dimensionen (1, 1, 28, 28) für ein Bild
        INDArray array = Nd4j.create(1, 1, 28, 28);

        // Iteration über das skalierte Bild, um die Pixelwerte in das INDArray zu setzen
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int pixel = scaledImage.getRGB(x, y);
                // Extrahieren Sie den Grauwert des Pixels und setzen Sie ihn im INDArray
                double grayValue = ((pixel >> 16) & 0xFF) * 0.299 + ((pixel >> 8) & 0xFF) * 0.587 + (pixel & 0xFF) * 0.114;
                array.putScalar(0, 0, y, x, grayValue);
            }
        }

        // Umwandlung des 4D-INDArray in ein 2D-Format (Vektor)
        return array.reshape(1, 784);
    }
//    public static INDArray imageToINDArray(BufferedImage image) {
//        int width = image.getWidth();
//        int height = image.getHeight();
//
//        // Erstellen eines INDArray mit den Dimensionen (1, 1, height, width) für ein Bild
//        INDArray array = Nd4j.create(1, 1, height, width);
//
//        // Iteration über das Bild, um die Pixelwerte in das INDArray zu setzen
//        for (int y = 0; y < height; y++) {
//            for (int x = 0; x < width; x++) {
//                int pixel = image.getRGB(x, y);
//                // Extrahieren Sie den Grauwert des Pixels und setzen Sie ihn im INDArray
//                double grayValue = ((pixel >> 16) & 0xFF) * 0.299 + ((pixel >> 8) & 0xFF) * 0.587 + (pixel & 0xFF) * 0.114;
//                array.putScalar(0, 0, y, x, grayValue);
//            }
//        }
//
//        return array;
//    }
}
