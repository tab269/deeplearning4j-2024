package org.example.feedforward;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MNISTClassifyOwnDataSet {

    public static final String MODEL_PATH = "/home/tom/tmp/MATHEMA-Campus-2024/DeepLearning4j/models/MNISTSingleLayer.2024-04-09_12-31-14.model.dl4j";
    public static final String IMAGE_PATH_PREFIX = "/home/tom/tmp/MATHEMA-Campus-2024/DeepLearning4j/datasets/mnistOwn/";
    public static final String IMAGE_PATH_SUFFIX = ".png";

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model = MultiLayerNetwork.load(new File(MODEL_PATH), true);

        for (int i = 0; i <= 9; i++) {
            File imageFile = new File(IMAGE_PATH_PREFIX + i + IMAGE_PATH_SUFFIX);
//            System.out.println(imageFile.getPath());
            INDArray input = imageToINDArray(ImageIO.read(imageFile));
            INDArray output = model.output(input);
            System.out.println(i + " => " + output);
        }
    }

    public static INDArray imageToINDArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Skalieren Sie das Bild auf 28x28, falls erforderlich
        BufferedImage scaledImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = scaledImage.createGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();

        // Erstellen eines INDArray mit den Dimensionen (1, 1, 28, 28) für ein Bild
        INDArray array = Nd4j.create(1, 1, height, width);

        // Iteration über das skalierte Bild, um die Pixelwerte in das INDArray zu setzen
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = scaledImage.getRGB(x, y);
                // Extrahieren Sie den Grauwert des Pixels und setzen Sie ihn im INDArray
                double grayValue = ((pixel >> 16) & 0xFF) * 0.299 + ((pixel >> 8) & 0xFF) * 0.587 + (pixel & 0xFF) * 0.114;
                array.putScalar(0, 0, y, x, grayValue);
            }
        }

        // Umwandlung des 4D-INDArray in ein 2D-Format (Vektor)
        return array.reshape(1, width * height);
    }
}
