package org.example.feedforward;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

/** By Manuel Kohler */
public class ImageToNumber {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("/home/tom/tmp/deeplearning4j-2024/models/MNSISTSingleLayerManuel.model.dl4j"), false);

        run(1, model, "/home/tom/tmp/deeplearning4j-2024/datasets/mnsistOwnManuel/1.png");
        run(3, model, "/home/tom/tmp/deeplearning4j-2024/datasets/mnsistOwnManuel/3.png");
        run(6, model, "/home/tom/tmp/deeplearning4j-2024/datasets/mnsistOwnManuel/6.png");
        run(4, model, "/home/tom/tmp/deeplearning4j-2024/datasets/mnsistOwnManuel/4.png");
    }

    private static void run(int r, MultiLayerNetwork model, String path) throws IOException {
        INDArray indArrayImage = loadImage(path);

        List<INDArray> indArrayList = model.feedForward(indArrayImage, false);
        INDArray indArray = indArrayList.get(indArrayList.size() - 1);

        float maxValue = 0;
        int indexMaxValue = 0;
        for(int i = 0; i < 10; i++) {
            if(indArray.getFloat(i) > maxValue){
                maxValue = indArray.getFloat(i);
                indexMaxValue = i;
            }
        }

        System.out.println(r + " wurde gelesen als: " + indexMaxValue);
        model.clear();
    }

    private static INDArray loadImage(String path) throws IOException {
        BufferedImage bufferedImage = ImageIO.read(new File(path));

        int[] data = bufferedImage.getData().getPixels(0,0,28,28, (int[]) null);
        int[] oneChannel = transformImageWith4ChannelsTo1(data);
        float[] imageFin = centerImage(oneChannel);
        //showImageInConsole(imageFin);

        float[][] t = new float[1][];
        t[0] = imageFin;
        return Nd4j.create(t);
    }

    private static float[] centerImage(int[] image) {
        float[] out = new float[28*28];

        int posX = 0;
        int posY = 0;
        int count = 0;

        for(int x = 0; x < 28; x++) {
            for(int y = 0; y < 28; y++) {
                int val = image[x * 28 + y] > 125 ? 1 : 0;
                if(val == 0) {
                    posX += x;
                    posY += y;
                    count += 1;
                }
            }
        }

        int offsetX = 14 - posX / count;
        int offsetY = 14 - posY / count;

        // Verschieben
        for(int x = 0; x < 28; x++) {
            for(int y = 0; y < 28; y++) {
                int indexX = x + offsetX;
                int indexY = y + offsetY;
                if(indexX >= 0 && indexX < 28 && indexY >= 0 && indexY < 28)
                    out[indexX * 28 + indexY] = (1f-(float)image[x * 28 + y]/256f);
            }
        }

        return out;
    }

    private static int[] transformImageWith4ChannelsTo1(int[] image) {
        int[] data = new int[28*28];

        for(int x = 0; x < 28; x++) {
            for(int y = 0; y < 28; y++) {
                data[x * 28 + y] = image[(x * 28 + y) * 4];
            }
        }

        return data;
    }

    private static void showImageInConsole(float[] image) {
        for(int x = 0; x < 28; x++) {
            for(int y = 0; y < 28; y++) {
                if(image[x * 28 + y] > 0.5f)
                    System.out.print(" ");
                else
                    System.out.print("@");
            }
            System.out.println("");
        }
    }
}

