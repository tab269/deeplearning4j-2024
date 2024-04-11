package org.example.feedforward;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class MNISTSingleLayerActivationVisualization /*extends Application*/ {
//
//    private static final int WIDTH = 28;
//    public static final int HEIGHT = 28;
//
//    public static void main(String[] args) {
//        launch(args);
//    }
//
//    @Override
//    public void start(Stage stage) throws Exception {
//        MultiLayerNetwork model;
//        try {
//            model = ModelSerializer.restoreMultiLayerNetwork(new File("/home/tom/tmp/MATHEMA-Campus-2024/DeepLearning4j/models/MNISTSingleLayer.2024-04-09_12-31-14.model.dl4j"));
//            NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH);
//            INDArray image;
//            image = loader.asMatrix(new File("/home/tom/tmp/MATHEMA-Campus-2024/DeepLearning4j/datasets/mnistOwn/1.png"));
//            List<INDArray> activations = model.feedForward(image);
//
//            HBox hbox = new HBox();
//            for (INDArray activation : activations) {
//                BufferedImage bufferedImage = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
//                for (int i = 0; i < WIDTH; i++) {
//                    for (int j = 0; i < HEIGHT; j++) {
//                        bufferedImage.setRGB(i, j, (int) (255 * activation.getFloat(i * WIDTH + j)));
//
//                    }
//                }
//                ImageView imageView = new ImageView(createImageFromINDArray(activation));
//                hbox.getChildren().add(imageView);
//            }
//
//            Scene scene = new Scene(hbox);
//            stage.setScene(scene);
//            stage.show();
//        } catch (IOException ioe) {
//            ioe.printStackTrace();
//            return;
//        }
//    }
//
//    private Image createImageFromINDArray(INDArray array) {
//        byte[] byteArray =  new byte[(int) array.length()];
//        for (int i = 0; i < array.length(); i++) {
//            byteArray[i] = (byte) (255 * array.getDouble(i));
//        }
//        return new Image(new java.io.ByteArrayInputStream(byteArray));
//    }
}

