package org.freeware.dl4j.modelx.utils;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Date;


@Slf4j
public class VisualisationUtils {

    private static JFrame frame;
    private static JPanel panel;

    private static    Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();

    public static void mnistVisualize(INDArray[] samples) {
        if (frame == null) {

            frame = new JFrame();
            frame.setTitle("Viz");
            frame.setBounds(200, 200, 295, 157);
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            panel = new JPanel();
            panel.setLayout(new GridLayout(samples.length / 3, 1, 8, 8));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }

        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            panel.add(getMnistImage(samples[i]));
        }

        frame.revalidate();
        frame.pack();
    }

    private static JLabel getMnistImage(INDArray tensor) {

        BufferedImage bi = convertToImage(tensor,28,28);

        ImageIcon orig = new ImageIcon(bi);

        Image imageScaled = orig.getImage().getScaledInstance((8 * 28), (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }

    @NotNull
    private static BufferedImage convertToImage(INDArray tensor,int width,int height) {

        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int i = 0; i < width*height; i++) {

            int pixel = (int)(((tensor.getDouble(i) + 1) * 2) * 255);

            bi.getRaster().setSample(i % width, i / height, 0, pixel);
        }

        return bi;
    }

    public static void saveAsImage(INDArray[] tensors,String savePath){

        ExtendedFileUtils.makeDirs(savePath);

        for (int k=0;k<tensors.length;k++){

            String fileName= DateUtils.format(new Date(), DateUtils.FORMAT_DATE_TIME_YYYYMMDDHHMMSS).concat("_"+k+"_.jpg");

            fileName=savePath.concat(File.separator).concat(fileName);

            saveAsImage(tensors[k],fileName,28,28);
        }
    }

    public static void saveAsImage(INDArray tensor,String fileName,int width,int height) {

            BufferedImage bi =convertToImage(tensor,width,height);
        try {
            ImageIO.write(bi, "jpg",new File(fileName));
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }



    public static void mnistVisualizeForConvolution2D(INDArray[] samples) {
        if (frame == null) {

            frame = new JFrame();
            frame.setTitle("Viz");
            frame.setBounds(200, 200, 295, 157);
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            panel = new JPanel();
            panel.setLayout(new GridLayout(samples.length / 3, 1, 8, 8));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }

        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            panel.add(getMnistImageForConvolution2D(samples[i]));
        }

        frame.revalidate();
        frame.pack();
    }

    private static JLabel getMnistImageForConvolution2D(INDArray tensor) {

        BufferedImage bi= java2DNativeImageLoader.asBufferedImage(tensor);

        ImageIcon orig = new ImageIcon(bi);

        Image imageScaled = orig.getImage().getScaledInstance((8 * 28), (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }


    public static void saveAsImageForConvolution2D(INDArray[] tensors,String savePath){

        ExtendedFileUtils.makeDirs(savePath);

        for (int k=0;k<tensors.length;k++){

            String fileName= DateUtils.format(new Date(), DateUtils.FORMAT_DATE_TIME_YYYYMMDDHHMMSS).concat("_"+k+"_.jpg");

            fileName=savePath.concat(File.separator).concat(fileName);

            saveAsImageForConvolution2D(tensors[k],fileName);
        }
    }

    public static void saveAsImageForConvolution2D(INDArray tensor,String fileName) {

        BufferedImage bi = java2DNativeImageLoader.asBufferedImage(tensor);
        try {
            ImageIO.write(bi, "jpg",new File(fileName));
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }



}
