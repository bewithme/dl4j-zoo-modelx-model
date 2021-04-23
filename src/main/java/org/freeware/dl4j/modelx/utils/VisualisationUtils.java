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

    private static void initFrameForMnist(Sample[] samples, String title) {
        if (frame == null) {

            frame = new JFrame();
            frame.setTitle(title);
            frame.setBounds(200, 200, 295, 157);
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());
            panel = new JPanel();
            panel.setLayout(new GridLayout(samples.length / 3, 1, 8, 8));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }
    }


    private static void initFrame(Sample[] samples, String title) {
        if (frame == null) {

            frame = new JFrame();
            frame.setTitle(title);
            frame.setBounds(200, 200, 1024, 1024);
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());
            panel = new JPanel();
            panel.setLayout(new GridLayout(samples.length / 2, 1, 8, 8));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }
    }


    public static void mnistVisualizeForConvolution2D(Sample[] samples,String title) {

        initFrameForMnist(samples, title);

        panel.removeAll();
        for (int i = 0; i < samples.length; i++) {
            panel.add(getLabelForConvolution2D(samples[i].getFeature(),samples[i].getLabel(),28,28,8));
        }
        frame.revalidate();
        frame.pack();
    }

    public static void visualizeForConvolution2D(Sample[] samples,String title) {

        initFrame(samples, title);

        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            INDArray feature=samples[i].getFeature();
            int h= (int)feature.size(2);
            int w= (int)feature.size(3);
            panel.add(getLabelForConvolution2D(feature,samples[i].getLabel(),w,h,1));
        }
        frame.revalidate();
        frame.pack();
    }

    private static JLabel getLabelForConvolution2D(INDArray tensor, String title, int width, int height, int scale) {

        BufferedImage bufferedImage= java2DNativeImageLoader.asBufferedImage(tensor);

        JLabel label = wrapToLabel(bufferedImage,title, width, height, scale);

        return label;
    }

    @NotNull
    private static JLabel wrapToLabel(BufferedImage bufferedImage,String title, int width, int height,int scale) {

        ImageIcon orig = new ImageIcon(bufferedImage);

        Image imageScaled = orig.getImage().getScaledInstance((scale * width), (scale * height), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        JLabel label = new JLabel(scaled);

        label.setText(title);

        label.setVerticalTextPosition(JLabel.TOP);

        label.setHorizontalTextPosition(JLabel.CENTER);

        return label;
    }


    public static void saveAsImageForConvolution2D(Sample[] tensors,String savePath){

        ExtendedFileUtils.makeDirs(savePath);

        for (int k=0;k<tensors.length;k++){

            String fileName= DateUtils.format(new Date(), DateUtils.FORMAT_DATE_TIME_YYYYMMDDHHMMSS).concat("_"+k+"_.jpg");

            fileName=savePath.concat(File.separator).concat(fileName);

            saveAsImageForConvolution2D(tensors[k].getFeature(),fileName);
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
