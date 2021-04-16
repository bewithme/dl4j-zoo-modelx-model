package org.freeware.dl4j.modelx.utils;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.Java2DNativeImageLoader;
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

    public static void mnistVisualize(INDArray[] samples) {
        if (frame == null) {

            frame = new JFrame();
            frame.setTitle("Viz");
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

        BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);

        for (int i = 0; i < 784; i++) {
            int pixel = (int)(((tensor.getDouble(i) + 1) * 2) * 255);
            bi.getRaster().setSample(i % 28, i / 28, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);

        Image imageScaled = orig.getImage().getScaledInstance((8 * 28), (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }

 

}
