package org.freeware.dl4j.modelx;

import org.datavec.image.recordreader.objdetect.ImageObject;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

public class YoloImageUtils {

    public static void drawBoundingBox(BufferedImage bufferedImage, ImageObject boundingBox, Color color) {

        Graphics2D graphics2D=bufferedImage.createGraphics();

        graphics2D.setColor(color);

        Font font = new Font( graphics2D.getFont().getFontName(), Font.PLAIN, 30);

        graphics2D.setFont(font);

        //矩形框(原点x坐标，原点y坐标，矩形的长，矩形的宽)
        graphics2D.drawRect(boundingBox.getX1(), boundingBox.getY1(), boundingBox.getX2()-boundingBox.getX1(), boundingBox.getY2()-boundingBox.getY1());

        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION,  RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        graphics2D.drawString(boundingBox.getLabel(), boundingBox.getX1(), boundingBox.getY2());

    }


    public static void drawBoundingBoxes(BufferedImage bufferedImage, List<ImageObject> boundingBoxList, Color color){
        for (ImageObject boundingBox:boundingBoxList){
            drawBoundingBox(bufferedImage,boundingBox,color);
        }
    }
}
