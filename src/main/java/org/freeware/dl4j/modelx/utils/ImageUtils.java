package org.freeware.dl4j.modelx.utils;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_core.CV_MAKETYPE;
import static org.bytedeco.opencv.global.opencv_core.FLOAT;
import static org.bytedeco.opencv.global.opencv_imgproc.warpAffine;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_64F;

@Slf4j
public class ImageUtils {



    /**
     * 调整图片大小
     * @param image
     * @param targetWidth
     * @param targetHeight
     * @return
     * @throws IOException
     */
    public static INDArray resize(INDArray image, int targetWidth, int targetHeight) throws IOException {

        Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();

        BufferedImage sourceImage=java2DNativeImageLoader.asBufferedImage(image);

        BufferedImage targetImage=resize(sourceImage,targetWidth,targetHeight);

        return java2DNativeImageLoader.asMatrix(targetImage);
    }

    /**
     * 调整图片大小
     * @param sourceImage
     * @param targetWidth
     * @param targetHeight
     * @return
     * @throws IOException
     */
   public static BufferedImage resize(BufferedImage sourceImage,int targetWidth, int targetHeight) throws IOException {

        int width = sourceImage.getWidth();

        int height = sourceImage.getHeight();

        BufferedImage targetImage  = new BufferedImage(targetWidth, targetHeight, sourceImage.getType());

        Graphics2D g = targetImage.createGraphics();

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        g.drawImage(sourceImage, 0, 0, targetWidth, targetHeight, 0, 0, width, height, null);

        g.dispose();

        return targetImage;
    }

    /**
     * 仿射变换
     * @param image
     * @param width
     * @param height
     * @param transMat
     * @return
     */
    public static INDArray cv2WarpAffine(INDArray image, int width, int height, int[][]transMat) {

        Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();

        Mat mat=java2DNativeImageLoader.asMat(image);

        int rows=transMat.length;

        int  cols=transMat[0].length;

        Mat transMatCvMat = toMat(transMat, rows, cols);

        Mat result = new Mat();

        INDArray 	warpAffineImage=null;

        try {
            warpAffine(mat,result,transMatCvMat,new Size(width,height));

            warpAffineImage=java2DNativeImageLoader.asMatrix(result);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return warpAffineImage;
    }

    @NotNull
    private static Mat toMat(int[][] transMat, int rows, int cols) {
        Mat transMatCvMat = new Mat((int)Math.min(rows, Integer.MAX_VALUE), (int)Math.min(cols, Integer.MAX_VALUE),
                CV_MAKETYPE(CV_32F, (int)Math.min(1, Integer.MAX_VALUE)));
        Indexer matIndexer= transMatCvMat.createIndexer(true);

        FloatIndexer idx = (FloatIndexer)matIndexer;

        for(long i=0;i<rows;i++){

            for(long j=0;j<cols;j++){

                float val=Float.parseFloat(String.valueOf(transMat[(int) i][(int) j]));

                idx.put(i,j,0,val);

            }

        }
        return transMatCvMat;
    }


}
