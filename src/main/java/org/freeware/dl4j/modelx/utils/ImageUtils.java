package org.freeware.dl4j.modelx.utils;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Random;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.opencv.core.CvType.CV_32F;


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





    /**
     * 随机扭曲图片
     * @param image
     * @return
     */
    public static INDArray randomDistortImage(INDArray image,Random random){

        float hue=18f, saturation=1.5f, exposure=1.5f;

        float dhue=RandomUtils.randomUniform(-hue,hue,random);

        float dsat=RandomUtils.randScale(saturation,random);

        float dexp=RandomUtils.randScale(exposure,random);

        image=ctvColor(image,COLOR_RGB2HSV);

        INDArrayIndex[] dsatIndex=new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all()};

        INDArrayIndex[] dexpIndex= new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(2),NDArrayIndex.all(),NDArrayIndex.all()};

        INDArray dsatArray=image.get(dsatIndex).mul(dsat);

        INDArray dexpArray=image.get(dexpIndex).mul(dexp);

        image.put(dsatIndex,dsatArray);

        image.put(dexpIndex,dexpArray);

        INDArrayIndex[] dhueIndex= new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all()};

        INDArray dhueArray=image.get(dhueIndex).add(dhue);

        INDArray replacedDhueArray=dhueArray.dup();

        BooleanIndexing.replaceWhere(replacedDhueArray, 180, Conditions.greaterThan(180));

        dhueArray=dhueArray.sub(replacedDhueArray);

        replacedDhueArray=dhueArray.dup();

        BooleanIndexing.replaceWhere(replacedDhueArray, 180, Conditions.lessThan(0));

        dhueArray=dhueArray.add(replacedDhueArray);

        image.put(dhueIndex,dhueArray);

        image=ctvColor(image,COLOR_HSV2RGB);

        return image;
    }

    /**
     * 改变颜色
     * @param image
     * @param code
     * @return
     */
    public static INDArray ctvColor(INDArray image,int code){

        Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();

        Mat mat=java2DNativeImageLoader.asMat(image);

        Mat result = new Mat();

        try {
            cvtColor(mat, result, code);
            return java2DNativeImageLoader.asMatrix(result);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 随机翻转
     * @param image
     * @param flip
     * @return
     */
    public static INDArray randomFLip(INDArray image,int flip){

        if(flip!=1){
            return image;
        }
        Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();
        Mat mat=java2DNativeImageLoader.asMat(image);
        Mat result = new Mat();
        try {
            flip(mat, result, flip);
            return java2DNativeImageLoader.asMatrix(result);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }




}
