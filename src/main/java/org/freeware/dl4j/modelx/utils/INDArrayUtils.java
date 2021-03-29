package org.freeware.dl4j.modelx.utils;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static org.bytedeco.opencv.global.opencv_imgproc.warpAffine;


public class INDArrayUtils {




    /**
     * 将小于min的值设为min
     * 将大于max的值设为max
     * @param data
     * @param min
     * @param max
     * @return
     */
    public static INDArray clip(INDArray data,float min,float max){

        BooleanIndexing.replaceWhere(data, min, Conditions.lessThan(min));

        BooleanIndexing.replaceWhere(data, max, Conditions.greaterThan(max));

        return data;
    }



    /**
     * for example you can flip
     *
     * [[[[    1.0000,    2.0000,    3.0000],
     *    [    4.0000,    5.0000,    6.0000],
     *    [    7.0000,    8.0000,    9.0000]],
     *
     *   [[   10.0000,   11.0000,   12.0000],
     *    [   13.0000,   14.0000,   15.0000],
     *    [   16.0000,   17.0000,   18.0000]],
     *
     *   [[   19.0000,   20.0000,   21.0000],
     *    [   22.0000,   23.0000,   24.0000],
     *    [   25.0000,   26.0000,   27.0000]]]]
     *
     *    to
     *
     * [[[[    3.0000,    2.0000,    1.0000],
     *    [    6.0000,    5.0000,    4.0000],
     *    [    9.0000,    8.0000,    7.0000]],
     *
     *   [[   12.0000,   11.0000,   10.0000],
     *    [   15.0000,   14.0000,   13.0000],
     *    [   18.0000,   17.0000,   16.0000]],
     *
     *   [[   21.0000,   20.0000,   19.0000],
     *    [   24.0000,   23.0000,   22.0000],
     *    [   27.0000,   26.0000,   25.0000]]]]
     *
     * Process finished with exit code 0
     * @param array
     * @return
     */
    public  static INDArray horizontalFlip(INDArray array){

        long[] shape=array.shape();

        long lastDimensionLen=shape[shape.length-1];

        INDArray newArray=array.dup();

        for(long lastDimensionIndex=0;lastDimensionIndex<lastDimensionLen;lastDimensionIndex++){

            INDArray col= array.get(getLastDimensionPoint(array,lastDimensionIndex));

            long[] colShape=col.shape();

            col= Nd4j.expandDims(col,colShape.length);

            long newArrayIndex=lastDimensionLen-1-lastDimensionIndex;

            newArray.put(getLastDimensionPoint(newArray,newArrayIndex),col);
        }

        return newArray;
    }

    /**
     * 仿射变换
     * @param image
     * @param width
     * @param height
     * @param transMat
     * @return
     */
    public static INDArray cv2WarpAffine(INDArray image, int width, int height, int[][] transMat) {

        Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();

        Mat mat=java2DNativeImageLoader.asMat(image);

        INDArray transMatNDArray= Nd4j.create(transMat).reshape(2,3);

        Mat transMatCvMat=java2DNativeImageLoader.asMat(transMatNDArray);

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




    public static INDArrayIndex[] getLastDimensionIndexes(INDArray array, INDArrayIndex lastDimensionIndex){

        INDArrayIndex[] indexes =new INDArrayIndex[array.shape().length];

        for(int i=0;i<array.shape().length;i++){

            if(i!=array.shape().length-1){

                indexes[i]=NDArrayIndex.all();

            }
        }
        indexes[array.shape().length-1]=lastDimensionIndex;

        return indexes ;
    }


    public static INDArrayIndex[] getLastTwoDimensionIndexes(INDArray array, INDArrayIndex firstToLastDimensionIndex,INDArrayIndex secondToLastDimensionIndex){

        INDArrayIndex[] indexes =new INDArrayIndex[array.shape().length];

        for(int i=0;i<array.shape().length;i++){

            if(i<array.shape().length-2){

                indexes[i]=NDArrayIndex.all();

            }
        }
        indexes[array.shape().length-1]=firstToLastDimensionIndex;
        indexes[array.shape().length-2]=secondToLastDimensionIndex;
        return indexes ;
    }


    public static INDArrayIndex[] getLastDimensionPointZero(INDArray array){
        return getLastDimensionIndexes(array,NDArrayIndex.point(0));
    }
    public static INDArrayIndex[] getLastDimensionPointOne(INDArray array){
        return getLastDimensionIndexes(array,NDArrayIndex.point(1));
    }
    public static INDArrayIndex[] getLastDimensionPointTwo(INDArray array){
        return getLastDimensionIndexes(array,NDArrayIndex.point(2));
    }
    public static INDArrayIndex[] getLastDimensionPointThree(INDArray array){
        return getLastDimensionIndexes(array,NDArrayIndex.point(3));
    }
    public static INDArrayIndex[] getLastDimensionPointFromZeroToTwo(INDArray array){
        return getLastDimensionIndexes(array,NDArrayIndex.interval(0,2));
    }
    public static INDArrayIndex[] getLastDimensionPointFromTwoToFour(INDArray array){
        return getLastDimensionIndexes(array,NDArrayIndex.interval(2,4));
    }
    public static INDArrayIndex[] getLastDimensionPoint(INDArray array,long point){
        return getLastDimensionIndexes(array,NDArrayIndex.point(point));
    }

    public static INDArrayIndex[] getLastTwoDimensionIndexes(INDArray array, long firstToLastDimensionIndexFrom,long firstToLastDimensionIndexTo,int secondToLastDimensionIndexFrom,int secondToLastDimensionIndexTo) {

        return getLastTwoDimensionIndexes(array,NDArrayIndex.interval(firstToLastDimensionIndexFrom,firstToLastDimensionIndexTo),NDArrayIndex.interval(secondToLastDimensionIndexFrom,secondToLastDimensionIndexTo));
    }

    }
