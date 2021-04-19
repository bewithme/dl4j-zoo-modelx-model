package org.freeware.dl4j.modelx.utils;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;




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

            INDArray col= array.get(getLastDimensionPoint(shape,lastDimensionIndex));

            long[] colShape=col.shape();

            col= Nd4j.expandDims(col,colShape.length);

            long newArrayIndex=lastDimensionLen-1-lastDimensionIndex;

            newArray.put(getLastDimensionPoint(shape,newArrayIndex),col);
        }

        return newArray;
    }

    /**
     * 按指定形状填充
     * np.full
     * @param shape
     * @param value
     * @return
     */
    public  static INDArray full(long[] shape,float value){

        INDArray array=Nd4j.zeros(shape);

        long lastDimensionLen=shape[shape.length-1];

        for(long lastDimensionIndex=0;lastDimensionIndex<lastDimensionLen;lastDimensionIndex++){

            array.put(getLastDimensionPoint(shape,lastDimensionIndex),value);
        }
        return  array;
    }


    /**
     * 获取最后一个维度的索引
     * @param shape
     * @param lastDimensionIndex
     * @return
     */
    public static INDArrayIndex[] getLastDimensionIndexes(long[] shape, INDArrayIndex lastDimensionIndex){

        INDArrayIndex[] indexes =new INDArrayIndex[shape.length];

        for(int i=0;i<shape.length;i++){

            if(i!=shape.length-1){

                indexes[i]=NDArrayIndex.all();

            }
        }
        indexes[shape.length-1]=lastDimensionIndex;

        return indexes ;
    }


    /**
     * 获取最后两个维度的索引
     * @param shape
     * @param firstToLastDimensionIndex
     * @param secondToLastDimensionIndex
     * @return
     */
    public static INDArrayIndex[] getLastTwoDimensionIndexes(long[] shape, INDArrayIndex firstToLastDimensionIndex,INDArrayIndex secondToLastDimensionIndex){

        INDArrayIndex[] indexes =new INDArrayIndex[shape.length];

        for(int i=0;i<shape.length;i++){

            if(i<shape.length-2){

                indexes[i]=NDArrayIndex.all();

            }
        }
        indexes[shape.length-1]=firstToLastDimensionIndex;
        indexes[shape.length-2]=secondToLastDimensionIndex;
        return indexes ;
    }

    /**
     * 获取最后一个维度的第0个索引
     * @param shape
     * @return
     */
    public static INDArrayIndex[] getLastDimensionPointZero(long[] shape){
        return getLastDimensionIndexes(shape,NDArrayIndex.point(0));
    }
    /**
     * 获取最后一个维度的第1个索引
     * @param shape
     * @return
     */
    public static INDArrayIndex[] getLastDimensionPointOne(long[] shape){
        return getLastDimensionIndexes(shape,NDArrayIndex.point(1));
    }
    /**
     * 获取最后一个维度的第2个索引
     * @param shape
     * @return
     */
    public static INDArrayIndex[] getLastDimensionPointTwo(long[] shape){
        return getLastDimensionIndexes(shape,NDArrayIndex.point(2));
    }
    /**
     * 获取最后一个维度的第3个索引
     * @param shape
     * @return
     */
    public static INDArrayIndex[] getLastDimensionPointThree(long[] shape){
        return getLastDimensionIndexes(shape,NDArrayIndex.point(3));
    }
    /**
     * 获取最后一个维度的第0-2个索引
     * @param shape
     * @return
     */
    public static INDArrayIndex[] getLastDimensionPointFromZeroToTwo(long[] shape){
        return getLastDimensionIndexes(shape,NDArrayIndex.interval(0,2));
    }
    /**
     * 获取最后一个维度的第2-4个索引
     * @param shape
     * @return
     */
    public static INDArrayIndex[] getLastDimensionPointFromTwoToFour(long[] shape){
        return getLastDimensionIndexes(shape,NDArrayIndex.interval(2,4));
    }

    /**
     * 获取最后一个维度的第point个索引
     * @param shape
     * @param point
     * @return
     */
    public static INDArrayIndex[] getLastDimensionPoint(long[] shape,long point){
        return getLastDimensionIndexes(shape,NDArrayIndex.point(point));
    }

    /**
     * 获取倒数第二个维度的第firstToLastDimensionIndexFrom到firstToLastDimensionIndexTo个索引
     * 倒数第一个维度的第secondToLastDimensionIndexFrom到secondToLastDimensionIndexTo个索引
     * @param shape
     * @param firstToLastDimensionIndexFrom
     * @param firstToLastDimensionIndexTo
     * @param secondToLastDimensionIndexFrom
     * @param secondToLastDimensionIndexTo
     * @return
     */
    public static INDArrayIndex[] getLastTwoDimensionIndexes(long[] shape, long firstToLastDimensionIndexFrom,long firstToLastDimensionIndexTo,int secondToLastDimensionIndexFrom,int secondToLastDimensionIndexTo) {

        return getLastTwoDimensionIndexes(shape,NDArrayIndex.interval(firstToLastDimensionIndexFrom,firstToLastDimensionIndexTo),NDArrayIndex.interval(secondToLastDimensionIndexFrom,secondToLastDimensionIndexTo));
    }


    public static INDArray getHalfOfFirstDimension(INDArray array){

        long[] shape=array.shape();

        INDArrayIndex[] indexes =new INDArrayIndex[shape.length];

        for(int i=0;i<shape.length;i++){

            if(i!=0){

                indexes[i]=NDArrayIndex.all();

            }
        }
        indexes[0]=NDArrayIndex.interval(0,array.size(0)/2);

        return array.get(indexes) ;
    }


}
