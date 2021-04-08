package org.freeware.dl4j.modelx.utils;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;

import java.util.Arrays;

@Slf4j
public class SameDiffUtils {

    public static SDIndex[] getLastDimensionIndexes(SDVariable array, SDIndex lastDimensionIndex){

        long[] shape=array.getShape();

        log.info("--->"+Arrays.toString(shape));

        int length=shape.length;

        SDIndex[] indexes =new SDIndex[length];

        for(int i=0;i<length;i++){

            if(i!=length-1){

                indexes[i]= SDIndex.all();

            }
        }
        indexes[length-1]=lastDimensionIndex;

        return indexes ;
    }


    public static SDIndex[] getLastTwoDimensionIndexes(SDVariable array, SDIndex firstToLastDimensionIndex,SDIndex secondToLastDimensionIndex){

        long[] shape=array.getShape();

        log.info("--->>"+Arrays.toString(shape));

        int length=shape.length;

        SDIndex[] indexes =new SDIndex[length];

        for(int i=0;i<length;i++){

            if(i!=length-2){

                indexes[i]= SDIndex.all();
            }
        }
        indexes[length-1]=firstToLastDimensionIndex;

        indexes[length-2]=secondToLastDimensionIndex;

        return indexes ;
    }


    public static SDIndex[] getLastDimensionPointZero(SDVariable array){
        return getLastDimensionIndexes(array,SDIndex.point(0));
    }
    public static SDIndex[] getLastDimensionPointOne(SDVariable array){
        return getLastDimensionIndexes(array,SDIndex.point(1));
    }
    public static SDIndex[] getLastDimensionPointTwo(SDVariable array){
        return getLastDimensionIndexes(array,SDIndex.point(2));
    }
    public static SDIndex[] getLastDimensionPointThree(SDVariable array){
        return getLastDimensionIndexes(array,SDIndex.point(3));
    }
    public static SDIndex[] getLastDimensionPointFromZeroToTwo(SDVariable array){
        return getLastDimensionIndexes(array,SDIndex.interval(0,2));
    }
    public static SDIndex[] getLastDimensionPointFromTwoToFour(SDVariable array){
        return getLastDimensionIndexes(array,SDIndex.interval(2,4));
    }
    public static SDIndex[] getLastDimensionPoint(SDVariable array,long point){
        return getLastDimensionIndexes(array,SDIndex.point(point));
    }

    public static SDIndex[] getLastTwoDimensionIndexes(SDVariable array, long firstToLastDimensionIndexFrom,long firstToLastDimensionIndexTo,int secondToLastDimensionIndexFrom,int secondToLastDimensionIndexTo) {
        return getLastTwoDimensionIndexes(array,SDIndex.interval(firstToLastDimensionIndexFrom,firstToLastDimensionIndexTo),SDIndex.interval(secondToLastDimensionIndexFrom,secondToLastDimensionIndexTo));
    }
}
