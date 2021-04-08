package org.freeware.dl4j.modelx.utils;

import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SDIndex;



@Slf4j
public class SameDiffUtils {



    @NotNull
    private static SDIndex[] getLastDimensionIndexes(long[] shape,SDIndex lastDimensionIndex) {

        int length=shape.length;

        SDIndex[] indexes =new SDIndex[length];

        for(int i=0;i<length;i++){

            if(i!=length-1){

                indexes[i]= SDIndex.all();

            }
        }
        indexes[length-1]=lastDimensionIndex;

        return indexes;
    }



    private static SDIndex[] getLastTwoDimensionIndexes(long[] shape,SDIndex firstToLastDimensionIndex, SDIndex secondToLastDimensionIndex) {

        int length=shape.length;

        SDIndex[] indexes =new SDIndex[length];

        for(int i=0;i<length;i++){

            if(i!=length-2){

                indexes[i]= SDIndex.all();
            }
        }
        indexes[length-1]=firstToLastDimensionIndex;

        indexes[length-2]=secondToLastDimensionIndex;
        return indexes;
    }


    public static SDIndex[] getLastDimensionPointZero(long[] shape){
        return getLastDimensionIndexes(shape,SDIndex.point(0));
    }
    public static SDIndex[] getLastDimensionPointOne(long[] shape){
        return getLastDimensionIndexes(shape,SDIndex.point(1));
    }
    public static SDIndex[] getLastDimensionPointTwo(long[] shape){
        return getLastDimensionIndexes(shape,SDIndex.point(2));
    }
    public static SDIndex[] getLastDimensionPointThree(long[] shape){
        return getLastDimensionIndexes(shape,SDIndex.point(3));
    }
    public static SDIndex[] getLastDimensionPointFromZeroToTwo(long[] shape){
        return getLastDimensionIndexes(shape,SDIndex.interval(0,2));
    }
    public static SDIndex[] getLastDimensionPointFromTwoToFour(long[] shape){
        return getLastDimensionIndexes(shape,SDIndex.interval(2,4));
    }
    public static SDIndex[] getLastDimensionPoint(long[] shape,long point){
        return getLastDimensionIndexes(shape,SDIndex.point(point));
    }

    public static SDIndex[] getLastTwoDimensionIndexes(long[] shape, long firstToLastDimensionIndexFrom,long firstToLastDimensionIndexTo,int secondToLastDimensionIndexFrom,int secondToLastDimensionIndexTo) {
        return getLastTwoDimensionIndexes(shape,SDIndex.interval(firstToLastDimensionIndexFrom,firstToLastDimensionIndexTo),SDIndex.interval(secondToLastDimensionIndexFrom,secondToLastDimensionIndexTo));
    }
}
