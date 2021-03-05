package org.freeware.dl4j.modelx;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class INDArrayUtil {


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
    public static INDArray leftToRightFlip(INDArray array){

        INDArray newArray=array.dup();

        long batchSize=array.shape()[0];

        long channel=array.shape()[1];

        for(long exampleIndex=0;exampleIndex<batchSize;exampleIndex++){

            INDArray single=array.get(NDArrayIndex.point(exampleIndex),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all());

            for(long channelIndex=0;channelIndex<channel;channelIndex++){

                INDArray hw=single.get(new INDArrayIndex[]{NDArrayIndex.point(channelIndex),NDArrayIndex.all(),NDArrayIndex.all()});

                long width=hw.shape()[1];

                for(long colIndex=0;colIndex<width;colIndex++){

                    INDArray col=hw.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(colIndex)});

                    newArray.put(new INDArrayIndex[]{NDArrayIndex.point(exampleIndex),NDArrayIndex.point(channelIndex),NDArrayIndex.all(),NDArrayIndex.point(width-1-colIndex)},col);
                }

            }

        }
        return newArray;
    }
}
