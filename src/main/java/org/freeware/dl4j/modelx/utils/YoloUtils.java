package org.freeware.dl4j.modelx.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class YoloUtils {

    private static INDArrayIndex[] INDEX_ZERO_5D =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(0)};
    private static INDArrayIndex[] INDEX_ONE_5D  =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(1)};
    private static INDArrayIndex[] INDEX_TWO_5D  =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(2)};
    private static INDArrayIndex[] INDEX_THREE_5D  =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(3)};
    private static INDArrayIndex[] INDEX_FROM_ZERO_TO_TWO_5D  =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,2)};
    private static INDArrayIndex[] INDEX_FROM_TWO_TO_FOUR_5D  =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(2,4)};

    private static INDArrayIndex[]  INDEX_ZERO_2D =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(0)};
    private static INDArrayIndex[]  INDEX_ONE_2D =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(1)};
    private static INDArrayIndex[]  INDEX_TWO_2D =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(2)};
    private static INDArrayIndex[]  INDEX_THREE_2D =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(3)};
    private static INDArrayIndex[]  INDEX_FROM_ZERO_TO_TWO_2D =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.interval(0,2)};
    private static INDArrayIndex[]  INDEX_FROM_TWO_TO_FOUR_2D =new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.interval(2,4)};

    /**
     * 获取每个单元格相对于最左上角的坐标
     * 输出形状为[batchSize,gridSize,gridSize,anchorQuantityPerGrid,2]
     * 最后一个维度用来存当前单元格相对于左上角的坐标(Cx,Cy)
     * @param gridSize 网格大小，有13，26，52
     * @param batchSize 批量大小
     * @param anchorQuantityPerCell 每个单元格负责检测的先验框数量，一般为3
     * @return
     */
    private static  INDArray getCxCy(int gridSize, int batchSize, int anchorQuantityPerCell) {

        //创建一个元素为0到gridSize-1一维数组
        INDArray gridCoordinatePoints= Nd4j.linspace(0,gridSize-1,gridSize);
        //将形状为[1,gridSize]的数组在[gridSize,1]形状上平铺
        INDArray x=Nd4j.tile(gridCoordinatePoints.reshape(new int[]{1,gridSize}),new int[]{gridSize,1});

        INDArray y=Nd4j.tile(gridCoordinatePoints.reshape(new int[]{gridSize,1}),new int[]{1,gridSize});
        //[gridSize,gridSize]-->[gridSize,gridSize,1]
        x=Nd4j.expandDims(x,2);
        //[gridSize,gridSize]-->[gridSize,gridSize,1]
        y=Nd4j.expandDims(y,2);
        //[gridSize,gridSize,1]-->[gridSize,gridSize,2]
        INDArray xy= Nd4j.concat(-1,x,y);
        //[gridSize,gridSize,2]-->[1,gridSize,gridSize,2]
        xy=Nd4j.expandDims(xy,0);
        //[1,gridSize,gridSize,2]-->[1,gridSize,gridSize,1,2]
        xy=Nd4j.expandDims(xy,3);
        //[1,gridSize,gridSize,1,2]-->[batchSize,gridSize,gridSize,anchorQuantityPerGrid,2]
        xy=Nd4j.tile(xy,new int[]{batchSize,1,1,anchorQuantityPerCell,1});

        return xy;
    }

    public   static INDArray get2DBoxIou(INDArray boundingBoxes1,INDArray boundingBoxes2){
        return getIou(INDEX_ZERO_2D,INDEX_ONE_2D,INDEX_TWO_2D,INDEX_THREE_2D,INDEX_FROM_ZERO_TO_TWO_2D,INDEX_FROM_TWO_TO_FOUR_2D,boundingBoxes1,boundingBoxes2);
    }

    public   static INDArray get5DBoxIou(INDArray boundingBoxes1,INDArray boundingBoxes2){
        return getIou(INDEX_ZERO_5D ,INDEX_ONE_5D ,INDEX_TWO_5D ,INDEX_THREE_5D ,INDEX_FROM_ZERO_TO_TWO_5D ,INDEX_FROM_TWO_TO_FOUR_5D ,boundingBoxes1,boundingBoxes2);
    }

    public   static INDArray get5DBoxGIou(INDArray boundingBoxes1,INDArray boundingBoxes2){
        return getGIou(INDEX_ZERO_5D ,INDEX_ONE_5D ,INDEX_TWO_5D ,INDEX_THREE_5D ,INDEX_FROM_ZERO_TO_TWO_5D ,INDEX_FROM_TWO_TO_FOUR_5D ,boundingBoxes1,boundingBoxes2);
    }

    /**
     * 批量计算两个边界框的IOU
     * A∩B/A∪B
     * @param boundingBoxes1
     * @param boundingBoxes2
     * @return
     */
    private  static INDArray getIou(INDArrayIndex[] indexZero,INDArrayIndex[] indexOne,INDArrayIndex[] indexTwo,INDArrayIndex[] indexThree,INDArrayIndex[] indexFromZeroToTwo, INDArrayIndex[] indexFromTwoToFour ,INDArray boundingBoxes1,INDArray boundingBoxes2) {

        //计算第一个边界框的面积
        INDArray boundingBoxes1Area= boundingBoxes1.get(indexTwo).mul(boundingBoxes1.get(indexThree));
        //计算第二个边界框的面积
        INDArray boundingBoxes2Area= boundingBoxes2.get(indexTwo).mul(boundingBoxes2.get(indexThree));
        //左上角坐标
        INDArray boundingBoxes1LeftTop= boundingBoxes1.get(indexFromZeroToTwo).sub(boundingBoxes1.get(indexFromTwoToFour).mul(0.5));
        //右下角坐标
        INDArray boundingBoxes1RightBottom= boundingBoxes1.get(indexFromZeroToTwo).add(boundingBoxes1.get(indexFromTwoToFour).mul(0.5));
        //转换为(top x,top y,bottom x, bottom y) 格式的bounding box
        boundingBoxes1=Nd4j.concat(-1,boundingBoxes1LeftTop,boundingBoxes1RightBottom);

        INDArray boundingBoxes2LeftTop= boundingBoxes2.get(indexFromZeroToTwo).sub(boundingBoxes2.get(indexFromTwoToFour).mul(0.5));

        INDArray boundingBoxes2RightBottom= boundingBoxes2.get(indexFromZeroToTwo).add(boundingBoxes2.get(indexFromTwoToFour).mul(0.5));

        boundingBoxes2=Nd4j.concat(-1,boundingBoxes2LeftTop,boundingBoxes2RightBottom);

        INDArray boundingBoxesLeftTop= Transforms.max(boundingBoxes1.get(indexFromZeroToTwo),boundingBoxes2.get(indexFromZeroToTwo));

        INDArray boundingBoxesRightBottom=Transforms.min(boundingBoxes1.get(indexFromTwoToFour),boundingBoxes2.get(indexFromTwoToFour));

        INDArray interSection=Transforms.max(boundingBoxesRightBottom.sub(boundingBoxesLeftTop),0.0);

        INDArray interArea=interSection.get(indexZero).mul(interSection.get(indexOne));

        INDArray unionArea=boundingBoxes1Area.add(boundingBoxes2Area).sub(interArea);

        INDArray iou=interArea.mul(1.0).mul(unionArea);

        return iou;
    }


    /**
     * 批量计算两个边界框的GIOU
     *
     * @param boundingBoxes1 [batchSize,gridSize,gridSize,anchorsSize,(x,y,h,w,confidence,classNum)]
     * @param boundingBoxes2
     * @return
     */
    private  static INDArray getGIou(INDArrayIndex[] indexZero,INDArrayIndex[] indexOne,INDArrayIndex[] indexTwo,INDArrayIndex[] indexThree,INDArrayIndex[] indexFromZeroToTwo,INDArrayIndex[] indexFromTwoToFour ,INDArray boundingBoxes1,INDArray boundingBoxes2) {


        //左上角坐标
        INDArray boundingBoxes1LeftTop= boundingBoxes1.get(indexFromZeroToTwo).sub(boundingBoxes1.get(indexFromTwoToFour).mul(0.5));
        //右下角坐标
        INDArray boundingBoxes1RightBottom= boundingBoxes1.get(indexFromZeroToTwo).add(boundingBoxes1.get(indexFromTwoToFour).mul(0.5));
        //转换为(top x,top y,bottom x, bottom y) 格式的bounding box
        boundingBoxes1=Nd4j.concat(-1,boundingBoxes1LeftTop,boundingBoxes1RightBottom);

        INDArray boundingBoxes2LeftTop= boundingBoxes2.get(indexFromZeroToTwo).sub(boundingBoxes2.get(indexFromTwoToFour).mul(0.5));

        INDArray boundingBoxes2RightBottom= boundingBoxes2.get(indexFromZeroToTwo).add(boundingBoxes2.get(indexFromTwoToFour).mul(0.5));
        //转换为(top x,top y,bottom x, bottom y) 格式的bounding box
        boundingBoxes2=Nd4j.concat(-1,boundingBoxes2LeftTop,boundingBoxes2RightBottom);
        //确保左上角的坐标小于右下角的坐标
        boundingBoxes1=Nd4j.concat(-1,Transforms.min(boundingBoxes1.get(indexFromZeroToTwo),boundingBoxes1.get(indexFromTwoToFour)),Transforms.max(boundingBoxes1.get(indexFromZeroToTwo),boundingBoxes1.get(indexFromTwoToFour)));
        //确保左上角的坐标小于右下角的坐标
        boundingBoxes2=Nd4j.concat(-1,Transforms.min(boundingBoxes2.get(indexFromZeroToTwo),boundingBoxes2.get(indexFromTwoToFour)),Transforms.max(boundingBoxes2.get(indexFromZeroToTwo),boundingBoxes2.get(indexFromTwoToFour)));
        //计算第一个边界框的面积
        INDArray boundingBoxes1Area= boundingBoxes1.get(indexTwo).sub(boundingBoxes1.get(indexZero)).mul(boundingBoxes1.get(indexThree).add(boundingBoxes1.get(indexOne)));
        //计算第二个边界框的面积
        INDArray boundingBoxes2Area=boundingBoxes2.get(indexTwo).sub(boundingBoxes2.get(indexZero)).mul(boundingBoxes2.get(indexThree).add(boundingBoxes2.get(indexOne)));

        INDArray boundingBoxesLeftTop= Transforms.max(boundingBoxes1.get(indexFromZeroToTwo),boundingBoxes2.get(indexFromZeroToTwo));

        INDArray boundingBoxesRightBottom=Transforms.min(boundingBoxes1.get(indexFromTwoToFour),boundingBoxes2.get(indexFromTwoToFour));

        INDArray interSection=Transforms.max(boundingBoxesRightBottom.sub(boundingBoxesLeftTop),0.0);

        INDArray interArea=interSection.get(indexZero).mul(interSection.get(indexOne));

        INDArray unionArea=boundingBoxes1Area.add(boundingBoxes2Area).sub(interArea);

        INDArray iou=interArea.mul(1.0).mul(unionArea);
        //计算boxes1和boxes2的最小凸集框的左上角和右下角坐标
        INDArray encloseBoundingBoxesLeftTop= Transforms.min(boundingBoxes1.get(indexFromZeroToTwo),boundingBoxes2.get(indexFromZeroToTwo));

        INDArray encloseBoundingBoxesRightBottom=Transforms.max(boundingBoxes1.get(indexFromTwoToFour),boundingBoxes2.get(indexFromTwoToFour));
        //计算最小凸集的边长
        INDArray enclose=Transforms.max(encloseBoundingBoxesRightBottom.sub(encloseBoundingBoxesLeftTop),0);
        //计算最小凸集的面积
        INDArray encloseArea=enclose.get(indexZero).mul(enclose.get(indexOne));
        //【最小凸集内不属于两个框的区域】与【最小凸集】的比值
        INDArray rate=encloseArea.sub(unionArea).mul(1.0).div(encloseArea);

        INDArray giou=iou.sub(rate);

        return giou;
    }
}
