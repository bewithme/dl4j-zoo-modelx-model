package org.freeware.dl4j.modelx.utils;

import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.Map;

@Slf4j
public class YoloUtils {


    /**
     * 获取每个单元格相对于最左上角的坐标
     * 输出形状为[batchSize,gridSize,gridSize,anchorQuantityPerGrid,2]
     * 最后一个维度用来存当前单元格相对于左上角的坐标(Cx,Cy)
     *
     * @param gridSize                            网格大小，有13，26，52
     * @param batchSize                           批量大小
     * @param numberOfPriorBoundingBoxPerGridCell 每个单元格负责检测的先验框数量，一般为3
     * @return
     */
    public static INDArray getCxCy(int gridSize, int batchSize, int numberOfPriorBoundingBoxPerGridCell) {

        //创建一个元素为0到gridSize-1一维数组
        INDArray gridCoordinatePoints = Nd4j.linspace(0, gridSize - 1, gridSize);
        //将形状为[1,gridSize]的数组在[gridSize,1]形状上平铺
        INDArray x = Nd4j.tile(gridCoordinatePoints.reshape(new int[]{1, gridSize}), new int[]{gridSize, 1});

        INDArray y = Nd4j.tile(gridCoordinatePoints.reshape(new int[]{gridSize, 1}), new int[]{1, gridSize});
        //[gridSize,gridSize]-->[gridSize,gridSize,1]
        x = Nd4j.expandDims(x, 2);
        //[gridSize,gridSize]-->[gridSize,gridSize,1]
        y = Nd4j.expandDims(y, 2);
        //[gridSize,gridSize,1]-->[gridSize,gridSize,2]
        INDArray xy = Nd4j.concat(-1, x, y);
        //[gridSize,gridSize,2]-->[1,gridSize,gridSize,2]
        xy = Nd4j.expandDims(xy, 0);
        //[1,gridSize,gridSize,2]-->[1,gridSize,gridSize,1,2]
        xy = Nd4j.expandDims(xy, 3);
        //[1,gridSize,gridSize,1,2]-->[batchSize,gridSize,gridSize,anchorQuantityPerGrid,2]
        xy = Nd4j.tile(xy, new int[]{batchSize, 1, 1, numberOfPriorBoundingBoxPerGridCell, 1});

        return xy;
    }


    /**
     * 批量计算两个边界框的IOU
     * 参考https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py
     * 中的IOU实现
     * A∩B/A∪B
     *
     * @param predictBoxes xywh
     * @param truthBoxes   xywh
     * @return
     */
    public static INDArray getIou(INDArray predictBoxes, INDArray truthBoxes) {
        SameDiff sd = SameDiff.create();
        SDVariable iou = computeIou(sd, predictBoxes, truthBoxes);
        sd.output(Collections.<String, INDArray>emptyMap(), "computeIou");
        return sd.getArrForVarName("computeIou");
    }

    public static INDArray getGradientOfIou(INDArray predictBoxes, INDArray truthBoxes) {
        SameDiff sd = SameDiff.create();
        SDVariable iou = computeIou(sd, predictBoxes, truthBoxes);
        sd.output(Collections.<String, INDArray>emptyMap(), "computeIou");
        Map<String, INDArray> gradients = sd.calculateGradients(null, "pX", "pY", "pW", "pH");
        //对x求偏导
        INDArray dLx = gradients.get("pX");
        INDArray dLy = gradients.get("pY");
        INDArray dLw = gradients.get("pW");
        INDArray dLh = gradients.get("pH");
        return dLx;
    }

    public static INDArray getGIou(INDArray predictBoxes, INDArray truthBoxes) {
        SameDiff sd = SameDiff.create();
        SDVariable iou = computeGIou(sd, predictBoxes, truthBoxes);
        sd.output(Collections.<String, INDArray>emptyMap(), "computeGIou");
        return sd.getArrForVarName("computeGIou");
    }



    private static SDVariable computeIou(SameDiff sd, INDArray predictBoxes, INDArray truthBoxes) {

        long[] shape = truthBoxes.shape();

        INDArray predictBoxesX = predictBoxes.get(INDArrayUtils.getLastDimensionPointZero(predictBoxes.shape()));
        INDArray predictBoxesY = predictBoxes.get(INDArrayUtils.getLastDimensionPointOne(predictBoxes.shape()));
        INDArray predictBoxesW = predictBoxes.get(INDArrayUtils.getLastDimensionPointTwo(predictBoxes.shape()));
        INDArray predictBoxesH = predictBoxes.get(INDArrayUtils.getLastDimensionPointThree(predictBoxes.shape()));
        predictBoxesX = Nd4j.expandDims(predictBoxesX, shape.length-1);
        predictBoxesY = Nd4j.expandDims(predictBoxesY, shape.length-1);
        predictBoxesW = Nd4j.expandDims(predictBoxesW, shape.length-1);
        predictBoxesH = Nd4j.expandDims(predictBoxesH, shape.length-1);
        INDArray truthBoxesX = truthBoxes.get(INDArrayUtils.getLastDimensionPointZero(truthBoxes.shape()));
        INDArray truthBoxesY = truthBoxes.get(INDArrayUtils.getLastDimensionPointOne(truthBoxes.shape()));
        INDArray truthBoxesW = truthBoxes.get(INDArrayUtils.getLastDimensionPointTwo(truthBoxes.shape()));
        INDArray truthBoxesH = truthBoxes.get(INDArrayUtils.getLastDimensionPointThree(truthBoxes.shape()));
        truthBoxesX = Nd4j.expandDims(truthBoxesX, shape.length-1);
        truthBoxesY = Nd4j.expandDims(truthBoxesY, shape.length-1);
        truthBoxesW = Nd4j.expandDims(truthBoxesW, shape.length-1);
        truthBoxesH = Nd4j.expandDims(truthBoxesH, shape.length-1);

        //创建变量x、p
        SDVariable tX = sd.var("tX");
        SDVariable tY = sd.var("tY");
        SDVariable tW = sd.var("tW");
        SDVariable tH = sd.var("tH");

        SDVariable pX = sd.var("pX");
        SDVariable pY = sd.var("pY");
        SDVariable pW = sd.var("pW");
        SDVariable pH = sd.var("pH");

        tX.setArray(truthBoxesX);
        tY.setArray(truthBoxesY);
        tW.setArray(truthBoxesW);
        tH.setArray(truthBoxesH);

        pX.setArray(predictBoxesX);
        pY.setArray(predictBoxesY);
        pW.setArray(predictBoxesW);
        pH.setArray(predictBoxesH);

        SDVariable iou = computeIou(sd, shape, tX, tY, tW, tH, pX, pY, pW, pH);


        return iou;


    }

    private static SDVariable computeGIou(SameDiff sd, INDArray predictBoxes, INDArray truthBoxes) {

        long[] shape = truthBoxes.shape();

        INDArray predictBoxesX = predictBoxes.get(INDArrayUtils.getLastDimensionPointZero(predictBoxes.shape()));
        INDArray predictBoxesY = predictBoxes.get(INDArrayUtils.getLastDimensionPointOne(predictBoxes.shape()));
        INDArray predictBoxesW = predictBoxes.get(INDArrayUtils.getLastDimensionPointTwo(predictBoxes.shape()));
        INDArray predictBoxesH = predictBoxes.get(INDArrayUtils.getLastDimensionPointThree(predictBoxes.shape()));
        predictBoxesX = Nd4j.expandDims(predictBoxesX, shape.length-1);
        predictBoxesY = Nd4j.expandDims(predictBoxesY, shape.length-1);
        predictBoxesW = Nd4j.expandDims(predictBoxesW, shape.length-1);
        predictBoxesH = Nd4j.expandDims(predictBoxesH, shape.length-1);
        INDArray truthBoxesX = truthBoxes.get(INDArrayUtils.getLastDimensionPointZero(truthBoxes.shape()));
        INDArray truthBoxesY = truthBoxes.get(INDArrayUtils.getLastDimensionPointOne(truthBoxes.shape()));
        INDArray truthBoxesW = truthBoxes.get(INDArrayUtils.getLastDimensionPointTwo(truthBoxes.shape()));
        INDArray truthBoxesH = truthBoxes.get(INDArrayUtils.getLastDimensionPointThree(truthBoxes.shape()));
        truthBoxesX = Nd4j.expandDims(truthBoxesX, shape.length-1);
        truthBoxesY = Nd4j.expandDims(truthBoxesY, shape.length-1);
        truthBoxesW = Nd4j.expandDims(truthBoxesW, shape.length-1);
        truthBoxesH = Nd4j.expandDims(truthBoxesH, shape.length-1);

        //创建变量x、p
        SDVariable tX = sd.var("tX");
        SDVariable tY = sd.var("tY");
        SDVariable tW = sd.var("tW");
        SDVariable tH = sd.var("tH");

        SDVariable pX = sd.var("pX");
        SDVariable pY = sd.var("pY");
        SDVariable pW = sd.var("pW");
        SDVariable pH = sd.var("pH");

        tX.setArray(truthBoxesX);
        tY.setArray(truthBoxesY);
        tW.setArray(truthBoxesW);
        tH.setArray(truthBoxesH);

        pX.setArray(predictBoxesX);
        pY.setArray(predictBoxesY);
        pW.setArray(predictBoxesW);
        pH.setArray(predictBoxesH);

        SDVariable iou = computeGIou(sd, shape, tX, tY, tW, tH, pX, pY, pW, pH);
        return iou;
    }




    /**
     * 计算IOU
     *
     * @param sd
     * @param shape
     * @param labelX
     * @param labelY
     * @param labelW
     * @param labelH
     * @param predictX
     * @param predictY
     * @param predictW
     * @param predictH
     * @return
     */
    public static SDVariable computeIou(SameDiff sd, long[] shape, SDVariable labelX, SDVariable labelY, SDVariable labelW, SDVariable labelH, SDVariable predictX, SDVariable predictY, SDVariable predictW, SDVariable predictH) {

        SDVariable labelBoxesXy = sd.concat(-1, labelX, labelY);

        SDVariable labelBoxesWh = sd.concat(-1, labelW, labelH);

        SDVariable predictBoxesXy = sd.concat(-1, predictX, predictY);

        SDVariable predictBoxesWh = sd.concat(-1, predictW, predictH);

        return computeIou(sd, shape, labelBoxesXy, labelBoxesWh, predictBoxesXy, predictBoxesWh);
    }

    public static SDVariable computeIou(SameDiff sd, long[] shape, SDVariable labelBoxesXyWh, SDVariable predictBoxesXyWh){

        SDVariable labelBoxesXy = labelBoxesXyWh.get(SameDiffUtils.getLastDimensionPointFromZeroToTwo(shape));

        SDVariable labelBoxesWh =labelBoxesXyWh.get(SameDiffUtils.getLastDimensionPointFromTwoToFour(shape));

        SDVariable predictBoxesXy = predictBoxesXyWh.get(SameDiffUtils.getLastDimensionPointFromZeroToTwo(shape));

        SDVariable predictBoxesWh = predictBoxesXyWh.get(SameDiffUtils.getLastDimensionPointFromTwoToFour(shape));

        return computeIou(sd,shape,labelBoxesXy,labelBoxesWh,predictBoxesXy,predictBoxesWh);
    }

    public static SDVariable computeIou(SameDiff sd, long[] shape, SDVariable labelBoxesXy, SDVariable labelBoxesWh, SDVariable predictBoxesXy, SDVariable predictBoxesWh) {

        shape[shape.length-1]=2;

        SDVariable zero=sd.var(Nd4j.zeros(shape));

        SDVariable labelBoxesWhHalf=labelBoxesWh.div(2);

        SDVariable labelBoxesMin=labelBoxesXy.sub(labelBoxesWhHalf);

        SDVariable labelBoxesMax=labelBoxesXy.add(labelBoxesWhHalf);

        SDVariable predictBoxesWhHalf=predictBoxesWh.div(2);

        SDVariable predictBoxesMin=predictBoxesXy.sub(predictBoxesWhHalf);

        SDVariable predictBoxesMax=predictBoxesXy.add(predictBoxesWhHalf);

        SDVariable intersectMin=sd.math().max(labelBoxesMin,predictBoxesMin);

        SDVariable intersectMax=sd.math().min(labelBoxesMax,predictBoxesMax);

        SDVariable intersectWh=sd.math().max(intersectMax.sub(intersectMin),zero);

        SDVariable intersectW=intersectWh.get(SameDiffUtils.getLastDimensionPoint(shape,0));

        SDVariable intersectH=intersectWh.get(SameDiffUtils.getLastDimensionPoint(shape,1));

        SDVariable intersectArea=intersectW.mul(intersectH);

        SDVariable predictBoxesArea=  predictBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,0)).mul(predictBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,1)));

        SDVariable trueBoxesArea=  labelBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,0)).mul(labelBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,1)));

        return intersectArea.div("computeIou",predictBoxesArea.add(trueBoxesArea).sub(intersectArea));
    }


    public static SDVariable computeGIou(SameDiff sd, long[] shape, SDVariable labelX, SDVariable labelY, SDVariable labelW, SDVariable labelH, SDVariable predictX, SDVariable predictY, SDVariable predictW, SDVariable predictH) {
        //因为这个
        shape[shape.length-1]=2;

        SDVariable zero=sd.var(Nd4j.zeros(shape));

        SDIndex[] indexPointZero=SameDiffUtils.getLastDimensionPointZero(shape);

        SDIndex[] indexPointOne=SameDiffUtils.getLastDimensionPointOne(shape);

        SDIndex[] indexPointTwo=SameDiffUtils.getLastDimensionPointTwo(shape);;

        SDIndex[] indexPointThree=SameDiffUtils.getLastDimensionPointThree(shape);

        SDIndex[] indexZeroToTwo=SameDiffUtils.getLastDimensionPointFromZeroToTwo(shape);

        SDIndex[] indexTwoToFour=SameDiffUtils.getLastDimensionPointFromTwoToFour(shape);

        SDVariable predictBoxes=sd.concat(-1,predictX,predictY,predictW,predictH);

        SDVariable labelBoxes=sd.concat(-1,labelX,labelY,labelW,labelH);

        predictBoxes=convertToTopLeftBottomRight(sd,predictBoxes,shape);

        labelBoxes=convertToTopLeftBottomRight(sd,labelBoxes,shape);

        predictBoxes=restrictBoxes(sd,predictBoxes,shape);

        labelBoxes=restrictBoxes(sd,labelBoxes,shape);

        SDVariable predictBoxesW= predictBoxes.get(indexPointTwo).sub(predictBoxes.get(indexPointZero));

        SDVariable predictBoxesH= predictBoxes.get(indexPointThree).sub(predictBoxes.get(indexPointOne));

        SDVariable labelBoxesW=   labelBoxes.get(indexPointTwo).sub(labelBoxes.get(indexPointZero));

        SDVariable labelBoxesH=   labelBoxes.get(indexPointThree).sub(labelBoxes.get(indexPointOne));

        //计算第一个边界框的面积
        SDVariable predictBoxesArea= predictBoxesW.mul(predictBoxesH);
        //计算第二个边界框的面积
        SDVariable labelBoxesArea=labelBoxesW.mul(labelBoxesH);

        SDVariable boundingBoxesLeftTop= sd.math.max(predictBoxes.get(indexZeroToTwo),labelBoxes.get(indexZeroToTwo));

        SDVariable boundingBoxesRightBottom= sd.math.min(predictBoxes.get(indexTwoToFour),labelBoxes.get(indexTwoToFour));

        SDVariable interSection= sd.math.max(boundingBoxesRightBottom.sub(boundingBoxesLeftTop),zero);

        SDVariable interArea=interSection.get(indexPointZero).mul(interSection.get(indexPointOne));

        SDVariable unionArea=predictBoxesArea.add(labelBoxesArea).sub(interArea);

        SDVariable iou=interArea.div(unionArea.add(1e-6));
        //计算boxes1和boxes2的最小凸集框的左上角和右下角坐标
        SDVariable encloseBoundingBoxesLeftTop=  sd.math.min(predictBoxes.get(indexZeroToTwo),labelBoxes.get(indexZeroToTwo));

        SDVariable encloseBoundingBoxesRightBottom= sd.math.max(predictBoxes.get(indexTwoToFour),labelBoxes.get(indexTwoToFour));
        //计算最小凸集的边长
        SDVariable enclose= sd.math.max(encloseBoundingBoxesRightBottom.sub(encloseBoundingBoxesLeftTop),zero);
        //计算最小凸集的面积
        SDVariable encloseArea=enclose.get(indexPointZero).mul(enclose.get(indexPointOne));
        //【最小凸集内不属于两个框的区域】与【最小凸集】的比值
        SDVariable rate=encloseArea.sub(unionArea).mul(1.0).div(encloseArea);

        SDVariable gIou=iou.sub("computeGIou",rate);

        return gIou;

    }


    /**
     * x,y,w,h 转换为(top x,top y,bottom x, bottom y) 格式
     * @param sd
     * @param boxes
     * @param shape
     * @return
     */
    private  static  SDVariable convertToTopLeftBottomRight(SameDiff sd,SDVariable boxes,long[] shape){

        SDIndex[] indexZeroToTwo=SameDiffUtils.getLastDimensionPointFromZeroToTwo(shape);

        SDIndex[] indexTwoToFour=SameDiffUtils.getLastDimensionPointFromTwoToFour(shape);

        SDVariable topLeft= boxes.get(indexZeroToTwo).sub(boxes.get(indexTwoToFour).mul(0.5));

        SDVariable rightBottom= boxes.get(indexZeroToTwo).add(boxes.get(indexTwoToFour).mul(0.5));

        return sd.concat(-1,topLeft,rightBottom);
    }

    /**
     * 确保左上角的坐标小于右下角的坐标
     * @param boxes
     * @return
     */
    private static SDVariable restrictBoxes(SameDiff sd,SDVariable boxes,long[] shape){

        SDIndex[] indexZeroToTwo=SameDiffUtils.getLastDimensionPointFromZeroToTwo(shape);

        SDIndex[] indexTwoToFour=SameDiffUtils.getLastDimensionPointFromTwoToFour(shape);

        SDVariable min= sd.math().min(boxes.get(indexZeroToTwo),boxes.get(indexTwoToFour));

        SDVariable max= sd.math().max(boxes.get(indexZeroToTwo),boxes.get(indexTwoToFour));

        return sd.concat(-1,min,max);
    }




    /**
     * focal=|labels-predict|^2
     * @param labels
     * @param predict
     * @return
     */
    public static INDArray focal( INDArray labels, INDArray predict){
        float  alpha=1, gamma=2;
        //构建SameDiff实例
        SameDiff sd=SameDiff.create();
        computeFocal(sd,labels, predict, alpha, gamma);
        return  sd.getArrForVarName("computeFocal");
    }

    /**
     * focal的梯度
     *
     * @param truth
     * @param predict
     * @return
     */
    public static INDArray gradientOfOfFocal(INDArray truth, INDArray predict){

        float  alpha=1, gamma=2;
        //构建SameDiff实例
        SameDiff sd=SameDiff.create();

        computeFocal(sd,truth, predict, alpha, gamma);

        Map<String,INDArray> gradients = sd.calculateGradients(null, "t", "p");
        //对x求偏导
        INDArray dLp = gradients.get("p");

        return dLp;
    }

    @NotNull
    public static SDVariable computeFocal(  SameDiff sd,INDArray labels, INDArray predict, float alpha, float gamma) {

        //创建变量x、z
        SDVariable t= sd.var("t");

        SDVariable p=sd.var("p");

        t.setArray(labels);

        p.setArray(predict);

        SDVariable f=computeFocal(alpha, gamma, sd, t, p);

        sd.output(Collections.<String, INDArray>emptyMap(), "computeFocal");

        return f;
    }

    public static SDVariable computeFocal(float alpha, float gamma, SameDiff sd, SDVariable t, SDVariable p) {

        SDVariable tSubB=sd.math().sub(t,p);

        SDVariable absSubB=sd.math().abs(tSubB);

        SDVariable powAbsSubB=sd.math().pow(absSubB,gamma);

        SDVariable f=powAbsSubB.mul("computeFocal",alpha);

        return f;
    }


    /**
     * max(x, 0) - x * z + log(1 + exp(-abs(x)))
     * @param labels
     * @param logits
     * @return
     */
    public static INDArray sigmoidCrossEntropyLossWithLogits(INDArray labels, INDArray logits) {
         SameDiff sd=SameDiff.create();
         computeSigmoidCrossEntropyLossWithLogits(sd,labels, logits);
         return sd.getArrForVarName("computeSigmoidCrossEntropyLossWithLogits");
    }

    /**
     * max(x, 0) - x * z + log(1 + exp(-abs(x)))
     * SigmoidCrossEntropyLossWithLogits的导数
     * @param labels
     * @param logits
     * @return
     */
    public  static  INDArray gradientOfSigmoidCrossEntropyLossWithLogits(INDArray labels, INDArray logits){
        //构建SameDiff实例
        SameDiff sd=SameDiff.create();

        computeSigmoidCrossEntropyLossWithLogits(sd,labels, logits);

        Map<String,INDArray> gradients = sd.calculateGradients(null, "x", "z");
        //对x求偏导
        INDArray dLx = gradients.get("x");

        return dLx;
    }

    public static SDVariable computeSigmoidCrossEntropyLossWithLogits(SameDiff sd,INDArray labels, INDArray logits) {

        long[] shape=logits.shape();

        //创建变量x、z
        SDVariable z= sd.var("z");

        SDVariable x=sd.var("x");

        z.setArray(labels);

        x.setArray(logits);

        SDVariable f= computeSigmoidCrossEntropyLossWithLogits(shape, sd, z, x);

        sd.output(Collections.<String, INDArray>emptyMap(), "computeSigmoidCrossEntropyLossWithLogits");

        return f;
    }

    /**
     *
     * @param shape
     * @param sd
     * @param z labels
     * @param x logits
     * @return
     */
    public static SDVariable computeSigmoidCrossEntropyLossWithLogits(long[] shape, SameDiff sd, SDVariable z, SDVariable x) {

        SDVariable zero=sd.var(Nd4j.zeros(shape));

        SDVariable one=sd.var(Nd4j.ones(shape));

        SDVariable max=sd.math().max(x, zero);

        SDVariable xz=z.mul(x);

        SDVariable absX=sd.math().abs(x);

        SDVariable negAbsX=sd.math().neg(absX);

        SDVariable expNegAbsX=sd.math().exp(negAbsX);

        SDVariable onePlusExpNegAbsX=one.add(expNegAbsX);

        SDVariable logOnePlusExpNegAbsX=sd.math().log(onePlusExpNegAbsX);

        SDVariable f=max.sub(xz).add(logOnePlusExpNegAbsX);

        return f;
    }








}
