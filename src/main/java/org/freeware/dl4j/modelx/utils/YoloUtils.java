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
     * @param gridSize 网格大小，有13，26，52
     * @param batchSize 批量大小
     * @param numberOfPriorBoundingBoxPerGridCell 每个单元格负责检测的先验框数量，一般为3
     * @return
     */
    public static  INDArray getCxCy(int gridSize, int batchSize, int numberOfPriorBoundingBoxPerGridCell) {

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
        xy=Nd4j.tile(xy,new int[]{batchSize,1,1,numberOfPriorBoundingBoxPerGridCell,1});

        return xy;
    }





    /**
     * 批量计算两个边界框的IOU
     * 参考https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py
     * 中的IOU实现
     * A∩B/A∪B
     * @param predictBoxes xywh
     * @param truthBoxes xywh
     * @return
     */
    public  static INDArray getIou(INDArray predictBoxes, INDArray truthBoxes) {
        SameDiff sd=SameDiff.create();
        SDVariable iou = computeIou(sd, predictBoxes, truthBoxes);
        sd.output(Collections.<String, INDArray>emptyMap(), "computeIou");
        return sd.getArrForVarName("computeIou");
    }

    public  static INDArray getGradientOfIou(INDArray predictBoxes, INDArray truthBoxes) {
        SameDiff sd=SameDiff.create();
        SDVariable iou = computeIou(sd, predictBoxes, truthBoxes);
        sd.output(Collections.<String, INDArray>emptyMap(), "computeIou");
        Map<String,INDArray> gradients = sd.calculateGradients(null,  "pX","pY","pW","pH");
        //对x求偏导
        INDArray dLx = gradients.get("pX");
        INDArray dLy = gradients.get("pY");
        INDArray dLw = gradients.get("pW");
        INDArray dLh = gradients.get("pH");
        return dLx;

    }

    public  static INDArray getGIou2(INDArray predictBoxes, INDArray truthBoxes) {
        SameDiff sd=SameDiff.create();
        SDVariable iou = computeGIou(sd, predictBoxes, truthBoxes);
        sd.output(Collections.<String, INDArray>emptyMap(), "computeGIou");
        return sd.getArrForVarName("computeGIou");
    }



    private static SDVariable computeIou(SameDiff sd,INDArray predictBoxes, INDArray truthBoxes) {

        long[] shape=truthBoxes.shape();
        INDArray predictBoxesX=predictBoxes.get(INDArrayUtils.getLastDimensionPointZero(predictBoxes.shape()));
        INDArray predictBoxesY=predictBoxes.get(INDArrayUtils.getLastDimensionPointOne(predictBoxes.shape()));
        INDArray predictBoxesW=predictBoxes.get(INDArrayUtils.getLastDimensionPointTwo(predictBoxes.shape()));
        INDArray predictBoxesH=predictBoxes.get(INDArrayUtils.getLastDimensionPointThree(predictBoxes.shape()));

        predictBoxesX=Nd4j.expandDims(predictBoxesX,4);
        predictBoxesY=Nd4j.expandDims(predictBoxesY,4);
        predictBoxesW=Nd4j.expandDims(predictBoxesW,4);
        predictBoxesH=Nd4j.expandDims(predictBoxesH,4);

        INDArray truthBoxesX=truthBoxes.get(INDArrayUtils.getLastDimensionPointZero(truthBoxes.shape()));
        INDArray truthBoxesY=truthBoxes.get(INDArrayUtils.getLastDimensionPointOne(truthBoxes.shape()));
        INDArray truthBoxesW=truthBoxes.get(INDArrayUtils.getLastDimensionPointTwo(truthBoxes.shape()));
        INDArray truthBoxesH=truthBoxes.get(INDArrayUtils.getLastDimensionPointThree(truthBoxes.shape()));

        truthBoxesX=Nd4j.expandDims(truthBoxesX,4);
        truthBoxesY=Nd4j.expandDims(truthBoxesY,4);
        truthBoxesW=Nd4j.expandDims(truthBoxesW,4);
        truthBoxesH=Nd4j.expandDims(truthBoxesH,4);

        //创建变量x、p
        SDVariable tX= sd.var("tX");
        SDVariable tY=sd.var("tY");
        SDVariable tW= sd.var("tW");
        SDVariable tH=sd.var("tH");

        SDVariable pX= sd.var("pX");
        SDVariable pY=sd.var("pY");
        SDVariable pW= sd.var("pW");
        SDVariable pH=sd.var("pH");

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

    private static SDVariable computeGIou(SameDiff sd,INDArray predictBoxes, INDArray truthBoxes) {

        long[] shape=truthBoxes.shape();
        INDArray predictBoxesX=predictBoxes.get(INDArrayUtils.getLastDimensionPointZero(predictBoxes.shape()));
        INDArray predictBoxesY=predictBoxes.get(INDArrayUtils.getLastDimensionPointOne(predictBoxes.shape()));
        INDArray predictBoxesW=predictBoxes.get(INDArrayUtils.getLastDimensionPointTwo(predictBoxes.shape()));
        INDArray predictBoxesH=predictBoxes.get(INDArrayUtils.getLastDimensionPointThree(predictBoxes.shape()));

        predictBoxesX=Nd4j.expandDims(predictBoxesX,4);
        predictBoxesY=Nd4j.expandDims(predictBoxesY,4);
        predictBoxesW=Nd4j.expandDims(predictBoxesW,4);
        predictBoxesH=Nd4j.expandDims(predictBoxesH,4);

        INDArray truthBoxesX=truthBoxes.get(INDArrayUtils.getLastDimensionPointZero(truthBoxes.shape()));
        INDArray truthBoxesY=truthBoxes.get(INDArrayUtils.getLastDimensionPointOne(truthBoxes.shape()));
        INDArray truthBoxesW=truthBoxes.get(INDArrayUtils.getLastDimensionPointTwo(truthBoxes.shape()));
        INDArray truthBoxesH=truthBoxes.get(INDArrayUtils.getLastDimensionPointThree(truthBoxes.shape()));

        truthBoxesX=Nd4j.expandDims(truthBoxesX,4);
        truthBoxesY=Nd4j.expandDims(truthBoxesY,4);
        truthBoxesW=Nd4j.expandDims(truthBoxesW,4);
        truthBoxesH=Nd4j.expandDims(truthBoxesH,4);

        //创建变量x、p
        SDVariable tX= sd.var("tX");
        SDVariable tY=sd.var("tY");
        SDVariable tW= sd.var("tW");
        SDVariable tH=sd.var("tH");

        SDVariable pX= sd.var("pX");
        SDVariable pY=sd.var("pY");
        SDVariable pW= sd.var("pW");
        SDVariable pH=sd.var("pH");

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
     * @param sd
     * @param shape
     * @param trueX
     * @param trueY
     * @param trueW
     * @param trueH
     * @param predictX
     * @param predictY
     * @param predictW
     * @param predictH
     * @return
     */
    public static SDVariable computeIou(SameDiff sd, long[] shape, SDVariable trueX, SDVariable trueY, SDVariable trueW, SDVariable trueH, SDVariable predictX, SDVariable predictY, SDVariable predictW, SDVariable predictH) {

        shape[shape.length-1]=2;

        SDVariable zero=sd.var("iou_zero");

        zero.setArray(Nd4j.zeros(shape));

        SDVariable trueBoxesXy=sd.concat(-1,trueX,trueY);

        SDVariable trueBoxesWh=sd.concat(-1,trueW,trueH);

        SDVariable trueBoxesWhHalf=trueBoxesWh.div(2);

        SDVariable trueBoxesMin=trueBoxesXy.sub(trueBoxesWhHalf);

        SDVariable trueBoxesMax=trueBoxesXy.add(trueBoxesWhHalf);

        SDVariable predictBoxesXy=sd.concat(-1,predictX,predictY);

        SDVariable predictBoxesWh=sd.concat(-1,predictW,predictH);

        SDVariable predictBoxesWhHalf=predictBoxesWh.div(2);

        SDVariable predictBoxesMin=predictBoxesXy.sub(predictBoxesWhHalf);

        SDVariable predictBoxesMax=predictBoxesXy.add(predictBoxesWhHalf);

        SDVariable intersectMin=sd.math().max(trueBoxesMin,predictBoxesMin);

        SDVariable intersectMax=sd.math().min(trueBoxesMax,predictBoxesMax);

        SDVariable intersectWh=sd.math().max(intersectMax.sub(intersectMin),zero);

        SDVariable intersectW=intersectWh.get(SameDiffUtils.getLastDimensionPoint(shape,0));

        SDVariable intersectH=intersectWh.get(SameDiffUtils.getLastDimensionPoint(shape,1));

        SDVariable intersectArea=intersectW.mul(intersectH);

        SDVariable predictBoxesArea=  predictBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,0)).mul(predictBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,1)));

        SDVariable trueBoxesArea=  trueBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,0)).mul(trueBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,1)));

        return intersectArea.div("computeIou",predictBoxesArea.add(trueBoxesArea).sub(intersectArea));
    }


    public static SDVariable computeGIou(SameDiff sd, long[] shape, SDVariable trueX, SDVariable trueY, SDVariable trueW, SDVariable trueH, SDVariable predictX, SDVariable predictY, SDVariable predictW, SDVariable predictH) {

        SDVariable zero=sd.var("gIou_zero");

        zero.setArray(Nd4j.zeros(shape));

        SDIndex[] indexPointZero=SameDiffUtils.getLastDimensionPointZero(shape);

        SDIndex[] indexPointOne=SameDiffUtils.getLastDimensionPointOne(shape);

        SDIndex[] indexPointTwo=SameDiffUtils.getLastDimensionPointTwo(shape);;

        SDIndex[] indexPointThree=SameDiffUtils.getLastDimensionPointThree(shape);

        SDIndex[] indexZeroToTwo=SameDiffUtils.getLastDimensionPointFromZeroToTwo(shape);

        SDIndex[] indexTwoToFour=SameDiffUtils.getLastDimensionPointFromTwoToFour(shape);

        SDVariable predictBoxes=sd.concat(-1,predictX,predictY,predictW,predictH);

        SDVariable truthBoxes=sd.concat(-1,trueX,trueY,trueW,trueH);

        predictBoxes=convertToTopLeftBottomRight(sd,predictBoxes,shape);

        truthBoxes=convertToTopLeftBottomRight(sd,truthBoxes,shape);

        predictBoxes=restrictBoxes(sd,predictBoxes,shape);

        truthBoxes=restrictBoxes(sd,truthBoxes,shape);

        SDVariable predictBoxesW= predictBoxes.get(indexPointTwo).sub(predictBoxes.get(indexPointZero));

        SDVariable predictBoxesH= predictBoxes.get(indexPointThree).sub(predictBoxes.get(indexPointOne));

        SDVariable truthBoxesW=   truthBoxes.get(indexPointTwo).sub(truthBoxes.get(indexPointZero));

        SDVariable truthBoxesH=   truthBoxes.get(indexPointThree).sub(truthBoxes.get(indexPointOne));

        //计算第一个边界框的面积
        SDVariable predictBoxesArea= predictBoxesW.mul(predictBoxesH);
        //计算第二个边界框的面积
        SDVariable truthBoxesArea=truthBoxesW.mul(truthBoxesH);

        SDVariable boundingBoxesLeftTop= sd.math.max(predictBoxes.get(indexZeroToTwo),truthBoxes.get(indexZeroToTwo));

        SDVariable boundingBoxesRightBottom= sd.math.min(predictBoxes.get(indexTwoToFour),truthBoxes.get(indexTwoToFour));

        SDVariable interSection= sd.math.max(boundingBoxesRightBottom.sub(boundingBoxesLeftTop),zero);

        SDVariable interArea=interSection.get(indexPointZero).mul(interSection.get(indexPointOne));

        SDVariable unionArea=predictBoxesArea.add(truthBoxesArea).sub(interArea);

        SDVariable iou=interArea.div(unionArea.add(1e-6));
        //计算boxes1和boxes2的最小凸集框的左上角和右下角坐标
        SDVariable encloseBoundingBoxesLeftTop=  sd.math.min(predictBoxes.get(indexZeroToTwo),truthBoxes.get(indexZeroToTwo));

        SDVariable encloseBoundingBoxesRightBottom= sd.math.max(predictBoxes.get(indexTwoToFour),truthBoxes.get(indexTwoToFour));
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
        SameDiff sd = computeFocal(labels, predict, alpha, gamma);
        return  sd.getArrForVarName("f");
    }

    /**
     * focal的梯度
     *
     * @param labels
     * @param predict
     * @return
     */
    public static INDArray gradientOfOfFocal(INDArray labels, INDArray predict){

        float  alpha=1, gamma=2;

        SameDiff sd = computeFocal(labels, predict, alpha, gamma);

        Map<String,INDArray> gradients = sd.calculateGradients(null, "t", "p");
        //对x求偏导
        INDArray dLp = gradients.get("p");

        return dLp;
    }

    @NotNull
    private static SameDiff computeFocal(INDArray labels, INDArray predict, float alpha, float gamma) {
        //构建SameDiff实例
        SameDiff sd=SameDiff.create();
        //创建变量x、z
        SDVariable t= sd.var("t");

        SDVariable p=sd.var("p");

        t.setArray(labels);

        p.setArray(predict);

        SDVariable tSubB=sd.math().sub(t,p);

        SDVariable absSubB=sd.math().abs(tSubB);

        SDVariable powAbsSubB=sd.math().pow(absSubB,gamma);

        SDVariable f=powAbsSubB.mul("f",alpha);

        sd.output(Collections.<String, INDArray>emptyMap(), "f");

        return sd;
    }


    /**
     * max(x, 0) - x * z + log(1 + exp(-abs(x)))
     * @param labels
     * @param logits
     * @return
     */
    public static INDArray sigmoidCrossEntropyLossWithLogits(INDArray labels, INDArray logits) {

         return computeSigmoidCrossEntropyLossWithLogits(labels, logits).getArrForVarName("f");
    }

    /**
     * max(x, 0) - x * z + log(1 + exp(-abs(x)))
     * SigmoidCrossEntropyLossWithLogits的导数
     * @param labels
     * @param logits
     * @return
     */
    public  static  INDArray gradientOfSigmoidCrossEntropyLossWithLogits(INDArray labels, INDArray logits){

        SameDiff sd=computeSigmoidCrossEntropyLossWithLogits(labels, logits);

        Map<String,INDArray> gradients = sd.calculateGradients(null, "x", "z");
        //对x求偏导
        INDArray dLx = gradients.get("x");

        return dLx;
    }

    private static SameDiff computeSigmoidCrossEntropyLossWithLogits(INDArray labels, INDArray logits) {

        //构建SameDiff实例
        SameDiff sd=SameDiff.create();
        //创建变量x、z
        SDVariable z= sd.var("z");

        SDVariable x=sd.var("x");

        z.setArray(labels);

        x.setArray(logits);

        SDVariable zero=sd.var("zero");

        SDVariable one=sd.var("one");

        zero.setArray(Nd4j.zeros(logits.shape()));

        one.setArray(Nd4j.ones(logits.shape()));

        SDVariable max=sd.math().max(x, zero);

        SDVariable xz=z.mul(x);

        SDVariable absX=sd.math().abs(x);

        SDVariable negAbsX=sd.math().neg(absX);

        SDVariable expNegAbsX=sd.math().exp(negAbsX);

        SDVariable onePlusExpNegAbsX=one.add(expNegAbsX);

        SDVariable logOnePlusExpNegAbsX=sd.math().log(onePlusExpNegAbsX);

        SDVariable f=max.sub(xz).add("f",logOnePlusExpNegAbsX);

        sd.output(Collections.<String, INDArray>emptyMap(), "f");

        return sd;
    }


    public static INDArray derivativeOfIou(INDArray predictBoxes,INDArray truthBoxes,Boolean gIou){

        INDArray predictBoxesTopBottomLeftRight=convertToTopBottomLeftRight(predictBoxes);

        INDArray truthBoxesTopBottomLeftRight=convertToTopBottomLeftRight(truthBoxes);

        long[] shape=predictBoxesTopBottomLeftRight.shape();

        long batchSize=shape[0];

        long gridWidth=shape[1];

        long gridHeight=shape[2];

        long numberOfPriorBoundingBoxPerGridCell=shape[3];

        INDArray derivative=Nd4j.zeros(new long[]{batchSize,gridWidth,gridHeight,numberOfPriorBoundingBoxPerGridCell,4});

        for(int exampleIndex=0;exampleIndex<batchSize;exampleIndex++){

            for(int gridWidthIndex=0;gridWidthIndex<gridWidth;gridWidthIndex++){

                for(int gridHeightIndex=0;gridHeightIndex<gridHeight;gridHeightIndex++){

                    for(int priorBoundingBoxIndex=0;priorBoundingBoxIndex<numberOfPriorBoundingBoxPerGridCell;priorBoundingBoxIndex++){

                        INDArray  singlePredictBoxes=predictBoxesTopBottomLeftRight.get(getSinglePredictBoxesIndexes(exampleIndex, gridWidthIndex, gridHeightIndex, priorBoundingBoxIndex));

                        INDArray  singleTruthBoxes=truthBoxesTopBottomLeftRight.get(getSinglePredictBoxesIndexes(exampleIndex, gridWidthIndex, gridHeightIndex, priorBoundingBoxIndex));

                        float[] singlePredictBoxesArray=singlePredictBoxes.toFloatVector();

                        float[] singleTruthBoxesArray=singleTruthBoxes.toFloatVector();

                        float pred_tblr_top=singlePredictBoxesArray[0];

                        float pred_tblr_bot=singlePredictBoxesArray[1];

                        float pred_tblr_left=singlePredictBoxesArray[2];

                        float pred_tblr_right=singlePredictBoxesArray[3];

                        float truth_tblr_top=singleTruthBoxesArray[0];

                        float truth_tblr_bot=singleTruthBoxesArray[1];

                        float truth_tblr_left=singleTruthBoxesArray[2];

                        float truth_tblr_right=singleTruthBoxesArray[3];

                        float[] dxs=dx_box_iou(pred_tblr_top,pred_tblr_bot,pred_tblr_left,pred_tblr_right,truth_tblr_top,truth_tblr_bot,truth_tblr_left,truth_tblr_right,gIou);

                        INDArray dxsArray=Nd4j.create(dxs).reshape(new int []{1,4});

                        derivative.put(getDerivativeOfIndexes(exampleIndex, gridWidthIndex, gridHeightIndex, priorBoundingBoxIndex),dxsArray);

                    }

                }


            }

        }



        return derivative;
    }


    private static INDArrayIndex[] getSinglePredictBoxesIndexes(int exampleIndex, int gridWidthIndex, int gridHeightIndex, int priorBoundingBoxIndex) {
        return new INDArrayIndex[]{NDArrayIndex.point(exampleIndex),NDArrayIndex.point(gridWidthIndex),NDArrayIndex.point(gridHeightIndex),NDArrayIndex.point(priorBoundingBoxIndex),NDArrayIndex.interval(0,4)};
    }

    private static INDArrayIndex[] getDerivativeOfIndexes(int exampleIndex, int gridWidthIndex, int gridHeightIndex, int priorBoundingBoxIndex) {
        return new INDArrayIndex[]{NDArrayIndex.point(exampleIndex),NDArrayIndex.point(gridWidthIndex),NDArrayIndex.point(gridHeightIndex),NDArrayIndex.point(priorBoundingBoxIndex),NDArrayIndex.all()};
    }

    private static float[] dx_box_iou(float pred_tblr_top,float pred_tblr_bot,float pred_tblr_left,float pred_tblr_right,float truth_tblr_top,float truth_tblr_bot,float truth_tblr_left,float truth_tblr_right, Boolean iou_loss) {

        float pred_t = Math.min(pred_tblr_top, pred_tblr_bot);
        float pred_b = Math.max(pred_tblr_top, pred_tblr_bot);
        float pred_l = Math.min(pred_tblr_left, pred_tblr_right);
        float pred_r = Math.max(pred_tblr_left, pred_tblr_right);

        float X = (pred_b - pred_t) * (pred_r - pred_l);
        float Xhat = (truth_tblr_bot - truth_tblr_top) * (truth_tblr_right - truth_tblr_left);
        float Ih = Math.min(pred_b, truth_tblr_bot) -   Math.max(pred_t, truth_tblr_top);
        float Iw = Math.min(pred_r, truth_tblr_right) - Math.max(pred_l, truth_tblr_left);
        float I = Iw * Ih;
        float U = X + Xhat - I;

        float Cw = Math.max(pred_r, truth_tblr_right) - Math.min(pred_l, truth_tblr_left);
        float Ch = Math.max(pred_b, truth_tblr_bot) - Math.min(pred_t, truth_tblr_top);
        float C = Cw * Ch;

        // float IoU = I / U;
        // Partial Derivatives, derivatives
        float dX_wrt_t = -1 * (pred_r - pred_l);
        float dX_wrt_b = pred_r - pred_l;
        float dX_wrt_l = -1 * (pred_b - pred_t);
        float dX_wrt_r = pred_b - pred_t;

        // gradient of I min/max in IoU calc (prediction)
        float dI_wrt_t = pred_t > truth_tblr_top ? (-1 * Iw) : 0;
        float dI_wrt_b = pred_b < truth_tblr_bot ? Iw : 0;
        float dI_wrt_l = pred_l > truth_tblr_left ? (-1 * Ih) : 0;
        float dI_wrt_r = pred_r < truth_tblr_right ? Ih : 0;
        // derivative of U with regard to x
        float dU_wrt_t = dX_wrt_t - dI_wrt_t;
        float dU_wrt_b = dX_wrt_b - dI_wrt_b;
        float dU_wrt_l = dX_wrt_l - dI_wrt_l;
        float dU_wrt_r = dX_wrt_r - dI_wrt_r;
        // gradient of C min/max in IoU calc (prediction)
        float dC_wrt_t = pred_t < truth_tblr_top ? (-1 * Cw) : 0;
        float dC_wrt_b = pred_b > truth_tblr_bot ? Cw : 0;
        float dC_wrt_l = pred_l < truth_tblr_left ? (-1 * Ch) : 0;
        float dC_wrt_r = pred_r > truth_tblr_right ? Ch : 0;

        // Final IOU loss (prediction) (negative of IOU gradient, we want the negative loss)
        float p_dt = 0;
        float p_db = 0;
        float p_dl = 0;
        float p_dr = 0;
        if (U > 0) {
            p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
            p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
            p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
            p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
        }

        if (iou_loss == Boolean.TRUE) {
            if (C > 0) {
                // apply "C" term from gIOU
                p_dt += ((C * dU_wrt_t) - (U * dC_wrt_t)) / (C * C);
                p_db += ((C * dU_wrt_b) - (U * dC_wrt_b)) / (C * C);
                p_dl += ((C * dU_wrt_l) - (U * dC_wrt_l)) / (C * C);
                p_dr += ((C * dU_wrt_r) - (U * dC_wrt_r)) / (C * C);
            }
        }
        float dx_dt = pred_tblr_top < pred_tblr_bot ? p_dt : p_db;
        float dx_db = pred_tblr_top < pred_tblr_bot ? p_db : p_dt;
        float dx_dl = pred_tblr_left < pred_tblr_right ? p_dl : p_dr;
        float dx_dr = pred_tblr_left < pred_tblr_right ? p_dr : p_dl;

        float w=dx_dr-dx_dl;

        float h=dx_dt-dx_db;

        float x=dx_dl+w/2;

        float y=dx_dt+h/2;

        //return new float[]{dx_dt,dx_db,dx_dl,dx_dr};

        return new float[]{x,y,w,h};
    }



    private static INDArray convertToTopBottomLeftRight(INDArray boxes){

        INDArray x=boxes.get(INDArrayUtils.getLastDimensionPointZero(boxes.shape()));

        INDArray y=boxes.get(INDArrayUtils.getLastDimensionPointOne(boxes.shape()));

        INDArray w=boxes.get(INDArrayUtils.getLastDimensionPointTwo(boxes.shape()));

        INDArray h=boxes.get(INDArrayUtils.getLastDimensionPointThree(boxes.shape()));

        x=Nd4j.expandDims(x,4);

        y=Nd4j.expandDims(y,4);

        w=Nd4j.expandDims(w,4);

        h=Nd4j.expandDims(h,4);

        INDArray t=y.sub(h.div(2));

        INDArray b=y.add(h.div(2));

        INDArray l=x.sub(w.div(2));

        INDArray r=x.add(w.div(2));

        INDArray retArray=Nd4j.concat(-1,t,b,l,r);

        return retArray;
    }

}
