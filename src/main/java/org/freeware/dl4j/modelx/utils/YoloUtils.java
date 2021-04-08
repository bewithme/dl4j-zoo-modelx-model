package org.freeware.dl4j.modelx.utils;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

@Slf4j
public class YoloUtils {


    //构建SameDiff实例
    public  static  SameDiff sd=SameDiff.create();

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
     * @param predictBoxes
     * @param truthBoxes
     * @return
     */
    public  static INDArray computeIou(INDArray predictBoxes,INDArray truthBoxes) {

        //创建变量x、p
        SDVariable t= sd.var("t");

        SDVariable p=sd.var("p");

        SDVariable zero=sd.var("zero");

        t.setArray(truthBoxes);

        p.setArray(predictBoxes);

        long[] shape=truthBoxes.shape();

        shape[shape.length-1]=2;

        zero.setArray(Nd4j.zeros(shape));

        SDVariable trueBoxesXy=t.get(SameDiffUtils.getLastDimensionPointFromZeroToTwo(shape));

        SDVariable trueBoxesWh=t.get(SameDiffUtils.getLastDimensionPointFromTwoToFour(shape));

        SDVariable trueBoxesWhHalf=trueBoxesWh.div("trueBoxesWhHalf",2);

        SDVariable trueBoxesMin=trueBoxesXy.sub("trueBoxesMin",trueBoxesWhHalf);

        SDVariable trueBoxesMax=trueBoxesXy.add("trueBoxesMax",trueBoxesWhHalf);

        SDVariable predictBoxesXy=p.get(SameDiffUtils.getLastDimensionPointFromZeroToTwo(shape));

        SDVariable predictBoxesWh=p.get(SameDiffUtils.getLastDimensionPointFromTwoToFour(shape));

        SDVariable predictBoxesWhHalf=predictBoxesWh.div("predictBoxesWhHalf",2);

        SDVariable predictBoxesMin=predictBoxesXy.sub("predictBoxesMin",predictBoxesWhHalf);

        SDVariable predictBoxesMax=predictBoxesXy.add("predictBoxesMax",predictBoxesWhHalf);

        SDVariable intersectMin=sd.math().max("intersectMin",trueBoxesMin,predictBoxesMin);

        SDVariable intersectMax=sd.math().min("intersectMax",trueBoxesMax,predictBoxesMax);

        SDVariable intersectWh=sd.math().max(intersectMax.sub(intersectMin),zero);

        SDVariable intersectW=intersectWh.get(SameDiffUtils.getLastDimensionPoint(shape,0));

        SDVariable intersectH=intersectWh.get(SameDiffUtils.getLastDimensionPoint(shape,1));

        SDVariable intersectArea=intersectW.mul(intersectH);

        SDVariable predictBoxesArea=  predictBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,0)).mul(predictBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,1)));

        SDVariable trueBoxesArea=  trueBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,0)).mul(trueBoxesWh.get(SameDiffUtils.getLastDimensionPoint(shape,1)));

        SDVariable iou=intersectArea.div("f",predictBoxesArea.add(trueBoxesArea).sub(intersectArea));

        sd.output(Collections.<String, INDArray>emptyMap(), "f");

        return sd.getArrForVarName("f");
    }


    /**
     * 批量计算两个边界框的GIOU
     * C-AUB/
     * @param predictBoxes [batchSize,gridSize,gridSize,boundingBoxPriorsQuantityPerGridCell,(x,y,w,h,confidence,classNum)]
     * @param truthBoxes
     * @return
     */
    public   static INDArray getGIou(INDArray predictBoxes,INDArray truthBoxes) {

        INDArrayIndex[] indexPointZero=INDArrayUtils.getLastDimensionPointZero(predictBoxes);

        INDArrayIndex[] indexPointOne=INDArrayUtils.getLastDimensionPointOne(predictBoxes);

        INDArrayIndex[] indexPointTwo=INDArrayUtils.getLastDimensionPointTwo(predictBoxes);;

        INDArrayIndex[] indexPointThree=INDArrayUtils.getLastDimensionPointThree(predictBoxes);

        INDArrayIndex[] indexZeroToTwo=INDArrayUtils.getLastDimensionPointFromZeroToTwo(predictBoxes);

        INDArrayIndex[] indexTwoToFour=INDArrayUtils.getLastDimensionPointFromTwoToFour(predictBoxes);

        //转换为(top x,top y,bottom x, bottom y) 格式的bounding box
        predictBoxes=convertToLeftTopBottomRight(predictBoxes);
        //转换为(top x,top y,bottom x, bottom y) 格式的bounding box
        truthBoxes=convertToLeftTopBottomRight(truthBoxes);
        //确保左上角的坐标小于右下角的坐标
        predictBoxes=restrictBoxes(predictBoxes);
        //确保左上角的坐标小于右下角的坐标
        truthBoxes=restrictBoxes(truthBoxes);

        INDArray predictBoxesW= predictBoxes.get(indexPointTwo).sub(predictBoxes.get(indexPointZero));

        INDArray predictBoxesH= predictBoxes.get(indexPointThree).sub(predictBoxes.get(indexPointOne));

        INDArray truthBoxesW=   truthBoxes.get(indexPointTwo).sub(truthBoxes.get(indexPointZero));

        INDArray truthBoxesH=   truthBoxes.get(indexPointThree).sub(truthBoxes.get(indexPointOne));

        //计算第一个边界框的面积
        INDArray predictBoxesArea= predictBoxesW.mul(predictBoxesH);
        //计算第二个边界框的面积
        INDArray truthBoxesArea=truthBoxesW.mul(truthBoxesH);

        INDArray boundingBoxesLeftTop= Transforms.max(predictBoxes.get(indexZeroToTwo),truthBoxes.get(indexZeroToTwo));

        INDArray boundingBoxesRightBottom=Transforms.min(predictBoxes.get(indexTwoToFour),truthBoxes.get(indexTwoToFour));

        INDArray interSection=Transforms.max(boundingBoxesRightBottom.sub(boundingBoxesLeftTop),0.0);

        INDArray interArea=interSection.get(indexPointZero).mul(interSection.get(indexPointOne));

        INDArray unionArea=predictBoxesArea.add(truthBoxesArea).sub(interArea);

        INDArray iou=interArea.div(unionArea.add(1e-6));
        //计算boxes1和boxes2的最小凸集框的左上角和右下角坐标
        INDArray encloseBoundingBoxesLeftTop= Transforms.min(predictBoxes.get(indexZeroToTwo),truthBoxes.get(indexZeroToTwo));

        INDArray encloseBoundingBoxesRightBottom=Transforms.max(predictBoxes.get(indexTwoToFour),truthBoxes.get(indexTwoToFour));
        //计算最小凸集的边长
        INDArray enclose=Transforms.max(encloseBoundingBoxesRightBottom.sub(encloseBoundingBoxesLeftTop),0);
        //计算最小凸集的面积
        INDArray encloseArea=enclose.get(indexPointZero).mul(enclose.get(indexPointOne));
        //【最小凸集内不属于两个框的区域】与【最小凸集】的比值
        INDArray rate=encloseArea.sub(unionArea).mul(1.0).div(encloseArea);

        INDArray gIou=iou.sub(rate);

        return gIou;
    }

    /**
     * x,y,w,h 转换为(top x,top y,bottom x, bottom y) 格式的bounding box
     * @param boxes
     * @return
     */
    private static INDArray convertToLeftTopBottomRight(INDArray boxes){

        INDArrayIndex[] indexZeroToTwo=INDArrayUtils.getLastDimensionPointFromZeroToTwo(boxes);

        INDArrayIndex[] indexTwoToFour=INDArrayUtils.getLastDimensionPointFromTwoToFour(boxes);
        //左上角坐标
        INDArray boxesLeftTop= boxes.get(indexZeroToTwo).sub(boxes.get(indexTwoToFour).mul(0.5));
        //右下角坐标
        INDArray boxesRightBottom= boxes.get(indexZeroToTwo).add(boxes.get(indexTwoToFour).mul(0.5));
        //转换为(top x,top y,bottom x, bottom y) 格式的bounding box
        return Nd4j.concat(-1,boxesLeftTop,boxesRightBottom);
    }

    /**
     * 确保左上角的坐标小于右下角的坐标
     * @param boxes
     * @return
     */
    private static INDArray restrictBoxes(INDArray boxes){

        INDArrayIndex[] indexZeroToTwo=INDArrayUtils.getLastDimensionPointFromZeroToTwo(boxes);

        INDArrayIndex[] indexTwoToFour=INDArrayUtils.getLastDimensionPointFromTwoToFour(boxes);

        return Nd4j.concat(-1,Transforms.min(boxes.get(indexZeroToTwo),boxes.get(indexTwoToFour)),Transforms.max(boxes.get(indexZeroToTwo),boxes.get(indexTwoToFour)));

    }


    /**
     * focal=|labels-predict|^2
     * @param labels
     * @param predict
     * @return
     */
    public static INDArray focal( INDArray labels, INDArray predict){
        float  alpha=1, gamma=2;
        INDArray abs=Transforms.abs(labels.sub(predict));
        return   Transforms.pow(abs,gamma).mul(alpha);
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

        //log.info(sd.getArrForVarName("f").toString());

        Map<String,INDArray> gradients = sd.calculateGradients(null, "t", "p");
        //对x求偏导
        INDArray dLp = gradients.get("p");

        return dLp;
    }


    /**
     * max(x, 0) - x * z + log(1 + exp(-abs(x)))
     * @param labels
     * @param logits
     * @return
     */
    public static INDArray sigmoidCrossEntropyLossWithLogits(INDArray labels, INDArray logits) {

        INDArray z=labels.dup();

        INDArray x=logits.dup();

        INDArray abs=Transforms.abs(x);

        abs=Transforms.neg(abs);

        abs=Transforms.exp(abs);

        abs=abs.add(1);

        return Transforms.max(x,0).sub(x.mul(z)).add(Transforms.log(abs));
    }

    /**
     * max(x, 0) - x * z + log(1 + exp(-abs(x)))
     * SigmoidCrossEntropyLossWithLogits的导数
     * @param labels
     * @param logits
     * @return
     */
    public  static  INDArray gradientOfSigmoidCrossEntropyLossWithLogits(INDArray labels, INDArray logits){
        //创建变量x、z
        SDVariable z= sd.var("z");

        SDVariable x=sd.var("x");

        SDVariable zero=sd.var("zero");

        SDVariable one=sd.var("one");

        zero.setArray(Nd4j.zeros(logits.shape()));

        one.setArray(Nd4j.ones(logits.shape()));

        z.setArray(labels);

        x.setArray(logits);

        SDVariable max=sd.math().max(x, zero);

        SDVariable xz=z.mul(x);

        SDVariable absX=sd.math().abs(x);

        SDVariable negAbsX=sd.math().neg(absX);

        SDVariable expNegAbsX=sd.math().exp(negAbsX);

        SDVariable onePlusExpNegAbsX=one.add(expNegAbsX);

        SDVariable logOnePlusExpNegAbsX=sd.math().log(onePlusExpNegAbsX);

        SDVariable f=max.sub(xz).add("f",logOnePlusExpNegAbsX);

        sd.output(Collections.<String, INDArray>emptyMap(), "f");

        Map<String,INDArray> gradients = sd.calculateGradients(null, "x", "z");
        //对x求偏导
        INDArray dLx = gradients.get("x");

        return dLx;
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

        INDArray x=boxes.get(INDArrayUtils.getLastDimensionPointZero(boxes));

        INDArray y=boxes.get(INDArrayUtils.getLastDimensionPointOne(boxes));

        INDArray w=boxes.get(INDArrayUtils.getLastDimensionPointTwo(boxes));

        INDArray h=boxes.get(INDArrayUtils.getLastDimensionPointThree(boxes));

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
