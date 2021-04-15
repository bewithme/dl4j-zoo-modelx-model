package org.freeware.dl4j.modelx.model.yolo;

import lombok.*;
import lombok.extern.slf4j.Slf4j;

import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.freeware.dl4j.modelx.utils.INDArrayUtils;
import org.freeware.dl4j.modelx.utils.YoloUtils;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;


@Slf4j
public class Yolo3OutputLayer extends AbstractLayer<Yolo3OutputLayerConfiguration> implements Serializable, IOutputLayer {

    private static final Gradient EMPTY_GRADIENT = new DefaultGradient();
    @Setter @Getter
    protected INDArray labels;

    public Yolo3OutputLayer(NeuralNetConfiguration conf, DataType networkDataType) {
        super(conf, networkDataType);
    }

    @Override
    public boolean needsLabels() {
        return false;
    }



    @Override
    public double computeScore(double fullNetworkRegScore, boolean training, LayerWorkspaceMgr workspaceMgr) {

        LossSameDiff lossSameDiff=computeLoss();

        INDArray gIouLoss=lossSameDiff.getGIouLossSameDiff().getArrForVarName("computeGIouLoss");
        //shape=[batchSize]
        gIouLoss=gIouLoss.sum(1,2,3,4);
        //置信度损失
        INDArray confidenceLoss =lossSameDiff.getConfidenceLossSameDiff().getArrForVarName("computeConfidenceLoss");
        //shape=[batchSize]
        confidenceLoss=confidenceLoss.sum(1,2,3,4);

        INDArray classLoss=lossSameDiff.getClassLossSameDiff().getArrForVarName("computeClassLoss");
        //shape=[batchSize]
        classLoss=classLoss.sum(1,2,3,4);
        //shape=1
        gIouLoss=gIouLoss.mean();
        //shape=1
        confidenceLoss=confidenceLoss.mean();
        //shape=1
        classLoss=classLoss.mean();

        INDArray totalScoreArray=gIouLoss.add(confidenceLoss).add(classLoss);

        double totalScore=totalScoreArray.sumNumber().doubleValue();

        return totalScore;
    }


    private LossSameDiff computeLoss(){

        labels=labels.castTo(input.dataType());

        INDArray priorBoundingBoxes=layerConf().getPriorBoundingBoxes();

        assertInputSet(true);

        Preconditions.checkState(getLabels() != null, "Cannot calculate gradients/score: labels are null");

        INDArray reshapeInput=reshapeInput();

        long batchSize=labels.shape()[0];

        long gridHeight=labels.shape()[1];

        long numberOfPriorBoundingBoxPerGridCell=labels.shape()[3];

        long classOneHotLength=labels.shape()[4]-5;

        long inputSize=416;

        long stride=inputSize/gridHeight;
        //（输入）预测值编码
        DecodeResult decodeResult=decode(reshapeInput,priorBoundingBoxes,classOneHotLength,gridHeight,batchSize,stride);
        //解码置信度
        INDArray decodePredictConfidence=decodeResult.getDecodePredictConfidence();
        //拼接解码的中心点坐标与宽高
        INDArray decodePredictXyWh=decodeResult.getDecodePredictBoxesXyWh();
        //输入）预测分类
        INDArray rawPredictClassOneHot= getPredictClassOneHot(reshapeInput, classOneHotLength);
        //标签的边界框信息
        INDArray labelXyWh= getLabelXyWh();
        //标签置信度,response box;
        INDArray labelConfidence= getLabelConfidence();
        //标签置分类
        INDArray labelClassOneHot= getLabelClassOneHot(classOneHotLength);

        SameDiff gIouLossSameDiff = createGIouLossSameDiff(inputSize, decodePredictXyWh, labelXyWh, labelConfidence);

        SameDiff confidenceLossSameDiff = createConfidenceLossSameDiff(reshapeInput, numberOfPriorBoundingBoxPerGridCell, decodePredictConfidence, decodePredictXyWh, labelConfidence);

        SameDiff classLossSameDiff = createClassLossSameDiff(rawPredictClassOneHot, labelConfidence, labelClassOneHot);

        return  new LossSameDiff(gIouLossSameDiff,classLossSameDiff,confidenceLossSameDiff);
    }

    @NotNull
    private SameDiff createConfidenceLossSameDiff(INDArray reshapeInput, long numberOfPriorBoundingBoxPerGridCell, INDArray decodePredictConfidence, INDArray decodePredictXyWh, INDArray labelConfidence) {

        SameDiff boxesBackgroundConfidenceSameDiff=SameDiff.create();

        INDArray boxesBackgroundConfidenceGroundTruthBoxesXyWh= getGroundTruthBoxesXyWh(numberOfPriorBoundingBoxPerGridCell);

        SDVariable boxesBackgroundConfidenceDecodePredictXyWhSdv=boxesBackgroundConfidenceSameDiff.var("decodePredictXyWh",decodePredictXyWh);

        SDVariable boxesBackgroundConfidenceGroundTruthBoxesXyWhSdv=boxesBackgroundConfidenceSameDiff.var("groundTruthBoxesXyWh",boxesBackgroundConfidenceGroundTruthBoxesXyWh);

        SDVariable boxesBackgroundConfidenceLabelConfidenceSdv=boxesBackgroundConfidenceSameDiff.var("labelConfidence",labelConfidence);

        long[] decodePredictXyWhShape= Nd4j.expandDims(decodePredictXyWh,4).shape();

        //背景自信度，即非检测物体的置信度
        INDArray boxesBackgroundConfidence=getBoxesBackgroundConfidence(boxesBackgroundConfidenceSameDiff,decodePredictXyWhShape,boxesBackgroundConfidenceLabelConfidenceSdv,boxesBackgroundConfidenceDecodePredictXyWhSdv,boxesBackgroundConfidenceGroundTruthBoxesXyWhSdv);

        SameDiff confidenceLossSameDiff=SameDiff.create();

        INDArray rawPredictConfidence= getPredictConfidence(reshapeInput);

        SDVariable decodePredictConfidenceSdv=confidenceLossSameDiff.var("decodePredictConfidence",decodePredictConfidence);

        SDVariable labelConfidenceSdv=confidenceLossSameDiff.var("labelConfidence",labelConfidence);

        SDVariable rawPredictConfidenceSdv=confidenceLossSameDiff.var("rawPredictConfidence",rawPredictConfidence);


        SDVariable bg=confidenceLossSameDiff.var("boxesBackgroundConfidence",boxesBackgroundConfidence);

        computeConfidenceLoss(confidenceLossSameDiff,decodePredictConfidence.shape(),decodePredictConfidenceSdv,labelConfidenceSdv,rawPredictConfidenceSdv,bg);

        confidenceLossSameDiff.output(Collections.<String, INDArray>emptyMap(), "computeConfidenceLoss");

        confidenceLossSameDiff.setLossVariables("computeConfidenceLoss");
        return confidenceLossSameDiff;
    }

    @NotNull
    private SameDiff createClassLossSameDiff(INDArray rawPredictClassOneHot, INDArray labelConfidence, INDArray labelClassOneHot) {

        SameDiff classLossSameDiff=SameDiff.create();

        SDVariable classLossLabelConfidenceSdv=classLossSameDiff.var("labelConfidence",  labelConfidence);

        SDVariable classLossLabelClassOneHotSdv=classLossSameDiff.var("labelClassOneHot",  labelClassOneHot);

        SDVariable classLossRawPredictClassOneHotSdv=classLossSameDiff.var("rawPredictClassOneHot",  rawPredictClassOneHot);

        computeClassLoss(classLossSameDiff,labelClassOneHot.shape(),classLossLabelConfidenceSdv,classLossLabelClassOneHotSdv,classLossRawPredictClassOneHotSdv);

        classLossSameDiff.output(Collections.<String, INDArray>emptyMap(), "computeClassLoss");

        classLossSameDiff.setLossVariables("computeClassLoss");

        return classLossSameDiff;
    }

    @NotNull
    private SameDiff createGIouLossSameDiff(long inputSize, INDArray decodePredictXyWh, INDArray labelXyWh, INDArray labelConfidence) {

        SameDiff gIouLossSameDiff=SameDiff.create();

        long[] xYwhShape=decodePredictXyWh.shape();

        INDArray decodePredictX= decodePredictXyWh.get(INDArrayUtils.getLastDimensionPointZero(xYwhShape));

        decodePredictX= Nd4j.expandDims(decodePredictX,xYwhShape.length-1);

        INDArray decodePredictY=decodePredictXyWh.get(INDArrayUtils.getLastDimensionPointOne(xYwhShape));

        decodePredictY=Nd4j.expandDims(decodePredictY,xYwhShape.length-1);

        INDArray decodePredictW= decodePredictXyWh.get(INDArrayUtils.getLastDimensionPointTwo(xYwhShape));

        decodePredictW=Nd4j.expandDims(decodePredictW,xYwhShape.length-1);

        INDArray decodePredictH=decodePredictXyWh.get(INDArrayUtils.getLastDimensionPointThree(xYwhShape));

        decodePredictH=Nd4j.expandDims(decodePredictH,xYwhShape.length-1);

        SDVariable decodePredictXSdv=gIouLossSameDiff.var("decodePredictX",decodePredictX);

        SDVariable decodePredictYSdv=gIouLossSameDiff.var("decodePredictY",decodePredictY);

        SDVariable decodePredictWSdv=gIouLossSameDiff.var("decodePredictW",decodePredictW);

        SDVariable decodePredictHSdv=gIouLossSameDiff.var("decodePredictH",decodePredictH);

        INDArray labelX=Nd4j.expandDims(labelXyWh.get(INDArrayUtils.getLastDimensionPointZero(labelXyWh.shape())),xYwhShape.length-1);

        INDArray labelY=Nd4j.expandDims(labelXyWh.get(INDArrayUtils.getLastDimensionPointOne(labelXyWh.shape())),xYwhShape.length-1);

        INDArray labelW=Nd4j.expandDims(labelXyWh.get(INDArrayUtils.getLastDimensionPointTwo(labelXyWh.shape())),xYwhShape.length-1);

        INDArray labelH=Nd4j.expandDims(labelXyWh.get(INDArrayUtils.getLastDimensionPointThree(labelXyWh.shape())),xYwhShape.length-1);

        SDVariable sdvLabelX=gIouLossSameDiff.var("labelX",labelX);

        SDVariable sdvLabelY=gIouLossSameDiff.var("labelY",labelY);

        SDVariable sdvLabelW=gIouLossSameDiff.var("labelW",labelW);

        SDVariable sdvLabelH=gIouLossSameDiff.var("labelH",labelH);

        SDVariable sdvLabelConfidence=gIouLossSameDiff.var("labelConfidence",  labelConfidence);

        computeGIouLoss(inputSize, gIouLossSameDiff, labelXyWh.shape(), sdvLabelX,  sdvLabelY,  sdvLabelW,  sdvLabelH,  decodePredictXSdv,  decodePredictYSdv,  decodePredictWSdv,  decodePredictHSdv,  sdvLabelConfidence);

        gIouLossSameDiff.output(Collections.<String, INDArray>emptyMap(), "computeGIouLoss");

        gIouLossSameDiff.setLossVariables("computeGIouLoss");

        return gIouLossSameDiff;
    }

    private SDVariable computeClassLoss(SameDiff sd,long[] shape,SDVariable labelConfidence, SDVariable labelClassOneHot, SDVariable rawPredictClassOneHot) {
        return labelConfidence.mul("computeClassLoss",YoloUtils.computeSigmoidCrossEntropyLossWithLogits(shape,sd,labelClassOneHot, rawPredictClassOneHot));
    }

    private SDVariable computeGIouLoss(long inputSize, SameDiff sd, long[] shape, SDVariable labelX, SDVariable labelY, SDVariable labelW, SDVariable labelH, SDVariable predictX, SDVariable predictY, SDVariable predictW, SDVariable predictH, SDVariable labelConfidence) {

        SDVariable gIou=YoloUtils.computeGIou( sd,shape, labelX,  labelY,  labelW,  labelH,  predictX, predictY,  predictW,  predictH);

        gIou= sd.expandDims(gIou,shape.length-1);

        SDVariable m=labelW.mul(labelH).mul(1.0).div(inputSize*inputSize);

        SDVariable lossScale= sd.math().neg(m).add(2);

        return labelConfidence.mul(lossScale).mul("computeGIouLoss",sd.math().neg(gIou).add(1));
    }

    private SDVariable computeConfidenceLoss(SameDiff sd,long[] confidenceShape,SDVariable decodePredictConfidence,  SDVariable labelConfidence,SDVariable rawPredictConfidence,SDVariable responseBoxesBackgroundConfidence) {

        float  alpha=1, gamma=2;

        SDVariable confidenceFocal=YoloUtils.computeFocal(alpha,gamma,sd,labelConfidence,decodePredictConfidence);

        SDVariable confidenceLoss1= labelConfidence.mul(YoloUtils.computeSigmoidCrossEntropyLossWithLogits(confidenceShape,sd,labelConfidence, rawPredictConfidence));

        SDVariable confidenceLoss2= responseBoxesBackgroundConfidence.mul(YoloUtils.computeSigmoidCrossEntropyLossWithLogits(confidenceShape,sd,labelConfidence, rawPredictConfidence));

        return confidenceFocal.mul("computeConfidenceLoss",confidenceLoss1.add(confidenceLoss2));
    }


    private INDArray getBoxesBackgroundConfidence(SameDiff sd, long[] shape, SDVariable labelConfidence, SDVariable decodePredictBoxesXyWh, SDVariable groundTruthBoxesXyWh) {
        //shape=[batchSize,gridSize,gridSize,3,4]->[batchSize,gridSize,gridSize,3,1,4]
        SDVariable decodePredictBoxesXyWhSixD= sd.expandDims(decodePredictBoxesXyWh,4);
        // [batchSize,numberOfPriorBoundingBoxPerGridCell-3,4]->[batchSize,1,1,1,numberOfPriorBoundingBoxPerGridCell-3,4]
        SDVariable groundTruthBoxesXyWhSixD = expandDimsToSix(sd,groundTruthBoxesXyWh);
        //[batchSize,gridSize,gridSize,3,numberOfPriorBoundingBoxPerGridCell-3]
        SDVariable iou= YoloUtils.computeIou(sd, shape,groundTruthBoxesXyWhSixD, decodePredictBoxesXyWhSixD);
        //[batchSize,gridSize,gridSize,3,1]
        SDVariable maxIou = getMaxIou(sd,iou);

        SDVariable result= sd.math().neg(labelConfidence).add(1.0).mul("computeBoxesBackgroundConfidence",maxIou);

        sd.output(Collections.<String, INDArray>emptyMap(), "computeBoxesBackgroundConfidence");

        return  sd.getArrForVarName("computeBoxesBackgroundConfidence");

    }




    @NotNull
    private SDVariable getMaxIou(SameDiff sd,SDVariable iou) {
        //[batchSize,gridSize,gridSize,3]
        SDVariable maxIou=iou.max(-1);
        //[batchSize,gridSize,gridSize,3,1]
        maxIou=sd.expandDims(maxIou,4);

        float iouLossThreshold=0.5f;

        sd.replaceWhere(maxIou,1,Conditions.lessThan(iouLossThreshold));

        sd.replaceWhere(maxIou,0,Conditions.greaterThan(iouLossThreshold));

        return maxIou;
    }

    /**
     * 扩充维度到[batchSize,1,1,1,numberOfPriorBoundingBoxPerGridCell-3,4]
     * @param boxesXyWh
     * @return
     */
    private SDVariable expandDimsToSix(SameDiff sd,SDVariable boxesXyWh) {

        boxesXyWh= sd.expandDims(boxesXyWh,1);

        boxesXyWh=sd.expandDims(boxesXyWh,1);
        //[batchSize,numberOfPriorBoundingBoxPerGridCell-3,4]->[batchSize,1,1,1,numberOfPriorBoundingBoxPerGridCell-3,4]
        boxesXyWh=sd.expandDims(boxesXyWh,1);

        return boxesXyWh;
    }










    private DecodeResult decode(INDArray input,INDArray priorBoundingBoxes,long classOneHotLength,long gridHeight,long batchSize,long stride){
        //得到所有预测边界框的中心点坐标
        INDArray predictBoxesXy= getPredictBoxesXy(input);
        //得到所有预测边界框的高和宽
        INDArray predictBoxesWh= getPredictBoxesWh(input);
        //得到所有预测边界框的置信度
        INDArray predictConfidence= getPredictConfidence(input);
        //得到所有预测边界框的分类
        INDArray predictClassOneHot= getPredictClassOneHot(input, classOneHotLength);
        //得到每个grid相对于最左上角的坐标
        INDArray cxCy=YoloUtils.getCxCy(Integer.parseInt(String.valueOf(gridHeight)),Integer.parseInt(String.valueOf(batchSize)),3);
        //解码中心点坐标
        INDArray decodePredictBoxesXy= Transforms.sigmoid(predictBoxesXy).add(cxCy).mul(stride);
        //解码宽高
        INDArray decodePredictBoxesWh= Transforms.exp(predictBoxesWh).mul(priorBoundingBoxes).mul(stride);
        //解码置信度
        INDArray decodePredictConfidence=Transforms.sigmoid(predictConfidence);
        //解码类别
        INDArray decodePredictClassOneHot=Transforms.sigmoid(predictClassOneHot);
        //拼接解码的中心点坐标与宽高
        INDArray decodePredictBoxesXyWh=Nd4j.concat(-1,decodePredictBoxesXy,decodePredictBoxesWh);

        return new DecodeResult(decodePredictBoxesXy,decodePredictBoxesWh,decodePredictConfidence,decodePredictClassOneHot,decodePredictBoxesXyWh);
    }

    private INDArray reshapeInput() {
        //NCHW --> NWHC
        INDArray reshapeInput=input.dup().permute(0,3,2,1);
        //NWHC --> [batch, grid_h, grid_w, 3, 4+1+nb_class]
        reshapeInput=reshapeInput.reshape(new long[]{reshapeInput.size(0),reshapeInput.size(1),reshapeInput.size(2),3,reshapeInput.size(3)/3});

        return reshapeInput;
    }



    private INDArray getGroundTruthBoxesXyWh(long numberOfPriorBoundingBoxPerGridCell) {
        return labels.get(new INDArrayIndex[]{all(), NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.interval(3,numberOfPriorBoundingBoxPerGridCell),NDArrayIndex.interval(0,4)});
    }



    private INDArray getLabelClassOneHot(long classOneHotLength) {
        return labels.get(new INDArrayIndex[]{all(), all(), all(), NDArrayIndex.interval(0,3),NDArrayIndex.interval(5,5+classOneHotLength)});
    }

    private INDArray getLabelConfidence() {
        return labels.get(new INDArrayIndex[]{all(), all(), all(), NDArrayIndex.interval(0,3),NDArrayIndex.interval(4,5)});
    }

    private INDArray getLabelXyWh() {
        return labels.get(new INDArrayIndex[]{all(), all(), all(), NDArrayIndex.interval(0,3),NDArrayIndex.interval(0,4)});
    }



    private INDArray getPredictClassOneHot(INDArray input, long classOneHotLength) {
        INDArrayIndex[] indexes=INDArrayUtils.getLastDimensionIndexes(input.shape(),NDArrayIndex.interval(5,5+classOneHotLength));
        return input.get(indexes);
    }

    private INDArray getPredictConfidence(INDArray input) {
        INDArrayIndex[] indexes=INDArrayUtils.getLastDimensionIndexes(input.shape(),NDArrayIndex.interval(4,5));
        return input.get(indexes);
    }

    private INDArray getPredictBoxesWh(INDArray input) {
        INDArrayIndex[] indexes=INDArrayUtils.getLastDimensionIndexes(input.shape(),NDArrayIndex.interval(2,4));
        return input.get(indexes);
    }

    private INDArray getPredictBoxesXy(INDArray input) {
        INDArrayIndex[] indexes=INDArrayUtils.getLastDimensionIndexes(input.shape(),NDArrayIndex.interval(0,2));
        return input.get(indexes);
    }

    @Override
    public INDArray computeScoreForExamples(double fullNetworkRegScore, LayerWorkspaceMgr workspaceMgr) {



        return null;
    }

    @Override
    public double f1Score(DataSet data) {
        return 0;
    }

    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        return 0;
    }

    @Override
    public int numLabels() {
        return 0;
    }

    @Override
    public void fit(DataSetIterator iter) {

    }

    @Override
    public int[] predict(INDArray examples) {
        return new int[0];
    }

    @Override
    public List<String> predict(DataSet dataSet) {
        return null;
    }

    @Override
    public void fit(INDArray examples, INDArray labels) {

    }

    @Override
    public void fit(DataSet data) {

    }

    @Override
    public void fit(INDArray examples, int[] labels) {

    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {

        LossSameDiff lossSameDiff=computeLoss();

        Map<String, INDArray> gIouLossGradientsMap=lossSameDiff.getGIouLossSameDiff().calculateGradients(null, "decodePredictX", "decodePredictY", "decodePredictW", "decodePredictH");

        INDArray gradientX=gIouLossGradientsMap.get("decodePredictX");
        INDArray gradientY=gIouLossGradientsMap.get("decodePredictY");
        INDArray gradientW=gIouLossGradientsMap.get("decodePredictW");
        INDArray gradientH=gIouLossGradientsMap.get("decodePredictH");

        Map<String, INDArray> classLossGradientsMap=lossSameDiff.getClassLossSameDiff().calculateGradients(null, "rawPredictClassOneHot");
        INDArray gradientClass=classLossGradientsMap.get("rawPredictClassOneHot");

        Map<String, INDArray> confidenceLossGradientsMap=lossSameDiff.getConfidenceLossSameDiff().calculateGradients(null, "rawPredictConfidence");

        INDArray gradientConfidence=confidenceLossGradientsMap.get("rawPredictConfidence");

        epsilon=Nd4j.concat(-1,gradientX,gradientY,gradientW,gradientH,gradientConfidence,gradientClass);

        //[batch, grid_h, grid_w, 3, 4+1+nb_class]--> [batch, grid_h, grid_w, 3*(4+1+nb_class)]
        epsilon=epsilon.reshape(new long[]{epsilon.shape()[0],epsilon.shape()[1],epsilon.shape()[2],epsilon.shape()[3]*epsilon.shape()[4]});
        //NWHC-->NCHW to match the input shape
        epsilon=epsilon.permute(0,3,1,2);

        IActivation activation =   new ActivationLReLU();

        INDArray gradient=activation.backprop(input.dup(),epsilon).getFirst();

        INDArray epsOut = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD,gradient);

        return new Pair<>(EMPTY_GRADIENT,epsOut);

    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return null;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {

    }

    @Data
    @AllArgsConstructor
    class  DecodeResult{

        //解码中心点坐标
        private INDArray decodePredictBoxesXy;
        //解码宽高
        private INDArray decodePredictBoxesWh;
        //解码置信度
        private  INDArray decodePredictConfidence;
        //解码类别
        private INDArray decodePredictClassOneHot;
        //拼接解码的中心点坐标与宽高
        private INDArray decodePredictBoxesXyWh;
    }


    @Data
    @AllArgsConstructor
    class  LossSameDiff{

        private  SameDiff gIouLossSameDiff;

        private  SameDiff classLossSameDiff;

        private  SameDiff confidenceLossSameDiff;

    }

}

