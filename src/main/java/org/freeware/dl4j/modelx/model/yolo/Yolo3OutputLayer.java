package org.freeware.dl4j.modelx.model.yolo;

import lombok.*;
import lombok.extern.slf4j.Slf4j;

import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.freeware.dl4j.modelx.utils.YoloUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;
import java.util.List;

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

        INDArray priorBoundingBoxes=layerConf().getPriorBoundingBoxes();

        assertInputSet(true);

        Preconditions.checkState(getLabels() != null, "Cannot calculate gradients/score: labels are null");

        reshapeInput();

        long batchSize=labels.shape()[0];

        long gridHeight=labels.shape()[1];

        long numberOfPriorBoundingBoxPerGridCell=labels.shape()[3];

        long classOneHotLength=labels.shape()[4]-5;

        long inputSize=416;

        long stride=inputSize/gridHeight;

        DecodeResult decodeResult=decode(input,priorBoundingBoxes,classOneHotLength,gridHeight,batchSize,stride);
        //解码置信度
        INDArray decodePredictConfidence=decodeResult.getDecodePredictConfidence();
        //拼接解码的中心点坐标与宽高
        INDArray decodePredictBoxesXyWh=decodeResult.getDecodePredictBoxesXyWh();
        //得到标签的边界框信息
        INDArray labelXyWh= getLabelXyWh();

        INDArray labelConfidence= getLabelConfidence();

        INDArray labelClassOneHot= getLabelClassOneHot(classOneHotLength);
        //取出所有标签
        INDArray groundTrueBoxesXyWh= getGroundTrueBoxesXyWh(numberOfPriorBoundingBoxPerGridCell);

        INDArray gIou=YoloUtils.getGIou(decodePredictBoxesXyWh,labelXyWh);

        gIou=Nd4j.expandDims(gIou,-1);

        INDArray labelBoxW= getLabelBoxW(labelXyWh);

        INDArray labelBoxH= getLabelBoxH(labelXyWh);

        INDArray lossScale=Transforms.neg(labelBoxW.mul(labelBoxH).mul(1.0).div(inputSize*inputSize)).add(2);

        INDArray gIouLoss=labelConfidence.mul(lossScale).mul(Transforms.neg(gIou).add(1));

        INDArray iou=YoloUtils.getIou(decodePredictBoxesXyWh,groundTrueBoxesXyWh);

        INDArray maxIou=iou.max(-1);

        float iou_loss_thresh=0.5f;

        INDArray backgroundConfidence=Transforms.neg(labelConfidence).add(1.0).mul(maxIou);

        INDArray confidenceFocal=YoloUtils.focal(labelConfidence,decodePredictConfidence);

        INDArray predictConfidence= getPredictConfidence(input);

        INDArray predictClassOneHot= getPredictClassOneHot(input, classOneHotLength);

        INDArray confidenceLoss1= YoloUtils.sigmoidCrossEntropyLossWithLogits(labelConfidence, predictConfidence).mul(labelConfidence);

        INDArray confidenceLoss2= YoloUtils.sigmoidCrossEntropyLossWithLogits(labelConfidence, predictConfidence).mul(backgroundConfidence);

        INDArray  confidenceLoss=confidenceFocal.mul(confidenceLoss1.add(confidenceLoss2));

        INDArray  classOneHotLoss= YoloUtils.sigmoidCrossEntropyLossWithLogits(labelClassOneHot, predictClassOneHot).mul(labelConfidence);

        double score=gIouLoss.add(confidenceLoss).add(classOneHotLoss).sumNumber().doubleValue();

        return score;
    }

    private void reshapeInput() {
        //NCHW --> NWHC
        input=input.permute(0,3,2,1);
        //NHWC --> [batch, grid_h, grid_w, 3, 4+1+nb_class]
        input=input.reshape(new long[]{input.size(0),input.size(1),input.size(2),3,input.size(3)/3});
    }

    private INDArray getLabelBoxH(INDArray labelXyWh) {
        return labelXyWh.get(new INDArrayIndex[]{all(), all(), all(), all(), NDArrayIndex.interval(3,4)});
    }

    private INDArray getLabelBoxW(INDArray labelXyWh) {
        return labelXyWh.get(new INDArrayIndex[]{all(), all(), all(), all(), NDArrayIndex.interval( 2,3)});
    }

    private INDArray getGroundTrueBoxesXyWh(long numberOfPriorBoundingBoxPerGridCell) {
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


    private DecodeResult decode(INDArray input,INDArray priorBoundingBoxes,long classOneHotLength,long gridHeight,long batchSize,long stride){
        //得到所有边界框的中心点坐标
        INDArray predictBoxesXy= getPredictBoxesXy(input);
        //得到所有边界框的高和宽
        INDArray predictBoxesWh= getPredictBoxesWh(input);

        INDArray predictConfidence= getPredictConfidence(input);

        INDArray predictClassOneHot= getPredictClassOneHot(input, classOneHotLength);

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

    private INDArray getPredictClassOneHot(INDArray input, long classOneHotLength) {
        //log.info(input.shapeInfoToString());
        return input.get(new INDArrayIndex[]{all(), all(), all(), all(), NDArrayIndex.interval(5,5+classOneHotLength)});
    }

    private INDArray getPredictConfidence(INDArray input) {
        return input.get(new INDArrayIndex[]{all(), all(), all(), all(), NDArrayIndex.interval(4,5)});
    }

    private INDArray getPredictBoxesWh(INDArray input) {
        return input.get(new INDArrayIndex[]{all(), all(), all(), all(), NDArrayIndex.interval(2,4)});
    }

    private INDArray getPredictBoxesXy(INDArray input) {
        return input.get(new INDArrayIndex[]{all(), all(), all(), all(), NDArrayIndex.interval(0,2)});
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

        INDArray priorBoundingBoxes=layerConf().getPriorBoundingBoxes();

        assertInputSet(true);

        Preconditions.checkState(getLabels() != null, "Cannot calculate gradients/score: labels are null");

        reshapeInput();

        long batchSize=labels.shape()[0];

        long gridHeight=labels.shape()[1];

        long numberOfPriorBoundingBoxPerGridCell=labels.shape()[3];

        long classOneHotLength=labels.shape()[4]-5;

        long inputSize=416;

        long stride=inputSize/gridHeight;

        DecodeResult decodeResult=decode(input,priorBoundingBoxes,classOneHotLength,gridHeight,batchSize,stride);
        //解码置信度
        INDArray decodePredictConfidence=decodeResult.getDecodePredictConfidence();
        //拼接解码的中心点坐标与宽高
        INDArray decodePredictBoxesXyWh=decodeResult.getDecodePredictBoxesXyWh();
        //得到标签的边界框信息
        INDArray labelXyWh= getLabelXyWh();

        INDArray labelConfidence= getLabelConfidence();

        INDArray labelClassOneHot= getLabelClassOneHot(classOneHotLength);
        //取出所有标签
        INDArray groundTrueBoxesXyWh= getGroundTrueBoxesXyWh(numberOfPriorBoundingBoxPerGridCell);

        INDArray gIou=YoloUtils.getGIou(decodePredictBoxesXyWh,labelXyWh);

        gIou=Nd4j.expandDims(gIou,4);

        INDArray labelBoxW= getLabelBoxW(labelXyWh);

        INDArray labelBoxH= getLabelBoxH(labelXyWh);

        INDArray lossScale=Transforms.neg(labelBoxW.mul(labelBoxH).mul(1.0).div(inputSize*inputSize)).add(2);

        INDArray gIouLoss=labelConfidence.mul(lossScale).mul(Transforms.neg(gIou).add(1));
        //shape=[batchSize,gridSize,gridSize,3,4]->[batchSize,gridSize,gridSize,3,1,4]
        INDArray decodePredictBoxesXyWhSixD=Nd4j.expandDims(decodePredictBoxesXyWh,4);

        INDArray groundTrueBoxesXyWhSixD=Nd4j.expandDims(groundTrueBoxesXyWh,1);

        groundTrueBoxesXyWhSixD=Nd4j.expandDims(groundTrueBoxesXyWhSixD,1);
        //[batchSize,numberOfPriorBoundingBoxPerGridCell-3,4]->[batchSize,1,1,1,numberOfPriorBoundingBoxPerGridCell-3,4]
        groundTrueBoxesXyWhSixD=Nd4j.expandDims(groundTrueBoxesXyWhSixD,1);
        //[batchSize,gridSize,gridSize,3,numberOfPriorBoundingBoxPerGridCell-3]
        INDArray iou=YoloUtils.getIou(decodePredictBoxesXyWhSixD,groundTrueBoxesXyWhSixD);
        //[batchSize,gridSize,gridSize,3]
        INDArray maxIou=iou.max(-1);
        //[batchSize,gridSize,gridSize,3,1]
        maxIou=Nd4j.expandDims(maxIou,4);

        float iou_loss_thresh=0.5f;

        INDArray backgroundConfidence=Transforms.neg(labelConfidence).add(1.0).mul(maxIou);

        INDArray confidenceFocal=YoloUtils.focal(labelConfidence,decodePredictConfidence);

        INDArray predictConfidence= getPredictConfidence(input);

        INDArray predictClassOneHot= getPredictClassOneHot(input, classOneHotLength);

        INDArray confidenceLoss1= YoloUtils.derivativeOfSigmoidCrossEntropyLossWithLogits(labelConfidence, predictConfidence).mul(labelConfidence);

        INDArray confidenceLoss2= YoloUtils.derivativeOfSigmoidCrossEntropyLossWithLogits(labelConfidence, predictConfidence).mul(backgroundConfidence);

        INDArray  confidenceLoss=confidenceFocal.mul(confidenceLoss1.add(confidenceLoss2));

        INDArray  classOneHotLoss= YoloUtils.derivativeOfSigmoidCrossEntropyLossWithLogits(labelClassOneHot, predictClassOneHot).mul(labelConfidence);


        INDArray values=gIouLoss.add(confidenceLoss).add(classOneHotLoss);


        return new Pair<>(EMPTY_GRADIENT,values);
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
}

