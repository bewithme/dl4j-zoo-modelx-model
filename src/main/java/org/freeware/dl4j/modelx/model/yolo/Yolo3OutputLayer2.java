package org.freeware.dl4j.modelx.model.yolo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
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
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;


@Slf4j
public class Yolo3OutputLayer2 extends AbstractLayer<Yolo3OutputLayerConfiguration> implements Serializable, IOutputLayer {

    private static final Gradient EMPTY_GRADIENT = new DefaultGradient();
    @Setter @Getter
    protected INDArray labels;

    public Yolo3OutputLayer2(NeuralNetConfiguration conf, DataType networkDataType) {
        super(conf, networkDataType);
    }

    @Override
    public boolean needsLabels() {
        return false;
    }



    @Override
    public double computeScore(double fullNetworkRegScore, boolean training, LayerWorkspaceMgr workspaceMgr) {

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
        //拼接解码的中心点坐标与宽高
        INDArray decodePredictBoxesXyWh=decodeResult.getDecodePredictBoxesXyWh();

        INDArray boxLossScale=getBoxLossScale(inputSize);
        //标签置信度
        INDArray objectMask= getLabelConfidence();
        //未处理真实标签
        INDArray groundTruthBoxesXyWh= getGroundTruthBoxesXyWh(numberOfPriorBoundingBoxPerGridCell);

        INDArray ignoreMask=getIgnoreMask(decodePredictBoxesXyWh,objectMask,groundTruthBoxesXyWh);
        //标签置分类
        INDArray labelClassOneHot= getLabelClassOneHot(classOneHotLength);

        INDArray xyLoss=computeXyLoss(objectMask,boxLossScale,getLabelXy(),decodeResult.getDecodePredictBoxesXy());

        INDArray whLoss=computeWhLoss(objectMask,boxLossScale,getLabelWh(),decodeResult.getDecodePredictBoxesWh());

        INDArray confidentLoss=computeConfidenceLoss(objectMask,ignoreMask,decodeResult.getDecodePredictConfidence());

        INDArray classLoss= computeClassLoss(objectMask,labelClassOneHot,decodeResult.getDecodePredictClassOneHot());

        double xyLossNum=xyLoss.sumNumber().doubleValue();

        double whLossNum=whLoss.sumNumber().doubleValue();

        double confidentLossNum=confidentLoss.sumNumber().doubleValue();

        double classLossNum=classLoss.sumNumber().doubleValue();

        double totalScore=xyLossNum+whLossNum+confidentLossNum+classLossNum;

        return totalScore;
    }

    private INDArray getBoxLossScale(long inputSize) {
        //标签的边界框信息
        INDArray labelXyWh= getLabelXyWh();

        INDArray labelBoxW= getLabelBoxW(labelXyWh);

        INDArray labelBoxH= getLabelBoxH(labelXyWh);

        INDArray boxLossScale= Transforms.neg(labelBoxW.mul(labelBoxH).mul(1.0).div(inputSize*inputSize)).add(2);

        return boxLossScale;
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

    private INDArray getIgnoreMask(INDArray decodePredictBoxesXyWh, INDArray objectMask, INDArray groundTruthBoxesXyWh) {

        //shape=[batchSize,gridSize,gridSize,3,4]->[batchSize,gridSize,gridSize,3,1,4]
        INDArray decodePredictBoxesXyWhSixD= Nd4j.expandDims(decodePredictBoxesXyWh,4);
        // [batchSize,numberOfPriorBoundingBoxPerGridCell-3,4]->[batchSize,1,1,1,numberOfPriorBoundingBoxPerGridCell-3,4]
        INDArray groundTruthBoxesXyWhSixD = expandDimsToSix(groundTruthBoxesXyWh);
        //[batchSize,gridSize,gridSize,3,numberOfPriorBoundingBoxPerGridCell-3]
        INDArray iou= YoloUtils.getIou(decodePredictBoxesXyWhSixD,groundTruthBoxesXyWhSixD);

        INDArray maxIou = getMaxIou(iou);

        return Transforms.neg(objectMask).add(1.0).mul(maxIou);
    }

    private INDArray getMaxIou(INDArray iou) {
        //[batchSize,gridSize,gridSize,3]
        INDArray maxIou=iou.max(-1);
        //[batchSize,gridSize,gridSize,3,1]
        maxIou= Nd4j.expandDims(maxIou,4);

        float iouLossThreshold=0.5f;

        BooleanIndexing.replaceWhere(maxIou,1, Conditions.lessThan(iouLossThreshold));

        BooleanIndexing.replaceWhere(maxIou,0, Conditions.greaterThan(iouLossThreshold));

        return maxIou;
    }
    /**
     * 扩充维度到[batchSize,1,1,1,numberOfPriorBoundingBoxPerGridCell-3,4]
     * @param groundTruthBoxesXyWh
     * @return
     */
    private INDArray expandDimsToSix(INDArray groundTruthBoxesXyWh) {

        groundTruthBoxesXyWh= Nd4j.expandDims(groundTruthBoxesXyWh,1);

        groundTruthBoxesXyWh=Nd4j.expandDims(groundTruthBoxesXyWh,1);
        //[batchSize,numberOfPriorBoundingBoxPerGridCell-3,4]->[batchSize,1,1,1,numberOfPriorBoundingBoxPerGridCell-3,4]
        groundTruthBoxesXyWh=Nd4j.expandDims(groundTruthBoxesXyWh,1);

        return groundTruthBoxesXyWh;
    }

    /**
     * 计算XY损失
     * @param objectMask
     * @param boxLossScale
     * @param rawTrueXy
     * @param rawPredictXy
     * @return
     */
    private INDArray computeXyLoss(INDArray objectMask ,INDArray boxLossScale,INDArray rawTrueXy,INDArray rawPredictXy){

        ILossFunction lossFunction=new LossBinaryXENT();

       return  objectMask.mul(boxLossScale).mul(lossFunction.computeScoreArray(rawTrueXy,rawPredictXy,new ActivationLReLU(),null));
    }

    /**
     * 计算XY损失梯度
     * @param objectMask
     * @param boxLossScale
     * @param rawTrueXy
     * @param rawPredictXy
     * @return
     */
    private INDArray computeGradientOfXyLoss(INDArray objectMask ,INDArray boxLossScale,INDArray rawTrueXy,INDArray rawPredictXy){

        ILossFunction lossFunction=new LossBinaryXENT();

        return  objectMask.mul(boxLossScale).mul(lossFunction.computeGradient(rawTrueXy,rawPredictXy,new ActivationLReLU(),null));
    }

    /**
     *  计算WH损失
     * @param objectMask
     * @param boxLossScale
     * @param rawTrueWh
     * @param rawPredictWh
     * @return
     */
    private INDArray computeWhLoss(INDArray objectMask ,INDArray boxLossScale,INDArray rawTrueWh,INDArray rawPredictWh){

        ILossFunction lossFunction=new LossL2();

        return  objectMask.mul(boxLossScale).mul(0.5).mul(lossFunction.computeScoreArray(rawTrueWh,rawPredictWh,new ActivationLReLU(),null));
    }

    /**
     * 计算WH损失梯度
     * @param objectMask
     * @param boxLossScale
     * @param rawTrueWh
     * @param rawPredictWh
     * @return
     */
    private INDArray computeGradientOfWhLoss(INDArray objectMask ,INDArray boxLossScale,INDArray rawTrueWh,INDArray rawPredictWh){

        ILossFunction lossFunction=new LossL2();

        return  objectMask.mul(boxLossScale).mul(0.5).mul(lossFunction.computeGradient(rawTrueWh,rawPredictWh,new ActivationLReLU(),null));
    }


    /**
     * 计算置信度损失
     * @param objectMask
     * @param ignoreMask
     * @param rawPredictConfidence
     * @return
     */
    private INDArray computeConfidenceLoss(INDArray objectMask ,INDArray ignoreMask,INDArray rawPredictConfidence){

        ILossFunction lossFunction=new LossBinaryXENT();

        INDArray loss1=  objectMask.mul(lossFunction.computeScoreArray(objectMask,rawPredictConfidence,new ActivationLReLU(),null));

        INDArray loss2=  Transforms.neg(objectMask).add(1).mul(ignoreMask).mul(lossFunction.computeScoreArray(objectMask,rawPredictConfidence,new ActivationLReLU(),null));

        return loss1.add(loss2);
    }

    /**
     * 计算置信度损失梯度
     * @param objectMask
     * @param ignoreMask
     * @param rawPredictConfidence
     * @return
     */
    private INDArray computeGradientOfConfidenceLoss(INDArray objectMask ,INDArray ignoreMask,INDArray rawPredictConfidence){

        ILossFunction lossFunction=new LossBinaryXENT();

        INDArray loss1=  objectMask.mul(lossFunction.computeGradient(objectMask,rawPredictConfidence,new ActivationLReLU(),null));

        INDArray loss2=  Transforms.neg(objectMask).add(1).mul(ignoreMask).mul(lossFunction.computeGradient(objectMask,rawPredictConfidence,new ActivationLReLU(),null));

        return loss1.add(loss2);
    }


    /**
     * 计算类别损失
     * @param objectMask
     * @param trueClassProbability
     * @param rawPredictClassProbability
     * @return
     */
    private INDArray computeClassLoss(INDArray objectMask , INDArray trueClassProbability, INDArray rawPredictClassProbability){

        ILossFunction lossFunction=new LossBinaryXENT();

        return  objectMask.mul(lossFunction.computeScoreArray(trueClassProbability,rawPredictClassProbability,new ActivationLReLU(),null));
    }

    /**
     * 计算类别损失梯度
     * @param objectMask
     * @param trueClassProbability
     * @param rawPredictClassProbability
     * @return
     */
    private INDArray computeGradientOfClassLoss(INDArray objectMask ,INDArray trueClassProbability,INDArray rawPredictClassProbability){

        ILossFunction lossFunction=new LossBinaryXENT();

        return  objectMask.mul(lossFunction.computeGradient(trueClassProbability,rawPredictClassProbability,new ActivationLReLU(),null));
    }





    private INDArray reshapeInput() {
        //NCHW --> NWHC
        INDArray reshapeInput=input.dup().permute(0,3,2,1);
        //NWHC --> [batch, grid_h, grid_w, 3, 4+1+nb_class]
        reshapeInput=reshapeInput.reshape(new long[]{reshapeInput.size(0),reshapeInput.size(1),reshapeInput.size(2),3,reshapeInput.size(3)/3});

        return reshapeInput;
    }




    private INDArray getLabelBoxH(INDArray labelXyWh) {
        return labelXyWh.get(new INDArrayIndex[]{all(), all(), all(), all(), NDArrayIndex.interval(3,4)});
    }

    private INDArray getLabelBoxW(INDArray labelXyWh) {
        return labelXyWh.get(new INDArrayIndex[]{all(), all(), all(), all(), NDArrayIndex.interval( 2,3)});
    }

    private INDArray getGroundTruthBoxesXyWh(long numberOfPriorBoundingBoxPerGridCell) {
        return labels.get(new INDArrayIndex[]{all(), NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.interval(3,numberOfPriorBoundingBoxPerGridCell),NDArrayIndex.interval(0,4)});
    }

    private INDArray getGroundTruthBoxesXyWhForDerivative(long numberOfPriorBoundingBoxPerGridCell) {
        return labels.get(new INDArrayIndex[]{all(), all(), all(), NDArrayIndex.interval(3,numberOfPriorBoundingBoxPerGridCell),NDArrayIndex.interval(0,4)});
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
    private INDArray getLabelXy() {
        return labels.get(new INDArrayIndex[]{all(), all(), all(), NDArrayIndex.interval(0,3),NDArrayIndex.interval(0,2)});
    }
    private INDArray getLabelWh() {
        return labels.get(new INDArrayIndex[]{all(), all(), all(), NDArrayIndex.interval(0,3),NDArrayIndex.interval(2,4)});
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
        //拼接解码的中心点坐标与宽高
        INDArray decodePredictBoxesXyWh=decodeResult.getDecodePredictBoxesXyWh();

        INDArray boxLossScale=getBoxLossScale(inputSize);
        //标签置信度
        INDArray objectMask= getLabelConfidence();
        //未处理真实标签
        INDArray groundTruthBoxesXyWh= getGroundTruthBoxesXyWh(numberOfPriorBoundingBoxPerGridCell);

        INDArray ignoreMask=getIgnoreMask(decodePredictBoxesXyWh,objectMask,groundTruthBoxesXyWh);
        //标签置分类
        INDArray labelClassOneHot= getLabelClassOneHot(classOneHotLength);

        INDArray xyLossGradient=computeGradientOfXyLoss(objectMask,boxLossScale,getLabelXy(),decodeResult.getDecodePredictBoxesXy());

        INDArray whLossGradient=computeGradientOfWhLoss(objectMask,boxLossScale,getLabelWh(),decodeResult.getDecodePredictBoxesWh());

        INDArray confidentLossGradient=computeGradientOfConfidenceLoss(objectMask,ignoreMask,decodeResult.getDecodePredictConfidence());

        INDArray classLossGradient=computeGradientOfClassLoss(objectMask,labelClassOneHot,decodeResult.getDecodePredictClassOneHot());

        epsilon=Nd4j.concat(-1,xyLossGradient,whLossGradient,confidentLossGradient,classLossGradient);
        //[batch, grid_h, grid_w, 3, 4+1+nb_class]--> [batch, grid_h, grid_w, 3*(4+1+nb_class)]
        epsilon=epsilon.reshape(new long[]{epsilon.shape()[0],epsilon.shape()[1],epsilon.shape()[2],epsilon.shape()[3]*epsilon.shape()[4]});
        //NWHC-->NCHW to match the input shape
        epsilon=epsilon.permute(0,3,1,2);

        IActivation activation = new ActivationLReLU();

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

}

