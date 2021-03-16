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

        long numberOfPriorBoundingBoxPerGridCell=labels.size(3);
        //NCHW --> NHWC
        input=input.permute(0,2,3,1);
        //NHWC --> [batch, grid_h, grid_w, 3, 4+1+nb_class]
        input=input.reshape(new long[]{input.size(0),input.size(1),input.size(2),3,input.size(3)/3});

        long batchSize=labels.shape()[0];

        long gridHeight=labels.shape()[1];

        long classOneHotSize=labels.shape()[3]-5;

        //得到所有边界框的中心点坐标
        INDArray predictXy=input.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,2)});
        //得到所有边界框的高和宽
        INDArray predictHw=input.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(2,4)});

        INDArray predictConfidence=input.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(4,5)});

        INDArray predictClassOneHot=input.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(5,classOneHotSize)});

        INDArray cxCy=YoloUtils.getCxCy(Integer.parseInt(String.valueOf(gridHeight)),Integer.parseInt(String.valueOf(batchSize)),3);

        long stride=416/gridHeight;
        //解码中心点坐标
        INDArray decodeInputXy= Transforms.sigmoid(predictXy).add(cxCy).mul(stride);
        //解码高和宽
        INDArray decodeInputHw= Transforms.eps(predictHw).mul(priorBoundingBoxes).mul(stride);

        predictConfidence=Transforms.sigmoid(predictConfidence);

        predictClassOneHot=Transforms.sigmoid(predictClassOneHot);

        INDArray decodeInputXyHw=Nd4j.concat(-1,decodeInputXy,decodeInputHw);


        //得到所有边界框的高和宽
        INDArray labelXyHw=labels.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,3),NDArrayIndex.interval(0,4)});

        INDArray labelConfidence=labels.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,3),NDArrayIndex.interval(4,5)});

        INDArray labelClassOneHot=labels.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,3),NDArrayIndex.interval(5,classOneHotSize)});
        //取出所有标签
        INDArray groundTrueBoxes=labels.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.interval(3,numberOfPriorBoundingBoxPerGridCell),NDArrayIndex.interval(0,4)});


        INDArray gIou=YoloUtils.get5DBoxGIou(decodeInputXyHw,labelXyHw);

        gIou=Nd4j.expandDims(gIou,-1);




        return 0;
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
        return new Pair<>(EMPTY_GRADIENT,input);
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
}

