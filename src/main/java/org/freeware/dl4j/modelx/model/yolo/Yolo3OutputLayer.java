package org.freeware.dl4j.modelx.model.yolo;

import lombok.*;
import lombok.extern.slf4j.Slf4j;

import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
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

        INDArray boundingBoxPriors=layerConf().getBoundingBoxPriors();

        assertInputSet(true);

        Preconditions.checkState(getLabels() != null, "Cannot calculate gradients/score: labels are null");

        long anchorBoxesQty=labels.size(3);

        //取出真实原始标签
        INDArray groundTrueBoxes=labels.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.interval(4,anchorBoxesQty+1),NDArrayIndex.interval(0,5)});

        //取出yolo标签
        INDArray yoloLabel=labels.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,4),NDArrayIndex.all()});

        //NCHW --> NHWC
        input=input.permute(0,2,3,1);

        //NHWC --> [batch, grid_h, grid_w, 3, 4+1+nb_class]
        input=input.reshape(new long[]{input.size(0),input.size(1),input.size(2),3,input.size(3)/3});


        long gridHeight=yoloLabel.shape()[1];

        long gridWidth=yoloLabel.shape()[2];

        long classNum=yoloLabel.shape()[3]-5;

        INDArray gridFactor= Nd4j.create(new float[]{gridWidth,gridHeight}).reshape(new long[]{1,1,1,1,2});

        INDArray netFactor= Nd4j.create(new float[]{416,416}).reshape(new long[]{1,1,1,1,2});

        //得到所有边界框的中心点坐标
        INDArray predictXy=input.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,2)});
        //得到所有边界框的高和宽
        INDArray predictHw=input.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(2,4)});

        INDArray predictConfidence=input.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(4,5)});

        INDArray predictClass=input.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(5,5+classNum)});

        //解码中心点坐标
        INDArray decodeInputXy= Transforms.sigmoid(predictXy);
        //解码高和宽
        INDArray decodeInputHw= Transforms.eps(predictHw);

        return 0;
    }

    /**
     * 获取每个单元格相对于最左上角的坐标
     * 输出形状为[batchSize,gridSize,gridSize,anchorQuantityPerGrid,2]
     * 最后一个维度用来存当前单元格相对于左上角的坐标(Cx,Cy)
     * @param gridSize 网格大小，有13，26，52
     * @param batchSize 批量大小
     * @param anchorQuantityPerCell 每个单元格负责检测的先验框数量，一般为3
     * @return
     */
    private  INDArray getCxCy(int gridSize, int batchSize, int anchorQuantityPerCell) {

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


    /**
     * 批量计算IOU
     * @param predictBoundingBoxes
     * @param labelBoundingBoxes
     * @return
     */
    private  INDArray getIou(INDArray predictBoundingBoxes,INDArray labelBoundingBoxes) {

        INDArrayIndex[] indexZeroToTwo=new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,2)};

        INDArrayIndex[] indexTwoToFour=new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(2,4)};

        INDArrayIndex[] indexTwo=new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(2)};
        INDArrayIndex[] indexThree=new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(3)};

        INDArrayIndex[] indexZero=new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(0)};
        INDArrayIndex[] indexOne=new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(1)};

        INDArray predictBoundingBoxesArea= predictBoundingBoxes.get(indexTwo).mul(predictBoundingBoxes.get(indexThree));

        INDArray labelBoundingBoxesArea= labelBoundingBoxes.get(indexTwo).mul(labelBoundingBoxes.get(indexThree));

        INDArray predictBoundingBoxesLeftTop= predictBoundingBoxes.get(indexZeroToTwo).sub(predictBoundingBoxes.get(indexTwoToFour).mul(0.5));

        INDArray predictBoundingBoxesRightBottom= predictBoundingBoxes.get(indexZeroToTwo).add(predictBoundingBoxes.get(indexTwoToFour).mul(0.5));

        predictBoundingBoxes=Nd4j.concat(-1,predictBoundingBoxesLeftTop,predictBoundingBoxesRightBottom);


        INDArray labelBoundingBoxesLeftTop= labelBoundingBoxes.get(indexZeroToTwo).sub(labelBoundingBoxes.get(indexTwoToFour).mul(0.5));

        INDArray labelBoundingBoxesRightBottom= labelBoundingBoxes.get(indexZeroToTwo).add(labelBoundingBoxes.get(indexTwoToFour).mul(0.5));

        labelBoundingBoxes=Nd4j.concat(-1,labelBoundingBoxesLeftTop,labelBoundingBoxesRightBottom);


        INDArray boundingBoxesLeftTop=Transforms.max(predictBoundingBoxes.get(indexZeroToTwo),labelBoundingBoxes.get(indexZeroToTwo));

        INDArray boundingBoxesRightBottom=Transforms.min(predictBoundingBoxes.get(indexTwoToFour),labelBoundingBoxes.get(indexTwoToFour));


        INDArray interSection=Transforms.max(boundingBoxesRightBottom.sub(boundingBoxesLeftTop),0.0);

        INDArray interArea=interSection.get(indexZero).mul(interSection.get(indexOne));

        INDArray unionArea=predictBoundingBoxesArea.add(labelBoundingBoxesArea).sub(interArea);

        INDArray iou=interArea.mul(1.0).mul(unionArea);

        return iou;
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

