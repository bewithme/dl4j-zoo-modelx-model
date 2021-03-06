package org.freeware.dl4j.modelx.model.cae;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.freeware.dl4j.modelx.model.ZooModelX;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.util.ArrayList;
import java.util.List;


/**
 * @author wenfengxu
 * 深度卷积自编码器
 */
@AllArgsConstructor
@Builder
@Slf4j
public class ConvolutionalAutoEncoder extends ZooModelX {

    @Builder.Default
    private long seed = 123456;

    @Builder.Default
    private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;

    private static final double LEARNING_RATE = 0.0003;

    private static final double GRADIENT_THRESHOLD = 100.0;

    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();

    @Builder.Default
    private int imageHeight = 256;

    @Builder.Default
    private int imageWidth = 256;

    @Builder.Default
    private int imageChannel = 3;

    @Builder.Default
    private IUpdater updater = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();





    /**
     * 编码器网络配置
     *
     * @return
     */
    public ComputationGraphConfiguration buildEncoderConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String[] inputs = {"input"};

        List<GraphLayerItem> encoderLayerItems= buildEncoderGraphLayerItems(inputs,updater);

        addGraphItems(graph,encoderLayerItems,false);

        String outputLayerName=getLastLayerName(encoderLayerItems);

        graph.addInputs(inputs);

        graph.setOutputs(outputLayerName);

        return graph.build();
    }


    /**
     * 编解码器网络配置
     * @return
     */
    public ComputationGraphConfiguration buildAutoEncoder() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .updater(updater)
                .seed(seed)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String[] inputs = {"input"};

        List<GraphLayerItem> encoderLayerItems= buildEncoderGraphLayerItems(inputs,updater);

        addGraphItems(graph,encoderLayerItems,false);

        String decoderInputName=getLastLayerName(encoderLayerItems);

        List<GraphLayerItem> decoderLayerItems= buildDecoderGraphLayerItems(new String[]{decoderInputName});

        addGraphItems(graph,decoderLayerItems,false);

        String output=getLastLayerName(decoderLayerItems);

        graph.addInputs(inputs);

        graph.setOutputs(output);

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return graph.build();
    }




    /**
     * 编码器
     * [N,C,256,256]->[N,C,128,128]->[N,C,64,64]->[N,C,32,32]->[N,C,16,16]
     * @param inputs
     * @return
     */
    public   List<GraphLayerItem> buildEncoderGraphLayerItems(String[] inputs,IUpdater updater){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem("en-01",
                new Convolution2D.Builder()
                        .nIn(imageChannel)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .updater(updater)
                        .nOut(32)
                        .build(),
                 inputs));

        graphItemList.add(new GraphLayerItem("en-02",
                new SubsamplingLayer.Builder(
                        SubsamplingLayer.PoolingType.AVG,
                        new int[]{2,2},
                        new int[]{ 2,2})

                        .build(),
                new String[]{"en-01"}));


        graphItemList.add(new GraphLayerItem("en-03",
                new Convolution2D.Builder()
                        .nIn(32)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .nOut(32)
                        .updater(updater)
                       .build(),
                new String[]{"en-02"}));

        graphItemList.add(new GraphLayerItem("en-04",
                new SubsamplingLayer.Builder(
                        SubsamplingLayer.PoolingType.AVG,
                        new int[]{2,2},
                        new int[]{ 2,2})
                        .build(),
                new String[]{"en-03"}));

      //shape= [N,32,16,16] if conMode=same outW=floor(inW/stride)

        return graphItemList;

    }

    /**
     * 解码器层列表
     * [N,C,16,16]->[N,C,32,32]->[N,C,64,64]->[N,C,128,128]->[N,C,256,256]
     * @param inputs
     * @return
     */
    private List<GraphLayerItem> buildDecoderGraphLayerItems(String[] inputs) {

        List<GraphLayerItem> graphItemList = new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem("de-01",
                new Deconvolution2D.Builder()
                        .nOut(32)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build(),
                inputs));

        graphItemList.add(new GraphLayerItem("de-02",
                new Deconvolution2D.Builder()
                        .nOut(32)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build(),
                new String[]{"de-01"}));

        graphItemList.add(new GraphLayerItem("de-03",
                new Deconvolution2D.Builder()
                        .nOut(32)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build(),
                new String[]{"de-02"}));

        graphItemList.add(new GraphLayerItem("de-04",
                new Deconvolution2D.Builder()
                        .nOut(3)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build(),
                new String[]{"de-03"}));


        graphItemList.add(new GraphLayerItem("output",
                new CnnLossLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.TANH).build(),
                new String[]{"de-04"}));

        return graphItemList;

    }





    @Override
    public void setInputShape(int[][] inputShape) {

    }


    /**
     * 初始化CAE
     * @return
     */
    public ComputationGraph init() {

        ComputationGraphConfiguration configuration= buildAutoEncoder();

        ComputationGraph gan = new ComputationGraph(configuration);

        gan.init();

        return gan;

    }

    /**
     * 初始化编码器
     * @return
     */
    public ComputationGraph initEncoder() {

        ComputationGraphConfiguration genConf= buildEncoderConfiguration();

        ComputationGraph model = new ComputationGraph(genConf);

        model.init();

        return model;
    }




    @Override
    public ModelMetaData metaData() {
        return null;
    }

    @Override
    public Class<? extends Model> modelType() {
        return null;
    }

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return 0;
    }
}
