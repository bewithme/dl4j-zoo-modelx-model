package org.freeware.dl4j.modelx.model.gan;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


@AllArgsConstructor
@Builder
@Slf4j
public class CycleGan extends AbsGan {

    @Builder.Default
    private long seed = 12345;

    @Builder.Default
    private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;

    private static final double LEARNING_RATE = 0.0003;

    private static final double GRADIENT_THRESHOLD = 100.0;

    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();

    @Builder.Default
    private int imageHeight = 128;

    @Builder.Default
    private int imageWidth = 128;

    @Builder.Default
    private int imageChannel = 3;

    @Builder.Default
    private IUpdater generatorUpdater = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    @Builder.Default
    private IUpdater discriminatorUpdater = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    @Builder.Default
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private static Random random = new Random(123456);


    private static int generatorFilters = 32;

    private static int discriminatorFilters = 64;


    /**
     * 生成器网络配置
     *
     * @return
     */
    public ComputationGraphConfiguration buildGeneratorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String inputs = "gen-input";

        String moduleName = "gen";

        List<GraphLayerItem> layerItems = buildGeneratorGraphLayerItems(inputs,moduleName,generatorUpdater);

        addGraphItems(graph, layerItems, Boolean.FALSE);

        graph.addInputs(inputs);

        graph.setOutputs(getLastLayerName(layerItems));

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return graph.build();
    }

    /**
     * 判别器网络配置
     *
     * @return
     */
    public ComputationGraphConfiguration buildDiscriminatorConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)

                .updater(discriminatorUpdater)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String input ="dis-input";

        List<GraphLayerItem> layerItems = buildDiscriminatorGraphLayerItems(input, discriminatorUpdater);

        addGraphItems(graph, layerItems, Boolean.FALSE);

        graph.addInputs(input);

        graph.setOutputs(getLastLayerName(layerItems));

        return graph.build();
    }


    public ComputationGraphConfiguration buildGanConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String genInputs = "gen-input";

        String moduleName = "gen";

        List<GraphLayerItem> genLayerItems = buildGeneratorGraphLayerItems(genInputs,moduleName,generatorUpdater);

        addGraphItems(graph, genLayerItems, Boolean.FALSE);

        String disInput = getLastLayerName(genLayerItems);
        //学习率为0，即判别器不会被训练，只训练生成器
        List<GraphLayerItem> disLayerItems = buildDiscriminatorGraphLayerItems(disInput, UPDATER_ZERO);

        addGraphItems(graph, disLayerItems, Boolean.FALSE);

        graph.addInputs(genInputs);

        graph.setOutputs(getLastLayerName(disLayerItems));

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return graph.build();
    }


    /**
     * 重建网络
     * A->B^->A^
     * B->A^->2^
     * @return
     */
    public ComputationGraphConfiguration buildReconstructNetworkConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String genInputs = "gen-input";

        String moduleName = "gen";

        List<GraphLayerItem> genLayerItems = buildGeneratorGraphLayerItems(genInputs,moduleName,generatorUpdater);

        addGraphItems(graph, genLayerItems, Boolean.TRUE);

        String reconGenInputs =getLastLayerName(genLayerItems);

        String reconModuleName = "gen-recon";

        List<GraphLayerItem> genReconLayerItems = buildGeneratorGraphLayerItems(reconGenInputs,reconModuleName,generatorUpdater);
        String outputLayerName="gen-recon-output";
        genReconLayerItems.add(new GraphLayerItem(outputLayerName,
                new CnnLossLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                        .updater(generatorUpdater)
                        .activation(Activation.IDENTITY).build(),
                new String[]{getLastLayerName(genReconLayerItems)}));

        addGraphItems(graph, genReconLayerItems, Boolean.TRUE);


        graph.addInputs(genInputs);

        graph.setOutputs(outputLayerName);

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return graph.build();
    }

    public ComputationGraphConfiguration buildIdentityMappingNetworkConfiguration() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        String genInputs = "gen-input";

        String moduleName = "gen-identity";

        List<GraphLayerItem> genLayerItems = buildGeneratorGraphLayerItems(genInputs,moduleName,generatorUpdater);

        String lossLayerInput =getLastLayerName(genLayerItems);

        String outputLayerName="gen-identity-output";

        genLayerItems.add(new GraphLayerItem(outputLayerName,
                new CnnLossLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                        .updater(generatorUpdater)
                        .activation(Activation.IDENTITY).build(),
                new String[]{lossLayerInput}));

        addGraphItems(graph, genLayerItems, Boolean.TRUE);

        graph.addInputs(genInputs);

        graph.setOutputs(outputLayerName);

        graph.setInputTypes(InputType.convolutional(imageHeight, imageWidth, imageChannel));

        return graph.build();
    }


    /**
     * 生成器计算图的层列表
     *
     *
     * @param input
     * @return
     */
    private List<GraphLayerItem> buildGeneratorGraphLayerItems(String input, String moduleName,IUpdater updater) {

        List<GraphLayerItem> graphItemList = new ArrayList<GraphLayerItem>(10);

        int modelIndex = 0;

        List<GraphLayerItem> downSamplingGraphLayerItemList0=convolution2D(moduleName, modelIndex, 0, input, imageChannel, generatorFilters,updater);

        String downSamplingLayerName0=getLastLayerName(downSamplingGraphLayerItemList0);

        List<GraphLayerItem> downSamplingGraphLayerItemList1=convolution2D(moduleName, modelIndex, 1, downSamplingLayerName0, generatorFilters, generatorFilters*2,updater);

        String downSamplingLayerName1=getLastLayerName(downSamplingGraphLayerItemList1);

        List<GraphLayerItem> downSamplingGraphLayerItemList2=convolution2D(moduleName, modelIndex, 2, downSamplingLayerName1, generatorFilters*2, generatorFilters*4,updater);

        String downSamplingLayerName2=getLastLayerName(downSamplingGraphLayerItemList2);

        List<GraphLayerItem> downSamplingGraphLayerItemList3=convolution2D(moduleName, modelIndex, 3, downSamplingLayerName2, generatorFilters*4, generatorFilters*8,updater);

        String downSamplingLayerName3=getLastLayerName(downSamplingGraphLayerItemList3);

        List<GraphLayerItem> upSamplingGraphLayerItemList0=deconvolution2D(moduleName,modelIndex,4,downSamplingLayerName3,downSamplingLayerName2,generatorFilters*8,generatorFilters*4,updater);

        String upSamplingLayerName0=getLastLayerName(upSamplingGraphLayerItemList0);

        List<GraphLayerItem> upSamplingGraphLayerItemList1=deconvolution2D(moduleName,modelIndex,5,upSamplingLayerName0,downSamplingLayerName1,generatorFilters*(4+4),generatorFilters*2,updater);

        String upSamplingLayerName1=getLastLayerName(upSamplingGraphLayerItemList1);

        List<GraphLayerItem> upSamplingGraphLayerItemList2=deconvolution2D(moduleName,modelIndex,6,upSamplingLayerName1,downSamplingLayerName0,generatorFilters*(2+2),generatorFilters,updater);

        String upSamplingLayerName2=getLastLayerName(upSamplingGraphLayerItemList2);

        graphItemList.addAll(downSamplingGraphLayerItemList0);

        graphItemList.addAll(downSamplingGraphLayerItemList1);

        graphItemList.addAll(downSamplingGraphLayerItemList2);

        graphItemList.addAll(downSamplingGraphLayerItemList3);

        graphItemList.addAll(upSamplingGraphLayerItemList0);

        graphItemList.addAll(upSamplingGraphLayerItemList1);

        graphItemList.addAll(upSamplingGraphLayerItemList2);

        String upSampling2DLayerName=createLayerName(moduleName, UP_SAMPLING_2D,0,7);

        graphItemList.add(new GraphLayerItem(upSampling2DLayerName,
                new Upsampling2D.Builder(2)
                        .build(),
                new String[]{upSamplingLayerName2}));

        String cnnLayerName=createLayerName(moduleName, CNN,0,8);

        graphItemList.add(new GraphLayerItem(cnnLayerName,
                new Convolution2D.Builder()
                        .kernelSize(4,4)
                        .stride(1,1)
                        .nOut(imageChannel)
                        .activation(Activation.TANH)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{upSampling2DLayerName}));

        return graphItemList;

    }

    /**
     * 判别器计算图的层列表
     *
     * @param input
     * @param updater
     * @return
     */
    private List<GraphLayerItem> buildDiscriminatorGraphLayerItems(String input, IUpdater updater) {

        List<GraphLayerItem> graphItemList = new ArrayList<GraphLayerItem>(10);

        int modelIndex = 0;

        String moduleName = "dis";

        List<GraphLayerItem> downSamplingGraphLayerItemList0=convolution2D(moduleName, modelIndex, 0, input, imageChannel, discriminatorFilters,4,Boolean.FALSE,updater);

        String downSamplingLayerName0=getLastLayerName(downSamplingGraphLayerItemList0);

        List<GraphLayerItem> downSamplingGraphLayerItemList1=convolution2D(moduleName, modelIndex, 1, downSamplingLayerName0, discriminatorFilters, discriminatorFilters*2,updater);

        String downSamplingLayerName1=getLastLayerName(downSamplingGraphLayerItemList1);

        List<GraphLayerItem> downSamplingGraphLayerItemList2=convolution2D(moduleName, modelIndex, 2, downSamplingLayerName1, discriminatorFilters*2, discriminatorFilters*4,updater);

        String downSamplingLayerName2=getLastLayerName(downSamplingGraphLayerItemList2);

        List<GraphLayerItem> downSamplingGraphLayerItemList3=convolution2D(moduleName, modelIndex, 3, downSamplingLayerName2, discriminatorFilters*4, discriminatorFilters*8,updater);

        String downSamplingLayerName3=getLastLayerName(downSamplingGraphLayerItemList3);

        graphItemList.addAll(downSamplingGraphLayerItemList0);

        graphItemList.addAll(downSamplingGraphLayerItemList1);

        graphItemList.addAll(downSamplingGraphLayerItemList2);

        graphItemList.addAll(downSamplingGraphLayerItemList3);

        String cnnLayerName=createLayerName(moduleName, CNN,0,4);

        graphItemList.add(new GraphLayerItem(cnnLayerName,
                new Convolution2D.Builder()
                        .kernelSize(4,4)
                        .nIn(discriminatorFilters*8)
                        .stride(1,1)
                        .nOut(1)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{downSamplingLayerName3}));

        String outputLayerName=createLayerName(moduleName, CNN,0,5);

        graphItemList.add(new GraphLayerItem(outputLayerName,
                new CnnLossLayer.Builder(LossFunctions.LossFunction.MSE)
                        .updater(updater)
                        .activation(Activation.IDENTITY).build(),
                new String[]{cnnLayerName}));

        return graphItemList;

    }

    private List<GraphLayerItem> convolution2D(String moduleName, int moduleIndex, int blockIndex, String layerInputName, int nIn, int nOut, IUpdater updater){

        return  convolution2D( moduleName, moduleIndex, blockIndex, layerInputName, nIn, nOut,4,Boolean.TRUE, updater);
    }

    private  List<GraphLayerItem> convolution2D(String moduleName,int moduleIndex,int blockIndex,String layerInputName,int nIn,int nOut,int kernelSize,boolean normalization,IUpdater updater){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        String cnnLayerName=createLayerName(moduleName, CNN,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(cnnLayerName,
                new Convolution2D.Builder()
                        .nIn(nIn)
                        .kernelSize(kernelSize,kernelSize)
                        .stride(2,2)
                        .nOut(nOut)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{layerInputName}));

        String activationLayerName=createLayerName(moduleName, ACTIVATION,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(activationLayerName,
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new String[]{cnnLayerName}));


        if(normalization){

            String batchNormLayerName=createLayerName(moduleName, BATCH_NORM,moduleIndex,blockIndex);

            graphItemList.add(new GraphLayerItem(batchNormLayerName,
                    new BatchNormalization.Builder()
                            .nIn(nOut)
                            .nOut(nOut)
                            .updater(updater)
                            .build(),
                    new String[]{activationLayerName}));

        }

        return graphItemList;

    }

        private  List<GraphLayerItem> deconvolution2D(String moduleName,int moduleIndex,int blockIndex,String layerInputName,String skipInputName,int nIn,int nOut,IUpdater updater){

         return  deconvolution2D( moduleName, moduleIndex, blockIndex, layerInputName, skipInputName, nIn, nOut, 4,0D, updater);

        }


        private  List<GraphLayerItem> deconvolution2D(String moduleName,int moduleIndex,int blockIndex,String layerInputName,String skipInputName,int nIn,int nOut,int kernelSize,double dropoutRate,IUpdater updater){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        String upSampling2DLayerName=createLayerName(moduleName, UP_SAMPLING_2D,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(upSampling2DLayerName,
                new Upsampling2D.Builder(2).build(),
                new String[]{layerInputName}));

        String cnnLayerName=createLayerName(moduleName, CNN,moduleIndex,blockIndex);


        graphItemList.add(new GraphLayerItem(cnnLayerName,
                new Convolution2D.Builder()
                        .nIn(nIn)
                        .kernelSize(kernelSize,kernelSize)
                        .stride(1,1)
                        .nOut(nOut)
                        .activation(Activation.RELU)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(updater).build(),
                new String[]{upSampling2DLayerName}));

        String bnLayerInput=cnnLayerName;

        if(dropoutRate!=0D){

            String dropOutLayerName=createLayerName(moduleName, DROP_OUT,moduleIndex,blockIndex);

            graphItemList.add(new GraphLayerItem(dropOutLayerName,
                    new DropoutLayer.Builder(dropoutRate).build(),
                    new String[]{cnnLayerName}));
            bnLayerInput=dropOutLayerName;
         }

        String batchNormLayerName=createLayerName(moduleName, BATCH_NORM,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(batchNormLayerName,
                new BatchNormalization.Builder()
                        .nIn(nOut)
                        .nOut(nOut)
                        .updater(updater)
                        .build(),
                new String[]{bnLayerInput}));

        String mergeVertexLayerName=createLayerName(moduleName, MERGE_VERTEX,moduleIndex,blockIndex);

        graphItemList.add(new GraphLayerItem(mergeVertexLayerName,
                new MergeVertex(),
                new String[]{batchNormLayerName,skipInputName}));

        return graphItemList;

    }







    @Override
    public void setInputShape(int[][] inputShape) {

    }


    /**
     * 初始化对抗网络
     * @return
     */
    public ComputationGraph init() {

        ComputationGraphConfiguration configuration=buildGanConfiguration();

        ComputationGraph gan = new ComputationGraph(configuration);

        gan.init();

        return gan;

    }

    /**
     * 初始化生成器
     * @return
     */
    public ComputationGraph initGenerator() {

        ComputationGraphConfiguration genConf=buildGeneratorConfiguration();

        ComputationGraph model = new ComputationGraph(genConf);

        model.init();

        return model;
    }

    /**
     * 初始化判别器
     * @return
     */
    public ComputationGraph initDiscriminator() {

        ComputationGraphConfiguration configuration=buildDiscriminatorConfiguration();

        ComputationGraph model = new ComputationGraph(configuration);

        model.init();

        return model;
    }


    public ComputationGraph initReconstructNetwork() {

        ComputationGraphConfiguration configuration= buildReconstructNetworkConfiguration();

        ComputationGraph model = new ComputationGraph(configuration);

        model.init();

        return model;
    }


    public ComputationGraph initIdentityMappingNetwork() {

        ComputationGraphConfiguration configuration=buildIdentityMappingNetworkConfiguration();

        ComputationGraph model = new ComputationGraph(configuration);

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
