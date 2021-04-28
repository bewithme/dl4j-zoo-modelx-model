package org.freeware.dl4j.modelx.model.gan;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public abstract class AbsGan extends ZooModel {


    protected static String ACTIVATION ="activation";

    protected static String ELEMENT_WISE_VERTEX ="element-wise-vertex";

    protected static String MERGE_VERTEX ="merge-vertex";

    protected static String MAX_POOLING ="max-pooling";

    protected static String AVG_POOLING ="avg-pooling";

    protected static String UP_SAMPLING_2D="up-sampling-2d";

    protected static String ZERO_PADDING ="zero-padding";

    protected static String CNN ="cnn";

    protected static String BATCH_NORM="batch-norm";

    protected static String DROP_OUT="drop-out";



    /**
     * 从对抗网络中把参数复制给生成器和判别器
     * @param generator
     * @param discriminator
     * @param gan
     */
    public void copyParamsFromGanToGeneratorAndDiscriminator(ComputationGraph generator,ComputationGraph discriminator, ComputationGraph gan) {
       copyParamsFromGanToGenerator(generator,gan);
       copyParamsFromGanToDiscriminator(discriminator,gan);
    }




    /**
     * 从对抗网络中把参数复制给生成器
     * @param generator
     * @param gan
     */
    public void copyParamsFromGanToGenerator(ComputationGraph generator, ComputationGraph gan) {
        int genLayerLen = generator.getLayers().length;
        for (int i = 0; i < genLayerLen; i++) {
            generator.getLayer(i).setParams(gan.getLayer(i).params());
        }
    }

    /**
     * 从对抗网络中把参数复制给判别器
     * @param discriminator
     * @param gan
     */
    public void copyParamsFromGanToDiscriminator(ComputationGraph discriminator, ComputationGraph gan) {

        int ganLayerLen =gan.getLayers().length;

        int disLayerLen=discriminator.getLayers().length;

        int ganLayerIndex = ganLayerLen-disLayerLen;

        for (int disLayerIndex =0; disLayerIndex < disLayerLen; disLayerIndex++) {

            INDArray params= gan.getLayer(ganLayerIndex).params();

            ganLayerIndex++;

            discriminator.getLayer(disLayerIndex).setParams(params);
        }
    }

    /**
     * 从判别器复制参数到对抗网络的判别器中
     * discriminator, gan
     * @param discriminator
     * @param gan
     */
    public  void copyParamsFromDiscriminatorToGanDiscriminator(ComputationGraph discriminator, ComputationGraph gan) {

        int disLayerLen=discriminator.getLayers().length;

        int ganLayerIndex= gan.getLayers().length-disLayerLen;

        for (int disLayerIndex = 0; disLayerIndex < disLayerLen; disLayerIndex++) {

            INDArray params=discriminator.getLayer(disLayerIndex ).params();

            gan.getLayer(ganLayerIndex).setParams(params);

            ganLayerIndex++;
        }
    }


    /**
     * 批量添加计算图的层或顶点
     * @param graphBuilder
     * @param graphLayerItems
     * @param frozen
     */
    protected void addGraphItems(ComputationGraphConfiguration.GraphBuilder graphBuilder, List<GraphLayerItem> graphLayerItems, Boolean frozen) {

        for(GraphLayerItem graphLayerItem:graphLayerItems) {

            if(graphLayerItem.getLayerOrVertex() instanceof MergeVertex) {

                MergeVertex mergeVertex=(MergeVertex)graphLayerItem.getLayerOrVertex();

                graphBuilder.addVertex(graphLayerItem.getLayerName(), mergeVertex, graphLayerItem.getLayerInputs());

            } else if(graphLayerItem.getLayerOrVertex() instanceof ElementWiseVertex) {

                ElementWiseVertex elementWiseVertex=(ElementWiseVertex)graphLayerItem.getLayerOrVertex();

                graphBuilder.addVertex(graphLayerItem.getLayerName(), elementWiseVertex, graphLayerItem.getLayerInputs());

            }else if (graphLayerItem.getLayerOrVertex() instanceof ReshapeVertex){

                ReshapeVertex reshapeVertex=(ReshapeVertex)graphLayerItem.getLayerOrVertex();

                graphBuilder.addVertex(graphLayerItem.getLayerName(), reshapeVertex, graphLayerItem.getLayerInputs());

            }else if (graphLayerItem.getLayerOrVertex() instanceof Layer){

                Layer layer=(Layer)graphLayerItem.getLayerOrVertex();

                graphBuilder.addLayer(graphLayerItem.getLayerName(), layer, graphLayerItem.getLayerInputs());

            }
        }
    }


    /**
     * 创建层名称
     * @param moduleName
     * @param layerName
     * @param moduleIndex
     * @param blockIndex
     * @return
     */
    protected String createLayerName(String moduleName, String layerName,Integer moduleIndex,Integer blockIndex) {

        String newLayerName=moduleName.concat("-").concat(layerName).concat("-").concat(String.valueOf(moduleIndex)).concat("-").concat(String.valueOf(blockIndex));

        return newLayerName;
    }

    /**
     * 获取最后一层名称
     * @param graphLayerItemList
     * @return
     */
    protected String getLastLayerName(List<GraphLayerItem> graphLayerItemList){
        return graphLayerItemList.size()==0?"":graphLayerItemList.get(graphLayerItemList.size()-1).getLayerName();
    }

    protected List<GraphLayerItem> getUNetBackbone(int nIn,String[] inputs,ConvolutionLayer.AlgoMode aLgoMode){

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem(
                "conv1-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nIn(nIn)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(),
                inputs));


        graphItemList.add(new GraphLayerItem(
                "conv1-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(),
                new String[]{"conv1-1"}));


        graphItemList.add(new GraphLayerItem(
                "pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(),
                new String[]{"conv1-2"}));

        graphItemList.add(new GraphLayerItem("conv2-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool1"}));

        graphItemList.add(new GraphLayerItem("conv2-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv2-1"}));
        graphItemList.add(new GraphLayerItem("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(), new String[]{"conv2-2"}));

        graphItemList.add(new GraphLayerItem("conv3-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool2"}));
        graphItemList.add(new GraphLayerItem("conv3-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv3-1"}));
        graphItemList.add(new GraphLayerItem("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(), new String[]{"conv3-2"}));
        graphItemList.add(new GraphLayerItem("conv4-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool3"}));
        graphItemList.add(new GraphLayerItem("conv4-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv4-1"}));
        graphItemList.add(new GraphLayerItem("drop4", new DropoutLayer.Builder(0.5).build(), new String[]{"conv4-2"}));

        graphItemList.add(new GraphLayerItem("pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                .build(), new String[]{"drop4"}));

        graphItemList.add(new GraphLayerItem("conv5-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"pool4"}));
        graphItemList.add(new GraphLayerItem("conv5-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv5-1"}));
        graphItemList.add(new GraphLayerItem("drop5", new DropoutLayer.Builder(0.5).build(), new String[]{"conv5-2"}));

        // up6
        graphItemList.add(new GraphLayerItem("up6-1", new Upsampling2D.Builder(2).build(), new String[]{"drop5"}));
        graphItemList.add(new GraphLayerItem("up6-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"up6-1"}));
        graphItemList.add(new GraphLayerItem("merge6", new MergeVertex(), new String[]{"drop4", "up6-2"}));
        graphItemList.add(new GraphLayerItem("conv6-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge6"}));
        graphItemList.add(new GraphLayerItem("conv6-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(512)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv6-1"}));

        // up7
        graphItemList.add(new GraphLayerItem("up7-1", new Upsampling2D.Builder(2).build(), new String[]{"conv6-2"}));
        graphItemList.add(new GraphLayerItem("up7-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"up7-1"}));
        graphItemList.add(new GraphLayerItem("merge7", new MergeVertex(), new String[]{"conv3-2", "up7-2"}));
        graphItemList.add(new GraphLayerItem("conv7-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge7"}));
        graphItemList.add(new GraphLayerItem("conv7-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(256)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv7-1"}));

        // up8
        graphItemList.add(new GraphLayerItem("up8-1", new Upsampling2D.Builder(2).build(), new String[]{"conv7-2"}));
        graphItemList.add(new GraphLayerItem("up8-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"up8-1"}));
        graphItemList.add(new GraphLayerItem("merge8", new MergeVertex(),new String[]{ "conv2-2", "up8-2"}));
        graphItemList.add(new GraphLayerItem("conv8-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge8"}));
        graphItemList.add(new GraphLayerItem("conv8-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(128)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv8-1"}));

        // up9
        graphItemList.add(new GraphLayerItem("up9-1", new Upsampling2D.Builder(2).build(), new String[]{"conv8-2"}));
        graphItemList.add(new GraphLayerItem("up9-2", new ConvolutionLayer.Builder(2,2)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"up9-1"}));
        graphItemList.add(new GraphLayerItem("merge9", new MergeVertex(), new String[]{"conv1-2", "up9-2"}));
        graphItemList.add(new GraphLayerItem("conv9-1", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"merge9"}));
        graphItemList.add(new GraphLayerItem("conv9-2", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(64)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv9-1"}));
        graphItemList.add(new GraphLayerItem("conv9-3", new ConvolutionLayer.Builder(3,3)
                .stride(1,1)
                .nOut(2)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(aLgoMode)
                .activation(Activation.RELU).build(), new String[]{"conv9-2"}));

        graphItemList.add(new GraphLayerItem("conv10", new ConvolutionLayer.Builder(1,1)
                .stride(1,1).nOut(3)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(aLgoMode)
                .activation(Activation.IDENTITY).build(), new String[]{"conv9-3"}));

        return graphItemList;

    }



    //ResNet50


    protected List<GraphLayerItem> getResNet50Backbone(int nIn,String[] inputs,ConvolutionLayer.AlgoMode aLgoMode,IUpdater updater) {

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);
               // stem
        graphItemList.add(new GraphLayerItem("stem-zero", new ZeroPaddingLayer.Builder(3, 3).build(), inputs));
        graphItemList.add(new GraphLayerItem("stem-cnn1",
                        new ConvolutionLayer.Builder(new int[] {7, 7}, new int[] {2, 2}).nOut(64)
                                .build(),
                        new String[]{"stem-zero"}));
        graphItemList.add(new GraphLayerItem("stem-batch1", new BatchNormalization(), new String[]{"stem-cnn1"}));
        graphItemList.add(new GraphLayerItem("stem-act1", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        new String[]{"stem-batch1"}));
        graphItemList.add(new GraphLayerItem("stem-maxpool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                        new int[] {3, 3}, new int[] {2, 2}).build(), new String[]{"stem-act1"}));


        graphItemList.addAll(convBlock( new int[] {3, 3}, new int[] {64, 64, 256}, "2", "a", new int[] {2, 2}, "stem-maxpool1",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {64, 64, 256}, "2", "b", "res2a_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {64, 64, 256}, "2", "c", "res2b_branch",aLgoMode,updater));

        graphItemList.addAll(convBlock(new int[] {3, 3}, new int[] {128, 128, 512}, "3", "a", "res2c_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {128, 128, 512}, "3", "b", "res3a_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {128, 128, 512}, "3", "c", "res3b_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {128, 128, 512}, "3", "d", "res3c_branch",aLgoMode,updater));

        graphItemList.addAll(convBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "a", "res3d_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "b", "res4a_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "c", "res4b_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "d", "res4c_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "e", "res4d_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "f", "res4e_branch",aLgoMode,updater));

        graphItemList.addAll(convBlock( new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "a", "res4f_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "b", "res5a_branch",aLgoMode,updater));
        graphItemList.addAll(identityBlock( new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "c", "res5b_branch",aLgoMode,updater));

        graphItemList.add(new GraphLayerItem("avgpool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3}).build(),
                new String[]{"res5c_branch"}));


        return graphItemList;
    }


        private List<GraphLayerItem> convBlock( int[] kernelSize, int[] filters,
                               String stage, String block, String input,ConvolutionLayer.AlgoMode aLgoMode, IUpdater updater) {
           return  convBlock( kernelSize, filters, stage, block, new int[] {2, 2}, input,aLgoMode,updater);
        }
        private List<GraphLayerItem> identityBlock(int[] kernelSize, int[] filters, String stage, String block, String input,ConvolutionLayer.AlgoMode aLgoMode, IUpdater updater) {

        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem(convName + "2a",
                new ConvolutionLayer.Builder(new int[] {1, 1})
                        .nOut(filters[0])
                        .updater(updater)
                        .cudnnAlgoMode(aLgoMode)
                        .build(),
                        new String[]{input}));
        graphItemList.add(new GraphLayerItem(batchName + "2a", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2a"}));
                        graphItemList.add(new GraphLayerItem(activationName + "2a",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                                        new String[]{batchName + "2a"}));

        graphItemList.add(new GraphLayerItem(convName + "2b", new ConvolutionLayer.Builder(kernelSize)
                .nOut(filters[1])
                .updater(updater)
                .cudnnAlgoMode(aLgoMode).convolutionMode(ConvolutionMode.Same).build(),
                        new String[]{activationName + "2a"}));
        graphItemList.add(new GraphLayerItem(batchName + "2b", new BatchNormalization.Builder()
                .updater(updater)
                .build(), new String[]{convName + "2b"}));
        graphItemList.add(new GraphLayerItem(activationName + "2b",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        new String[]{batchName + "2b"}));

        graphItemList.add(new GraphLayerItem(convName + "2c",
                        new ConvolutionLayer.Builder(new int[] {1, 1}).nOut(filters[2])
                                .updater(updater)
                                .cudnnAlgoMode(aLgoMode).build(),
                        new String[]{activationName + "2b"}));
        graphItemList.add(new GraphLayerItem(batchName + "2c", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2c"}));

        graphItemList.add(new GraphLayerItem(shortcutName, new ElementWiseVertex(ElementWiseVertex.Op.Add),new String[]{ batchName + "2c",
                        input}));
        graphItemList.add(new GraphLayerItem(convName, new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{shortcutName}));

        return graphItemList;
    }

    private List<GraphLayerItem>   convBlock(int[] kernelSize, int[] filters, String stage, String block, int[] stride, String input,ConvolutionLayer.AlgoMode aLgoMode, IUpdater updater) {
        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        List<GraphLayerItem>  graphItemList=new ArrayList<GraphLayerItem>(10);

        graphItemList.add(new GraphLayerItem(convName + "2a", new ConvolutionLayer.Builder(new int[] {1, 1}, stride)
                .cudnnAlgoMode(aLgoMode)
                .updater(updater)
                .nOut(filters[0]).build(),
                        new String[]{input}));
        graphItemList.add(new GraphLayerItem(batchName + "2a", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2a"}));
        graphItemList.add(new GraphLayerItem(activationName + "2a",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        new String[]{batchName + "2a"}));

        graphItemList.add(new GraphLayerItem(convName + "2b",
                        new ConvolutionLayer.Builder(kernelSize).nOut(filters[1])
                                .updater(updater)
                                .convolutionMode(ConvolutionMode.Same).build(),
                        new String[]{activationName + "2a"}));
        graphItemList.add(new GraphLayerItem(batchName + "2b", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2b"}));
        graphItemList.add(new GraphLayerItem(activationName + "2b",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        new String[]{batchName + "2b"}));
        graphItemList.add(new GraphLayerItem(convName + "2c",
                        new ConvolutionLayer.Builder(new int[] {1, 1})
                                .cudnnAlgoMode(aLgoMode)
                                .nOut(filters[2])
                                .updater(updater)
                                .build(),
                        new String[]{activationName + "2b"}));
        graphItemList.add(new GraphLayerItem(batchName + "2c", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "2c"}));

                // shortcut
        graphItemList.add(new GraphLayerItem(convName + "1",
                        new ConvolutionLayer.Builder(new int[] {1, 1}, stride)
                                .cudnnAlgoMode(aLgoMode)
                                .nOut(filters[2])
                                .updater(updater)
                                .build(),
                        new String[]{input}));
        graphItemList.add(new GraphLayerItem(batchName + "1", new BatchNormalization.Builder()
                .updater(updater).build(), new String[]{convName + "1"}));


        graphItemList.add(new GraphLayerItem(shortcutName,
                new ElementWiseVertex(ElementWiseVertex.Op.Add),
                new String[]{batchName + "1"}));

        graphItemList.add(new GraphLayerItem(convName, new ActivationLayer.Builder().activation(Activation.RELU).build(),
                new String[]{shortcutName}));

        return graphItemList;
    }





}
