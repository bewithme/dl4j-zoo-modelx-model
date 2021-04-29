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
import java.util.Arrays;
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
            log.info(graphLayerItem.getLayerName()+ "  "+graphLayerItem.getLayerOrVertex().getClass().getName()+" "+Arrays.toString(graphLayerItem.getLayerInputs()));
            if(graphLayerItem.getLayerOrVertex() instanceof MergeVertex) {

                MergeVertex mergeVertex=(MergeVertex)graphLayerItem.getLayerOrVertex();

                graphBuilder.addVertex(graphLayerItem.getLayerName(), mergeVertex, graphLayerItem.getLayerInputs());

            } else if(graphLayerItem.getLayerOrVertex() instanceof ElementWiseVertex) {

                ElementWiseVertex elementWiseVertex=(ElementWiseVertex)graphLayerItem.getLayerOrVertex();

                if(graphLayerItem.getLayerInputs()[0].contains("bn2a_branch2c")){
                    log.info(graphLayerItem.getLayerInputs()[0]+graphLayerItem.getLayerInputs()[1]);
                }

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









}
