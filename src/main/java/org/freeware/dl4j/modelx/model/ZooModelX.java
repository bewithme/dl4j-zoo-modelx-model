package org.freeware.dl4j.modelx.model;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.freeware.dl4j.nn.GraphLayerItem;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

@Slf4j
public abstract class ZooModelX extends ZooModel {


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

    protected static String DENSE_LAYER="dense-layer";

    protected static String OUTPUT_LAYER="output-layer";

    protected static String SUB_SAMPLING_LAYER = "subsampling-layer";



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


    /**
     * 当to为from的一部分时
     * 从from复制参数到to
     * 会把层名称相同的参数复制
     * @param from
     * @param to
     */
    public void copyParamsWhenToIsPartOfFromByName(ComputationGraph from, ComputationGraph to) {
            
            org.deeplearning4j.nn.api.Layer[] toLayers=to.getLayers();

            for(org.deeplearning4j.nn.api.Layer toLayer:toLayers){

                String toLayerName=toLayer.getConfig().getLayerName();
                
                toLayer.setParams( from.getLayer(toLayerName).params());
                
            }
    }
    /**
     * 当from为to的一部分时
     * 从from复制参数到to
     * 会把层名称相同的参数复制
     * @param from
     * @param to
     */
    public void copyParamsWhenFromIsPartOfToByName(ComputationGraph from, ComputationGraph to) {

        org.deeplearning4j.nn.api.Layer[] fromLayers=from.getLayers();

        for(org.deeplearning4j.nn.api.Layer fromLayer:fromLayers){

            String toLayerName=fromLayer.getConfig().getLayerName();

            to.getLayer(toLayerName).setParams( from.getLayer(toLayerName).params());

        }
    }





}
