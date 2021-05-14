package org.freeware.dl4j.modelx.train;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class AbsTrainer {

    protected static void setListeners(ComputationGraph... computationGraphs) {

        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();

        uiServer.attach(statsStorage);

        for(ComputationGraph computationGraph:computationGraphs){

            computationGraph.setListeners(new PerformanceListener(10, true),new StatsListener(statsStorage));

        }
    }


    protected static int getBatchSize(INDArray array){
        int batchSize=(int)array.size(0);
        return batchSize;
    }

    protected static int clipBatchSize(int batchSize,int max){
        if(batchSize>max){
            return max;
        }
        return batchSize;
    }
}
