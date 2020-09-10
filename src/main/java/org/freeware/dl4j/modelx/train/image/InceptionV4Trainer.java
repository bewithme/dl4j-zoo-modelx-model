package org.freeware.dl4j.modelx.train.image;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.freeware.dl4j.modelx.DataSetUtils;

import org.freeware.dl4j.modelx.model.image.InceptionV4;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;

import java.io.File;
import java.io.IOException;

@Slf4j
public class InceptionV4Trainer {

    public static void main(String[] args) {


        String dataPath="/Users/wenfengxu/Desktop/临时文件/stars_face";

        int numPossibleLabels= DataSetUtils.getFileDirectoriesCount(dataPath);

        DataSetIterator dataSetIterator= null;
        try {
            dataSetIterator = DataSetUtils.getDataSetIterator(dataPath,2,numPossibleLabels,299,299,3);
        } catch (IOException e) {
            log.error("",e);
        }

        InceptionV4 inceptionV4= InceptionV4.builder().numClasses(numPossibleLabels).build();

        ComputationGraph model=inceptionV4.init();

        log.info(model.summary());

        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();

        uiServer.attach(statsStorage);

        model.setListeners(new StatsListener(statsStorage));

        model.fit(dataSetIterator);

        try {
            ModelSerializer.writeModel(model,new File("InceptionV4.zip"),true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
