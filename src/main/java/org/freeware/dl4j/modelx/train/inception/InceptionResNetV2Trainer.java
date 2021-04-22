package org.freeware.dl4j.modelx.train.inception;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.freeware.dl4j.modelx.utils.DataSetUtils;
import org.freeware.dl4j.modelx.model.inception.InceptionResNetV2;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class InceptionResNetV2Trainer {

	public static void main(String[] args) {


		String dataPath="/Users/wenfengxu/Downloads/59760_840806_bundle_archive/raw-img";
		
		int numPossibleLabels=DataSetUtils.getFileDirectoriesCount(dataPath);

		DataSetIterator dataSetIterator= null;
		try {
			dataSetIterator = DataSetUtils.getDataSetIterator(dataPath,2,numPossibleLabels,299,299,3);
		} catch (IOException e) {
			e.printStackTrace();
		}

		InceptionResNetV2 inceptionResNetV2= InceptionResNetV2.builder().numClasses(numPossibleLabels).build();

		
		ComputationGraph model=inceptionResNetV2.init();
		
		log.info(model.summary());

		UIServer uiServer = UIServer.getInstance();

		StatsStorage statsStorage = new InMemoryStatsStorage();

		uiServer.attach(statsStorage);

		model.setListeners(new StatsListener(statsStorage));
		
		model.fit(dataSetIterator);

		try {
			ModelSerializer.writeModel(model,new File("InceptionResnetV2.zip"),true);
		} catch (IOException e) {
			e.printStackTrace();
		}


	}
	
	

}
