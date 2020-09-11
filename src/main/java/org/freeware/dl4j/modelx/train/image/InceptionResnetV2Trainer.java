package org.freeware.dl4j.modelx.train.image;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.freeware.dl4j.modelx.DataSetUtils;
import org.freeware.dl4j.modelx.model.image.InceptionResnetV2;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

@Slf4j
public class InceptionResnetV2Trainer {

	public static void main(String[] args) {


		String dataPath="/Users/wenfengxu/Downloads/59760_840806_bundle_archive/raw-img";
		
		int numPossibleLabels=DataSetUtils.getFileDirectoriesCount(dataPath);

		DataSetIterator dataSetIterator= null;
		try {
			dataSetIterator = DataSetUtils.getDataSetIterator(dataPath,2,numPossibleLabels,299,299,3);
		} catch (IOException e) {
			e.printStackTrace();
		}

		InceptionResnetV2 inceptionResNetV2= InceptionResnetV2.builder().numClasses(numPossibleLabels).build();

		
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
