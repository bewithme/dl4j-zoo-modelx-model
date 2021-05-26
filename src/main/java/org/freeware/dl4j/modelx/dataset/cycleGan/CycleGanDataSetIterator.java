package org.freeware.dl4j.modelx.dataset.cycleGan;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.freeware.dl4j.modelx.utils.ExtendedFileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Random;


/**
 *

 * @author wenfengxu
 *
 */
@Slf4j
public class CycleGanDataSetIterator implements MultiDataSetIterator {


	private static final String IMAGES_FOLDER="trainA";

	private static final String LABELS_FOLDER ="trainB";

	protected int   batchSize = 10;

	private   int   currentBatch=0;

	private   int    totalBatches=1;

	private  List<File> featureFiles;

	private  List<File> labelFiles;

	private NativeImageLoader nativeImageLoader ;


	private Random random =new Random(123456);

	private String labelPath;

	private String featurePath;


	public CycleGanDataSetIterator(String dataSetPath,
                                   int batchSize, int imageHeight, int imageWidth, int channels) {

		super();

        this.nativeImageLoader=new NativeImageLoader(imageHeight,imageWidth,channels);

      //检查并设置目录
		checkAndSetDirectory(dataSetPath);


		//设置小批量大小
		this.batchSize = batchSize;
		//获取所有特征数据
    	this.featureFiles = ExtendedFileUtils.listFiles(featurePath, new String[] {"jpg","png"}, false);

		this.labelFiles = ExtendedFileUtils.listFiles(labelPath, new String[] {"jpg","png"}, false);

		//设置总的小批次数量
		setTotalBatches(batchSize);
		//随机打乱
       	Collections.shuffle(featureFiles, random);
	}

	private void setTotalBatches(int batchSize) {
		if(featureFiles.size()%batchSize==0){
			this.totalBatches=featureFiles.size()/batchSize;
		}else {
			this.totalBatches=featureFiles.size()/batchSize+1;
		}
	}



	/**
	 * 检查并设置特征与标签目录
	 * @param dataSetPath
	 */
	private void checkAndSetDirectory(String dataSetPath) {

		if(!new File(dataSetPath).exists()){
			throw new IllegalStateException(dataSetPath.concat("directory does not exist"));
		}

		if(!dataSetPath.endsWith(File.separator)){
			dataSetPath=dataSetPath.concat(File.separator);
		}

		labelPath=dataSetPath.concat(LABELS_FOLDER).concat(File.separator);

		if(!new File(labelPath).exists()){
			throw new IllegalStateException(labelPath.concat("directory does not exist"));
		}
		featurePath=dataSetPath.concat(IMAGES_FOLDER).concat(File.separator);

		if(!new File(featurePath).exists()){
			throw new IllegalStateException(featurePath.concat("directory does not exist"));
		}
	}



	/**
	 * 
	 */
	private static final long serialVersionUID = -6357192255171476363L;

	@Override
	public boolean hasNext() {
		
		return this.currentBatch<this.totalBatches;
	}

	@Override
	public MultiDataSet next() {
		
		return next(batchSize);
	}

	@Override
	public MultiDataSet next(int batchSize) {
		//文件读取开始指针
		int startIndex=batchSize*currentBatch;
		//文件读取结束指针
		int endIndex=batchSize*(currentBatch+1);

		if(endIndex>featureFiles.size()){

			endIndex=featureFiles.size();

		}
		int realBatchSize=endIndex-startIndex;
		//特征存放，便于拼接
		INDArray[] imageFeatureList=new INDArray[realBatchSize] ;

		INDArray[] imageLabelList=new INDArray[realBatchSize] ;

		int exampleCount=0;

		for(int exampleIndex=startIndex;exampleIndex<endIndex;exampleIndex++) {
			
			File featureFile=featureFiles.get(exampleIndex);

			File labelFile=labelFiles.get(exampleIndex);
			
			try {
				//得到图片特征[N,C,H,W]
				INDArray imageFeature = nativeImageLoader.asMatrix(featureFile);

				INDArray label = nativeImageLoader.asMatrix(labelFile);

				imageFeatureList[exampleCount]=imageFeature;

				imageLabelList[exampleCount]=label;

				exampleCount=exampleCount+1;

			} catch (Exception e) {

				log.error("",e);
			}

    	}
		//按小批量维度拼接标签数据
		INDArray imageFeature=Nd4j.concat(0,imageFeatureList);

		INDArray[] featuresArray=new INDArray[] {imageFeature};

		INDArray imageLabel=Nd4j.concat(0,imageLabelList);

		INDArray[] labelsArray=new INDArray[] {imageLabel};

		MultiDataSet multiDataSet=new MultiDataSet(featuresArray,labelsArray);

        //小批量计数器加1
		currentBatch++;

		return multiDataSet;

	}









	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

	}

	@Override
	public MultiDataSetPreProcessor getPreProcessor() {
		return null;
	}









	@Override
	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return true;
	}




	@Override
	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}




	@Override
	public void reset() {
		 Collections.shuffle(featureFiles, random);
		 this.currentBatch=0;
		
	}








}
