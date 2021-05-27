package org.freeware.dl4j.modelx.dataset.mnist;

import io.jhdf.HdfFile;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import java.io.File;



/**
 * 3维Mnist数据迭代器
 * 数据下载地址
 * https://www.kaggle.com/daavoo/3d-mnist
 * @author wenfengxu
 *
 */
@Slf4j
public class Mnist3dDataSetIterator implements MultiDataSetIterator {


	protected int   batchSize = 10;

	private   int   currentBatch=0;

	private   int    totalBatches=1;

	private INDArray features;

	private INDArray labels;


	public Mnist3dDataSetIterator(String dataSetPath, String dataSetType, int batchSize) {

		super();

        //检查并设置目录
		checkAndSetDirectory(dataSetPath);

		File file = new File(dataSetPath);

		try (HdfFile hdfFile = new HdfFile(file)) {

			String featurePath="X_".concat(dataSetType);

			String labelPath="y_".concat(dataSetType);

			Object featureData = hdfFile.getDatasetByPath(featurePath).getData();

			Object labelData =hdfFile.getDatasetByPath(labelPath).getData();

			double[][] featuresArray= (double[][]) featureData;

			long[] labelsArray= (long[]) labelData;

			features=Nd4j.create (featuresArray);

			labels= Nd4j.create ( labelsArray);

			this.batchSize= batchSize;

		}

		//设置总的小批次数量
		setTotalBatches(batchSize);

	}

	private void setTotalBatches(int batchSize) {
		if(this.features.size(0)%batchSize==0){
			this.totalBatches=(int)this.features.size(0)/batchSize;
		}else {
			this.totalBatches=(int)this.features.size(0)/batchSize+1;
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

		if(endIndex>this.features.size(0)){

			endIndex=(int)this.features.size(0);

		}

		INDArray batchFeatures=this.features.get(new INDArrayIndex[]{NDArrayIndex.interval(startIndex,endIndex),NDArrayIndex.all()});

		//batchFeatures=batchFeatures.reshape(realBatchSize,1,16,16,16);

		INDArray[] featuresArray=new INDArray[] {batchFeatures};

		INDArray batchLabels=this.labels.get(new INDArrayIndex[]{NDArrayIndex.interval(startIndex,endIndex),NDArrayIndex.all()});

		INDArray[] labelsArray=new INDArray[] {batchLabels};

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


		 this.currentBatch=0;
		
	}






}
