package org.freeware.dl4j.modelx.dataset;


import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.freeware.dl4j.modelx.ExtendedFileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.text.DecimalFormat;
import java.util.*;


/**
 *
 * Data set Iterator for yolo3
 *  @author wenfengxu
 *
 */

@Slf4j
public class Yolo3DataSetIterator implements MultiDataSetIterator {


	private static final String IMAGES_FOLDER="JPEGImages";

	private static final String ANNOTATIONS_FOLDER="Annotations";

	protected int   batchSize = 10;
	
	private   int   currentBatch=0;
	
	private   int    totalBatches=1;
	
	private  List<File> featureFiles;
	
	private NativeImageLoader nativeImageLoader ;
	
    private DataSetPreProcessor preProcessor;
	
	private Random rnd=new Random(123456);

	private String labelPath;

	private String featurePath;

	private int maxBoxPerImage=0;

	private  ImageObjectLabelProvider labelProvider;


	private int[] inputShape={3,416,416};

	private String[] labels={};

	private List<ImageObject> anchors=new ArrayList<>(9);

	public Yolo3DataSetIterator(String dataSetPath,
								int batchSize,
								String[] labels ,
								int[][] bigBoundingBoxPriors,
								int[][] mediumBoundingBoxPriors,
								int[][] smallBoundingBoxPriors) {

		super();


		this.labels=labels;

        this.nativeImageLoader=new NativeImageLoader(inputShape[1],inputShape[2],inputShape[0]);

        setAnchors(bigBoundingBoxPriors, mediumBoundingBoxPriors, smallBoundingBoxPriors);

		//检查并设置目录
		checkAndSetDirectory(dataSetPath);

		this.labelProvider=new VocLabelProvider(dataSetPath);

		this.maxBoxPerImage=getMaxBoxPerImage();

		//设置小批量大小
		this.batchSize = batchSize;
		//删除无效数据，即把关键点缺失的数据都删除

		//获取所有特征数据
    	this.featureFiles = ExtendedFileUtils.listFiles(featurePath, new String[] {"jpg","png"}, false);
    	//设置总的小批次数量
		if(featureFiles.size()%batchSize==0){
			this.totalBatches=featureFiles.size()/batchSize;
		}else {
			this.totalBatches=featureFiles.size()/batchSize+1;
		}
		//随机打乱
       	Collections.shuffle(featureFiles,rnd);
	}

	private void setAnchors(int[][] bigBoundingBoxPriors, int[][] mediumBoundingBoxPriors, int[][] smallBoundingBoxPriors) {
		for(int i=0;i<smallBoundingBoxPriors.length;i++){

			anchors.add(new ImageObject(0,0,smallBoundingBoxPriors[i][0],smallBoundingBoxPriors[i][1],""));
		}
		for(int i=0;i<mediumBoundingBoxPriors.length;i++){

			anchors.add(new ImageObject(0,0,mediumBoundingBoxPriors[i][0],mediumBoundingBoxPriors[i][1],""));
		}
		for(int i=0;i<bigBoundingBoxPriors.length;i++){

			anchors.add(new ImageObject(0,0,bigBoundingBoxPriors[i][0],bigBoundingBoxPriors[i][1],""));
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

		this.labelPath=dataSetPath.concat(ANNOTATIONS_FOLDER).concat(File.separator);

		if(!new File(this.labelPath).exists()){
			throw new IllegalStateException(this.labelPath.concat("directory does not exist"));
		}
		this.featurePath=dataSetPath.concat(IMAGES_FOLDER).concat(File.separator);

		if(!new File(this.featurePath).exists()){
			throw new IllegalStateException(this.featurePath.concat("directory does not exist"));
		}
	}

	/**
	 * 遍历的所有标签文件，统计每个文件的边界框数量
	 * 返回最大值
	 * @return
	 */
	private int getMaxBoxPerImage(){

		List<File> labelFileList=ExtendedFileUtils.listFiles(this.labelPath, new String[] {"xml"}, false);

		List<Integer> imageObjectSizeList=new ArrayList<>(100);

		for(File labelFile:labelFileList){

			List<ImageObject> imageObjectList=labelProvider.getImageObjectsForPath(labelFile.getAbsolutePath());

			imageObjectSizeList.add(imageObjectList.size());
		}
		return Collections.max(imageObjectSizeList);
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

		if(endIndex>this.featureFiles.size()){

			endIndex=this.featureFiles.size();

			startIndex=endIndex-this.batchSize;
		}

		int realBatchSize=endIndex-startIndex;
		//特征存放，便于拼接
		INDArray[] featureList=new INDArray[realBatchSize] ;
		//标签存放数，便于拼接
		INDArray[] labelList=new INDArray[realBatchSize];

		INDArray labelBig=Nd4j.zeros(realBatchSize,13,13,3,4+1+this.labels.length);

		INDArray labelMedium=Nd4j.zeros(realBatchSize,26,26,3,4+1+this.labels.length);

		INDArray labelSmall=Nd4j.zeros(realBatchSize,52,52,3,4+1+this.labels.length);

		INDArray[] labels=new INDArray[]{labelBig,labelMedium,labelSmall} ;

		int exampleCount=0;

		for(int exampleIndex=startIndex;exampleIndex<endIndex;exampleIndex++) {
			
			File featureFile=featureFiles.get(exampleIndex);
			
			try {

				//得到特征值
				INDArray feature = nativeImageLoader.asMatrix(featureFile);

				featureList[exampleIndex]=feature;
				//通过特征文件名来得到标签文件名
				String   labelFileName = this.labelPath.concat(featureFile.getName().replace(".jpg",".xml").replace(".png",".xml"));
                //得到指定标签文件的所有边界框
				List<ImageObject> imageObjectList=labelProvider.getImageObjectsForPath(labelFileName);

				for(ImageObject imageObject:imageObjectList){

					ImageObject maxAnchor=null;

					int maxIndex=-1;

					double maxIou=-1D;

					ImageObject shiftedBox=new ImageObject(0,0,imageObject.getX2()-imageObject.getX1(),imageObject.getY2()-imageObject.getY1(),"");

                    for(int anchorIndex=0;anchorIndex<anchors.size();anchorIndex++){

                    	ImageObject anchor=this.anchors.get(anchorIndex);

						double iou=iou(shiftedBox,anchor);

                    	if(maxIou<iou){
                    		maxIou=iou;
                    		maxIndex=anchorIndex;
							maxAnchor=anchor;
						}

					}
                    //边界框放于与之iou最大的anchor所在的输出
					INDArray currentLabel=labels[maxIndex/3];

                    long gridWidth=currentLabel.shape()[1];

					long gridHeight=currentLabel.shape()[2];

					double centerX = .5*(imageObject.getX1() + imageObject.getX2());
					// sigma(t_x) + c_x
					centerX = centerX / inputShape[1] * gridWidth ;

					double centerY = .5*(imageObject.getY1() + imageObject.getY2());
					// sigma(t_y) + c_y
					centerY = centerY / inputShape[2] * gridHeight ;

					// determine the sizes of the bounding box 数据归一化
					double	width = Math.log(imageObject.getX2()- imageObject.getX1()) / maxAnchor.getX2();

					double	height = Math.log(imageObject.getY2()- imageObject.getY1()) / maxAnchor.getY2();

                    double[] box={centerX,centerY,width,height};

                    int classIndex= Arrays.asList(labels).indexOf(imageObject.getLabel());

                    int gridX=(int)Math.floor(centerX);

					int gridY=(int)Math.floor(centerY);

                    int anchorIndex=maxIndex%3;

                    currentLabel.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount), NDArrayIndex.point(gridX), NDArrayIndex.point(gridY), NDArrayIndex.point(anchorIndex),NDArrayIndex.all()},0);

					for(int boxValueIndex=0;boxValueIndex<box.length;boxValueIndex++){

						currentLabel.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount), NDArrayIndex.point(gridX), NDArrayIndex.point(gridY), NDArrayIndex.point(anchorIndex),NDArrayIndex.point(boxValueIndex)},box[boxValueIndex]);
					}
                    currentLabel.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount), NDArrayIndex.point(gridX), NDArrayIndex.point(gridY), NDArrayIndex.point(anchorIndex),NDArrayIndex.point(4)},1.0);

					currentLabel.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount), NDArrayIndex.point(gridX), NDArrayIndex.point(gridY), NDArrayIndex.point(anchorIndex),NDArrayIndex.point(4+1+classIndex)},1.0);

				}
				exampleCount=exampleCount+1;

				log.info(labels.hashCode()+"----");

			} catch (Exception e) {

				log.error("",e);
			}
			
    	}
		//按小批量维度拼接标签数据
		INDArray features=Nd4j.concat(0,featureList);

		INDArray[] featuresArray=new INDArray[] {features};

		// we have three outputs here ,big medium and small
		INDArray[] labelsArray=new INDArray[] {labelBig,labelMedium,labelSmall};

		MultiDataSet multiDataSet=new MultiDataSet(featuresArray,labelsArray);

        //小批量计数器加1
		currentBatch++;

		return multiDataSet;

	}

   private double iou(ImageObject imageObject1,ImageObject imageObject2){

	   DetectedObject detectedObject1=convertToDetectedObject(imageObject1);

	   DetectedObject detectedObject2=convertToDetectedObject(imageObject2);

	   return YoloUtils.iou(detectedObject1,detectedObject2);
   }

   private DetectedObject convertToDetectedObject(ImageObject imageObject){

	  return  new DetectedObject(0,  imageObject.getX2()/2,  imageObject.getY2()/2,  imageObject.getX2(),  imageObject.getY2(),null, 0);

	}



	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

	}

	@Override
	public MultiDataSetPreProcessor getPreProcessor() {
		return null;
	}


	/**
	 * 标签归一化
	 * @param labels
	 */
	public void transformLabel(INDArray labels) {
		  //96是输入图片的大小
		  labels.divi(96);
		  
		  labels.subi(0.5);
	}

	/**
	 * 标签还原（返归一化）
	 * @param labels
	 */
	public void revertLabel(INDArray labels) {
		  
		  labels.addi(0.5);
		  
		  labels.muli(96);
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
		 Collections.shuffle(featureFiles,rnd);
		 this.currentBatch=0;
		
	}






}
