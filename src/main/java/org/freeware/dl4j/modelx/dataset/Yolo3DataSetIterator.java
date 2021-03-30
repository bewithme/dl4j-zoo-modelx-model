package org.freeware.dl4j.modelx.dataset;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;

import org.freeware.dl4j.modelx.utils.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;


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
	
	private Random random =new Random(123456);

	private String labelPath;

	private String featurePath;

	private int maxBoxPerImage=0;

	private  ImageObjectLabelProvider labelProvider;

	private int[] inputShape={3,416,416};

	private int[] strides={32,16,8};

	private String[] labels={};
    //shape=[3,3,2]保存所有先验框
	private INDArray priorBoundingBoxes;

	private float jitter=0.3f;
	//每个单元格负责检测的先验框数量，一般为3
	private  int numberOfPriorBoundingBoxPerGridCell=3;

	public Yolo3DataSetIterator(String dataSetPath,
								int batchSize,
								String[] labels ,
								int[][] bigPriorBoundingBoxes,
								int[][] mediumPriorBoundingBoxes,
								int[][] smallPriorBoundingBoxes) {

		super();

		this.labels=labels;

        this.nativeImageLoader=new NativeImageLoader();

        setPriorBoundingBoxes(bigPriorBoundingBoxes, mediumPriorBoundingBoxes, smallPriorBoundingBoxes);
		//检查并设置目录
		checkAndSetDirectory(dataSetPath);

		this.labelProvider=new VocLabelProvider(dataSetPath);

		this.maxBoxPerImage=getMaxBoxPerImage();
		//设置小批量大小
		this.batchSize = batchSize;
		//获取所有特征数据
    	this.featureFiles = ExtendedFileUtils.listFiles(featurePath, new String[] {"jpg","png"}, false);
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

	private void setPriorBoundingBoxes(int[][] bigPriorBoundingBoxes, int[][] mediumPriorBoundingBoxes, int[][] smallPriorBoundingBoxes) {
		int[] shape=new int[]{1,3,2};
		INDArray big=Nd4j.create(bigPriorBoundingBoxes).reshape(shape);
		INDArray medium=Nd4j.create(mediumPriorBoundingBoxes).reshape(shape);
		INDArray small=Nd4j.create(smallPriorBoundingBoxes).reshape(shape);
		this.priorBoundingBoxes =Nd4j.concat(0,big,medium,small);
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

		labelPath=dataSetPath.concat(ANNOTATIONS_FOLDER).concat(File.separator);

		if(!new File(labelPath).exists()){
			throw new IllegalStateException(labelPath.concat("directory does not exist"));
		}
		featurePath=dataSetPath.concat(IMAGES_FOLDER).concat(File.separator);

		if(!new File(featurePath).exists()){
			throw new IllegalStateException(featurePath.concat("directory does not exist"));
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

		if(endIndex>featureFiles.size()){

			endIndex=featureFiles.size();

		}

		int realBatchSize=endIndex-startIndex;
		//特征存放，便于拼接
		INDArray[] imageFeatureList=new INDArray[realBatchSize] ;

		INDArray[] labelBigMediumSmall = getLabelBigMediumSmall(realBatchSize);

		int exampleCount=0;

		for(int exampleIndex=startIndex;exampleIndex<endIndex;exampleIndex++) {
			
			File featureFile=featureFiles.get(exampleIndex);
			
			try {
				//得到图片特征[N,C,H,W]
				INDArray imageFeature = nativeImageLoader.asMatrix(featureFile);
				//得到当前图片对应的所有边界框
				List<ImageObject> boundingBoxesList=getBoundingBoxes(featureFile);
                //得到图片增强结果
				ImageAndBoundingBoxes imageAugmentResult=augmentImage(imageFeature,boundingBoxesList,inputShape[1],inputShape[2]);
				//设小批量中的置图片特征
				imageFeatureList[exampleCount]=imageAugmentResult.getImage();
                //得到图片增强后的边界框
				boundingBoxesList=imageAugmentResult.getBoundingBoxesList();
                //用来统计参与预测的大中小边界框
				INDArray boxesCount=Nd4j.zeros(3);

				for(ImageObject boundingBox:boundingBoxesList){
                    //[4]
					INDArray  smoothClassOneHot=getSmoothClassOneHot(boundingBox);
					//[4]
					INDArray centerXyWhBoundingBox = getCenterXyWhBoundingBox(boundingBox);
					//[3,4]
					INDArray bigMediumSmallScaledBoundingBox= getBigMediumSmallScaledBoundingBox(centerXyWhBoundingBox);

					List<INDArray> iouList=new ArrayList<>(10);

					Boolean existPositive = Boolean.FALSE;

					for(int labelIndex=0;labelIndex<labelBigMediumSmall.length;labelIndex++){
						//[3,4]
						INDArray threeBoundingBoxPriors = getThreeBoundingBoxPriors(bigMediumSmallScaledBoundingBox, labelIndex);
                        //[4]
						INDArray scaledBoundingBox=bigMediumSmallScaledBoundingBox.get(new INDArrayIndex[]{NDArrayIndex.point(labelIndex),NDArrayIndex.all()});
                        //[1,4]
						INDArray expandDimsScaledBoundingBox=Nd4j.expandDims(scaledBoundingBox,0);
                        //[3]
						INDArray scaledIou= YoloUtils.getIou(threeBoundingBoxPriors,expandDimsScaledBoundingBox);

						float[] scaledIouVector=scaledIou.toFloatVector();

						float scaledIouThreshold=0.3f;

						iouList.add(scaledIou);

						int greaterThanScaledIouThresholdCount = getGreaterThanScaledIouThresholdCount(scaledIou, scaledIouThreshold);

						if(greaterThanScaledIouThresholdCount>0){

							INDArray label=labelBigMediumSmall[labelIndex];

							for(int scaledIouVectorIndex=0;scaledIouVectorIndex<scaledIouVector.length;scaledIouVectorIndex++){

								if(scaledIouVector[scaledIouVectorIndex]>scaledIouThreshold){

									setLabelValues(label, exampleCount,  scaledIouVectorIndex, scaledBoundingBox,smoothClassOneHot);

								}
							}

							setExtraValues(exampleCount, boxesCount, labelIndex, scaledBoundingBox, label);

							existPositive = Boolean.TRUE;
						}
					}

					if(existPositive==Boolean.FALSE){

						int bestPriorBoundingBoxIndex = getBestPriorBoundingBoxIndex(iouList);

						int bestPriorBoundingBoxCategoryIndex=bestPriorBoundingBoxIndex/numberOfPriorBoundingBoxPerGridCell;

						int bestPriorBoundingBoxCategoryIndexMode=bestPriorBoundingBoxIndex%numberOfPriorBoundingBoxPerGridCell;
						//[4]
						INDArray scaledBoundingBox=bigMediumSmallScaledBoundingBox.get(new INDArrayIndex[]{NDArrayIndex.point(bestPriorBoundingBoxCategoryIndex),NDArrayIndex.all()});

						INDArray label=labelBigMediumSmall[bestPriorBoundingBoxCategoryIndex];

						setLabelValues(label, exampleCount,  bestPriorBoundingBoxCategoryIndexMode, scaledBoundingBox,smoothClassOneHot);

						setExtraValues(exampleCount, boxesCount, bestPriorBoundingBoxCategoryIndex, scaledBoundingBox, label);

					}
				}
				exampleCount=exampleCount+1;

			} catch (Exception e) {

				log.error("",e);
			}
			
    	}
		//按小批量维度拼接标签数据
		INDArray imageFeature=Nd4j.concat(0,imageFeatureList);

		INDArray[] featuresArray=new INDArray[] {imageFeature};

		MultiDataSet multiDataSet=new MultiDataSet(featuresArray,labelBigMediumSmall);

        //小批量计数器加1
		currentBatch++;

		return multiDataSet;

	}

	private ImageAndBoundingBoxes augmentImage(INDArray image, List<ImageObject> boundingBoxesList, int inputHeight, int inputWidth) throws IOException {

		ImageAndBoundingBoxes imageAndBoundingBoxes=randomHorizontalFlip(image,boundingBoxesList);

		imageAndBoundingBoxes=randomCrop(imageAndBoundingBoxes.getImage(),imageAndBoundingBoxes.getBoundingBoxesList());

		imageAndBoundingBoxes=randomTranslate(imageAndBoundingBoxes.getImage(),imageAndBoundingBoxes.getBoundingBoxesList());

		imageAndBoundingBoxes=ctvColorAndPadding(imageAndBoundingBoxes.getImage(),imageAndBoundingBoxes.getBoundingBoxesList(),inputHeight,inputWidth);

		List<ImageObject> clippedBoxes=clipBoxes(imageAndBoundingBoxes.getImage(),imageAndBoundingBoxes.getBoundingBoxesList());

		imageAndBoundingBoxes.setBoundingBoxesList(clippedBoxes);

		return imageAndBoundingBoxes;
	}



		/**
         * 随机水平翻转
         * @param image
         * @param boundingBoxesList
         * @return
         */
	private ImageAndBoundingBoxes randomHorizontalFlip(INDArray image, List<ImageObject> boundingBoxesList){

		INDArray flipImage=INDArrayUtils.horizontalFlip(image);

		long width=image.shape()[image.shape().length-1];

		List<ImageObject> flipBoundingBoxesList=new ArrayList<>(5);

		for(ImageObject imageObject:boundingBoxesList){

			int x1=Integer.parseInt(String.valueOf(width-imageObject.getX2()));

			int x2=Integer.parseInt(String.valueOf(width-imageObject.getX1()));

			ImageObject flipImageObject=new ImageObject(x1,imageObject.getY1(),x2,imageObject.getY2(),imageObject.getLabel());

			flipBoundingBoxesList.add(flipImageObject);
		}

		ImageAndBoundingBoxes imageAndBoundingBoxes =new ImageAndBoundingBoxes(flipImage,flipBoundingBoxesList);

		return imageAndBoundingBoxes;
	}

	/**
	 * 随机载剪
	 * @param image
	 * @param boundingBoxesList
	 * @return
	 */
	private ImageAndBoundingBoxes randomCrop(INDArray image, List<ImageObject> boundingBoxesList){

		ImageObject maxBox=getMaxBox(boundingBoxesList);

		int width=Integer.parseInt(String.valueOf(image.shape()[image.shape().length-1]));

		int height=Integer.parseInt(String.valueOf(image.shape()[image.shape().length-2]));

		int max_l_trans = maxBox.getX1();

		int max_u_trans = maxBox.getY1();

		int max_r_trans = width - maxBox.getX2();

		int max_d_trans = height  - maxBox.getY2();

		int crop_xmin = Math.max(0, maxBox.getX1() - randomUniform(0, max_l_trans));

		int crop_ymin = Math.max(0, maxBox.getY1() - randomUniform(0, max_u_trans));

		int crop_xmax = Math.max(width,  maxBox.getX2() + randomUniform(0, max_r_trans));

		int crop_ymax = Math.max(height, maxBox.getY2() + randomUniform(0, max_d_trans));

		INDArrayIndex[] cropImageIndexes=INDArrayUtils.getLastTwoDimensionIndexes(image,crop_xmin,crop_xmax,crop_ymin,crop_ymax);

		INDArray cropImage=image.get(cropImageIndexes);

		List<ImageObject> cropBoundingBoxesList=new ArrayList<>(5);

		for(ImageObject imageObject:boundingBoxesList){

			int x1=imageObject.getX1()-crop_xmin;

			int x2=imageObject.getX2()-crop_xmin;

			int y1=imageObject.getY1()-crop_ymin;

			int y2=imageObject.getY2()-crop_ymin;

			ImageObject corpImageObject=new ImageObject(x1,y1,x2,y2,imageObject.getLabel());

			cropBoundingBoxesList.add(corpImageObject);
		}
		ImageAndBoundingBoxes imageAndBoundingBoxes =new ImageAndBoundingBoxes(cropImage,cropBoundingBoxesList);

		return imageAndBoundingBoxes;
	}


	/**
	 * 仿射变换
	 * @param image
	 * @param boundingBoxesList
	 * @return
	 */
	private ImageAndBoundingBoxes randomTranslate(INDArray image, List<ImageObject> boundingBoxesList){

		ImageObject maxBox=getMaxBox(boundingBoxesList);

		int width=Integer.parseInt(String.valueOf(image.shape()[image.shape().length-1]));

		int height=Integer.parseInt(String.valueOf(image.shape()[image.shape().length-2]));

		int max_l_trans = maxBox.getX1();

		int max_u_trans = maxBox.getY1();

		int max_r_trans = width - maxBox.getX2();

		int max_d_trans = height  - maxBox.getY2();

		int tx =randomUniform(-(max_l_trans - 1), (max_r_trans - 1));

		int ty =randomUniform(-(max_u_trans - 1), (max_d_trans - 1));

		int[][] transMat=new int[][]{{1, 0, tx},{0, 1, ty}};

		INDArray warpAffineImage= ImageUtils.cv2WarpAffine(image, width, height, transMat);

		List<ImageObject> translateBoundingBoxesList=new ArrayList<>(5);

		for(ImageObject imageObject:boundingBoxesList){

			int x1=imageObject.getX1()+tx;

			int x2=imageObject.getX2()+tx;

			int y1=imageObject.getY1()+ty;

			int y2=imageObject.getY2()+ty;

			ImageObject translateImageObject=new ImageObject(x1,y1,x2,y2,imageObject.getLabel());

			translateBoundingBoxesList.add(translateImageObject);
		}

		ImageAndBoundingBoxes imageAndBoundingBoxes =new ImageAndBoundingBoxes(warpAffineImage,translateBoundingBoxesList);

		return imageAndBoundingBoxes;
	}


	/**
	 * 填充
	 * @param image
	 * @param boundingBoxesList
	 * @param targetWidth
	 * @param targetHeight
	 * @return
	 * @throws IOException
	 */
	private ImageAndBoundingBoxes ctvColorAndPadding(INDArray image, List<ImageObject> boundingBoxesList,int targetHeight,int targetWidth) throws IOException {

		int iw=targetWidth;

		int ih=targetHeight;

		image=ctvColor(image,COLOR_RGB2HSV);

		int width=Integer.parseInt(String.valueOf(image.shape()[image.shape().length-1]));

		int height=Integer.parseInt(String.valueOf(image.shape()[image.shape().length-2]));

		float scale = Math.min(Float.parseFloat(String.valueOf(iw))/Float.parseFloat(String.valueOf(width)), Float.parseFloat(String.valueOf(ih))/Float.parseFloat(String.valueOf(height)));

		int nw= (int)(scale * width);

		int nh=(int) (scale * height);

		INDArray imageResized=ImageUtils.resize(image,nw,nh);

		INDArray imagePadded=INDArrayUtils.full(new long[]{image.shape()[0],3,ih,iw},128.0f) ;

		int dw =(iw-nw) /2;

		int	dh =(ih-nh) /2;

		imagePadded.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(dh,nh+dh),NDArrayIndex.interval(dw,nw+dw)},imageResized);

		//imagePadded=imagePadded.div(255.f);

		List<ImageObject> paddedBoundingBoxesList=new ArrayList<>(5);

		for(ImageObject imageObject:boundingBoxesList){

			int x1=(int)(imageObject.getX1()* scale) + dw;

			int x2=(int)(imageObject.getX2()* scale) + dw;

			int y1=(int)(imageObject.getY1()* scale) + dh;

			int y2=(int)(imageObject.getY2()* scale) + dh;

			ImageObject paddedImageObject=new ImageObject(x1,y1,x2,y2,imageObject.getLabel());

			paddedBoundingBoxesList.add(paddedImageObject);
		}

		ImageAndBoundingBoxes imageAndBoundingBoxes =new ImageAndBoundingBoxes(imagePadded,paddedBoundingBoxesList);

		return imageAndBoundingBoxes;

	}



	/**
	 * 截取有效边界框
	 * @param image
	 * @param boundingBoxesList
	 * @return
	 */
	private List<ImageObject> clipBoxes(INDArray image,List<ImageObject> boundingBoxesList){

		int width=Integer.parseInt(String.valueOf(image.shape()[image.shape().length-1]));

		int height=Integer.parseInt(String.valueOf(image.shape()[image.shape().length-2]));

		List<ImageObject> paddedBoundingBoxesList=new ArrayList<>(5);

		for(ImageObject imageObject:boundingBoxesList){

			int x1=clip(imageObject.getX1(),0,width);

			int x2=clip(imageObject.getX2(),0,width);

			int y1=clip(imageObject.getY1(),0,height);

			int y2=clip(imageObject.getY2(),0,height);

			ImageObject clipImageObject=new ImageObject(x1,y1,x2,y2,imageObject.getLabel());

			paddedBoundingBoxesList.add(clipImageObject);
		}

		return paddedBoundingBoxesList;
	}

	private int clip(int toClip,int min,int max){
		if(toClip<min){
			return min;
		}
		if(toClip>max){
			return max;
		}
		return toClip;
	}

	private ImageObject getMaxBox(List<ImageObject> boundingBoxesList){

		ImageObject firstImageObject=boundingBoxesList.get(0);

		int minX1=firstImageObject.getX1();

		int minY1=firstImageObject.getY1();

		int maxX2=firstImageObject.getX2();

		int maxY2=firstImageObject.getY2();

		for(ImageObject imageObject:boundingBoxesList){

			if(imageObject.getX1()<minX1){
				minX1=imageObject.getX1();
			}
			if(imageObject.getY1()<minY1){
				minY1=imageObject.getY1();
			}
			if(imageObject.getX2()>maxX2){
				maxX2=imageObject.getX2();
			}
			if(imageObject.getY2()>maxY2){
				maxY2=imageObject.getY2();
			}

		}
		ImageObject maxImageObject=new ImageObject(minX1,minY1,maxX2,maxY2,"");

		return maxImageObject;
	}




	private void setExtraValues(int exampleCount, INDArray boxesCount, int labelIndex, INDArray scaledBoundingBox, INDArray label) {

		int bigMediumSmallScaledBoundingBoxIndex=boxesCount.get(new INDArrayIndex[]{NDArrayIndex.point(labelIndex)}).toIntVector()[0];
		int priorBoundingBoxIndex=bigMediumSmallScaledBoundingBoxIndex%maxBoxPerImage;
		long gridWidth=label.shape()[1];
		long gridHeight=label.shape()[2];
		for (int gridWidthIndex=0;gridWidthIndex<gridWidth;gridWidthIndex++){
			for (int gridHeightIndex=0;gridHeightIndex<gridHeight;gridHeightIndex++){
				label.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount),NDArrayIndex.point(gridWidthIndex),NDArrayIndex.point(gridHeightIndex),NDArrayIndex.point(numberOfPriorBoundingBoxPerGridCell+priorBoundingBoxIndex),NDArrayIndex.interval(0,4)},scaledBoundingBox);
			}
		}
		bigMediumSmallScaledBoundingBoxIndex++;
		boxesCount.put(new INDArrayIndex[]{NDArrayIndex.point(labelIndex)},bigMediumSmallScaledBoundingBoxIndex);

	}

	private void setLabelValues(INDArray label, int exampleCount, int priorBoundingBoxIndex, INDArray scaledBoundingBox, INDArray smoothClassOneHot) {

		INDArray  centerXy=scaledBoundingBox.get(new INDArrayIndex[]{NDArrayIndex.interval(0,2)});

		INDArray  gridXy= Transforms.floor(centerXy);

		int gridX=gridXy.toIntVector()[0];

		int gridY=gridXy.toIntVector()[1];

		label.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount),NDArrayIndex.point(gridX),NDArrayIndex.point(gridY),NDArrayIndex.point(priorBoundingBoxIndex),NDArrayIndex.interval(0,4)},scaledBoundingBox);

		label.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount),NDArrayIndex.point(gridX),NDArrayIndex.point(gridY),NDArrayIndex.point(priorBoundingBoxIndex),NDArrayIndex.interval(4,5)},1.0);

		label.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount),NDArrayIndex.point(gridX),NDArrayIndex.point(gridY),NDArrayIndex.point(priorBoundingBoxIndex),NDArrayIndex.interval(5,5+labels.length)},smoothClassOneHot);
	}

	private int getBestPriorBoundingBoxIndex(List<INDArray> iouList) {

		INDArray iouArray= Nd4j.create(new long[]{iouList.size(),numberOfPriorBoundingBoxPerGridCell});

		for(int iouIndex=0;iouIndex<iouList.size();iouIndex++){
			iouArray.put(new INDArrayIndex[]{NDArrayIndex.point(iouIndex),NDArrayIndex.all()},iouList.get(iouIndex));
		}
		return iouArray.reshape(-1).argMax(-1).toIntVector()[0];
	}


	private INDArray[] getLabelBigMediumSmall(int realBatchSize) {

		INDArray labelBig= Nd4j.zeros(realBatchSize,13,13,3+maxBoxPerImage,4+1+labels.length);

		INDArray labelMedium=Nd4j.zeros(realBatchSize,26,26,3+maxBoxPerImage,4+1+labels.length);

		INDArray labelSmall=Nd4j.zeros(realBatchSize,52,52,3+maxBoxPerImage,4+1+labels.length);

		return new INDArray[]{labelBig,labelMedium,labelSmall};
	}

	private int getGreaterThanScaledIouThresholdCount(INDArray scaledIou, float scaledIouThreshold) {
		//统计数组中数值大于0.3的个数
		MatchCondition matchCondition = new MatchCondition(scaledIou, Conditions.greaterThan(scaledIouThreshold));

		return Nd4j.getExecutioner().exec(matchCondition).toIntVector()[0];
	}

	/**
	 * 获取分类的one hot编码
	 * @param boundingBox
	 * @return
	 */
	private INDArray getSmoothClassOneHot(ImageObject boundingBox) {

		INDArray classOneHot= Nd4j.zeros(new int[]{labels.length});

		int classOneHotIndex= Arrays.asList(labels).indexOf(boundingBox.getLabel());

		classOneHot=classOneHot.put(new INDArrayIndex[]{NDArrayIndex.point(classOneHotIndex)},1.0);

		INDArray uniformDistribution=Nd4j.zeros(new int[]{labels.length});

		uniformDistribution=uniformDistribution.put(new INDArrayIndex[]{NDArrayIndex.all()},1.0f/labels.length);

		float  deta = 0.01f;

		INDArray smoothClassOneHot=classOneHot.mul(1-deta).add(uniformDistribution.mul(deta));

		return smoothClassOneHot;
	}

	/**
	 * 转换为[centerX,centerY,W,H]格式
	 * @param boundingBox
	 * @return
	 */
	private INDArray getCenterXyWhBoundingBox(ImageObject boundingBox) {

		INDArray coordinate= Nd4j.create(new float[]{boundingBox.getX1(),boundingBox.getY1(),boundingBox.getX2(),boundingBox.getY2()});

		INDArray centerXy=coordinate.get(new INDArrayIndex[]{NDArrayIndex.interval(2,4)}).add(coordinate.get(new INDArrayIndex[]{NDArrayIndex.interval(0,2)})).mul(0.5);

		INDArray wh=coordinate.get(new INDArrayIndex[]{NDArrayIndex.interval(2,4)}).sub(coordinate.get(new INDArrayIndex[]{NDArrayIndex.interval(0,2)}));

		return Nd4j.concat(-1,centerXy,wh);
	}

	/**
	 * 获取三个先验框的数组
	 * 它的centerX centerY是从缩放的边界框获取
	 * 它的wh是在用户提供的先验框获取
	 * @param bigMediumSmallScaledBoundingBox
	 * @param labelIndex
	 * @return [3,4]
	 */

	private INDArray getThreeBoundingBoxPriors(INDArray bigMediumSmallScaledBoundingBox, int labelIndex) {
		//创建一个保存三个先验框的数组
		INDArray threeBoundingBoxPriors= Nd4j.zeros(new int[]{3,4});

		INDArray scaledBoundingBoxXy=bigMediumSmallScaledBoundingBox.get(new INDArrayIndex[]{NDArrayIndex.point(labelIndex),NDArrayIndex.interval(0,2)});

		scaledBoundingBoxXy= Transforms.floor(scaledBoundingBoxXy).add(0.5);

		float centerX=scaledBoundingBoxXy.toFloatVector()[0];

		float centerY=scaledBoundingBoxXy.toFloatVector()[1];
		//它的centerX  centerY是从缩放的边界框获取
		threeBoundingBoxPriors.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(0)},centerX);

		threeBoundingBoxPriors.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(1)},centerY);

		INDArray boundingBoxPrior= priorBoundingBoxes.get(new INDArrayIndex[]{NDArrayIndex.point(labelIndex),NDArrayIndex.all()});
		//它的wh是在用户提供的先验框获取
		threeBoundingBoxPriors.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.interval(2,4)},boundingBoxPrior);

		return threeBoundingBoxPriors;
	}


	/**
	 * 得到相对于大中小三个输出的xywh
	 *
	 * @param centerXyWhBoundingBox
	 * @return [3,4]
	 */
	private INDArray getBigMediumSmallScaledBoundingBox(INDArray centerXyWhBoundingBox){

		INDArray bigMediumSmallScaledBoundingBox=null;

		for(int stridesIndex=0;stridesIndex<strides.length;stridesIndex++){

			INDArray scaledBoundingBox=centerXyWhBoundingBox.div(strides[stridesIndex]);

			scaledBoundingBox=Nd4j.expandDims(scaledBoundingBox,0);

			if(bigMediumSmallScaledBoundingBox==null){

				bigMediumSmallScaledBoundingBox=scaledBoundingBox;

			}else{

				bigMediumSmallScaledBoundingBox=Nd4j.concat(0,bigMediumSmallScaledBoundingBox,scaledBoundingBox);
			}

		}
		return bigMediumSmallScaledBoundingBox;
	}


	/**
	 * 保存数据增强的结果
	 * @param featureFile
	 * @param imageAndBoundingBoxes
	 * @throws IOException
	 */
	private void saveImageAndBoundingBoxes(File featureFile, ImageAndBoundingBoxes imageAndBoundingBoxes) throws IOException {

		Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();

		BufferedImage bufferedImage=java2DNativeImageLoader.asBufferedImage(imageAndBoundingBoxes.getImage());

		YoloImageUtils.drawBoundingBoxes(bufferedImage,imageAndBoundingBoxes.getBoundingBoxesList(), Color.green);

		String saveDir=featureFile.getParentFile().getParentFile().getAbsolutePath()+"/augment_image/";

		ExtendedFileUtils.makeDirs(saveDir);

		ImageIO.write(bufferedImage,"jpg",new File(saveDir+featureFile.getName()));
	}

	/**
	 * 获取边界框
	 * @param featureFile
	 * @return
	 */
	private  List<ImageObject> getBoundingBoxes(File featureFile){

		//通过特征文件名来得到标签文件名
		String   labelFileName =labelPath.concat(featureFile.getName().replace(".jpg",".xml").replace(".png",".xml"));
		//得到指定标签文件的所有边界框
		List<ImageObject> boundingBoxList=labelProvider.getImageObjectsForPath(labelFileName);

		return boundingBoxList;
	}







	/**
	 * 图片数据增强
	 * @param image
	 * @param boundingBoxesList
	 * @param inputHeight
	 * @param inputWidth
	 * @return
	 * @throws IOException
	 */
	private ImageAndBoundingBoxes augmentImage1(INDArray image, List<ImageObject> boundingBoxesList, int inputHeight, int inputWidth) throws IOException {

		long imageHeight = image.shape()[2];

		long imageWidth= image.shape()[3];

		float downImageWidth = jitter * imageWidth;

		float downImageHeight = jitter * imageHeight;

		float newAr = (imageWidth + randomUniform(-downImageWidth, downImageWidth)) / (imageHeight + randomUniform(-downImageHeight, downImageHeight));

		float scale = randomUniform(0.25f, 2f);

		int newInputHeight=0;

		int newInputWidth=0;

		if (newAr < 1) {
			newInputHeight = (int) (scale * inputHeight);
			newInputWidth = (int) (inputHeight * newAr);
		}else{
			newInputWidth = (int) (scale * inputWidth);
			newInputHeight = (int) (inputWidth / newAr);
		}
		int dx = (int) randomUniform(0, inputWidth - newInputWidth);
		int dy = (int) randomUniform(0, inputHeight - newInputHeight);

        image=applyRandomScaleAndCrop(image, newInputWidth, newInputHeight, inputWidth, inputHeight, dx, dy);

		image=randomDistortImage(image);

		int flip= random.nextInt(2);

		image=randomFLip(image,flip);

		List<ImageObject> 	correctBoundingBoxesList=correctBoundingBoxes(boundingBoxesList,newInputWidth,newInputHeight,inputWidth,inputHeight,dx,dy,flip,imageWidth,imageHeight);

		return new ImageAndBoundingBoxes(image,correctBoundingBoxesList);
	}



	/**
	 * 修正边界框
	 * @param imageObjectList
	 * @param newInputWidth
	 * @param newInputHeight
	 * @param inputWidth
	 * @param inputHeight
	 * @param dx
	 * @param dy
	 * @param flip
	 * @param imageWidth
	 * @param imageHeight
	 * @return
	 */
	private List<ImageObject> correctBoundingBoxes(List<ImageObject> imageObjectList, int newInputWidth, int newInputHeight, int inputWidth, int inputHeight, int dx,int  dy,int  flip, long  imageWidth, long imageHeight){

		Collections.shuffle(imageObjectList);
		float sx = Float.parseFloat(String.valueOf(newInputWidth))/imageWidth;
		float sy=Float.parseFloat(String.valueOf(newInputHeight))/imageHeight;
		List<ImageObject>  correctBoxesList = new ArrayList<>();
		for(ImageObject imageObject:imageObjectList){
			int xMin = constrain(0, inputWidth, (int) (imageObject.getX1()*sx + dx));
			int xMax = constrain(0, inputWidth, (int) (imageObject.getX2()*sx + dx));
			int yMin = constrain(0, inputHeight, (int) (imageObject.getY1()*sy + dy));
			int yMax = constrain(0, inputHeight, (int) (imageObject.getY2()*sy + dy));
			if (xMax<= xMin || yMax<= yMin){
				continue;
			}
			if (flip == 1) {
				int swap = xMin;
				xMin = inputWidth - xMax;
				xMax = inputWidth - swap;
			}
			correctBoxesList.add(new ImageObject(xMin,yMin,xMax,yMax,imageObject.getLabel()));
		}

		return correctBoxesList;
	}


	/**
	 * 随机缩放与裁剪
	 * @param image
	 * @param newInputWidth
	 * @param newInputHeight
	 * @param netInputWidth
	 * @param netInputHeight
	 * @param dx
	 * @param dy
	 * @return
	 * @throws IOException
	 */
	private INDArray applyRandomScaleAndCrop(INDArray image,int  newInputWidth, int newInputHeight, int netInputWidth, int netInputHeight, int dx, int dy) throws IOException {

		INDArray resizedImage = ImageUtils.resize(image, newInputWidth, newInputHeight);

		long[] shape=resizedImage.shape();

		long imageHeight = shape[2];

		long imageWidth= shape[3];

		if (dx > 0) {
			//resizedImage的宽左边加dx列并用127填充
			resizedImage=Nd4j.pad(resizedImage,new int[][]{{0,0},{0,0},{0,0},{dx,0}}, Pad.Mode.CONSTANT,127);
		}else {
			resizedImage=resizedImage.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(-dx,imageWidth)} );
		}
		if ((newInputWidth + dx) < netInputWidth){
			//resizedImage的宽右边加netInputWidth - (newInputWidth+dx)列并用127填充
			resizedImage=Nd4j.pad(resizedImage,new int[][]{{0,0},{0,0},{0,0},{0,netInputWidth - (newInputWidth+dx)}}, Pad.Mode.CONSTANT,127);
		}

		if (dy > 0){
			//把resizedImage高上边加dy行，并用127填充
			resizedImage=Nd4j.pad(resizedImage,new int[][]{{0,0},{0,0},{dy,0},{0,0}}, Pad.Mode.CONSTANT,127);
		}else{

			resizedImage=resizedImage.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(-dy,imageHeight),NDArrayIndex.all()} );
		}
		if ((newInputHeight + dy) < netInputHeight){
			//把resizedImage高下边加netInputHeight - (newInputHeight+dy)行，并用127填充
			resizedImage=Nd4j.pad(resizedImage,new int[][]{{0,0},{0,0},{0, netInputHeight - (newInputHeight+dy)},{0,0}}, Pad.Mode.CONSTANT,127);
		}

		return resizedImage.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,netInputHeight),NDArrayIndex.interval(0,netInputWidth)});

    }



	/**
	 * 随机扭曲图片
	 * @param image
	 * @return
	 */
	private INDArray randomDistortImage(INDArray image){

		float hue=18f, saturation=1.5f, exposure=1.5f;

		float dhue=randomUniform(-hue,hue);

		float dsat=randScale(saturation);

		float dexp=randScale(exposure);

		image=ctvColor(image,COLOR_RGB2HSV);

		INDArrayIndex[] dsatIndex=new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.all()};
		INDArrayIndex[] dexpIndex= new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(2),NDArrayIndex.all(),NDArrayIndex.all()};

		INDArray dsatArray=image.get(dsatIndex).mul(dsat);
		INDArray dexpArray=image.get(dexpIndex).mul(dexp);

		image.put(dsatIndex,dsatArray);
		image.put(dexpIndex,dexpArray);

		INDArrayIndex[] dhueIndex= new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all()};
		INDArray dhueArray=image.get(dhueIndex).add(dhue);

		INDArray replacedDhueArray=dhueArray.dup();
		BooleanIndexing.replaceWhere(replacedDhueArray, 180, Conditions.greaterThan(180));
		dhueArray=dhueArray.sub(replacedDhueArray);

		replacedDhueArray=dhueArray.dup();
		BooleanIndexing.replaceWhere(replacedDhueArray, 180, Conditions.lessThan(0));
		dhueArray=dhueArray.add(replacedDhueArray);

		image.put(dhueIndex,dhueArray);

		image=ctvColor(image,COLOR_HSV2RGB);

		return image;
	}

	/**
	 * 改变颜色
	 * @param image
	 * @param code
	 * @return
	 */
	private INDArray ctvColor(INDArray image,int code){

		Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();

		Mat mat=java2DNativeImageLoader.asMat(image);

		Mat result = new Mat();

		try {
			cvtColor(mat, result, code);
			return java2DNativeImageLoader.asMatrix(result);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * 随机翻转
	 * @param image
	 * @param flip
	 * @return
	 */
	private INDArray randomFLip(INDArray image,int flip){

		if(flip!=1){
			return image;
		}
		Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();
		Mat mat=java2DNativeImageLoader.asMat(image);
		Mat result = new Mat();
		try {
			flip(mat, result, flip);
			return java2DNativeImageLoader.asMatrix(result);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * 随机缩放
	 * @param scale
	 * @return
	 */
	private float randScale(float scale){
		scale=randomUniform(1,scale);
		if(random.nextInt(2)==0){
			return scale;
		}
		return 1.0f/scale;
	}

	/**
	 * 获取随机数
	 * @param min
	 * @param max
	 * @return
	 */
	private float randomUniform(float min,float max){

		return min + ((max - min) * random.nextFloat());
	}

	/**
	 * 获取随机数
	 * @param min
	 * @param max
	 * @return
	 */
	private int randomUniform(int min,int max){
		int bound=max - min+1;
		if(bound<=0){
          return 0;
		}
		int randomInt=random.nextInt(bound)+min;
		return randomInt;
	}

	/**
	 * 约束值的范围
	 * @param min
	 * @param max
	 * @param value
	 * @return
	 */
	private  int constrain(int min,int  max, int value){
		if (value < min){
			return min;
		}
		if (value > max){
			return max;
		}
		return value;
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

	@Data
	@AllArgsConstructor
   class ImageAndBoundingBoxes {
	   private INDArray image;
	   private List<ImageObject> boundingBoxesList;
   }






}
