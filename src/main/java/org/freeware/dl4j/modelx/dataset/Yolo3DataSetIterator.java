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
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.freeware.dl4j.modelx.utils.ExtendedFileUtils;
import org.freeware.dl4j.modelx.utils.YoloImageUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
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

	private String[] labels={};

	private List<ImageObject> anchors=new ArrayList<>(9);

	private float jitter=0.3f;

	public Yolo3DataSetIterator(String dataSetPath,
								int batchSize,
								String[] labels ,
								int[][] bigBoundingBoxPriors,
								int[][] mediumBoundingBoxPriors,
								int[][] smallBoundingBoxPriors) {

		super();


		this.labels=labels;

        this.nativeImageLoader=new NativeImageLoader();

        setAnchors(bigBoundingBoxPriors, mediumBoundingBoxPriors, smallBoundingBoxPriors);

		//检查并设置目录
		checkAndSetDirectory(dataSetPath);

		this.labelProvider=new VocLabelProvider(dataSetPath);

		this.maxBoxPerImage=getMaxBoxPerImage();

		//设置小批量大小
		this.batchSize = batchSize;

		//获取所有特征数据
    	this.featureFiles = ExtendedFileUtils.listFiles(featurePath, new String[] {"jpg","png"}, false);
    	//设置总的小批次数量
		if(featureFiles.size()%batchSize==0){
			this.totalBatches=featureFiles.size()/batchSize;
		}else {
			this.totalBatches=featureFiles.size()/batchSize+1;
		}
		//随机打乱
       	Collections.shuffle(featureFiles, random);
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

			startIndex=endIndex-batchSize;
		}

		int realBatchSize=endIndex-startIndex;
		//特征存放，便于拼接
		INDArray[] imageFeatureList=new INDArray[realBatchSize] ;

		INDArray groundTrue=Nd4j.zeros(realBatchSize,1,1,1,maxBoxPerImage,4);

		INDArray featureBig=Nd4j.zeros(realBatchSize,13,13,3,4+1+labels.length);

		INDArray featureMedium=Nd4j.zeros(realBatchSize,26,26,3,4+1+labels.length);

		INDArray featureSmall=Nd4j.zeros(realBatchSize,52,52,3,4+1+labels.length);

		INDArray[] features=new INDArray[]{featureBig,featureMedium,featureSmall} ;

		int exampleCount=0;

		int groundTrueIndex=0;

		for(int exampleIndex=startIndex;exampleIndex<endIndex;exampleIndex++) {
			
			File featureFile=featureFiles.get(exampleIndex);
			
			try {

				//得到特征
				INDArray imageFeature = nativeImageLoader.asMatrix(featureFile);
				//得到边界框标签
				List<ImageObject> boundingBoxesList=getBoundingBoxes(featureFile);

				ImageAugmentResult imageAugmentResult=augmentImage(imageFeature,boundingBoxesList,inputShape[1],inputShape[2]);

				imageFeatureList[exampleCount]=imageAugmentResult.getImage();

				boundingBoxesList=imageAugmentResult.getBoundingBoxesList();

				for(ImageObject boundingBox:boundingBoxesList){

					ImageObject maxAnchor=null;

					int maxIndex=-1;

					double maxIou=-1D;

					ImageObject shiftedBox=new ImageObject(0,0,boundingBox.getX2()-boundingBox.getX1(),boundingBox.getY2()-boundingBox.getY1(),"");

                    for(int anchorIndex=0;anchorIndex<anchors.size();anchorIndex++){

                    	ImageObject anchor=anchors.get(anchorIndex);

						double iou=iou(shiftedBox,anchor);

                    	if(maxIou<iou){
                    		maxIou=iou;
                    		maxIndex=anchorIndex;
							maxAnchor=anchor;
						}

					}

                    int currentLabelIndex=maxIndex/3;

                    //边界框放于与之iou最大的anchor所在的输出
					INDArray currentFeature=features[currentLabelIndex];

                    long gridWidth=currentFeature.shape()[1];

					long gridHeight=currentFeature.shape()[2];

					double centerX = .5*(boundingBox.getX1() + boundingBox.getX2());
					// 归一化
					centerX = centerX / inputShape[1] * gridWidth ;

					double centerY = .5*(boundingBox.getY1() + boundingBox.getY2());
					// 归一化
					centerY = centerY / inputShape[2] * gridHeight ;

					// 归一化
					double	width = Math.log(boundingBox.getX2()- boundingBox.getX1()) / maxAnchor.getX2();

					double	height = Math.log(boundingBox.getY2()- boundingBox.getY1()) / maxAnchor.getY2();

                    double[] box={centerX,centerY,width,height};

                    int gridX=(int)Math.floor(centerX);

					int gridY=(int)Math.floor(centerY);

                    int anchorIndex=maxIndex%3;

					log.info(currentFeature.shapeInfoToString());

					currentFeature.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount), NDArrayIndex.point(gridX), NDArrayIndex.point(gridY), NDArrayIndex.point(anchorIndex),NDArrayIndex.all()},0);

					int classIndex= Arrays.asList(features).indexOf(boundingBox.getLabel());

					for(int boxValueIndex=0;boxValueIndex<box.length;boxValueIndex++){

						currentFeature.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount), NDArrayIndex.point(gridX), NDArrayIndex.point(gridY), NDArrayIndex.point(anchorIndex),NDArrayIndex.point(boxValueIndex)},box[boxValueIndex]);
					}
                    currentFeature.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount), NDArrayIndex.point(gridX), NDArrayIndex.point(gridY), NDArrayIndex.point(anchorIndex),NDArrayIndex.point(4)},1.0);

					currentFeature.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount), NDArrayIndex.point(gridX), NDArrayIndex.point(gridY), NDArrayIndex.point(anchorIndex),NDArrayIndex.point(4+1+classIndex)},1.0);

					double[] groundTrueLabelValues=new double[]{centerX,centerY,boundingBox.getX2()-boundingBox.getX1(),boundingBox.getY2()-boundingBox.getY1()};

                    for(int groundTrueLabelValueIndex=0;groundTrueLabelValueIndex<groundTrueLabelValues.length;groundTrueLabelValueIndex++){
						groundTrue.put(new INDArrayIndex[]{NDArrayIndex.point(exampleCount),NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.point(0),NDArrayIndex.point(groundTrueIndex),NDArrayIndex.point(groundTrueLabelValueIndex)},groundTrueLabelValues[groundTrueLabelValueIndex]);
                    }
					groundTrueIndex=groundTrueIndex+1;

					groundTrueIndex=groundTrueIndex%maxBoxPerImage;

				}
				exampleCount=exampleCount+1;



			} catch (Exception e) {

				log.error("",e);
			}
			
    	}
		//按小批量维度拼接标签数据
		INDArray imageFeature=Nd4j.concat(0,imageFeatureList);

		INDArray[] featuresArray=new INDArray[] {imageFeature,featureBig,featureMedium,featureSmall,groundTrue};

		INDArray dummyLabel1 = Nd4j.zeros(realBatchSize,1);

		INDArray dummyLabel2 = Nd4j.zeros(realBatchSize,1);

		INDArray dummyLabel3 = Nd4j.zeros(realBatchSize,1);
		// we have three outputs here ,big medium and small
		INDArray[]  labelsArray=new INDArray[] {dummyLabel1,dummyLabel2,dummyLabel3};

		MultiDataSet multiDataSet=new MultiDataSet(featuresArray,labelsArray);

        //小批量计数器加1
		currentBatch++;

		return multiDataSet;

	}

	/**
	 * 保存数据增强的结果
	 * @param featureFile
	 * @param imageAugmentResult
	 * @throws IOException
	 */
	private void saveImageAugmentResult(File featureFile, ImageAugmentResult imageAugmentResult) throws IOException {
		Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();
		BufferedImage bufferedImage=java2DNativeImageLoader.asBufferedImage(imageAugmentResult.getImage());
		YoloImageUtils.drawBoundingBoxes(bufferedImage,imageAugmentResult.getBoundingBoxesList(), Color.green);
		ImageIO.write(bufferedImage,"jpg",new File("images/"+featureFile.getName()));
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
	private ImageAugmentResult augmentImage(INDArray image,List<ImageObject> boundingBoxesList,int inputHeight,int inputWidth) throws IOException {

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

		return new ImageAugmentResult(image,correctBoundingBoxesList);
	}

	private float randomUniform(float min,float max){
		return min + ((max - min) * random.nextFloat());
	}

	private  int constrain(int min,int  max, int value){
		if (value < min){
			return min;
		}
		if (value > max){
			return max;
		}
		return value;
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

		Java2DNativeImageLoader java2DNativeImageLoader=new Java2DNativeImageLoader();

		BufferedImage sourceImage=java2DNativeImageLoader.asBufferedImage(image);

		BufferedImage targetImage=resize(sourceImage,newInputWidth,newInputHeight);

		INDArray resizedImage=java2DNativeImageLoader.asMatrix(targetImage);

		long[] shape=resizedImage.shape();

		long imageHeight = shape[2];

		long imageWidth= shape[3];

		if (dx > 0) {
			resizedImage=Nd4j.pad(resizedImage,new int[][]{{0,0},{0,0},{0,0},{dx,0}}, Pad.Mode.CONSTANT,127);
		}else {
			resizedImage=resizedImage.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(-dx,imageWidth)} );
		}
		if ((newInputWidth + dx) < netInputWidth){
			resizedImage=Nd4j.pad(resizedImage,new int[][]{{0,0},{0,0},{0,0},{0,netInputWidth - (newInputWidth+dx)}}, Pad.Mode.CONSTANT,127);
		}

		if (dy > 0){
			resizedImage=Nd4j.pad(resizedImage,new int[][]{{0,0},{0,0},{dy,0},{0,0}}, Pad.Mode.CONSTANT,127);
		}else{
			resizedImage=resizedImage.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(-dy,imageHeight),NDArrayIndex.all()} );
		}

		if ((newInputHeight + dy) < netInputHeight){
			resizedImage=Nd4j.pad(resizedImage,new int[][]{{0,0},{0,0},{0, netInputHeight - (newInputHeight+dy)},{0,0}}, Pad.Mode.CONSTANT,127);
		}

		return resizedImage.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,netInputHeight),NDArrayIndex.interval(0,netInputWidth)});

    }

	/**
	 * 调整图片大小
	 * @param sourceImage
	 * @param targetWidth
	 * @param targetHeight
	 * @return
	 * @throws IOException
	 */
	private BufferedImage resize(BufferedImage sourceImage,int targetWidth, int targetHeight) throws IOException {

		int width = sourceImage.getWidth();

		int height = sourceImage.getHeight();

		BufferedImage targetImage  = new BufferedImage(targetWidth, targetHeight, sourceImage.getType());

		Graphics2D g = targetImage.createGraphics();

		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

		g.drawImage(sourceImage, 0, 0, targetWidth, targetHeight, 0, 0, width, height, null);

		g.dispose();

		return targetImage;
	}

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

	private float randScale(float scale){
		scale=randomUniform(1,scale);
		if(random.nextInt(2)==0){
			return scale;
		}
		return 1.0f/scale;
	}


	private double iou(ImageObject imageObject1,ImageObject imageObject2){

	   DetectedObject detectedObject1=convertToDetectedObject(imageObject1);

	   DetectedObject detectedObject2=convertToDetectedObject(imageObject2);

	   return YoloUtils.iou(detectedObject1,detectedObject2);
   }

   private DetectedObject convertToDetectedObject(ImageObject imageObject){

	  return  new DetectedObject(0,  imageObject.getX2()/2,  imageObject.getY2()/2,  imageObject.getX2()-imageObject.getX1(),  imageObject.getY2()-imageObject.getY1(),null, 0);

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
   class ImageAugmentResult{
	   private INDArray image;
	   private List<ImageObject> boundingBoxesList;
   }



}
