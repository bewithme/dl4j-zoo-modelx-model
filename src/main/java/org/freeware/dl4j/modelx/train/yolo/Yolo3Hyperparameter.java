package org.freeware.dl4j.modelx.train.yolo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Yolo3Hyperparameter {
	
	    private  String name;
	     
	    private  String[] labels ;
	    //small med  big
	    private  int[][] priorBoundingBoxes;

	    private  double learningRate;
	    
	    private  String dataDir;
	    
	    private  int batchSize ;
	    
	    private  int epochs ;
	   
	    private  int randomSeed ;

	    private  double lamdbaCoord;
	    
	    private  double lamdbaNoObject;
	    
	    private  String  imageFormat;
	    
	    private  String  modelSavePath;
	
	

}
