package org.freeware.dl4j.modelx;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;


public class ExtendedFileUtils extends FileUtils{
	
	
	
	
	
	public  static void makeDirs(String dirs){
		File saveDirFile = new File(dirs);
		if (!saveDirFile.exists()) {
			saveDirFile.mkdirs();
		}
	}
	
	

	public static List<String> getFileNameList(File fileDir) {
		
		List<String> fileNameList=new ArrayList<String>();
		
		if(fileDir.isDirectory()){
			
 		File[] fileList=fileDir.listFiles();
 		
			for(File file:fileList){
				
				fileNameList.add(file.getName());
			}
		}
		return fileNameList;
	}
	
	public static List<File> listFiles(String  directory,String[] extensions,Boolean recursive) {
		
		File fileDirectory=new File(directory);
		
		Collection<File> fileList=FileUtils.listFiles(fileDirectory, extensions, recursive);
		 
		String uselessFilePrefix="._";
		
		Iterator<File>  iterator=fileList.iterator();
		
		List<File> retList=new ArrayList<File>();
		
		while(iterator.hasNext()) {
			
			File file=iterator.next();
			
			if(!file.getName().startsWith(uselessFilePrefix)) {
				
				retList.add(file);
			}
			
		}
	
		return retList;
	}
}
