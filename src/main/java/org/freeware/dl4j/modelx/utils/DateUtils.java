package org.freeware.dl4j.modelx.utils;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

public class DateUtils {
	
	public static void main(String[] args) {
	
		
		parseDate("2019-5-31 05:48", "yyyy-M-dd HH:mm");
		
	}
	
	public static final String FORMAT_DATE_TIME_DEFAULT="yyyy-MM-dd HH:mm:ss";
	
	public static final String FORMAT_DATE_DEFAULT="yyyy-MM-dd";
	
	public static final String FORMAT_DATE_TIME_YYYYMMDDHHMMSS="yyyyMMddHHmmss";
	
	public static final String FORMAT_DATE_YYYYMMDD="yyyyMMdd";
	
	public static final String DATE_START_TIME=" 00:00:00";
	
	public static final String DATE_END_TIME=" 23:59:59";
	
	
	/**
	 * 获取指定时间的日期的开始时间
	 * @param date
	 * @return
	 */
	public static Date getStartDateTime(String date){
		date=date.concat(DATE_START_TIME);
		return parseDate(date,FORMAT_DATE_TIME_DEFAULT);
	}
	
	/**
	 * 获取指定时间的结束
	 * @param date
	 * @return
	 */
	public static Date getEndDateTime(String date){
		date=date.concat(DATE_END_TIME);
		return parseDate(date,FORMAT_DATE_TIME_DEFAULT);
	}
	
	
	public static Date parseDate(String text,String format){
		SimpleDateFormat sdf=new SimpleDateFormat(format);
		try {
			return sdf.parse(text);
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static String format(Date date,String format){
		SimpleDateFormat sdf=new SimpleDateFormat(format);
	    return sdf.format(date);
	}
	
	/**
	 * 得到给定日期的年
	 * @param date
	 * @return
	 */
	public static String getYear(Date date){
		String dateStr=format(date, FORMAT_DATE_DEFAULT);
		String[] array=dateStr.split("-");
		return array[0];
	}
	/**
	 * 得到给定日期的月
	 * @param date
	 * @return
	 */
	public static String getMonth(Date date){
		String dateStr=format(date, FORMAT_DATE_DEFAULT);
		String[] array=dateStr.split("-");
		return array[1];
	}
	/**
	 * 得到给定日期的日
	 * @param date
	 * @return
	 */
	public static String getDate(Date date){
		String dateStr=format(date, FORMAT_DATE_DEFAULT);
		String[] array=dateStr.split("-");
		return array[2];
	}
	
	/**
	 * 得到给定日期所在月份的第一天
	 * @param date
	 * @return
	 */
	 public static Date getMonthStart(Date date) {
	        Calendar calendar = Calendar.getInstance();
	        calendar.setTime(date);
	        int index = calendar.get(Calendar.DAY_OF_MONTH);
	        calendar.add(Calendar.DATE, (1 - index));
	        return calendar.getTime();
	    }
	 
	 /**
	  * 得到给定日期所在月份的最后一天
	  * @param date
	  * @return
	  */
	 public static Date getMonthEnd(Date date) {
	        Calendar calendar = Calendar.getInstance();
	        calendar.setTime(date);
	        calendar.add(Calendar.MONTH, 1);
	        int index = calendar.get(Calendar.DAY_OF_MONTH);
	        calendar.add(Calendar.DATE, (-index));
	        return calendar.getTime();
	  }
	 /**
	  * 得到上个月的月初
	  * @param date
	  * @return
	  */
	 public static Date getLastMonthStart(Date date) {
		 Calendar calendar = Calendar.getInstance();
         calendar.setTime(date);
         calendar.add(Calendar.MONTH, -1);
         int index = calendar.get(Calendar.DAY_OF_MONTH);
         calendar.add(Calendar.DATE, (1 - index));
         return calendar.getTime();
	  }
	 /**
	  * 得到上个月的月末
	  * @param date
	  * @return
	  */
	 public static Date getLastMonthEnd(Date date) {
	        Calendar calendar = Calendar.getInstance();
	        calendar.setTime(date);
	        calendar.add(Calendar.MONTH,0);
	        int index = calendar.get(Calendar.DAY_OF_MONTH);
	        calendar.add(Calendar.DATE, (-index));
	        return calendar.getTime();
	  }
	 
	 /**
	  * 得到给定日期的下一天
	  * @param date
	  * @return
	  */
	 public static Date getNext(Date date) {
	        Calendar calendar = Calendar.getInstance();
	        calendar.setTime(date);
	        calendar.add(Calendar.DATE, 1);
	        return calendar.getTime();
	 }
	 
	 /**
	  * 得到给定日期的昨天
	  * @param date
	  * @return
	  */
	 public static Date getYesterday(Date date){
		  Calendar cal=Calendar.getInstance();
		  cal.setTime(date);
		  cal.add(Calendar.DATE,-1);
		  return cal.getTime();
	 }
	 
	 /**
	  * 得到上周一
	  * @param date
	  * @return
	  */
	 public static Date getLastMonday(Date date) {
		    Calendar cal = Calendar.getInstance();
		    cal.setTime(date);
		    cal.setFirstDayOfWeek(Calendar.MONDAY);//将每周第一天设为星期一，默认是星期天
		    cal.add(Calendar.DATE, -1*7);
		    cal.set(Calendar.DAY_OF_WEEK,Calendar.MONDAY);
		    return cal.getTime();
	}
	 /**
	  * 得到上周日
	  * @param date
	  * @return
	  */
	 public static Date getLastSunday(Date date) {
		    Calendar cal = Calendar.getInstance();
		    cal.setTime(date);
		    cal.setFirstDayOfWeek(Calendar.MONDAY);//将每周第一天设为星期一，默认是星期天
		    cal.add(Calendar.DATE, -1*7);
		    cal.set(Calendar.DAY_OF_WEEK, Calendar.SUNDAY);
		    return cal.getTime();
	 }
	 
	 /**
	  * 得到周一
	  * @param date
	  * @return
	  */
	 public static Date getMonday(Date date) {
		    Calendar cal = Calendar.getInstance();
		    cal.setTime(date);
		    cal.setFirstDayOfWeek(Calendar.MONDAY);//将每周第一天设为星期一，默认是星期天
		    cal.add(Calendar.DATE, 1);
		    cal.set(Calendar.DAY_OF_WEEK,Calendar.MONDAY);
		    return cal.getTime();
	}
	 /**
	  * 得到当前周日
	  * @param date
	  * @return
	  */
	 public static Date getSunday(Date date) {
		    Calendar cal = Calendar.getInstance();
		    cal.setTime(date);
		    cal.setFirstDayOfWeek(Calendar.MONDAY);//将每周第一天设为星期一，默认是星期天
		    cal.add(Calendar.DATE, 1);
		    cal.set(Calendar.DAY_OF_WEEK, Calendar.SUNDAY);
		    return cal.getTime();
	 }
	 /**
	  * 获取给定日期的前n天
	  * @param d
	  * @param day
	  * @return
	  */
	 public static Date getDateBefore(Date d, int day) {  
	        Calendar now = Calendar.getInstance();  
	        now.setTime(d);  
	        now.set(Calendar.DATE, now.get(Calendar.DATE) - day);  
	        return now.getTime();  
	 }    
	 /***
	  * 获取给定日期的后n天
	  * @param d
	  * @param day
	  * @return
	  */
	 public static Date getDateAfter(Date d, int day) {  
	        Calendar now = Calendar.getInstance();  
	        now.setTime(d);  
	        now.set(Calendar.DATE, now.get(Calendar.DATE) + day);  
	        return now.getTime();  
	 }
	 
	 /**
	     * 当前年的开始时间，即2012-01-01 00:00:00
	     * 
	     * @return
	     */
	    public  static Date getCurrentYearStartTime() {
	        Calendar c = Calendar.getInstance();
	    	SimpleDateFormat shortSdf=new SimpleDateFormat("yyyy-MM-dd");;
	    
	        Date now = null;
	        try {
	            c.set(Calendar.MONTH, 0);
	            c.set(Calendar.DATE, 1);
	            now = shortSdf.parse(shortSdf.format(c.getTime()));
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
	        return now;
	    }

	    /**
	     * 当前年的结束时间，即2012-12-31 23:59:59
	     * 
	     * @return
	     */
	    public  static Date getCurrentYearEndTime() {
	    	SimpleDateFormat shortSdf=new SimpleDateFormat("yyyy-MM-dd");;
	    	SimpleDateFormat longSdf= new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	        Calendar c = Calendar.getInstance();
	        Date now = null;
	        try {
	            c.set(Calendar.MONTH, 11);
	            c.set(Calendar.DATE, 31);
	            now = longSdf.parse(shortSdf.format(c.getTime()) + " 23:59:59");
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
	        return now;
	    }
	    
	    /**
	     * 获取前/后半年的开始时间
	     * @return
	     */
	    public  static Date getHalfYearStartTime(){
	    	SimpleDateFormat shortSdf=new SimpleDateFormat("yyyy-MM-dd");;
	    	SimpleDateFormat longSdf= new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	        Calendar c = Calendar.getInstance();
	        int currentMonth = c.get(Calendar.MONTH) + 1;
	        Date now = null;
	        try {
	            if (currentMonth >= 1 && currentMonth <= 6){
	                c.set(Calendar.MONTH, 0);
	            }else if (currentMonth >= 7 && currentMonth <= 12){
	                c.set(Calendar.MONTH, 6);
	            }
	            c.set(Calendar.DATE, 1);
	            now = longSdf.parse(shortSdf.format(c.getTime()) + " 00:00:00");
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
	        return now;
	        
	    }
	    /**
	     * 获取前/后半年的结束时间
	     * @return
	     */
	    public  static Date getHalfYearEndTime(){
	    	SimpleDateFormat shortSdf=new SimpleDateFormat("yyyy-MM-dd");;
	    	SimpleDateFormat longSdf= new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	        Calendar c = Calendar.getInstance();
	        int currentMonth = c.get(Calendar.MONTH) + 1;
	        Date now = null;
	        try {
	            if (currentMonth >= 1 && currentMonth <= 6){
	                c.set(Calendar.MONTH, 5);
	                c.set(Calendar.DATE, 30);
	            }else if (currentMonth >= 7 && currentMonth <= 12){
	                c.set(Calendar.MONTH, 11);
	                c.set(Calendar.DATE, 31);
	            }
	            now = longSdf.parse(shortSdf.format(c.getTime()) + " 23:59:59");
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
	        return now;
	    }
	    
	     /**
	       * 得到上半年或下半年每个月的开始日期
	       * @return
	       */
	      public static List<Date> getAllMonthStartDateForHalfYear(Date date,Boolean firstHalf){
	    	  
	    	  String year= DateUtils.getYear(date);
	    	  int month=1;
	    	  if(null==firstHalf){
	    		  month=Integer.parseInt(DateUtils.getMonth(date));
	    	  }
	    	  if(null!=firstHalf&&firstHalf==false){
	    		  month=7;
	    	  }
	    	  if(null!=firstHalf&&firstHalf==true){
	    		  month=1;
	    	  }
	          
	          List<Date> halfYearMonthStartDateList=new ArrayList<Date>();
	    	  
	    	  if(month<=6){//上半年
	    		  for(int i=1;i<7;i++){
	    			  halfYearMonthStartDateList.add(DateUtils.parseDate(year+"-0"+i+"-01", DateUtils.FORMAT_DATE_DEFAULT));
	    		  }
	       	  }else{//下半年
	    		  for(int i=7;i<13;i++){
	    			  if(i<10){
	    				  halfYearMonthStartDateList.add(DateUtils.parseDate(year+"-0"+i+"-01", DateUtils.FORMAT_DATE_DEFAULT));
	    			  }else{
	    				  halfYearMonthStartDateList.add(DateUtils.parseDate(year+"-"+i+"-01", DateUtils.FORMAT_DATE_DEFAULT));
	    			  }
	       		  }
	      	  }
	    	  
	    	 
	       	  return halfYearMonthStartDateList;
	      }
	      
	      
	      /**
	       * 获取当前日期是星期几<br>
	       * 
	       * @param date
	       * @return 当前日期是星期几
	       */
	      public static String getWeekOfDate(Date date) {
	          String[] weekDays = {"星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"};
	          
	          Calendar cal = Calendar.getInstance();
	          cal.setTime(date);
              int w = cal.get(Calendar.DAY_OF_WEEK) - 1;
	          if (w < 0){
	              w = 0;
	          }

	          return weekDays[w];
	      }

}
