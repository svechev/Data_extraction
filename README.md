# Extracting information from bar plots

## Requirements
For running the code - Python 3.
Libraries used in the scripts that need to be installed: 
- opencv-python
- easyocr
- XlsxWriter
- matplotlib
- numpy
  
The following libraries are only used in the script graph_statistics.py
- pandas
- dtw-python (only used in the script graph_statistics.py)

## Usage 
Run the extract_data.py script from inside the scripts directory. Example:

C:\..\some directories\..\bar_plot_extraction\scripts> python extract_data.py

Before running the script, read the provided configuration details in the beginning of the source code.

## Contents
- scripts/
 - extract_data.py - extracts data from the image in img/. Saves the result to results/FigureData1.xlsx or results/FigureData2.xlsx 
 - extract_diff_windows_test.py - like extract_data, but works for the solar.ts.png image. Runs 3 times with different configuration parameter values
 - graph_statistics.py - shows various information and plots that compare the extracted data from the bar plot to its actual data
 - solar_statistics - shows various information and plots that compare the extracted data from the curve plot (only beginning of each year) to its actual data
 - solar_statistics_6months.py - shows various information and plots that compare the extracted data from the curve plot (beginning and middle of each year) to its actual data
   
- data/
  - virus-data-report.ods - the Microsoft Excel spreadsheet that contains the actual data, that the bar plot image uses
  - solar.csv - comma separated values file, contains the actual data, that the curve plot image uses
    
- img/
  - solar.ts.png - curve plot, the image that the script runs on
    
- img_backup/ - this directory has no effect on the scripts, it's used just for storage
  - figure1_no_line.png - bar plot, the image that the script runs on
  
- results/
  - FigureData1.xlsx - stores the result of the extract_data.py script after it was used on the bar plot
  - FigureData2.xlsx - stores the result of the extract_data.py script after it was used on the curve plot
    
- statistics_figures/ - contains images of plots, produced by the statistics scripts
  - dtw.png
  - residual.png
  - solar_residual.png
- .gitignore
- README.md

## Additional information
The script is only used on one image, and it needs to be in the img directory.
To use with a different image, you need to replace the current one.
If the new image has a different name:
In the beginning of the extract_data.py script, change the variable img_path1 or img_path2 to your desired image name
 
