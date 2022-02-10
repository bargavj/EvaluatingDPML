from bs4 import BeautifulSoup
import zipfile
import requests
import wget
from os import listdir,makedirs
from os.path import isfile, join,isdir
import argparse

global dataFolder, url,blackListFiles
#data download page
url = "https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/"
dataFolder = "../dataset/census/"
blackListFiles = []
targetFiles = ["csv_pus.zip"]
def isFilePresent(fileName):
    """Returns whether a given data file found on the download web page is already present in the local folder.
    Also checks if the file is in the file blacklist (certain unwanted files)
    """
    global dataFolder,blackListFiles
    allDataFiles = [f for f in listdir(dataFolder) if (isfile(join(dataFolder, f)) and f.endswith('.zip'))]
    return fileName in allDataFiles and not (fileName in blackListFiles)

def crawl_data():
    print("### Starting to crawl data.")
    global dataFolder, url
    #create census dataset folder if not exists
    if not isdir(dataFolder):
        makedirs(dataFolder)
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data,"lxml")
    downloadZips = []
    for link in soup.find_all('a'):
        href = str(link.get('href'))
        #if the data file fits the desired format and it's not already downloaded, then download the file.
        if filePatternTest(href) and not (isFilePresent(href)):
            downloadZips.append(href)
    for href in downloadZips:
        downloadLink = url+href
        #the wget might fail. try until the file is downloaded sucessfully
        while True:
            try:
                wget.download(downloadLink,out = dataFolder)
            except:
                continue
            else:
                break
    print()
    print("### Finished crawling data.")
    return 0
def processZips():
    print("### Starting to unzip data.")
    global dataFolder
    allDataFiles = [f for f in listdir(dataFolder) if (isfile(join(dataFolder, f)) and f.endswith('.zip'))]
    for file in allDataFiles:
        zipFileDir = dataFolder+file
        print(zipFileDir)
        with zipfile.ZipFile(zipFileDir, 'r') as zip_ref:
            zip_ref.extractall(dataFolder)
    print("### Finished unzipping data.")
def filePatternTest(string:str):
    #change this condition for finding data download files that are not the census data.
    # condition = string.endswith(".zip") and string.startswith("csv_p")
    condition = string in targetFiles
    return condition
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_census_data',type=str,default='1year')
    args = parser.parse_args()
    if args.target_census_data=='5year':
        url = "https://www2.census.gov/programs-surveys/acs/data/pums/2019/5-Year/"
    
    statusCode = crawl_data()
    if statusCode==0:
        processZips()