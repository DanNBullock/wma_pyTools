#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:42:56 2022

@author: dan
"""

def searchAndLoad_configJSON(path=None):
    """
    searchAndLoad_configJSON(path=None)
    Searches for, parses, and, if possible, loads the config.json file

    Parameters
    ----------
    path : TYPE, optional
        Path to the config.json file or the directory containing the config.json
        If there is no path passed, the current working directory will be used.
        The default is None.

    Returns
    -------
    config : either empty list or dictionary
        If a "config.json" file is found, it will be loaded and passed out
        as the config output.  If none is found, an empty list wil be passed out 

    """  
    
    import os
    import json
    
    #if the input path is None
    if path==None:
        #set the current path to the cwd
        path=os.getcwd()
    
    #in either case proceed
        
    #if this is a directory
    if os.path.isdir(path):
        #check to see if if there's a config.json in it.

        #check to see if there is a config.json here
        #create path to hypothetical .json file
        jsonPath=os.path.join(path,'config.json')
        #check to see if this exists
        if os.path.isfile(jsonPath):
            #if so, load it
            with open('config.json') as config_json:
                config = json.load(config_json)
            #if it's a directory without a config.json, pass out nothing
        else: 
            config=[]
    #if this is instead a file
    elif os.path.isfile(path):
        #check to see if it's basename is config.json
        if os.path.basename(path)=='config.json':
            #if so load it as the config .json
            with open('config.json') as config_json:
                config = json.load(config_json)
        #if the file is not config.json, set config to empty
        else:
            config=[]
    #if there is not a config.json in the input path, or, as relevant, the current directory
    #output an empty config
    config=[]
    return config
    
    #check to see if there is a config.json here
    #create path to hypothetical .json file
    jsonPath=os.path.join(path,'config.json')
    
def parseInputSysArgs():
    
    import sys
    import numpy as np
    import warnings
    
    #unpack the contents of the sysargs
    argsIn=[iArg for iArg in sys.argv]
    
    #I think the args all come in in the form of strings
    #first arg is necessarily python script name.
    scriptName=argsIn[0]
    
    #check to see if any of the inputs are equals or if there are equals in any of the inputs
    #or if there are dashes leading any of the args
    isEquals=[iArg=='=' for iArg in argsIn]
    hasEquals=['=' in iArg for iArg in argsIn]
    #gotta be careful and use a loop I guess
    dashLeads=[[] for iArg in argsIn]
    for iterator,iArg in enumerate(argsIn):
        #if the input argument is 2, couldn't just have a singleton dash
        if len(iArg) == 2:
            if iArg[0]=='-':
                dashLeads[iterator]=True
        #if the input argument is longer than 2
        elif len(iArg) > 2:
            #double dash case
            if iArg[0:2]=='--':
                dashLeads[iterator]=True
                #otherwise
        else:
            dashLeads[iterator]=False
            
    #begin building non positional vars
    
    #keep track of which ones we've taken care of
    argsManaged=[0]
    
    #start with equals
    # they can't be equals nothing, right?
    isEquals_ArgLocations=np.unique(np.where(isEquals))
    isEqualsDict={}
    #iterate across these locations
    for iArgLocations in isEquals_ArgLocations:
        #get the input value before this equals
        priorArgInput=f'{argsIn[iArgLocations-1]}'
        #get the input value after this equals
        nextArgInput=f'{argsIn[iArgLocations+1]}'
        isEqualsDict[priorArgInput]=nextArgInput
        #record the ones we have managed with this
        argsManaged.append(iArgLocations-1)
        argsManaged.append(iArgLocations)
        argsManaged.append(iArgLocations+1)
        
        
    #next to hasEquals
    hasEquals_ArgLocations=np.unique(np.where(hasEquals))
    hasEqualsDict={}
    #iterate across these locations
    for iArgLocations in hasEquals_ArgLocations:
        #determine if the first or last character is the equals
        #first char case
        #why the zero index? to hit he first character
        if argsIn[iArgLocations][0]=='=':
            warnings.warning('Poorly formatted input detected for ' + argsIn[iArgLocations][0] + '\nRecommend more rigid structure in input')
        
            #stealing from the previous example even though it isn't exactly the same
            #because the = is on the left, we can assume the thing prior was the prior input
            priorArgInput=f'{argsIn[iArgLocations-1]}'
            #get the input value after this equals
            nextArgInput=f'{argsIn[iArgLocations]}'.split('=')[1]
            hasEqualsDict[priorArgInput]=nextArgInput
            #record the ones we have managed with this
            argsManaged.append(iArgLocations-1)
            argsManaged.append(iArgLocations)   
            
        #last char case
        elif argsIn[iArgLocations][-1]=='=':
            warnings.warning('Poorly formatted input detected for ' + argsIn[iArgLocations][0] + '\nRecommend more rigid structure in input')
        
            #stealing from the previous example even though it isn't exactly the same
            #because the = is on the left, we can assume the thing prior was the prior input
            priorArgInput=f'{argsIn[iArgLocations]}'.split('=')[0]
            #get the input value after this equals
            nextArgInput=f'{argsIn[iArgLocations+1]}'
            hasEqualsDict[priorArgInput]=nextArgInput
            #record the ones we have managed with this
            argsManaged.append(iArgLocations)
            argsManaged.append(iArgLocations+1)
            
        #somewhere in the middle
        else:
            #store it for a moment as a string
            #currArgInput=f'{argsIn[iArgLocations]=}'.split('=')[1]
            #split it at the '=' char
            argParts=argsIn[iArgLocations].split('=')
            #set the arg parts in the appropriate  locations
            hasEqualsDict[argParts[0]]=argParts[1]
            #record the ones we have managed with this
            argsManaged.append(iArgLocations)

    #now comes the confusing part, dealing with dash locations            
    dashLeads_ArgLocations=np.unique(np.where(dashLeads))
    dashLeadsDict={}
    #iterate across these locations
    for iArgLocations in dashLeads_ArgLocations:
        #the "prior arg input" is easy to determine
        priorArgInput=f'{argsIn[iArgLocations]}'
        #but now we have to be concerned if the next location has been already
        #taken care of OR if it is another dash location
        #if its in the managed list
        if not iArgLocations+1 in argsManaged:
            #if its not in the upcoming dashLeads locations
            if not iArgLocations+1 in dashLeads_ArgLocations:
                #then you can set the next arg as the thing that follows
                nextArgInput=f'{argsIn[iArgLocations+1]}'
                #record the ones we have managed with this
                argsManaged.append(iArgLocations)
                argsManaged.append(iArgLocations+1)
            
                #if it fails these conditions, its a singleton input
            else:
                nextArgInput=[]
                argsManaged.append(iArgLocations)
        else:  
            nextArgInput=[]
            argsManaged.append(iArgLocations)
        
        #set the dictionary entries
        dashLeadsDict[priorArgInput]=nextArgInput
        
    #ok, now we have taken care of all paired input arguments (theoretically)
    #its safe to assume everything left is positional
    positional_ArgLocations=[x for x in range(len(argsIn)) if x not in argsManaged]
    positionalArgs=[argsIn[argLocations] for argLocations in positional_ArgLocations]
    
    #should probably do some sort of sanity check here and throw an error if things are going wrong
    
    #merge all of the dictionaries
    
    nonPositionalArgs = dashLeadsDict | hasEqualsDict | isEqualsDict
    
    return positionalArgs , nonPositionalArgs

def cmdORconfig_input():
    """
    If system arguments have been passed in, parses these and passes them out
    in two separate objects.  As though *args and **kwargs.  If no such system
    arguments have been passed in, looks for a config.json file--either in the
    current working directory or at/in location sys.argv[1] and parses that
    into nonPositionalArgs

    Returns
    -------
    positionalArgs : list or tuple?
        The ORDERED list of positional varaibles that have been extracted
    nonPositionalArgs : dict
        The keyword arguments and their values, as a dictionary.

    """   
    
    import os
    import json
    import sys

    
    #use nonlocal to pass out variables
    
    #if there are any sys args passed in
    if len(sys.argv) == 0:
        #get the first arg, which is probably the python script
        pythonScript=sys.argv[0]
        
    if len(sys.argv) >= 2:
        #if the inputs are longer than 2, try and load the second item as a config.json
        
        tempConfig=searchAndLoad_configJSON(sys.argv[1])
        
        #if there are 3 or more arguments 
        if len(sys.argv) >= 3:
            #use these inputs as the config
            [positionalArgs , nonPositionalArgs]=parseInputSysArgs()
        
        #in the case that it is merely 2 long, we have to determine if temp config is empty
        else:
            #if tempconfig is empty
            if not tempConfig:
                #use the paraseInputSysArgs to get that input
                #nonpositional should be empty
                [positionalArgs , nonPositionalArgs]=parseInputSysArgs()
            
            #else, convert the temp config into nonPositionalArgs and just set positionalArgs to empty
            else:
                positionalArgs=[]
                nonPositionalArgs=nonPositionalArgs
    
    #whatever the case, we should now have arguments parsed to some degree
    #now pass out the positional args to the outer function
    
    #just pass these out as args and kwargs
    #use them like so:
    # funtion(*args,**kwargs)
    # funtion(*positionalArgs,**nonPositionalArgs)
    return positionalArgs,nonPositionalArgs


def is_docker():
    """
    Determines if current script is being executed in a docker environment
    
    Taken from:
    https://stackoverflow.com/questions/43878953/how-does-one-detect-if-one-is-running-within-a-docker-container-within-python
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    import os
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path)) or
        os.path.exists('/singularity') or #I don't know if this one or the next one triggers it
        os.path.exists('/.singularity.d') 
    )

def parcJSON_to_LUT(pathOrDict):
    import json
    import pandas as pd
    if isinstance(pathOrDict,str):
        with open(pathOrDict) as label_json:
            label_dict_list = json.load(label_json)
    else:
        #just assume it's been loaded already and is a list?  I guess
        pass
    
    #create a dataframe
    lookupTable=pd.DataFrame(columns=['labelNumber' , 'labelNames'])
    
    for iterator,iLabels in enumerate(label_dict_list):
        currVals=[iLabels['voxel_value'],iLabels['name']]
        toAppendSeries=pd.Series(currVals, index=lookupTable.columns)
        lookupTable=lookupTable.append(toAppendSeries,ignore_index=True)
    
    return lookupTable
        
def bl_conmat_fromDIPYandParc(M,lookupTable,outdir):
    """
    Produces a Brainlife compatible conmat datatype given the relevant inputs
    https://brainlife.io/datatype/5d34d9f744947d8aea0e0d2f/detail

    Parameters
    ----------
    M : numpy.ndarray
        The N X N array output from dipy.tracking.utils.connectivity_matrix
    lookupTable : pandas.DataFrame
        The reduced lookup table derived from wmaPyTools.analysisTools.reduceAtlasAndLookupTable
    outdir : string path
        Path into which the connmat data type objects should be stored
    
    Returns
    -------
    None.  Saves down relevant outputs.

    """
    import numpy as np
    import os
    import json
    
    #index.Json out
    indexOut={}
    indexOut['filename']='connectivity.csv'
    indexOut['unit']='stream_count'
    
    #check if matrix is symmetric
    isSymmetric=np.allclose(M, M.T, rtol=1e-05, atol=1e-08)
    
    #create the index.json out
    if isSymmetric:
        totalStreams=int(np.sum(M)*.5)
        indexOut['desc']='Symmetric structural connectivity of ' + str(totalStreams) + ' across ' +str(M.shape[0]) + ' regions'
        indexOut['name']='Structural connectivity'
    else:
        totalStreams=np.sum(M)
        indexOut['desc']='Non-symmetric structural connectivity of ' + str(totalStreams) + ' across ' +str(M.shape[0]) + ' regions'
        indexOut['name']='Structural connectivity'
    
    #create the label.json structure     
    label=[{'name':irois['labelNames'], 'label':irois['labelNumber'], 'voxel_value':iterator} for iterator,irois in lookupTable.iterrows()]
    
    #create csv out dir
    if not os.path.exists(os.path.join(outdir,'csv')):
    		os.makedirs(os.path.join(outdir,'csv'))
    
    #save CSV        
    np.savetxt(os.path.join(outdir,'csv',indexOut['filename']), M, delimiter=",")
    
    #dumpy the json files
    with open(os.path.join(outdir,"label.json"), "w") as outfile:
        #dump or dumps?  I have no idea
        json.dump(label, outfile)
        

    with open(os.path.join(outdir,"index.json"), "w") as outfile:
        #dump or dumps?  I have no idea
        json.dump(indexOut, outfile)

      
    
#NO, WE'RE NOT DOING THIS.        

# def conmat_to_JGFZ(arrayORcsv,indexIn,labelIn):
#     import os
#     from collections import OrderedDict
#     import json
#     import numpy as np
#     from glob import glob
#     import sys
    
#     #stolen from:
#     # https://github.com/filipinascimento/bl-conmat2network/blob/0.2/main.py
    
#     #load or parse input connectivity
#     if isinstance(arrayORcsv,np.ndarray):
#         conMatrix=arrayORcsv
#     elif isinstance(arrayORcsv,str):
#         #if it's a directory, as is the case with the conventional conmat standard
#         if os.path.isdir(arrayORcsv):
#             #throw a value error if there are multiple or no csvs in here
#             csvPaths=glob(os.path.join(arrayORcsv,'*.csv'))
#             if not len(csvPaths) == 1:
#                 ValueError('Specific Csv file not found on provided path')
#             else:
#                 conMatrix=np.loadtxt(csvPaths[0],delimiter=",")
#         #if its a (csv) file
#         if os.path.isfile(arrayORcsv):
#             conMatrix=np.loadtxt(arrayORcsv,delimiter=",")
        
#     #load or parse input index and label files
#     if isinstance(indexIn,dict):
#         indexDict=indexIn
#     elif isinstance(indexIn,str):
#         with open(indexIn, "r") as indexJson:
#             indexDict = json.load(indexJson)    
    
#     if isinstance(labelIn,dict):
#         labelDict=labelIn
#     elif isinstance(labelIn,str):
#         with open(labelIn, "r") as labelJson:
#             labelDict = json.load(labelJson)
    
#     #prep outfile paths
#     outputDirectory = "output"
#     if not os.path.exists(outputDirectory):
#     		os.makedirs(outputDirectory)
#     outputFile = os.path.join(outputDirectory,"network.json.gz")
    
#     if not os.path.exists(outputDirectory):
#     		os.makedirs(outputDirectory)
    
#     matrices = []
#     networkProperties = []
#     labels = []
#     #should only be one, but ok
#     #if it's a singleton dict, convert it to a list
#     if isinstance(indexDict,dict):
#         indexDict=[indexDict]
    
#     for entry in indexDict:
#     	entryFilename = entry["filename"]
#     	networkPropertiesDictionary = entry.copy()
    	
#     	label = ""
#     	if("name" in entry):
#     		label = entry["name"]
#     		del networkPropertiesDictionary["name"]
#     	del networkPropertiesDictionary["filename"]
    
#     	#adjacencyMatrix = loadCSVMatrix(os.path.join(CSVDirectory, entryFilename))
#     	matrices.append(conMatrix)
    
    
#     	if(len(labelDict)>len(conMatrix)):
#     		for key,value in labelDict[0].items():
#     			networkPropertiesDictionary["extra_"+key] = value
#     		labelDataHasHeader = True
    	
#     	networkProperties.append(networkPropertiesDictionary)
#     	labels.append(label)
#     #shouldn't this be computed on a per matrix basis in case they are different?
#     #https://github.com/filipinascimento/bl-conmat2network/blob/977d5d2a8a32b3dbef0ffea28e926e193b2657d7/main.py#L67-L76
#     #whatever
    
#     if(len(labelDict)>len(matrices[0])):
#     	labelDict = labelDict[1:]
    
#     nodesProperties = OrderedDict()
#     if(len(labelDict)>0):
#     		for nodeIndex,labelInformation in enumerate(labelDict):
#     			for key,value in labelInformation.items():
#     				if(key not in nodesProperties):
#     					nodesProperties[key] = OrderedDict()
#     				nodesProperties[key][nodeIndex] = value
    
    
#     #stolen from:
#     #https://github.com/filipinascimento/jgf/blob/6558ba152937bdb4814190e4c1a89c7ade5bdfaf/jgf/conmat.py#L140
#     #because I don't need an additional packaged dependancy to just do one thing
#     def _JGFAddGraphAttribute(graph,key,value):
#     	if(key in _nonMetaGraphAttributes):
#     		graph[key] = value
#     	else:
#     		if ("metadata" not in graph):
#     			graph["metadata"] = OrderedDict()
#     		graph["metadata"][key] = value
    
#     def save1(graphs,filename="",compressed=None):
#     	"""
#     	Writes a list of JXNF – Json compleX Network Format – dictionaries to 
#     	a JGF(Z) – Json Graph Format (gZipped) – file.
    	
#     	Parameters
#     	----------
#     	graphs : list of dict
#     			List of dictionaries in JXNF.
#     	filename : str or file handle
#     			Path to the file or a file handle to be used as output.
#     	compressed : bool
#     			If true, the input file will be interpreted as being compressed.
#     			If not provided, this will be guessed from the file extension.
#     			Use '.jgfz' for compressed files.
#     	"""
    	
#     	shallCleanupHandler = False;
#     	if(isinstance(filename, str)):
#     		shallCleanupHandler = True
#     		if(compressed is None):
#     			fileExtension = os.path.splitext(filename)[1]
#     			if(fileExtension==".jgfz"):
#     				compressed = True
#     			else:
#     				compressed = False
#     		if(compressed):
#     			filehandler = gzip.open(filename,"wt")
#     		else:
#     			filehandler = open(filename,"wt")
#     	else:
#     		shallCleanupHandler=False
#     		if(compressed is None):
#     			compressed = False
#     		filehandler = filename
    
#     	if(not isinstance(graphs, list)):
#     		if(isinstance(graphs, dict)):
#     			graphs = [graphs]
#     		else:
#     			raise TypeError(f"Argument graphs must be of type dict or a list of dicts, not {type(graphs)}")
    
#     	exportGraphs = []
#     	for graph in graphs:
#     		exportGraphs.append(_convertToJGFEntry(graph))
    	
#     	exportJSON={}
#     	if(len(graphs)==1):
#     		exportJSON["graph"] = exportGraphs[0]
#     	else:
#     		exportJSON["graphs"] = exportGraphs
    	
#     	json.dump(exportJSON,filehandler,cls=NumpyEncoder)
#     	if(shallCleanupHandler):
#     		filehandler.close()

    
#     jgf.conmat.save(matrices,outputFile, compressed=True,
#     	label=labels,
#     	networkProperties=networkProperties,
#     	nodeProperties=nodesProperties)
    
#     #cleanup
#     #I'll not be installing unnecessary packages.
#     if os.file.exists('jgf.tar.gz'):
#         os.remove('jgf.tar.gz')
#     if os.path.isdir('jgf'):
#         os.remove('jgf')
        
            