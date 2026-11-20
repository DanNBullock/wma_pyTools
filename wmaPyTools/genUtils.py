#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:42:56 2022

@author: dan
"""

# Claude note: --------------------------------------------------------------
# Stage timing (added 2026-11).  Several functions used to time themselves
# with inline time.process_time() pairs and print the result.  process_time
# is CPU time summed over every thread of the process, so with multithreaded
# numpy / BLAS it overstates the time a user actually waits, and those prints
# could neither be turned off nor collected.  StageTimer measures wall-clock
# time with time.perf_counter (keeping the CPU time alongside it), prints
# only while timing messages are switched on (the default, so output is as
# before), and reports every stage to any collectStageTimings() block, e.g.
# for a per-stage profiling DataFrame.
# ---------------------------------------------------------------------------

import contextlib as _contextlib
import time as _time

#Claude note: module-wide switch for the timing messages; see setTimingVerbose
_TIMING_VERBOSE = True
#Claude note: record lists of the currently open collectStageTimings blocks
_ACTIVE_RECORDS = []


def setTimingVerbose(verbose):
    """
    Claude note: switch the per-stage timing messages on or off.

    Parameters
    ----------
    verbose : bool
        True (the default state) prints a message as each timed stage
        finishes, as wmaPyTools always has; False silences them, e.g. inside
        loops over many ROIs.

    Returns
    -------
    previous : bool
        The setting before this call, so it can be restored.
    """
    global _TIMING_VERBOSE
    previous = _TIMING_VERBOSE
    _TIMING_VERBOSE = bool(verbose)
    return previous


@_contextlib.contextmanager
def collectStageTimings():
    """
    Claude note: collect every StageTimer that finishes inside the block.

    Usage::

        with collectStageTimings() as records:
            segmentationTools.segmentTractMultiROI(...)
        pd.DataFrame(records)   # columns: stage, wallSeconds, cpuSeconds

    Yields
    ------
    records : list of dict
        One {'stage', 'wallSeconds', 'cpuSeconds'} dict per finished stage,
        in finishing order.  Blocks may be nested; each sees the stages that
        finish while it is open.
    """
    records = []
    _ACTIVE_RECORDS.append(records)
    try:
        yield records
    finally:
        #Claude note: remove this block's own list by identity (list.remove
        #compares by equality, and nested blocks can hold equal lists)
        for iList in range(len(_ACTIVE_RECORDS) - 1, -1, -1):
            if _ACTIVE_RECORDS[iList] is records:
                del _ACTIVE_RECORDS[iList]
                break


class StageTimer(object):
    """
    Claude note: wall-clock (and CPU) timer for one stage of a computation.

    Starts when created; finish() stops it, reports it to any open
    collectStageTimings() blocks and, if timing messages are on, prints a
    message.

    Usage::

        stageTimer = StageTimer('subsetStreamsByROIboundingBox')
        ...work...
        stageTimer.finish(lambda seconds: 'subsetting complete in ' + str(seconds))

    Parameters
    ----------
    label : str
        Name of the stage, used in the records and the default message.

    Attributes
    ----------
    wallSeconds, cpuSeconds : float or None
        Elapsed times, set by finish().
    """
    def __init__(self, label):
        self.label = label
        self.wallSeconds = None
        self.cpuSeconds = None
        self._wallStart = _time.perf_counter()
        self._cpuStart = _time.process_time()

    def finish(self, message=None):
        """
        Claude note: stop the timer, record it, and print its message.

        Parameters
        ----------
        message : callable, optional
            Function of the elapsed wall-clock seconds returning the text to
            print.  Default: "<label> complete in <seconds> seconds".

        Returns
        -------
        wallSeconds : float
        """
        self.wallSeconds = _time.perf_counter() - self._wallStart
        self.cpuSeconds = _time.process_time() - self._cpuStart
        for records in _ACTIVE_RECORDS:
            records.append({'stage': self.label, 'wallSeconds': self.wallSeconds, 'cpuSeconds': self.cpuSeconds})
        if _TIMING_VERBOSE:
            if message is None:
                print(self.label + ' complete in ' + str(self.wallSeconds) + ' seconds')
            else:
                print(message(self.wallSeconds))
        return self.wallSeconds


def searchAndLoad_configJSON(path=None):
    """
    searchAndLoad_configJSON(path=None)
    Searches for, parses, and, if possible, loads the config.json file

    Parameters
    ----------
    path : string path, optional
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
    if path is None:
        #set the current path to the cwd
        path=os.getcwd()
    
    #Claude note: figure out where the candidate config.json would be
    #Claude note: a previous version of this function (1) opened the literal relative
    #path 'config.json' rather than the path it had just computed, so it only
    #ever worked from inside the config's directory, and (2) ended with an
    #unconditional config=[] that overwrote whatever had been loaded.  So it
    #always returned [].
    #if this is a directory, look for a config.json in it
    if os.path.isdir(path):
        jsonPath=os.path.join(path,'config.json')
    #Claude note: if this is instead a file, it's only a candidate if it's named config.json
    elif os.path.isfile(path) and os.path.basename(path)=='config.json':
        jsonPath=path
    #Claude note: otherwise there's nothing to find
    else:
        jsonPath=None
    
    #Claude note: if there is a config.json at the computed location, load it
    if jsonPath is not None and os.path.isfile(jsonPath):
        with open(jsonPath) as config_json:
            config = json.load(config_json)
    #if there is not a config.json in the input path, or, as relevant, the current directory
    #output an empty config
    else:
        config=[]
    return config
    
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
            warnings.warn('Poorly formatted input detected for ' + argsIn[iArgLocations] + '\nRecommend more rigid structure in input')
        
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
            warnings.warn('Poorly formatted input detected for ' + argsIn[iArgLocations] + '\nRecommend more rigid structure in input')
        
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
    positionalArgs : list
        The ORDERED list of positional varaibles that have been extracted
    nonPositionalArgs : dict
        The keyword arguments and their values, as a dictionary.

    """   
    
    import sys
    
    #Claude note: sys.argv always contains at least the script name (argv[0]), so
    #"no arguments" means len(sys.argv)==1.  A previous version checked for
    #len==0 (unreachable) and, in the config-found branch, assigned
    #nonPositionalArgs to itself before it existed, so it could never
    #actually return a loaded config.
    
    #Claude note: if no arguments were passed, look for a config.json in the cwd
    if len(sys.argv) == 1:
        tempConfig=searchAndLoad_configJSON()
        positionalArgs=[]
        #Claude note: an empty config results in empty kwargs
        nonPositionalArgs=dict(tempConfig) if tempConfig else {}
    
    #Claude note: if there are 3 or more arguments, treat them as command line inputs
    elif len(sys.argv) >= 3:
        [positionalArgs , nonPositionalArgs]=parseInputSysArgs()
    
    #Claude note: in the case that it is merely 2 long, it could be either a path to a
    #config.json, or a single command line argument
    else:
        tempConfig=searchAndLoad_configJSON(sys.argv[1])
        #if tempconfig is empty
        if not tempConfig:
            #use the paraseInputSysArgs to get that input
            #nonpositional should be empty
            [positionalArgs , nonPositionalArgs]=parseInputSysArgs()
        
        #else, convert the temp config into nonPositionalArgs and just set positionalArgs to empty
        else:
            positionalArgs=[]
            nonPositionalArgs=dict(tempConfig)
    
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
    """
    Converts a brainlife parcellation/volume label.json into a lookup table.

    Parameters
    ----------
    pathOrDict : string path or list of dicts
        Either a path to a label.json file, or the already-loaded contents of
        one: a list of dicts, each with (at minimum) 'voxel_value' and 'name'
        keys.

    Returns
    -------
    lookupTable : pandas.DataFrame
        A two column table, with columns 'labelNumber' and 'labelNames'
    """
    import json
    import pandas as pd
    if isinstance(pathOrDict,str):
        with open(pathOrDict) as label_json:
            label_dict_list = json.load(label_json)
    else:
        #Claude note: assume it's been loaded already and is a list of dicts
        #(previously this branch didn't assign label_dict_list, so it raised a
        #NameError below)
        label_dict_list = pathOrDict
    
    #Claude note: build the rows first, then the dataframe in one go
    #(DataFrame.append was removed in pandas 2, and was O(n^2) besides)
    rowList=[[iLabels['voxel_value'],iLabels['name']] for iLabels in label_dict_list]
    lookupTable=pd.DataFrame(rowList, columns=['labelNumber' , 'labelNames'])
    
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
        
            