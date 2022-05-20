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
        priorArgInput=f'{argsIn[iArgLocations-1]=}'.split('=')[1]
        #get the input value after this equals
        nextArgInput=f'{argsIn[iArgLocations+1]=}'.split('=')[1]
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
        if argsIn[iArgLocations][0]=='=':
            warnings.warning('Poorly formatted input detected for ' + argsIn[iArgLocations][0] + '\nRecommend more rigid structure in input')
        
            #stealing from the previous example even though it isn't exactly the same
            #because the = is on the left, we can assume the thing prior was the prior input
            priorArgInput=f'{argsIn[iArgLocations-1]=}'.split('=')[1]
            #get the input value after this equals
            nextArgInput=f'{argsIn[iArgLocations]=}'.split('=')[1]
            hasEqualsDict[priorArgInput]=nextArgInput
            #record the ones we have managed with this
            argsManaged.append(iArgLocations-1)
            argsManaged.append(iArgLocations)   
            
        #last char case
        elif argsIn[iArgLocations][-1]=='=':
            warnings.warning('Poorly formatted input detected for ' + argsIn[iArgLocations][0] + '\nRecommend more rigid structure in input')
        
            #stealing from the previous example even though it isn't exactly the same
            #because the = is on the left, we can assume the thing prior was the prior input
            priorArgInput=f'{argsIn[iArgLocations]=}'.split('=')[1]
            #get the input value after this equals
            nextArgInput=f'{argsIn[iArgLocations+1]=}'.split('=')[1]
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
        priorArgInput=f'{argsIn[iArgLocations]=}'.split('=')[1]
        #but now we have to be concerned if the next location has been already
        #taken care of OR if it is another dash location
        #if its in the managed list
        if not iArgLocations+1 in argsManaged:
            #if its not in the upcoming dashLeads locations
            if not iArgLocations+1 in dashLeads_ArgLocations:
                #then you can set the next arg as the thing that follows
                nextArgInput=f'{argsIn[iArgLocations+1]=}'.split('=')[1]
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
     